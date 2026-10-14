#if defined(__APPLE__)
#define _DARWIN_C_SOURCE 1
#else
#define _POSIX_C_SOURCE 200809L
#endif
#include "fixture.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__APPLE__)
#include <malloc/malloc.h>
#elif defined(__GLIBC__)
#include <malloc.h>
#endif
static void need(int ok,const char *why) { if (!ok) { fprintf(stderr,"FAIL: %s\n",why);exit(1); } }
static uint64_t now(void) {
    struct timespec t;need(clock_gettime(CLOCK_MONOTONIC,&t)==0,"clock");
    return (uint64_t)t.tv_sec*1000000000u+(uint64_t)t.tv_nsec;
}
static size_t usable(void *p) {
#if defined(__APPLE__)
    return malloc_size(p);
#elif defined(__GLIBC__)
    return malloc_usable_size(p);
#else
    (void)p;return 0;
#endif
}
static int guarded(const unsigned char *p,size_t n) {
    for (size_t i=0;i<64;++i) if(p[i]!=0xa5 || p[n+64+i]!=0xa5) return 0;
    return 1;
}
static void observed(size_t id,const float *data,size_t count,void *context) {
    (void)id;(void)data;(void)count;*(size_t *)context+=1;
}
/* Includes batch packing/zero-padding, input copy, validation, kernels, finite
 * checks and argmax/checksum. No Python/allocator/observer/JSON in timed passes. */
static size_t pass(const ta_graph *g,const ta_plan *p,unsigned char *arena,size_t cap,
                   float *saved,int *predictions) {
    size_t batch=g->nodes[0].rows,correct=0;
    float input[batch*64];const float *inputs[]={input};size_t counts[]={batch*64};
    for(size_t start=0;start<DIGITS_COUNT;start+=batch) {
        size_t rows=DIGITS_COUNT-start<batch?DIGITS_COUNT-start:batch;
        memset(input,0,sizeof(input));memcpy(input,digits_inputs()+start*64,rows*64*sizeof(float));
        need(ta_run(g,p,arena,cap,inputs,counts,1,NULL,NULL)==TA_OK,"production inference");
        const float *y=(const float *)(arena+p->offset[3]);
        for(size_t i=0;i<rows;++i) {
            size_t best=0;
            for(size_t j=1;j<10;++j) if(y[i*10+j]>y[i*10+best]) best=j;
            correct+=best==digits_labels()[start+i];
            if(saved) memcpy(saved+(start+i)*10,y+i*10,10*sizeof(float));
            if(predictions) predictions[start+i]=(int)best;
        }
    }
    return correct;
}
typedef struct {size_t sample,batch,mode,kind;uint64_t elapsed;} timing;
int main(int argc,char **argv) {
    need(argc==3,"run samples");char *end;
    size_t run=(size_t)strtoull(argv[1],&end,10);need(*end=='\0' && run<24,"run");
    size_t samples=(size_t)strtoull(argv[2],&end,10);need(*end=='\0' && samples>=1 && samples<=20,"samples");
    struct timespec res;need(clock_getres(CLOCK_MONOTONIC,&res)==0,"resolution");
    printf("{\"run\":%zu,\"samples\":%zu,\"planning_repetitions\":1024,\"inference_repetitions\":4,\"clock_resolution_ns\":%" PRIu64 ",\"cases\":[",run,samples,(uint64_t)res.tv_sec*1000000000u+(uint64_t)res.tv_nsec);
    timing timings[20*2*2*2];size_t nt=0;volatile size_t sink=0;
    for(size_t order=0;order<2;++order) {
        size_t batch=(order+run)%2?16:1,budget=batch==1?512:7168;
        ta_graph g;ta_node nodes[4];size_t output;digits_graph(batch,&g,nodes,&output);
        ta_plan plans[2];ta_scratch scratch;unsigned char *allocations[2],*arenas[2];
        unsigned char *limited=aligned_alloc(64,budget+128);need(limited!=NULL,"budget allocation");
        printf("%s{\"batch\":%zu,\"budget_bytes\":%zu,\"budget_request_bytes\":%zu,\"budget_usable_bytes\":%zu,\"plan_bytes\":%zu,\"scratch_bytes\":%zu,\"weight_bytes\":%zu,\"graph_bytes\":%zu,\"input_buffer_bytes\":%zu,\"modes\":[",order?",":"",batch,budget,budget+128,usable(limited),sizeof(ta_plan),sizeof(ta_scratch),digits_weight_bytes(),sizeof(g)+sizeof(nodes)+sizeof(output),batch*64*sizeof(float));
        for(size_t mode=0;mode<2;++mode) {
            ta_plan *p=&plans[mode];need(ta_plan_build(&g,(ta_mode)mode,p,&scratch)==TA_OK,"plan");
            memset(limited,0xa5,budget+128);float input[batch*64];memcpy(input,digits_inputs(),sizeof(input));
            const float *inputs[]={input};size_t counts[]={batch*64},seen=0;
            ta_status status=ta_run(&g,p,limited+64,budget,inputs,counts,1,observed,&seen);
            if(mode==0) {
                need(status==TA_CAPACITY && seen==0,"baseline capacity rejection before execution");
                for(size_t i=0;i<budget+128;++i) need(limited[i]==0xa5,"rejected run wrote memory");
            } else need(status==TA_OK && seen==4,"reuse fits budget");
            need(guarded(limited,budget),"budget guard");
            float budget_logits[DIGITS_COUNT*10];
            size_t budget_correct=mode==1?pass(&g,p,limited+64,budget,budget_logits,NULL):0;
            need(guarded(limited,budget),"whole-test budget guard");
            allocations[mode]=aligned_alloc(64,p->arena_bytes+128);need(allocations[mode]!=NULL,"timing arena");
            memset(allocations[mode],0xa5,p->arena_bytes+128);arenas[mode]=allocations[mode]+64;
            float logits[DIGITS_COUNT*10];int predictions[DIGITS_COUNT];
            size_t correct=pass(&g,p,arenas[mode],p->arena_bytes,logits,predictions);
            need(guarded(allocations[mode],p->arena_bytes),"timing arena guard");
            if(mode==1) need(budget_correct==correct && memcmp(budget_logits,logits,sizeof(logits))==0,"budget outputs mismatch");
            size_t fixed=sizeof(ta_plan)+sizeof(ta_scratch)+digits_weight_bytes()+sizeof(g)+sizeof(nodes)+sizeof(output)+sizeof(input);
            printf("%s{\"mode\":%zu,\"arena_bytes\":%zu,\"allocation_request_bytes\":%zu,\"allocator_usable_bytes\":%zu,\"budget_status\":%d,\"budget_observed_tensors\":%zu,\"rejection_untouched\":%s,\"guard_ok\":true,\"correct\":%zu,\"budget_correct\":%zu,\"accounted_bytes\":%zu,\"accounted_budget_bytes\":%zu,\"predictions\":[",mode?",":"",mode,p->arena_bytes,p->arena_bytes+128,usable(allocations[mode]),status,seen,mode==0?"true":"false",correct,budget_correct,fixed+p->arena_bytes,fixed+budget);
            for(size_t i=0;i<DIGITS_COUNT;++i) printf("%s%d",i?",":"",predictions[i]);
            printf("],\"logits\":[");for(size_t i=0;i<DIGITS_COUNT*10;++i) printf("%s%.9g",i?",":"",(double)logits[i]);printf("]}");
            for(size_t warm=0;warm<2;++warm) sink+=pass(&g,p,arenas[mode],p->arena_bytes,NULL,NULL);
        }
        printf("]}");
        for(size_t sample=0;sample<samples;++sample) for(size_t pos=0;pos<2;++pos) {
            size_t mode=(pos+sample+run)%2;
            for(size_t kind=0;kind<2;++kind) {
                uint64_t begin=now();
                if(kind==0) for(size_t rep=0;rep<1024;++rep) {
                    ta_plan p;need(ta_plan_build(&g,(ta_mode)mode,&p,&scratch)==TA_OK,"timed plan");sink^=p.arena_bytes;
                } else for(size_t rep=0;rep<4;++rep) sink+=pass(&g,&plans[mode],arenas[mode],plans[mode].arena_bytes,NULL,NULL);
                uint64_t elapsed=now()-begin;need(elapsed>0,"positive time");
                timings[nt++]=(timing){sample,batch,mode,kind,elapsed};
            }
        }
        for(size_t mode=0;mode<2;++mode) {need(guarded(allocations[mode],plans[mode].arena_bytes),"timed guards");free(allocations[mode]);}
        free(limited);
    }
    printf("],\"raw\":[");for(size_t i=0;i<nt;++i) {
        timing t=timings[i];printf("%s{\"sample\":%zu,\"batch\":%zu,\"mode\":%zu,\"kind\":%zu,\"elapsed_ns\":%" PRIu64 "}",i?",":"",t.sample,t.batch,t.mode,t.kind,t.elapsed);
    }
    printf("],\"sink\":%zu}\n",(size_t)sink);return 0;
}
