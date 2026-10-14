#if defined(__APPLE__)
#define _DARWIN_C_SOURCE 1
#else
#define _POSIX_C_SOURCE 200809L
#endif
#include "workloads.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__APPLE__)
#include <malloc/malloc.h>
#elif defined(__GLIBC__)
#include <malloc.h>
#endif

static void need(int ok, const char *message) {
    if (!ok) { fprintf(stderr,"FAIL: %s\n",message); exit(1); }
}
static uint64_t now_ns(void) {
    struct timespec ts;
    need(clock_gettime(CLOCK_MONOTONIC,&ts)==0,"monotonic clock");
    return (uint64_t)ts.tv_sec*UINT64_C(1000000000)+(uint64_t)ts.tv_nsec;
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
typedef struct { const ta_plan *plan; const unsigned char *base; size_t seen, high; int valid; } trace;
typedef struct { size_t sample, graph; ta_mode mode; uint64_t elapsed; } timing;
static void observe(size_t id,const float *data,size_t count,void *context) {
    trace *t=context;
    size_t offset=(size_t)((const unsigned char *)data-t->base);
    if (id!=t->seen || offset!=t->plan->offset[id]) t->valid=0;
    ++t->seen;
    if (offset+count*sizeof(float)>t->high) t->high=offset+count*sizeof(float);
}
static void array(const size_t *values,size_t n) {
    putchar('[');for (size_t i=0;i<n;++i) printf("%s%zu",i?",":"",values[i]);putchar(']');
}
static void describe(const ta_fixture *f,ta_mode mode) {
    const ta_graph *g=ta_fixture_graph(f);
    ta_plan p;ta_scratch s;
    need(ta_plan_build(g,mode,&p,&s)==TA_OK,"describe plan");
    size_t requested=p.arena_bytes+2*TA_ALIGNMENT;
    unsigned char *allocation=aligned_alloc(TA_ALIGNMENT,requested);
    need(allocation!=NULL,"arena allocation");
    memset(allocation,0xa5,requested);
    unsigned char *arena=allocation+TA_ALIGNMENT;
    trace t={&p,arena,0,0,1};
    need(ta_run(g,&p,arena,p.arena_bytes,ta_fixture_inputs(f),ta_fixture_input_counts(f),
        ta_fixture_input_count(f),observe,&t)==TA_OK,"production run");
    for (size_t i=0;i<TA_ALIGNMENT;++i) need(allocation[i]==0xa5 &&
        allocation[requested-TA_ALIGNMENT+i]==0xa5,"arena boundary guard");
    need(t.valid && t.seen==g->count && t.high<=p.arena_bytes,"actual tensor address trace");
    printf("{\"arena_bytes\":%zu,\"live_lower_bound_bytes\":%zu,\"allocation_requested_bytes\":%zu,"
           "\"allocator_usable_bytes\":%zu,\"guard_bytes\":%d,\"observed_highwater_bytes\":%zu,"
           "\"observed_tensors\":%zu,\"guard_ok\":true,\"offset\":",p.arena_bytes,p.live_lower_bound_bytes,
           requested,usable(allocation),2*TA_ALIGNMENT,t.high,t.seen);
    array(p.offset,g->count);printf(",\"bytes\":");array(p.bytes,g->count);
    printf(",\"birth\":");array(p.birth,g->count);printf(",\"death\":");array(p.death,g->count);
    printf(",\"outputs\":[");
    for (size_t out=0;out<g->output_count;++out) {
        size_t id=g->outputs[out],count=g->nodes[id].rows*g->nodes[id].cols;
        const float *data=(const float *)(arena+p.offset[id]);
        printf("%s[",out?",":"");
        for (size_t j=0;j<count;++j) printf("%s%.9g",j?",":"",(double)data[j]);
        putchar(']');
    }
    printf("]}");free(allocation);
}
int main(int argc,char **argv) {
    size_t run=0,samples=20,repetitions=1024;
    if (argc==4) {
        char *end;
        run=(size_t)strtoull(argv[1],&end,10);need(*end=='\0',"run number");
        samples=(size_t)strtoull(argv[2],&end,10);need(*end=='\0',"sample number");
        repetitions=(size_t)strtoull(argv[3],&end,10);need(*end=='\0',"repetition number");
    } else need(argc==1,"expected run samples repetitions");
    need(run<24 && samples>=1 && samples<=100 && repetitions>=128 && repetitions<=4096,"bounds");
    struct timespec resolution;
    need(clock_getres(CLOCK_MONOTONIC,&resolution)==0,"clock resolution");
    uint64_t resolution_ns=(uint64_t)resolution.tv_sec*UINT64_C(1000000000)+(uint64_t)resolution.tv_nsec;
    ta_fixture *fixtures[TA_CASES];
    for (size_t i=0;i<TA_CASES;++i) {
        fixtures[i]=ta_fixture_create(i);need(fixtures[i]!=NULL,"fixture");
    }
    ta_plan p;ta_scratch s;volatile size_t sink=0;
    for (size_t c=0;c<TA_CASES;++c) for (size_t mode=0;mode<2;++mode)
        for (size_t warm=0;warm<8;++warm)
            need(ta_plan_build(ta_fixture_graph(fixtures[c]),(ta_mode)mode,&p,&s)==TA_OK,"warm plan");
    printf("{\"schema\":1,\"run\":%zu,\"samples\":%zu,\"repetitions\":%zu,"
        "\"plan_metadata_bytes\":%zu,\"planner_scratch_bytes\":%zu,\"clock_resolution_ns\":%" PRIu64 ",\"graphs\":[",run,samples,repetitions,
        sizeof(ta_plan),sizeof(ta_scratch),resolution_ns);
    for (size_t c=0;c<TA_CASES;++c) {
        const ta_graph *g=ta_fixture_graph(fixtures[c]);
        printf("%s{\"name\":\"%s\",\"node_count\":%zu,\"graph_descriptor_bytes\":%zu,\"separate\":",
               c?",":"",ta_fixture_name(c),g->count,sizeof(*g)+g->count*sizeof(ta_node)+g->output_count*sizeof(size_t));
        describe(fixtures[c],TA_SEPARATE);printf(",\"reuse\":");describe(fixtures[c],TA_REUSE);printf("}");
    }
    timing timings[100*TA_CASES*2];size_t measured=0;
    for (size_t sample=0;sample<samples;++sample) for (size_t order=0;order<TA_CASES;++order) {
        size_t c=(order+sample+run)%TA_CASES;
        for (size_t position=0;position<2;++position) {
            ta_mode mode=(ta_mode)((position+run+sample+c)%2);
            uint64_t start=now_ns();
            for (size_t repetition=0;repetition<repetitions;++repetition) {
                need(ta_plan_build(ta_fixture_graph(fixtures[c]),mode,&p,&s)==TA_OK,"timed plan");
                sink^=p.arena_bytes;
            }
            uint64_t elapsed=now_ns()-start;need(elapsed>0,"positive duration");
            timings[measured++]=(timing){sample,c,mode,elapsed};
        }
    }
    printf("],\"raw\":[");
    for (size_t i=0;i<measured;++i) {
        const timing *t=&timings[i];
        printf("%s{\"sample\":%zu,\"graph\":\"%s\",\"mode\":\"%s\",\"elapsed_ns\":%" PRIu64 "}",
            i?",":"",t->sample,ta_fixture_name(t->graph),t->mode==TA_SEPARATE?"separate":"reuse",t->elapsed);
    }
    printf("],\"sink\":%zu}\n",(size_t)sink);
    for (size_t i=0;i<TA_CASES;++i) ta_fixture_free(fixtures[i]);
    return 0;
}
