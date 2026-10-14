#include "workloads.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); } } while (0)

int main(void) {
    float weights[]={1,2,3,4},bias[]={-1,1},input[]={2,-1};
    ta_node nodes[]={
        {TA_INPUT,TA_NONE,TA_NONE,1,2,NULL,NULL,0,0},
        {TA_DENSE,0,TA_NONE,1,2,weights,bias,4,2},
        {TA_RELU,1,TA_NONE,1,2,NULL,NULL,0,0},
        {TA_ADD,2,0,1,2,NULL,NULL,0,0}};
    size_t out[]={1,3},counts[]={2};const float *inputs[]={input};
    ta_graph graph={4,2,nodes,out};ta_plan p;ta_scratch s;
    for (size_t mode=0;mode<2;++mode) {
        CHECK(ta_plan_build(&graph,(ta_mode)mode,&p,&s)==TA_OK);
        unsigned char *base=aligned_alloc(TA_ALIGNMENT,p.arena_bytes+128);CHECK(base);
        memset(base,0xa5,p.arena_bytes+128);void *arena=base+64;
        CHECK(ta_run(&graph,&p,arena,p.arena_bytes,inputs,counts,1,NULL,NULL)==TA_OK);
        const float *dense=(const float *)((unsigned char *)arena+p.offset[1]);
        const float *result=(const float *)((unsigned char *)arena+p.offset[3]);
        CHECK(dense[0]==-2 && dense[1]==1 && result[0]==2 && result[1]==0);
        for (size_t j=0;j<64;++j) CHECK(base[j]==0xa5 && base[p.arena_bytes+64+j]==0xa5);
        CHECK(ta_run(&graph,&p,arena,p.arena_bytes-1,inputs,counts,1,NULL,NULL)==TA_CAPACITY);
        CHECK(ta_run(&graph,&p,base+65,p.arena_bytes,inputs,counts,1,NULL,NULL)==TA_CAPACITY);
        input[0]=NAN;CHECK(ta_run(&graph,&p,arena,p.arena_bytes,inputs,counts,1,NULL,NULL)==TA_BAD_INPUT);input[0]=2;
        counts[0]=1;CHECK(ta_run(&graph,&p,arena,p.arena_bytes,inputs,counts,1,NULL,NULL)==TA_BAD_INPUT);counts[0]=2;
        const float *alias[]={arena};CHECK(ta_run(&graph,&p,arena,p.arena_bytes,alias,counts,1,NULL,NULL)==TA_BAD_INPUT);
        ta_plan corrupt=p;corrupt.offset[3]=corrupt.offset[2];
        CHECK(ta_plan_validate(&graph,&corrupt,&s)==TA_BAD_PLAN);
        corrupt=p;corrupt.death[1]=1;CHECK(ta_plan_validate(&graph,&corrupt,&s)==TA_BAD_PLAN);
        corrupt=p;corrupt.offset[0]=SIZE_MAX;CHECK(ta_plan_validate(&graph,&corrupt,&s)==TA_BAD_PLAN);
        nodes[3].a=3;CHECK(ta_plan_build(&graph,(ta_mode)mode,&corrupt,&s)==TA_BAD_GRAPH);nodes[3].a=2;
        nodes[1].weight_count=3;CHECK(ta_plan_build(&graph,(ta_mode)mode,&corrupt,&s)==TA_BAD_GRAPH);nodes[1].weight_count=4;
        nodes[0].rows=SIZE_MAX;CHECK(ta_plan_build(&graph,(ta_mode)mode,&corrupt,&s)==TA_OVERFLOW);nodes[0].rows=1;
        weights[0]=NAN;CHECK(ta_run(&graph,&p,arena,p.arena_bytes,inputs,counts,1,NULL,NULL)==TA_NUMERICAL);weights[0]=1;
        free(base);
    }
    for (size_t id=0;id<TA_CASES;++id) {
        ta_fixture *f=ta_fixture_create(id);CHECK(f);
        for (size_t mode=0;mode<2;++mode) {
            CHECK(ta_plan_build(ta_fixture_graph(f),(ta_mode)mode,&p,&s)==TA_OK);
            unsigned char *base=aligned_alloc(TA_ALIGNMENT,p.arena_bytes+128);CHECK(base);
            memset(base,0xa5,p.arena_bytes+128);
            CHECK(ta_run(ta_fixture_graph(f),&p,base+64,p.arena_bytes,ta_fixture_inputs(f),
                ta_fixture_input_counts(f),ta_fixture_input_count(f),NULL,NULL)==TA_OK);
            for (size_t j=0;j<64;++j) CHECK(base[j]==0xa5 && base[p.arena_bytes+64+j]==0xa5);
            free(base);
        }
        ta_fixture_free(f);
    }
    CHECK(ta_plan_build(NULL,TA_REUSE,&p,&s)==TA_BAD_GRAPH);
    CHECK(ta_plan_build(&graph,(ta_mode)2,&p,&s)==TA_BAD_PLAN);
    puts("PASS: native hand-derived outputs, pins, guards, aliases, capacities, stale plans, invalid graphs and numerics");
    return 0;
}
