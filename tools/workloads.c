#include "workloads.h"
#include <stdint.h>
#include <stdlib.h>

/* Benchmark-only allocation/parameters, outside planner and all timed regions. */
struct ta_fixture {
    ta_graph graph;
    ta_node nodes[TA_MAX_NODES];
    size_t outputs[4], input_counts[4], input_count;
    const float *inputs[4];
    float parameters[70000], input_data[2048];
    size_t used;
    uint32_t rng;
};
const char *ta_fixture_name(size_t id) {
    static const char *names[] = {"chain", "residual", "fork_outputs", "residual_stack"};
    return id<TA_CASES ? names[id] : NULL;
}
static float *parameter(ta_fixture *f, size_t count) {
    if (count>70000-f->used) return NULL;
    float *out=f->parameters+f->used;
    for (size_t i=0;i<count;++i) {
        f->rng=f->rng*1664525u+1013904223u;
        out[i]=(float)((int)((f->rng>>8)%101)-50)*0.001f;
    }
    f->used+=count;
    return out;
}
static size_t add_node(ta_fixture *f, ta_op op, size_t a, size_t b, size_t cols) {
    size_t id=f->graph.count++;
    ta_node *n=&f->nodes[id];
    n->op=op;n->a=a;n->b=b;n->rows=f->nodes[a].rows;n->cols=cols;
    if (op==TA_DENSE) {
        n->weight_count=f->nodes[a].cols*cols;n->bias_count=cols;
        n->weights=parameter(f,n->weight_count);n->bias=parameter(f,cols);
    }
    return id;
}
ta_fixture *ta_fixture_create(size_t id) {
    if (id>=TA_CASES) return NULL;
    ta_fixture *f=calloc(1,sizeof(*f));
    if (!f) return NULL;
    f->rng=20261014u+(uint32_t)id;
    f->graph.nodes=f->nodes;f->graph.outputs=f->outputs;f->graph.count=1;
    size_t rows=id==1 ? 4 : id==2 ? 3 : 1;
    size_t cols=id==0 ? 64 : id==2 ? 17 : 32;
    f->nodes[0]=(ta_node){TA_INPUT,TA_NONE,TA_NONE,rows,cols,NULL,NULL,0,0};
    f->input_count=1;f->input_counts[0]=rows*cols;f->inputs[0]=f->input_data;
    for (size_t i=0;i<rows*cols;++i)
        f->input_data[i]=(float)((int)((i*17+id*11)%31)-15)/32.0f;
    size_t current=0;
    if (id==0) {
        const size_t widths[]={96,64,32,16};
        for (size_t i=0;i<4;++i) {
            current=add_node(f,TA_DENSE,current,TA_NONE,widths[i]);
            current=add_node(f,TA_RELU,current,TA_NONE,widths[i]);
        }
    } else if (id==1) {
        current=add_node(f,TA_DENSE,0,TA_NONE,64);
        current=add_node(f,TA_RELU,current,TA_NONE,64);
        current=add_node(f,TA_DENSE,current,TA_NONE,32);
        current=add_node(f,TA_ADD,current,0,32);
        current=add_node(f,TA_RELU,current,TA_NONE,32);
        current=add_node(f,TA_DENSE,current,TA_NONE,16);
    } else if (id==2) {
        current=add_node(f,TA_DENSE,0,TA_NONE,29);
        current=add_node(f,TA_RELU,current,TA_NONE,29);
        f->outputs[0]=add_node(f,TA_DENSE,current,TA_NONE,7);
        current=add_node(f,TA_DENSE,current,TA_NONE,23);
        current=add_node(f,TA_RELU,current,TA_NONE,23);
        current=add_node(f,TA_DENSE,current,TA_NONE,11);
        f->graph.output_count=2;f->outputs[1]=current;
        return f;
    } else {
        for (size_t block=0;block<16;++block) {
            size_t skip=current;
            current=add_node(f,TA_DENSE,current,TA_NONE,64);
            current=add_node(f,TA_RELU,current,TA_NONE,64);
            current=add_node(f,TA_DENSE,current,TA_NONE,32);
            current=add_node(f,TA_ADD,current,skip,32);
        }
    }
    f->graph.output_count=1;f->outputs[0]=current;
    return f;
}
void ta_fixture_free(ta_fixture *f) { free(f); }
const ta_graph *ta_fixture_graph(const ta_fixture *f) { return f ? &f->graph : NULL; }
const float *const *ta_fixture_inputs(const ta_fixture *f) { return f ? f->inputs : NULL; }
const size_t *ta_fixture_input_counts(const ta_fixture *f) { return f ? f->input_counts : NULL; }
size_t ta_fixture_input_count(const ta_fixture *f) { return f ? f->input_count : 0; }
