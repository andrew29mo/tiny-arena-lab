#include "tiny_arena.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

ta_status ta_run(const ta_graph *g, const ta_plan *p, void *arena, size_t capacity,
                 const float *const *inputs, const size_t *input_counts, size_t input_count,
                 ta_observer observer, void *context) {
    ta_scratch scratch;
    ta_status status=ta_plan_validate(g,p,&scratch);
    if (status!=TA_OK) return status;
    if (!arena || (uintptr_t)arena%TA_ALIGNMENT || capacity<p->arena_bytes) return TA_CAPACITY;
    if (!inputs || !input_counts || (uintptr_t)arena>UINTPTR_MAX-p->arena_bytes) return TA_BAD_INPUT;
    size_t expected_inputs=0;
    while (expected_inputs<g->count && g->nodes[expected_inputs].op==TA_INPUT) ++expected_inputs;
    if (input_count!=expected_inputs) return TA_BAD_INPUT;
    /* Validate all external inputs before writing arena; no overlapping input range. */
    for (size_t i=0;i<input_count;++i) {
        size_t count=g->nodes[i].rows*g->nodes[i].cols, bytes=count*sizeof(float);
        uintptr_t start=(uintptr_t)inputs[i],base=(uintptr_t)arena;
        if (!inputs[i] || input_counts[i]!=count || start>UINTPTR_MAX-bytes ||
            (start<base+p->arena_bytes && base<start+bytes)) return TA_BAD_INPUT;
        for (size_t j=0;j<count;++j) if (!isfinite(inputs[i][j])) return TA_BAD_INPUT;
    }
    for (size_t i=0;i<g->count;++i) {
        const ta_node *n=&g->nodes[i];
        size_t count=n->rows*n->cols;
        float *y=(float *)((unsigned char *)arena+p->offset[i]);
        if (n->op==TA_INPUT) memcpy(y,inputs[i],count*sizeof(float));
        else {
            const float *a=(const float *)((unsigned char *)arena+p->offset[n->a]);
            if (n->op==TA_DENSE) {
                size_t in=g->nodes[n->a].cols;
                for (size_t row=0;row<n->rows;++row) for (size_t col=0;col<n->cols;++col) {
                    float sum=n->bias[col];
                    for (size_t k=0;k<in;++k) sum+=a[row*in+k]*n->weights[k*n->cols+col];
                    y[row*n->cols+col]=sum;
                }
            } else if (n->op==TA_RELU) {
                for (size_t j=0;j<count;++j) y[j]=fmaxf(0.0f,a[j]);
            } else {
                const float *b=(const float *)((unsigned char *)arena+p->offset[n->b]);
                for (size_t j=0;j<count;++j) y[j]=a[j]+b[j];
            }
        }
        for (size_t j=0;j<count;++j) if (!isfinite(y[j])) return TA_NUMERICAL;
        if (observer) observer(i,y,count,context);
    }
    return TA_OK;
}
