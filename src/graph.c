#include "internal.h"

ta_status ta_analyze(const ta_graph *g, ta_scratch *s, size_t *lower_bound) {
    if (!g || !s || !lower_bound || !g->nodes || !g->outputs || g->count==0 ||
        g->count>TA_MAX_NODES || g->output_count==0 || g->output_count>g->count)
        return TA_BAD_GRAPH;
    int past_inputs=0;
    for (size_t i=0;i<g->count;++i) {
        const ta_node *n=&g->nodes[i];
        if (!n->rows || !n->cols) return TA_BAD_GRAPH;
        if (n->rows>SIZE_MAX/n->cols || n->rows*n->cols>SIZE_MAX/sizeof(float)) return TA_OVERFLOW;
        size_t bytes=n->rows*n->cols*sizeof(float);
        if (bytes>SIZE_MAX-(TA_ALIGNMENT-1)) return TA_OVERFLOW;
        s->bytes[i]=(bytes+TA_ALIGNMENT-1)/TA_ALIGNMENT*TA_ALIGNMENT;
        s->birth[i]=n->op==TA_INPUT ? 0 : i;
        s->death[i]=s->birth[i];
        s->order[i]=i;
        if (n->op!=TA_DENSE && (n->weights || n->bias || n->weight_count || n->bias_count))
            return TA_BAD_GRAPH;
        if (n->op==TA_INPUT) {
            if (past_inputs || n->a!=TA_NONE || n->b!=TA_NONE) return TA_BAD_GRAPH;
            continue;
        }
        past_inputs=1;
        if (!i || n->a>=i) return TA_BAD_GRAPH;
        const ta_node *a=&g->nodes[n->a];
        if (n->rows!=a->rows) return TA_BAD_GRAPH;
        if (n->op==TA_DENSE) {
            if (a->cols>SIZE_MAX/n->cols) return TA_OVERFLOW;
            if (a->cols*n->cols>SIZE_MAX/sizeof(float)) return TA_OVERFLOW;
            if (n->b!=TA_NONE || !n->weights || !n->bias ||
                n->weight_count!=a->cols*n->cols || n->bias_count!=n->cols) return TA_BAD_GRAPH;
        } else if (n->op==TA_RELU) {
            if (n->b!=TA_NONE || n->cols!=a->cols) return TA_BAD_GRAPH;
        } else if (n->op==TA_ADD) {
            if (n->b>=i || n->cols!=a->cols || g->nodes[n->b].rows!=n->rows ||
                g->nodes[n->b].cols!=n->cols) return TA_BAD_GRAPH;
        } else return TA_BAD_GRAPH;
    }
    if (g->nodes[0].op!=TA_INPUT) return TA_BAD_GRAPH;
    for (size_t i=0;i<g->count;++i) {
        const ta_node *n=&g->nodes[i];
        if (n->op==TA_INPUT) continue;
        if (s->death[n->a]<i) s->death[n->a]=i;
        if (n->op==TA_ADD && s->death[n->b]<i) s->death[n->b]=i;
    }
    for (size_t i=0;i<g->output_count;++i) {
        if (g->outputs[i]>=g->count) return TA_BAD_GRAPH;
        for (size_t j=0;j<i;++j) if (g->outputs[i]==g->outputs[j]) return TA_BAD_GRAPH;
        s->death[g->outputs[i]]=g->count; /* pin all outputs */
    }
    *lower_bound=0;
    for (size_t step=0;step<=g->count;++step) {
        size_t live=0;
        for (size_t i=0;i<g->count;++i) if (s->birth[i]<=step && step<=s->death[i]) {
            if (s->bytes[i]>SIZE_MAX-live) return TA_OVERFLOW;
            live+=s->bytes[i];
        }
        if (live>*lower_bound) *lower_bound=live;
    }
    return TA_OK;
}
