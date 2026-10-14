#include "internal.h"
#include <string.h>

ta_status ta_plan_validate(const ta_graph *g, const ta_plan *p, ta_scratch *s) {
    if (!p || !s || (p->mode!=TA_SEPARATE && p->mode!=TA_REUSE)) return TA_BAD_PLAN;
    size_t lower;
    ta_status status=ta_analyze(g,s,&lower);
    if (status!=TA_OK) return status;
    if (p->count!=g->count || p->live_lower_bound_bytes!=lower) return TA_BAD_PLAN;
    size_t end=0, sequential=0;
    for (size_t i=0;i<g->count;++i) {
        if (p->bytes[i]!=s->bytes[i] || p->birth[i]!=s->birth[i] || p->death[i]!=s->death[i] ||
            p->offset[i]%TA_ALIGNMENT || p->bytes[i]>SIZE_MAX-p->offset[i]) return TA_BAD_PLAN;
        size_t top=p->offset[i]+p->bytes[i];
        if (top>p->arena_bytes) return TA_BAD_PLAN;
        if (top>end) end=top;
        if (p->mode==TA_SEPARATE && p->offset[i]!=sequential) return TA_BAD_PLAN;
        if (p->mode==TA_SEPARATE) {
            if (p->bytes[i]>SIZE_MAX-sequential) return TA_BAD_PLAN;
            sequential+=p->bytes[i];
        }
        for (size_t j=0;j<i;++j) if (ta_live_overlap(i,j,s) &&
            ta_memory_overlap(p->offset[i],p->bytes[i],p->offset[j],p->bytes[j])) return TA_BAD_PLAN;
    }
    return p->arena_bytes==end && end>=lower ? TA_OK : TA_BAD_PLAN;
}

ta_status ta_plan_build(const ta_graph *g, ta_mode mode, ta_plan *p, ta_scratch *s) {
    if (!p || !s || (mode!=TA_SEPARATE && mode!=TA_REUSE)) return TA_BAD_PLAN;
    size_t lower;
    ta_status status=ta_analyze(g,s,&lower);
    if (status!=TA_OK) return status;
    memset(p,0,sizeof(*p));
    p->count=g->count; p->mode=mode; p->live_lower_bound_bytes=lower;
    for (size_t i=0;i<g->count;++i) {
        p->bytes[i]=s->bytes[i];p->birth[i]=s->birth[i];p->death[i]=s->death[i];
    }
    if (mode==TA_REUSE) {
        /* Stable descending-size insertion sort, then lowest feasible aligned offset.
         * This heuristic does not claim a globally optimal static allocation. */
        for (size_t i=1;i<g->count;++i) {
            size_t id=s->order[i],j=i;
            while (j && s->bytes[s->order[j-1]]<s->bytes[id]) {
                s->order[j]=s->order[j-1];--j;
            }
            s->order[j]=id;
        }
    }
    for (size_t k=0;k<g->count;++k) {
        size_t id=mode==TA_SEPARATE ? k : s->order[k];
        size_t offset=mode==TA_SEPARATE ? p->arena_bytes : 0;
        for (;;) {
            if (p->bytes[id]>SIZE_MAX-offset) return TA_OVERFLOW;
            size_t next=offset;
            if (mode==TA_REUSE) for (size_t j=0;j<k;++j) {
                size_t other=s->order[j];
                if (ta_live_overlap(id,other,s) &&
                    ta_memory_overlap(offset,p->bytes[id],p->offset[other],p->bytes[other])) {
                    size_t top=p->offset[other]+p->bytes[other];
                    if (top>next) next=top;
                }
            }
            if (next==offset) break;
            offset=next;
        }
        p->offset[id]=offset;
        size_t end=offset+p->bytes[id];
        if (end>p->arena_bytes) p->arena_bytes=end;
    }
    return ta_plan_validate(g,p,s);
}
