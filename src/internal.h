#ifndef TA_INTERNAL_H
#define TA_INTERNAL_H
#include "tiny_arena.h"
#include <stdint.h>
static inline int ta_live_overlap(size_t a, size_t b, const ta_scratch *s) {
    return s->birth[a] <= s->death[b] && s->birth[b] <= s->death[a];
}
static inline int ta_memory_overlap(size_t a, size_t na, size_t b, size_t nb) {
    /* Called only after all ends have passed overflow checks. */
    return a < b+nb && b < a+na;
}
#endif
