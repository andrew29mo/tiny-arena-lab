#ifndef TA_WORKLOADS_H
#define TA_WORKLOADS_H
#include "tiny_arena.h"
#define TA_CASES 4
typedef struct ta_fixture ta_fixture;
const char *ta_fixture_name(size_t id);
ta_fixture *ta_fixture_create(size_t id);
void ta_fixture_free(ta_fixture *f);
const ta_graph *ta_fixture_graph(const ta_fixture *f);
const float *const *ta_fixture_inputs(const ta_fixture *f);
const size_t *ta_fixture_input_counts(const ta_fixture *f);
size_t ta_fixture_input_count(const ta_fixture *f);
#endif
