#ifndef DIGITS_FIXTURE_H
#define DIGITS_FIXTURE_H
#include "tiny_arena.h"
#define DIGITS_COUNT 450
void digits_graph(size_t rows,ta_graph *graph,ta_node *nodes,size_t *output);
const float *digits_inputs(void);
const unsigned char *digits_labels(void);
size_t digits_weight_bytes(void);
#endif
