#ifndef TINY_ARENA_H
#define TINY_ARENA_H
#include <stddef.h>

#define TA_MAX_NODES 128
#define TA_ALIGNMENT 64
#define TA_NONE ((size_t)-1)

typedef enum { TA_OK=0, TA_BAD_GRAPH=1, TA_OVERFLOW=2, TA_BAD_PLAN=3,
               TA_CAPACITY=4, TA_BAD_INPUT=5, TA_NUMERICAL=6 } ta_status;
typedef enum { TA_INPUT=0, TA_DENSE=1, TA_RELU=2, TA_ADD=3 } ta_op;
typedef enum { TA_SEPARATE=0, TA_REUSE=1 } ta_mode;

typedef struct {
    ta_op op;
    size_t a, b, rows, cols;
    const float *weights, *bias;
    size_t weight_count, bias_count;
} ta_node;
typedef struct {
    size_t count, output_count;
    const ta_node *nodes;
    const size_t *outputs;
} ta_graph;
typedef struct {
    ta_mode mode;
    size_t count, arena_bytes, live_lower_bound_bytes;
    size_t offset[TA_MAX_NODES], bytes[TA_MAX_NODES];
    size_t birth[TA_MAX_NODES], death[TA_MAX_NODES];
} ta_plan;
typedef struct {
    size_t bytes[TA_MAX_NODES], birth[TA_MAX_NODES];
    size_t death[TA_MAX_NODES], order[TA_MAX_NODES];
} ta_scratch;

/* Fixed topological order; inputs form a nonempty prefix. Static float32 shapes.
 * Lifetimes include both producer and last consumer step. Outputs stay live
 * through graph completion. No in-place operation or dead-code elimination.
 * Weights/bias and graph descriptors are borrowed, immutable, and outside arena.
 * All pointer/count claims must be truthful; raw pointer validity is a caller duty.
 * Planner and validator use no heap. Scratch/plan/graph must not overlap.
 */
ta_status ta_analyze(const ta_graph *g, ta_scratch *s, size_t *lower_bound);
ta_status ta_plan_build(const ta_graph *g, ta_mode mode, ta_plan *p, ta_scratch *s);
ta_status ta_plan_validate(const ta_graph *g, const ta_plan *p, ta_scratch *s);

/* Arena base and offsets are 64-byte aligned. Inputs are separate external arrays,
 * copied into the same arena under either plan. Caller owns all buffers.
 * Execution validates the plan using a local ta_scratch on the stack. The observer
 * is optional, read-only, untimed instrumentation; data can expire after return.
 * Only designated outputs remain readable after successful execution. All arena
 * contents expire at the next run. Independent arenas permit concurrent runs.
 */
typedef void (*ta_observer)(size_t tensor, const float *data, size_t count, void *context);
ta_status ta_run(const ta_graph *g, const ta_plan *p, void *arena, size_t capacity,
                 const float *const *inputs, const size_t *input_counts, size_t input_count,
                 ta_observer observer, void *context);
#endif
