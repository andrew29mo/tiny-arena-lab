# Real-model and budget review

Reviewed `e260d87`, including measurement source `edfce00`, the production C
engine, example harness, generated fixture, independent references and raw data.
The core engine, model weights and dataset are unchanged by this review.

## Findings and repairs

**Timed-path verification gap (fixed).** The captured-output C pass checked all
450 predictions/logits, while warmup/timed calls used NULL output pointers and
only added the returned accuracy count to a sink. No assertion checked that
those calls completed all rows. A genuinely compiled variant that skipped the
last batch only when outputs were not captured was accepted by the old raw-report
validator. The original normal loop was correct; this was a reproducible hole in
what the timing validation could establish.

Protocol v2 always counts classified rows and engine calls and fingerprints each
row index, predicted class and all ten float32 logits, regardless of output
capture. Every warmup/timed pass must match its independently checked captured
pass. Timed-pass comparisons occur after the interval; counters, fingerprints,
argmax, data packing and production inference remain inside it. Raw groups record
actual aggregated counts. Planning groups also count and check completed builds.
The fingerprint is a regression check, not a cryptographic attestation.

Four compiled harness faults must fail: a missing final batch only in the timed
NULL-output path, a missing final valid row on that path, a small logit change on
that path even if classes stay the same, and half the claimed planning iterations.
Warmup remains correct in the three timed-path faults. Normal and optimized Python
both execute these checks. The existing compiled production-kernel missing-ReLU
fault also remains rejected by the independent numerical reference.

**Reference provenance hardened.** In addition to parameter comparisons and
numeric references, tests now compare all 28,800 normalized input values and all
450 labels directly through the same C fixture linked into the benchmark. The
optional installed-sklearn check verifies the full recorded upstream training
configuration. Dependency-free tests reject changed activation or training seed.

New measurements are collected from a new committed source revision. The v1
archive is retained unchanged and is no longer the current timing claim. Extra
output-audit work is included symmetrically in both layouts' v2 times; differences
between v1/v2 times must not be presented as planner/kernel improvements.

## Training and test separation

The model file is byte-identical to `models/digits/model.json` at tiny-infer-lab
revision `1b5f64859433f83bf59962cd601911c4d769da26`. The training function at that
revision fits only `x[train_ids], y[train_ids]`, with test probabilities generated
after fitting. It uses fixed pixel/16 normalization (no full-dataset fitted
statistics), Adam, a fixed seed and training-loss stopping with test-based early
stopping disabled. No training or hyperparameter selection occurs in this example.

The installed dataset's canonical full-data/label hash, stratified 1,347/450
split, exact test pixels/labels, training-image hashes and exported probabilities
are checked. No overlapping indices or exact duplicate images across train/test
were found. Policy/dataset evidence supports the stated train-only path; weights
alone cannot independently reconstruct every historical training decision.

These test images were already evaluated upstream. Accuracy is a transfer check
on that recorded split, not a fresh blind test. Writer identity is unavailable;
no writer-independent generalization or original UCI protocol result is claimed.

## Production execution and tail coverage

Both the shared library used by numerical tests and the C executable link the
same graph/planner/executor objects and fixture object. Every intermediate tensor
and actual address is checked against independent Python intervals/float64 fsum.
Parameters, inputs and labels must exactly match committed artifacts. Final
classes/logits are compared with sklearn/NumPy references. Budget rejection must
occur before any observer callback or arena write; guard regions stay intact.

At batch 16, all 450 unique held-out IDs are scored exactly once. There are 29
engine calls and 464 computed rows, including two valid and fourteen padded rows
in the final call. Both layouts compute the same padding. Each timed group has
four complete passes: 1,800 valid rows, 116 calls, and 1,856 computed rows. Raw
output evidence and timed-path fingerprints cover the final valid rows.

## Fairness and limits

Both layouts use identical weights, batches, kernels, input copies, validation,
output lifetimes and full-test workloads. Separate is one adequately allocated
arena during successful-inference timing; its smaller-budget capacity rejection
is not scored as faster inference. Both are warmed up, mode order alternates,
batch order rotates across processes, and raw group timings are aggregated as
medians of process medians. There is no CPU/frequency isolation or statistical
significance claim. Plans are still validated on each inference.

Payload limits remain 512 B and 7,168 B; host allocations additionally include
128-byte redzones and allocator rounding. Requested/usable sizes are observed,
not estimated from tensor dimensions alone. The storage subtotal includes one
arena/plan/scratch, parameters, graph descriptors and an active input batch;
it excludes other stack/allocator/runtime/test-harness costs. The harness keeps
additional comparison and capture objects. Neither the payload budget nor this
subtotal proves a physical whole-process RAM ceiling or MCU feasibility.

## Verification and current results

The corrected v2 run uses source `1357529`, six fresh processes and eleven groups
per process. Accuracy remains 438/450 (97.33%) for both layouts/batches; arena and
accounted storage sizes are unchanged. At batch 16 the observed median inference
cost is 0.685 vs 0.686 µs per valid example; process ranges overlap. Output auditing
is included in both figures and no speedup is claimed. Full values and raw records
are linked from the [current protocol/results](README.md).

Validation covers eight real-model tests in normal and optimized Python,
ASan/UBSan, the production-kernel missing-ReLU fault, four compiled harness faults,
semantic raw-record corruption tests, independent installed dataset/sklearn checks
and regenerated report/diagram tables. The existing synthetic core tests and
their historical evidence remain applicable because their sources are unchanged.
