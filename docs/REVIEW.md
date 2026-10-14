# MVP review

Historical review of the synthetic MVP. The later [real-model review](../examples/digits/REVIEW.md)
and [release review](RELEASE_REVIEW.md) describe the expanded project.

Reviewed the MVP at `6dd000f` and its selected measurement source `b149010`.
Scope: production lifetime analysis, placement, validation and execution;
tests against those exact C objects; and the memory/planning comparison.

## Findings and changes

No incorrect overlap or lifetime was reproduced within the documented API
contract. Both operands extend their sources through the last consumer step.
Inputs start live at step zero, dead tensors still get a producer step, and all
designated outputs (including inputs) stay live through graph completion.
Placement uses inclusive lifetime conflicts and checked, padded, aligned ranges.
The production validator independently recomputes graph metadata but shares the
C overlap helper with placement; the separate Python interval check is therefore
necessary, rather than treating C self-validation alone as proof.

Two validation gaps were addressed:

- Historical report checks and matching README numbers did not establish that
  the current measured implementation still matched the selected run. The
  publication check now compares current C/header, fixture, build, input/oracle
  bridge and timing-driver hashes to that run. Tests demonstrate that changing
  these files fails the check; changing tests alone does not invalidate timings.
- Random DAGs covered fanout and multiple inputs but there was no explicit
  changing-input, repeated-plan/arena regression. A directed case now combines
  Add's second operand at its last read, fanout's later first-operand read, a dead
  node, non-aligned logical lengths, an input output and an early pinned output.
  It checks hard-coded lifetimes, rejects a corrupted last-consumer layout,
  compares every intermediate to an independent reference and bitwise between
  layouts, checks final outputs and guards, and repeats with/without observation
  on the same arena and plan. Two additional compiled faults omit the second
  operand or retain only the first consumer; both the fixture checks and this
  directed regression reject them under normal Python and `python -O`.

The README additionally shows arena + one plan + one explicit scratch subtotal,
and explains warm repeated planning, uncontrolled host scheduling/frequency and
per-inference validation. No production C, fixture, build recipe or timing driver
was changed. Existing measurements retain their original source revision;
test/document changes are not represented as a new performance run.

## Comparison assessment

The measured comparison is fair for the stated static-layout question:

- The same fixture instance, graph order, shapes, inputs, weights and kernels
  execute under both plans. No fusion, in-place operators, pruning or different
  numerical implementation advantages are added to reuse.
- Separate is one arena with distinct aligned regions, not repeated heap calls.
  Reuse has the same input-copy and output-preservation policy. Both actual
  arenas are allocated, their tensor addresses and boundary guards inspected,
  and their final outputs read.
- Complete C plan-build groups include analysis, placement and final validation,
  loop/status checks and a volatile sink. There is no LTO or timing across Python
  calls. Allocation, inference, observation and JSON formatting are untimed.
  Raw schedules rotate graph positions and alternate modes. Six process medians
  are aggregated from 20 groups of 1,024 plans per graph/mode per process.
- Persisted raw samples reproduce published statistics; independent references,
  intervals and observed sizes reproduce correctness/layout claims. Rehashed
  corruptions must fail semantic checks. Hashes document provenance rather than
  providing signed execution attestation or independent performance replication.

The reported savings are tensor-arena reductions on four synthetic graphs, not
whole-process RAM reductions or faster inference. Plan/scratch are fixed-capacity
objects; the longest graph's arena is 12,416 to 768 B, while arena + plan + scratch
is 20,640 to 8,992 B. Even that subtotal excludes weights, graph descriptions,
external inputs, other stack frames, allocator overhead and host scaffolding.
Allocator usable sizes include guarded host allocations and are not RSS.

Reuse's median complete planning cost is higher on all four measured graphs.
The heuristic exceeds the live-byte lower bound on two of them. The lower bound
does not prove an attainable optimal layout. Host clock resolution and raw
process variation remain recorded; there is no CPU affinity/frequency control,
confidence interval, cold-start distribution or MCU hardware measurement.
`ta_run` validates the plan on each inference, so reusing a built plan does not
eliminate execution-side validation cost. No inference latency claim is supported.

## Verification

`make test sanitize mutation publication-check` passes locally: native C checks,
six planner tests (including 100 seeded DAGs and the 128-node boundary), six
reporting tests, normal/optimized Python, ASan/UBSan, five compiled faults,
production-object allocator-symbol checks, stored-evidence verification, current
source freshness, numerical README tables and exact SVG regeneration.

The targeted publication guard checks tracked and reachable historical content;
it is not an exhaustive secret audit. GitHub remains Private with no added
collaborators. Remote CI runs the same correctness/publication checks plus a
fresh real C timing smoke on macOS and Linux; timings are not CI speed gates.
