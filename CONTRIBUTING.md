# Contributing

Run `make test sanitize mutation publication-check` on macOS or Linux with a C11
compiler, Make, Git and Python 3.9+. No numerical Python packages are required.
Use a full clone so historical measurement source revisions are available.

Preserve inclusive last-read lifetimes, pinned outputs, alignment, checked sizes,
and the single fixed execution order. Tests must exercise the real planner and
kernels, with independent lifetimes and intermediate numerical references.
Changes that introduce in-place execution, graph reordering or new allocation
semantics need their own contract and tests; do not silently change the baseline.

Keep memory quantities separate: tensor arena, alignment/padding, allocator
usable bytes, plan metadata, scratch and process RSS are different observations.
Report planner overhead and heuristic gaps as well as storage reductions.
Performance measurements are not shared CI pass/fail speed gates.

Commit measurement sources before running `tools/run_benchmark.py`. Keep local
builds and trial runs ignored; copy only verified selected evidence into
`results/published/`. Historical measurements keep their original Git revision.
Render figures from verified offsets/lifetimes and check their byte scales.

Contributions are under the [MIT License](LICENSE). Preserve attribution and
identify any third-party material rather than assuming MIT relicenses it.
