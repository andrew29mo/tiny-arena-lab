# Current synthetic measurements

Source `b149010`, four static graphs, six processes and twenty sample groups per
process. This measures complete C planning and tensor-arena allocations; it does
not measure inference latency. See [current tables and limits](../../../README.md).

Verify with `python3 tools/verify_report.py results/published/host-20261014` from
the repository root. The current measured implementation still matches these
sources; newer tests/docs do not relabel the run as a new performance experiment.

Synthetic inputs/weights are generated; no third-party dataset is included.
Original code, evidence and tooling are under the [MIT license](../../../LICENSE).
