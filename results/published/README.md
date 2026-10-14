# Published measurement inventory

| Folder | Status | Measured source | Scope |
| --- | --- | --- | --- |
| [host-20261014](host-20261014/README.md) | Current synthetic evidence | b149010 | Four static graphs; arena sizes and complete planning cost |
| [digits-budget-reviewed-20261014](digits-budget-reviewed-20261014/README.md) | Current real-model evidence, protocol v2 | 1357529 | Real digits model, fixed arena budgets, audited full-test inference |
| [digits-budget-20261014](digits-budget-20261014/README.md) | Superseded, protocol v1 | edfce00 | Historical run before timed-path coverage/fingerprint checks |

Original records retain their measured source and hashes. v1/v2 timings include
different validation work and must not be presented as a speed comparison.
The current numerical tables are in the [main README](../../README.md) and
[real-model README](../../examples/digits/README.md). All host timing results are
workload/machine observations, not portable speed or whole-process RAM guarantees.

Run `python3 tools/check_release.py` from the repository root to verify all three
sets. The superseded run uses an isolated copy of its original committed verifier;
this does not change the checkout or Git history.

Digits dataset-derived content retains the repository's
[CC BY 4.0 attribution](../../THIRD_PARTY_NOTICES.md). Original project code,
tooling, model weights and measurement metadata retain the MIT license.
