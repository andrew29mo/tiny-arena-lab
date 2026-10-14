# Current real-model evidence — protocol v2

Measured source `1357529`, six processes and eleven sample groups per process.
Both layouts use the same trained model, data, batches and kernels. v2 includes
row/call counters and a fingerprint of every valid row, class and logit in timed
passes. Final padding is included in computation and timing.

See [current numerical tables, memory accounting and limits](../../../examples/digits/README.md)
and the [review](../../../examples/digits/REVIEW.md). The payload budget and storage
subtotal are not whole-process RAM or MCU feasibility claims.

From the repository root:
`python3 examples/digits/verify.py results/published/digits-budget-reviewed-20261014 --docs`.

Dataset-derived data retain [CC BY 4.0 attribution](../../../THIRD_PARTY_NOTICES.md);
project code, trained weights and original measurement metadata retain MIT.
