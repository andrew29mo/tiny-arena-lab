# Superseded real-model evidence — protocol v1

Historical measured source `edfce00`. These files are retained unchanged for
provenance. They are superseded by the
[current v2 measurements](../digits-budget-reviewed-20261014/README.md).

The original correct loop did include the final batch, but its validator could
accept a compiled fault that dropped tail work only on the uncaptured timing
path. v2 checks every timed pass's counts and output fingerprint. See the
[review](../../../examples/digits/REVIEW.md). Do not treat these v1 timings as the
current claim or compare v1/v2 differences as a kernel/planner speed improvement.

`python3 tools/check_release.py` verifies this archive using its own historical
source/verifier. The current v2 verifier intentionally rejects protocol v1.
Dataset-derived data retain [CC BY 4.0 attribution](../../../THIRD_PARTY_NOTICES.md);
project code, trained weights and original measurement metadata retain MIT.
