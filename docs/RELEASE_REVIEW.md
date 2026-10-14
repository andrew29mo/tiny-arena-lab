# Release candidate review — 2026-10-14

Reviewed the project at `733bac3` before these publication/documentation checks.
No production C, trained weights, dataset, timing driver or selected numerical
measurement was changed. GitHub remains Private; this review does not publish it.

## Corrections

- The main README's scope now includes the trained digits model as well as
  synthetic graphs; its limits no longer describe the whole project as synthetic.
- Repository-wide [license scope](../THIRD_PARTY_NOTICES.md) explicitly covers
  dataset-derived normalized pixels and other data in both published digits
  report folders, in addition to the fixture. Original code/weights/tooling retain
  MIT; dataset attribution, CC BY 4.0 links and transformation notices are retained.
- The [measurement inventory](../results/published/README.md) and per-folder
  READMEs label current synthetic/v2 evidence and the superseded v1 archive.
  Old timings are not silently relabeled or presented as a current speed claim.
- A complete release check now runs in CI. It validates current reports, the v1
  archive with its original committed verifier, all Markdown local links, license
  notices, report inventory and file modes. It complements numerical tests.
- Publication guards and ignore rules now also cover virtual environments,
  additional build outputs, Windows user paths and disguised binary content.
  Tests exercise representative private-content/credential/binary cases without
  storing literal credentials in test fixtures.

## Consistency and evidence

The core implements static float32 DAGs with Dense+bias, ReLU and Add, inclusive
last-consumer lifetimes, pinned outputs and a non-optimal greedy reuse heuristic.
Independent intervals and float64 references check actual production C tensors
and addresses. Tests cover seeded DAGs, repeated arenas, capacities, numeric
errors, sanitizers and compiled faults, including the genuinely timed tail path.

Current synthetic source `b149010` still matches the measured implementation.
Its table, allocator observations and same-scale diagrams regenerate from raw
evidence. The real-model v2 source `1357529` also matches current measured files;
all tables and diagrams agree with verified records. Superseded v1 `edfce00`
retains its original source and is separately verified, not accepted as v2.

Accuracy remains 438/450 (97.33%) on the previously evaluated recorded split.
No fresh blind-test or writer-independent accuracy is claimed. Full batch-16
passes score 450 rows in 29 calls, including two valid and fourteen padded rows
in the last call; timed coverage is checked explicitly.

Tensor-arena bytes, guarded/native allocation sizes, fixed metadata/scratch,
storage subtotals and whole-process RAM are distinguished. Current real-model
timings include counters/output fingerprints and show overlapping process ranges;
no reliable inference speedup is claimed. The experiments do not establish an
enforced whole-process RAM budget, optimal allocation or MCU hardware performance.

## Publication hygiene

Tracked files and all reachable historical blobs were inspected for private home
paths, credential patterns, local-only files and compiled/binary outputs. Build
directories and trial results are ignored. Commit author/committer metadata uses
the project's public name and GitHub noreply email; no private email was found.
The UCI digits data are intentional attributed public research data. Hardware/OS
and UTC measurement metadata are retained as technical provenance.

No targeted private-content finding or unwanted tracked artifact was found.
This is a scoped audit, not proof of the absence of every possible secret or bug.
The MIT project license is unchanged; optional package distributions are installed
separately rather than bundled into the repository.

## Verification recipe

From the repository root:

```sh
make test sanitize mutation
python3 examples/digits/run.py test
python3 examples/digits/run.py sanitize
python3 tools/check_release.py
```

The pinned optional environment additionally runs
`python examples/digits/prepare.py --check` against installed scikit-learn data
and reference outputs. CI performs these checks on macOS/Linux and fresh C timing
smokes; observed benchmark speed is not a CI pass/fail threshold.
