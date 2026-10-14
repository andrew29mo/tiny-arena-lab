# License scope and third-party data

Original project code, documentation, generated allocation diagrams, measurement
tooling and project-trained model weights are under the [MIT License](LICENSE),
Copyright (c) 2026 Andrew Yuan. MIT does not replace the following dataset terms.

## Handwritten digits data — CC BY 4.0

E. Alpaydin and C. Kaynak (1998), **Optical Recognition of Handwritten Digits**,
UCI Machine Learning Repository, DOI
[10.24432/C50P49](https://doi.org/10.24432/C50P49).
The [UCI dataset page](https://archive.ics.uci.edu/dataset/80/optical%2Brecognition%2Bof%2Bh)
identifies the license as [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
([legal text](https://creativecommons.org/licenses/by/4.0/legalcode.en)).

This attribution covers dataset-derived portions of:

- `examples/digits/model.json`: dataset-derived split indices and exported
  reference-probability metadata; the project-trained weights themselves retain MIT.
- `examples/digits/fixture.json`: selected images, labels, split indices,
  training-image hashes and reference predictions/logits.
- `results/published/digits-budget-20261014/` and
  `results/published/digits-budget-reviewed-20261014/`: normalized input pixels
  stored among intermediate tensors, selected indices, labels/hashes and
  dataset-derived reference/prediction evidence.
- Locally generated, ignored C fixture data containing those images/labels.

Changes include selecting and ordering 450 images from the scikit-learn subset,
representing data as JSON/C arrays, applying fixed pixel/16 scaling, and computing
hashes, intermediate values and predictions. Retain this attribution and the
license link when redistributing those data or their adaptations. No endorsement
by the dataset creators is implied. See [model/data provenance](examples/digits/NOTICE.md).

Published reports mix dataset-derived data with original timing/allocation
metadata and project-trained model weights. This notice does not relicense the
latter as third-party dataset material. Synthetic workloads contain generated
parameters/inputs and do not redistribute the digits dataset.

## Model provenance and optional dependencies

The digits model is copied unchanged from the same author's MIT-licensed
[tiny-infer-lab](https://github.com/andrew29mo/tiny-infer-lab), revision
`1b5f64859433f83bf59962cd601911c4d769da26`; its hash and original split are retained.

Optional NumPy, SciPy, scikit-learn, joblib and threadpoolctl packages are installed
separately through their package distributions. Their source, wheels and virtual
environments are not bundled in this repository; their respective licenses still
apply to those separate distributions. The C core has no such numerical dependency.
