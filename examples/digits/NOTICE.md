# Model and dataset attribution

The model is the existing project-trained float32 64→32→10 ReLU classifier from
[andrew29mo/tiny-infer-lab](https://github.com/andrew29mo/tiny-infer-lab), revision
`1b5f64859433f83bf59962cd601911c4d769da26`, `models/digits/model.json`.
The file is copied unchanged and its SHA-256 is recorded in `fixture.json`.
Code and project-trained weights: MIT, Copyright (c) 2026 Andrew Yuan;
see the repository [MIT license](../../LICENSE).
Dataset-derived split/reference metadata carried inside `model.json` retain the
dataset attribution below; this does not change the weights' MIT license.

Dataset: E. Alpaydin and C. Kaynak (1998), **Optical Recognition of Handwritten
Digits**, UCI Machine Learning Repository, DOI
[10.24432/C50P49](https://doi.org/10.24432/C50P49),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The [UCI license and dataset page](https://archive.ics.uci.edu/dataset/80/optical%2Brecognition%2Bof%2Bh)
and [scikit-learn load_digits](https://scikit-learn.org/1.6/modules/generated/sklearn.datasets.load_digits.html)
describe the source. MIT does not replace these dataset terms.

`fixture.json` redistributes the selected 450 held-out 8×8 pixel images and labels,
split indices, training-image hashes and derived reference predictions/logits.
Pixel images/labels and their dataset-derived selection are attributed under
CC BY 4.0. Changes: selected/reordered by the recorded split, represented as JSON,
scaled by fixed pixel/16 for model input, and used to generate predictions/hashes.
Generated C data is a local, ignored build artifact with the same attribution.
No endorsement by the dataset creators is implied.

This attribution also covers dataset-derived content in both published digits
report folders, including normalized input pixels retained among intermediate
tensors, labels, selected indices, hashes and reference/predicted results. Reports
also contain project-trained weights and original measurement metadata; those
parts are not relicensed as dataset material. See the repository-wide
[license scope](../../THIRD_PARTY_NOTICES.md).

The 1,797-image sklearn subset is from the original UCI test partition. The
recorded 1,347/450 stratified split is this project's experiment, not the original
UCI protocol. Writer IDs are unavailable; accuracy is not writer-independent.
