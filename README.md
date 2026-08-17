# QFormer-RevMark

**No-Shift Prediction-Error Expansion for High-Fidelity Reversible Colour Image Watermarking, a Predictor-Agnostic Framework with Cross-Channel Capacity Analysis**

> Anonymous review snapshot. Author and affiliation information is withheld during peer review.

## Overview

This repository accompanies the manuscript and contains the full PyTorch implementation, trained checkpoints, and result tables for a predictor-agnostic reversible colour-image watermarking framework. Three contributions are realised in code:

1. **No-shift PEE** — a variant of prediction-error expansion that externalises the embed/no-embed location encoding to a side-channel bitmap. Lifts marked-image PSNR from 51 dB to 74 dB at low payloads on Kodak, with bit-exact recovery preserved.
2. **Cross-channel floor reduction (Theorem 1)** — adding cross-channel context lowers the entropy-power floor on prediction-error variance by exactly `2^(-2 I(X_c; N_-c | N_c))`, for any host distribution and any predictor. The Gaussian linear-MMSE case, where the floor is attained, is the corollary. Checked empirically on Kodak (Spearman ρ = −0.891 over 72 image-channel pairs).
3. **Two predictor instantiations** — a classical 4-neighbour rhombus filter and a quaternion self-attention network, occupying complementary corners of the PSNR-capacity Pareto front.

## Repository structure

```
qformer-revmark/
├── src/
│   ├── models/          # Quaternion / scalar Transformer predictors
│   ├── data/            # DataLoaders + RGB <-> quaternion pack/unpack
│   ├── pee/             # Adaptive Q-PEE and no-shift PEE embed/extract
│   ├── training/        # Self-supervised training loop, integer-aware loss
│   └── utils/           # PSNR, SSIM, MS-SSIM, LPIPS, MI estimators
├── scripts/             # Reproduction entry points (one per table/figure)
├── tests/               # Unit tests, including bit-exact reversibility
├── checkpoints/         # Pretrained weights (v2, v3, v4, v5, scalar baselines)
├── results/             # JSON tables and PDF/PNG figures from the paper
├── paper/               # LaTeX source + compiled PDF
├── requirements.txt     # Pinned Python dependencies
├── setup.sh             # Environment bootstrap script
├── REPRODUCE.md         # Table-by-table and figure-by-figure reproduction guide
└── LICENSE              # MIT
```

## Setup

```bash
git clone <this-repo>
cd qformer-revmark
bash setup.sh
source .venv/bin/activate
```

Tested with Python 3.12.3 and PyTorch 2.4.1 (CUDA 12.1) on a single NVIDIA L40S GPU (46 GB). Training a model variant takes 5 to 9 minutes; full evaluation of all paper tables takes well under one hour.

### Datasets

The code automatically downloads or links the following datasets:

| Dataset | Use | Source | Approx. size |
|---|---|---|---|
| Kodak (24 images) | Canonical colour benchmark | http://r0k.us/graphics/kodak/ | 50 MB |
| USC-SIPI Misc (14) | Classic test images | https://sipi.usc.edu/database/ | 100 MB |
| CLIC-2020 validation (41) | High-resolution natural | https://www.compression.cc/ | 3 GB |
| COCO val-2017 (1000 subset) | Diversity | https://cocodataset.org/ | 500 MB |
| ISIC Archive (100 subset) | Medical dermoscopy | https://www.isic-archive.com/ | 200 MB |

See `scripts/download_datasets.py` for download helpers. All images are centre-cropped to 256×256 RGB unless stated otherwise.

## Quick start

Reproduce the headline no-shift result on Kodak (around 60 seconds on an L40S):

```bash
python scripts/eval_noshift.py --checkpoint checkpoints/qformer_v5_heldout_best.pt \
    --dataset kodak --payloads 1000,5000,10000
```

Reproduce the standard-PEE T-sweep capacity table (Table 3 in the paper):

```bash
python scripts/eval_capacity_curve.py --checkpoint checkpoints/qformer_v4_intaware_best.pt \
    --dataset kodak
```

Each script writes its JSON into `results/` and prints a summary. The `results/`
directory already ships the exact files behind every number in the paper, so the
tables can be checked without rerunning anything.

A complete table-by-table guide is in [REPRODUCE.md](REPRODUCE.md).

## Pretrained checkpoints

| File | Variant | Train epochs | Params | Notes |
|---|---|---|---|---|
| `qformer_v2_best.pt` | $d_q{=}48$, depth 6, MSE | 50 | 925 k | Standard predictor |
| `qformer_v3_best.pt` | $d_q{=}64$, depth 8, MSE | 100 | 1.93 M | Larger predictor |
| `qformer_v4_intaware_best.pt` | $d_q{=}48$, depth 6, integer-aware loss | 80 | 925 k | Best $T{=}0$ capacity |
| `qformer_v5_heldout_best.pt` | $d_q{=}48$, COCO+CLIC only (Kodak excluded) | 80 | 925 k | Held-out generalisation check |
| `scalar_v1_best.pt` | Scalar Transformer, matched feature dim | 50 | 2.97 M | Ablation baseline |
| `scalar_v2_matchparams_best.pt` | Scalar Transformer, matched param count | 50 | 948 k | Ablation baseline |

## Headline numbers (Kodak 256×256 RGB, 10 000-bit no-shift target)

The deep predictors saturate below the 10 000-bit target because their tighter error
distribution leaves fewer `|e|=0` candidates, so the target and the number of bits
actually embedded differ. Both are reported.

| Method | Embedded | PSNR (dB) | MS-SSIM | LPIPS | Bit-exact |
|---|---|---|---|---|---|
| QFormer-v5 (held-out predictor) | 8 348 | **65.03 ± 0.53** | **0.99998** | **<10⁻⁵** | 24/24 |
| QFormer-v4 (integer-aware) | 8 400 | 64.99 | 0.9998 | <10⁻⁵ | 24/24 |
| QFormer-v3 (larger) | 8 782 | 64.76 | 0.9998 | <10⁻⁵ | 24/24 |
| Classical 4-neighbour | 10 000 | 64.09 | 0.9998 | <10⁻⁵ | 24/24 |

For the grayscale 512×512 comparison against Hu 2021 (CNNP) and Qiu 2025 (ICNNP), and
for the colour 512×512 rows matched to He and Cai 2024, see Table 7 in the paper and
`scripts/eval_grayscale.py` / `scripts/verify_revision_experiments.py`.

## Revision findings worth knowing

- **Theorem 1** states the cross-channel gain over entropy-power floors, so it holds for
  any host distribution and any predictor, nonlinear ones included. The Gaussian
  linear-MMSE identity is the corollary. See `scripts/measure_attainment.py` for the
  empirical check (Spearman ρ = −0.891 over 72 image-channel pairs).
- **Quaternion attention is a negative result here.** Trained long enough it matches or
  beats a matched-parameter scalar Transformer on accuracy, but its residuals retain
  *more* cross-channel structure at every budget, and it is ~5× slower.
  See `scripts/measure_quaternion_role.py`.
- **The location map cannot be reconstructed from the marked image** (map-free recovery
  sits at chance, 0.499). Under partial corruption a plain positional map cascades; a
  256-flag block-synchronised map bounds the damage. See `scripts/measure_map_corruption.py`.

## Tests

```bash
pytest tests/                 # Unit tests including bit-exact reversibility
pytest tests/test_reversibility.py -v
```

The reversibility test verifies that for every test image, predictor variant, and threshold, the extracted payload equals the embedded payload exactly and the recovered host equals the original host exactly.

## Project size

Approximately 5 500 lines of tested Python, organised as:
- Training scripts and self-supervised loop
- Evaluation harness for capacity, fixed-payload, and no-shift modes
- Ablation configurations (predictor architecture, integer-aware loss, scalar baselines)
- Figure generators (Pareto curves, Q-Q plots, sample visualisations)
- Result JSON files covering every operating point reported in the paper

## License

MIT License, see [LICENSE](LICENSE).

## Citation

A BibTeX entry will be added upon de-anonymisation following peer-review acceptance.
