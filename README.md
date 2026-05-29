# QFormer-RevMark

**No-Shift Prediction-Error Expansion for High-Fidelity Reversible Color Image Watermarking, a Predictor-Agnostic Framework with Cross-Channel Capacity Analysis**

> Anonymous review snapshot. Author and affiliation information is withheld during peer review.

## Overview

This repository accompanies the manuscript and contains the full PyTorch implementation, trained checkpoints, and result tables for a predictor-agnostic reversible color-image watermarking framework. Three contributions are realised in code:

1. **No-shift PEE** — a variant of prediction-error expansion that externalises the embed/no-embed location encoding to a side-channel bitmap. Lifts marked-image PSNR from 51 dB to 74 dB at low payloads on Kodak, with bit-exact recovery preserved.
2. **Cross-channel capacity bound (Proposition 1)** — a closed-form lower bound on the per-pixel embedding capacity that a cross-channel predictor enjoys over an independent three-channel predictor, validated empirically on five datasets (mutual information 0.13 to 0.73 bits/pixel).
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
| Kodak (24 images) | Canonical color benchmark | http://r0k.us/graphics/kodak/ | 50 MB |
| USC-SIPI Misc (14) | Classic test images | https://sipi.usc.edu/database/ | 100 MB |
| CLIC-2020 validation (41) | High-resolution natural | https://www.compression.cc/ | 3 GB |
| COCO val-2017 (1000 subset) | Diversity | https://cocodataset.org/ | 500 MB |
| ISIC Archive (100 subset) | Medical dermoscopy | https://www.isic-archive.com/ | 200 MB |

See `scripts/download_datasets.py` for download helpers. All images are centre-cropped to 256×256 RGB unless stated otherwise.

## Quick start

Reproduce the headline 10 000-bit no-shift result on Kodak (around 60 seconds on an L40S):

```bash
python scripts/eval_noshift.py \
    --checkpoint checkpoints/qformer_v5_heldout_best.pt \
    --dataset kodak \
    --payload 10000 \
    --output results/noshift_kodak_v5.json
```

Reproduce the standard-PEE T-sweep capacity table (Table 1 in the paper):

```bash
python scripts/eval_capacity_curve.py \
    --checkpoint checkpoints/qformer_v4_intaware_best.pt \
    --dataset kodak \
    --t-values 0,1,2,3,5 \
    --output results/capacity_curve_kodak_v4.json
```

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

## Headline numbers (Kodak 256×256 RGB, 10 000-bit no-shift payload)

| Method | PSNR (dB) | MS-SSIM | LPIPS | Bit-exact |
|---|---|---|---|---|
| QFormer-v5 (held-out predictor) | **65.03 ± 0.53** | **0.99998** | **<10⁻⁵** | 24/24 |
| QFormer-v4 (integer-aware) | 64.99 | 0.9998 | <10⁻⁵ | 24/24 |
| QFormer-v3 (larger) | 64.88 | 0.9998 | <10⁻⁵ | 24/24 |
| Classical 4-neighbour | 64.09 | 0.9998 | <10⁻⁵ | 24/24 |

For grayscale 512×512 comparison against Hu 2021 (CNNP) and Qiu 2024 (ICNNP), see Table 3 in the paper and `scripts/eval_noshift.py --grayscale --size 512`.

## Tests

```bash
pytest tests/                 # Unit tests including bit-exact reversibility
pytest tests/test_reversibility.py -v
```

The reversibility test verifies that for every test image, predictor variant, and threshold, the extracted payload equals the embedded payload exactly and the recovered host equals the original host exactly.

## Project size

Approximately 2 700 lines of tested Python, organised as:
- Training scripts and self-supervised loop
- Evaluation harness for capacity, fixed-payload, and no-shift modes
- Ablation configurations (predictor architecture, integer-aware loss, scalar baselines)
- Figure generators (Pareto curves, Q-Q plots, sample visualisations)
- Result JSON files covering every operating point reported in the paper

## License

MIT License, see [LICENSE](LICENSE).

## Citation

A BibTeX entry will be added upon de-anonymisation following peer-review acceptance.
