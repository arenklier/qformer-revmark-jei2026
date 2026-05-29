# Reproduction guide

Every table and figure in the manuscript has a one-liner command that regenerates the corresponding result. Approximate runtimes are reported for a single NVIDIA L40S (46 GB) GPU.

## Prerequisites

```bash
bash setup.sh
source .venv/bin/activate
python scripts/download_datasets.py --all  # downloads / symlinks Kodak, SIPI, CLIC, COCO, ISIC
```

## Table 1. Standard PEE PSNR vs capacity on Kodak

```bash
python scripts/eval_capacity_curve.py \
    --checkpoint checkpoints/qformer_v4_intaware_best.pt \
    --variant qformer_v4 \
    --dataset kodak \
    --t-values 0,1,2,3,5 \
    --output results/capacity_curve_v4_kodak.json
```

Runtime: about 5 seconds. Repeat with `qformer_v2_best.pt` and `qformer_v3_best.pt` for the other columns. The classical 4-neighbour column uses `--variant classical` and needs no checkpoint.

## Table 2. No-shift PEE on Kodak at fixed payloads

```bash
for payload in 1000 5000 10000; do
    python scripts/eval_noshift.py \
        --checkpoint checkpoints/qformer_v5_heldout_best.pt \
        --dataset kodak \
        --payload ${payload} \
        --output results/noshift_kodak_v5_${payload}.json
done
```

Runtime: about 60 seconds per payload. For the bottom-block standard-PEE baselines, use `scripts/eval_capacity_curve.py --t-values 0`.

## Table 3. Comparison to prior art (grayscale 512x512 Hu / Qiu setup)

Hu and Qiu's CNNP weights are mirrored at the bottom of `checkpoints/`. To rerun their published pipeline on the four shipped standard images (Barbara, Lena, Peppers, yacht):

```bash
python scripts/baseline_comparison.py \
    --baseline hu2021 \
    --mode histogram_shifting \
    --dataset hu_std_4 \
    --payload 10000 \
    --output results/hu2021_histshift_4img.json
```

Runtime: about 90 seconds. Repeat with `--mode expansion_embedding`, and with `--dataset kodak_gray --size 512` for the larger grayscale Kodak evaluation.

For our method on the same setup:

```bash
python scripts/eval_noshift.py \
    --predictor classical \
    --grayscale --size 512 \
    --dataset hu_std_4 \
    --payload 10000 \
    --output results/ours_classical_hu_std_4.json
```

## Table 4. Empirical cross-channel mutual information (Proposition 1)

```bash
python scripts/measure_mi_errors.py \
    --checkpoint checkpoints/qformer_v2_best.pt \
    --datasets kodak,sipi,clic,coco,isic \
    --bins 32 \
    --output results/mi_lemma_validation.json
```

Runtime: about 4 minutes (the COCO subset dominates).

## Table 5. Fragility under image-processing attacks

```bash
python scripts/eval_robustness.py \
    --checkpoint checkpoints/qformer_v5_heldout_best.pt \
    --dataset kodak \
    --payload 5000 \
    --attacks jpeg_90,jpeg_70,jpeg_50,gauss_2,gauss_5,gauss_10,crop_5,crop_10 \
    --output results/robustness_kodak.json
```

Runtime: about 3 minutes. The `--audit-bypass` flag additionally runs the consistency-check-bypassed extraction to verify the 0.500 fair-coin bit-match rate reported in the paper.

## Figure 2. Kodak Pareto front

```bash
python scripts/plot_3method_pareto.py \
    --classical results/capacity_curve_classical_kodak.json \
    --qformer results/capacity_curve_v4_kodak.json \
    --noshift results/noshift_kodak_v5_10000.json,results/noshift_kodak_v5_5000.json,results/noshift_kodak_v5_1000.json \
    --output results/fig_pareto.pdf
```

Runtime: under a second. Requires Table 1 and Table 2 to have been generated first.

## Figure 3. Q-Q plots of prediction errors

```bash
python scripts/measure_mi_errors.py \
    --checkpoint checkpoints/qformer_v5_heldout_best.pt \
    --dataset kodak \
    --plot-qq results/fig_qq.pdf
```

Runtime: about 30 seconds.

## Figure 4. Sample visualisations

```bash
python scripts/make_sample_figures.py \
    --checkpoint checkpoints/qformer_v5_heldout_best.pt \
    --images kodak_01,kodak_23,sipi_baboon,sipi_peppers \
    --modes standard,noshift \
    --output results/fig_samples.pdf
```

Runtime: about 15 seconds.

## End-to-end retraining (optional)

If full retraining is desired rather than using the bundled checkpoints:

```bash
# Standard predictor (v2): 5 minutes
python scripts/train.py --config configs/v2.yaml

# Larger predictor (v3): 9 minutes
python scripts/train.py --config configs/v3.yaml

# Integer-aware loss (v4): 6 minutes
python scripts/train.py --config configs/v4.yaml

# Held-out (v5, Kodak excluded from training corpus): 5 minutes
python scripts/train.py --config configs/v5_heldout.yaml
```

All training uses `torch.manual_seed(0)` and `numpy.random.seed(0)`. DataLoader shuffling is seeded but remains stochastic; we recommend `torch.backends.cudnn.deterministic = True` if exact reproducibility of the location-map cardinality is required.

## Reversibility verification

```bash
pytest tests/test_reversibility.py -v
```

This embeds and extracts a random payload on every Kodak image with every variant at every operating point, then asserts that the extracted payload and the recovered host match the originals exactly. The full test suite runs in under 90 seconds and serves as the primary integrity check.

## Reading the bundled JSON results

All results files under `results/` use the same schema:

```json
{
    "method": "qformer_v5_noshift",
    "dataset": "kodak",
    "payload_target": 10000,
    "per_image": [
        {"image": "kodak01", "psnr_db": 65.21, "ms_ssim": 0.99998, "lpips": 1e-6, "side_bits": 30340, "bit_exact": true},
        ...
    ],
    "summary": {"psnr_mean": 65.03, "psnr_ci95": 0.53, "msssim_mean": 0.99998, "all_bit_exact": true}
}
```
