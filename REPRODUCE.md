# Reproduction guide

Every table and figure in the manuscript maps to a script in `scripts/`. All commands are run from the repository root with the virtual environment active. Approximate runtimes are for a single NVIDIA L40S (46 GB) GPU.

Each script writes its output JSON into `results/` and prints a summary table to stdout. The `results/` directory already contains the exact JSON files behind every number in the paper, so the tables can be checked without rerunning anything.

## Prerequisites

```bash
bash setup.sh
source .venv/bin/activate
python scripts/download_datasets.py --all     # or --kodak --sipi --clic --coco --isic
```

## Table 1. Standard PEE, PSNR versus capacity (Kodak, RGB 256)

```bash
python scripts/eval_capacity_curve.py --checkpoint checkpoints/qformer_v4_intaware_best.pt --dataset kodak
```

Runtime: a few seconds. Repeat with `--checkpoint checkpoints/qformer_v2_best.pt` and `checkpoints/qformer_v3_best.pt` for the other columns. Pre-computed: `results/capacity_curve_*.json`.

## Table 2. No-shift PEE at fixed payload targets (Kodak, RGB 256)

```bash
python scripts/eval_noshift.py --checkpoint checkpoints/qformer_v5_heldout_best.pt --dataset kodak --payloads 1000,5000,10000
```

Runtime: about 1 minute per payload. The `embedded` field in the output JSON is the "Emb." column of Table 2, which differs from the target where the candidate set saturates. Pre-computed: `results/noshift_qformer_v5_heldout_best_kodak_T0.json`.

MS-SSIM and LPIPS for the same operating points come from:

```bash
python scripts/eval_full_metrics.py --checkpoint checkpoints/qformer_v5_heldout_best.pt --dataset kodak --payloads 1000,5000,10000
```

Pre-computed: `results/fullmetrics_qformer_v5_heldout_best_kodak.json`.

## Table 3. Comparison to prior art

Grayscale 512x512 no-shift with the classical predictor, on Kodak, USC-SIPI, and Hu's four standard images:

```bash
python scripts/eval_grayscale.py --dataset kodak   --image-size 512 --payloads 10000,20000
python scripts/eval_grayscale.py --dataset sipi    --image-size 512 --payloads 10000
python scripts/eval_grayscale.py --dataset hu_std  --image-size 512 --payloads 10000
```

Pre-computed: `results/grayscale512_classical_noshift_{kodak,sipi,hu_std}.json`.

Hu 2021's pretrained CNNP predictor plugged into our no-shift pipeline:

```bash
python scripts/eval_hu2021_compare.py --dataset kodak  --image-size 512 --payloads 10000
python scripts/eval_hu2021_compare.py --dataset sipi   --image-size 512 --payloads 10000
python scripts/eval_hu2021_compare.py --dataset hu_std --image-size 512 --payloads 10000
```

Pre-computed: `results/hu2021_predictor_noshift_{kodak,sipi,hu_std}.json`.

The resolution sweep and the colour 512x512 rows matched to He and Cai 2024 come from the revision script:

```bash
python scripts/verify_revision_experiments.py
```

This prints the 128/256/512/1024 sweep at a fixed 500-bit payload, the extended attack table, the map-free extraction audit, and the 20000/40000-bit colour 512 comparison. Runtime: about 20 minutes end to end.

## Table 4. Feature-level comparison

Qualitative table compiled from the cited papers. No script.

## Table 5. Empirical cross-channel mutual information

```bash
python scripts/measure_mi_errors.py --checkpoint checkpoints/qformer_v2_best.pt --datasets kodak,sipi,clic,coco,isic --mi-bins 32
```

Runtime: about 4 minutes, dominated by the COCO subset. Pre-computed: `results/mi_lemma_validation.json`.

## Table 6. Five-dataset generalisation and CIEDE2000

```bash
python scripts/verify_color_generalization.py
```

Prints the CIEDE2000 values for the classical and v5 predictors on Kodak, then the five-dataset table at a fixed 5000-bit payload. Runtime: about 10 minutes.

## Table 7. Fragility under attacks

```bash
python scripts/eval_robustness.py --checkpoint checkpoints/qformer_v5_heldout_best.pt --dataset kodak --n-payload 5000
```

Runtime: about 3 minutes. The impulse-noise, median-filter, and sigma=1 rows added in revision come from `scripts/verify_revision_experiments.py`. The bit-match audit under a bypassed consistency check is produced by `scripts/measure_runtime_and_fpr.py`.

## Figure 2. Kodak Pareto front

```bash
python scripts/plot_3method_pareto.py
```

Requires the Table 1 and Table 2 JSON files to be present in `results/`. Output: `results/fig_pareto.pdf`.

## Figure 3. Q-Q plots of prediction errors

```bash
python scripts/make_qq_plot.py --checkpoint checkpoints/qformer_v5_heldout_best.pt --dataset kodak
```

Output: `results/fig_qq.pdf`.

## Figure 4. Sample visualisations

```bash
python scripts/make_sample_figures.py --checkpoint checkpoints/qformer_v5_heldout_best.pt
```

Output: `results/fig_samples.pdf`.

## Appendix A. Runtime and memory

```bash
python scripts/measure_runtime_and_fpr.py
```

Reports the predictor forward pass, the full embed and extract pipeline latencies, and the bypassed-check bit-match rates.

## Retraining from scratch (optional)

Training is driven by CLI flags. The entry point is `src/training/train.py`.

```bash
# v2: d_quat=48, depth 6, 50 epochs, about 5 minutes
python -m src.training.train --d-quat 48 --depth 6 --epochs 50 --save-name qformer_v2

# v3: d_quat=64, depth 8, 100 epochs, about 9 minutes
python -m src.training.train --d-quat 64 --depth 8 --epochs 100 --save-name qformer_v3

# v4 (integer-aware loss): 80 epochs, about 6 minutes
python -m src.training.train --d-quat 48 --depth 6 --epochs 80 --save-name qformer_v4_intaware

# v5 (held out, Kodak excluded from the training corpus): 80 epochs, about 5 minutes
python -m src.training.train --d-quat 48 --depth 6 --epochs 80 --save-name qformer_v5_heldout
```

Run `python -m src.training.train --help` for the full flag list. All training uses `torch.manual_seed(0)` and `numpy.random.seed(0)`. DataLoader shuffling is seeded but remains stochastic; set `torch.backends.cudnn.deterministic = True` if exact reproduction of the location-map cardinality is required.

## Reversibility test

```bash
pytest tests/test_reversibility.py -v
```

Embeds and extracts a random payload on every test image at every operating point, then asserts that both the recovered host and the extracted payload match the originals exactly.

## Result JSON schema

```json
{
  "checkpoint": "checkpoints/qformer_v5_heldout_best.pt",
  "dataset": "kodak",
  "image_size": 256,
  "T_select": 0,
  "results": [
    {"n_target": 10000, "psnr_mean": 65.03, "ssim_mean": 0.9998,
     "embedded_mean": 8348.0, "side_bytes_mean": 3795, "n_imgs": 24, "n_bit_exact": 24}
  ]
}
```
