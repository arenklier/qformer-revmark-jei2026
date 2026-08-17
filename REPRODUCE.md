# Reproduction guide

Every table and figure in the manuscript maps to a script in `scripts/`. All commands are run from the repository root with the virtual environment active. Approximate runtimes are for a single NVIDIA L40S (46 GB) GPU.

Each script writes its output JSON into `results/` and prints a summary table to stdout. The `results/` directory already contains the exact JSON files behind every number in the paper, so the tables can be checked without rerunning anything.

## Prerequisites

```bash
bash setup.sh
source .venv/bin/activate
python scripts/download_datasets.py --all     # or --kodak --sipi --clic --coco --isic
```

## Table 1. Location-map corruption, plain versus block-synchronised

```bash
python scripts/measure_map_corruption.py
```

Embeds a 2000-bit payload on 24 Kodak images, flips map bits at rates 0 to 1e-2 with three draws each, and extracts with both the plain positional map and a 256-flag block-synchronised map. Reports payload bit-match and refusal counts. Runtime: about 25 minutes.

## Table 2. Cross-channel gain and the entropy-power floor

```bash
python scripts/measure_attainment.py
```

Fits linear MMSE predictors on Kodak with and without cross-channel context, estimates the conditional mutual information with a Gaussian-copula estimator, and reports the achieved variance ratio against the floor ratio of Theorem 1, plus the rank correlation over all 72 image-channel pairs. Runtime: about 3 minutes.

Per-image records, plus the clustered rank tests that treat the 24 images rather than the 72 image-channel pairs as the sampling unit:

```bash
python scripts/attainment_json.py
```

Pre-computed: `results/attainment_kodak.json`. Note that every column of Table 2 is a per-image mean, so the achieved-ratio column is the mean of per-image ratios and does not equal the quotient of the two mean-variance columns, and the floor column is the mean of per-image `2^(-2I)` rather than `2^(-2*mean I)`. Both gaps are Jensen's inequality and the per-image identity is exact.

The distribution-free algebra behind Theorem 1 can be checked independently with:

```bash
python scripts/verify_entropy_power.py
```

## Table 3. Standard PEE, PSNR versus capacity (Kodak, RGB 256)

```bash
python scripts/eval_capacity_curve.py --checkpoint checkpoints/qformer_v4_intaware_best.pt --dataset kodak
```

Runtime: a few seconds. Repeat with `--checkpoint checkpoints/qformer_v2_best.pt` and `checkpoints/qformer_v3_best.pt` for the other columns. Pre-computed: `results/capacity_curve_*.json`.

## Table 4. Complexity-ordering probe

```bash
python scripts/probe_complexity_sort.py
```

Sorts the T_sel=0 candidates of each Kodak image by local complexity, keeps only the leading k, and reports the resulting fill factor, map size, payload, and net payload. Shows that concentrating the map raises the fill from 16.4 percent to 43.4 percent but never reaches the 77.3 percent break-even at which Hb(p) drops below p. Runtime: about 2 minutes.

## Table 5. Standard PEE versus no-shift at matched payloads

```bash
python scripts/measure_matched_payload.py
```

Runs canonical standard PEE and no-shift PEE at the same payloads on grayscale Kodak and reports PSNR and embedded counts for both. Confirms that standard-PEE distortion is payload-independent, so the like-for-like gap is 17.5 dB at 1000 bits. Runtime: about 25 minutes.

Two supporting analyses of the side-channel cost:

```bash
python scripts/measure_net_payload.py      # net = payload - map entropy, and the crossover
python scripts/probe_complexity_sort.py    # does complexity ordering reach the break-even fill?
```

## Table 6. No-shift PEE at fixed payload targets (Kodak, RGB 256)

```bash
python scripts/eval_noshift.py --checkpoint checkpoints/qformer_v5_heldout_best.pt --dataset kodak --payloads 1000,5000,10000
```

Runtime: about 1 minute per payload. The `embedded` field in the output JSON is the "Emb." column of Table 5, which differs from the target where the candidate set saturates. Pre-computed: `results/noshift_qformer_v5_heldout_best_kodak_T0.json`.

MS-SSIM and LPIPS for the same operating points come from:

```bash
python scripts/eval_full_metrics.py --checkpoint checkpoints/qformer_v5_heldout_best.pt --dataset kodak --payloads 1000,5000,10000
```

Pre-computed: `results/fullmetrics_qformer_v5_heldout_best_kodak.json`.

## Table 7. Comparison to prior art

Grayscale 512x512 no-shift with the classical predictor, on Kodak, USC-SIPI, and Hu's four standard images:

```bash
python scripts/eval_grayscale.py --dataset kodak   --image-size 512 --payloads 10000,20000
python scripts/eval_grayscale.py --dataset sipi    --image-size 512 --payloads 10000
python scripts/eval_grayscale.py --dataset hu_std  --image-size 512 --payloads 10000
```

Pre-computed: `results/grayscale512_classical_noshift_{kodak,sipi,hu_std}.json`.

Colour 512x512 at He and Cai's payloads, which produce the 67.09 and 64.11 dB rows:

```bash
python scripts/verify_revision_experiments.py
```

Pre-computed: `results/rgb512_hecai_comparison_kodak.json`.

One caveat on the Hu-CNNP rows. Hu's wrapper replicates each grayscale image across three channels and embeds into all of them, so its three-channel totals (3963 on Kodak, 3969 on USC-SIPI) correspond to about 1321 and 1323 bits per channel. Table 7 quotes those rows per channel so that they sit on the same basis as the single-channel classical rows.

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

## Table 8. Feature-level comparison

Qualitative table compiled from the cited papers. No script.

## Table 9. Resolution sweep with side-channel cost

```bash
python scripts/resolution_side.py
```

Runs the classical no-shift predictor at 128, 256, 512, and 1024 on grayscale Kodak at a fixed 500-bit payload, and reports PSNR together with the raw map, fill factor, entropy-coded map, and net payload. The fixed payload avoids confounding resolution with candidate saturation, since a 10000-bit target does not fit at 128. Pre-computed: `results/resolution_side.json`. Runtime: about 12 minutes.

## Table 10. Quaternion versus scalar attention

```bash
python scripts/quatrole_final.py
```

Compares three quaternion checkpoints (v2 at 50 epochs, v4 and v5 at 80) against the matched-parameter scalar Transformer at **both** 50 and 80 epochs, so every quaternion variant faces an equally trained opponent. Reports mean squared prediction error, the `|e|=0` population, residual cross-channel mutual information, and forward-pass latency.

Two details make the numbers comparable across tables. The mutual information uses the same estimator and the same Cross-plus-Dot sample construction as Table 11, so the shared QFormer-v2 entry reads 0.677 in both. The `mean e2` column is the per-sample mean squared error pooled over the three channels, not a sum. Requires `checkpoints/scalar_v3_80ep_best.pt`; see the retraining section if it is absent. Pre-computed: `results/quatrole_final_kodak.json`. Runtime: about 5 minutes.

## Table 11. Empirical cross-channel mutual information

```bash
python scripts/measure_mi_errors.py --checkpoint checkpoints/qformer_v2_best.pt --datasets kodak,sipi,clic,coco,isic --mi-bins 32
```

The checkpoint matters. This table is measured on QFormer-v2, and running the same script with another checkpoint overwrites `results/mi_lemma_validation.json` with different numbers. Runtime: about 4 minutes, dominated by the COCO subset. Pre-computed: `results/mi_lemma_validation.json`.

## Table 12. Five-dataset generalisation and CIEDE2000

```bash
python scripts/verify_color_generalization.py
```

Prints the CIEDE2000 values for the classical and v5 predictors on Kodak, then the five-dataset table at a fixed 5000-bit payload. Runtime: about 10 minutes.

## Table 13. Tamper sensitivity

```bash
python scripts/tamper_sensitivity.py
```

Embeds a 5000-bit payload on grayscale Kodak at 256, perturbs k randomly chosen pixels by plus or minus one for k from 1 to 1024, and classifies each outcome as refused, clean, or silently corrupted. Locates the detection boundary that the 0/24 attack table leaves open. Pre-computed: `results/tamper_sensitivity.json`. Runtime: about 20 minutes.

## Table 14. Fragility under attacks

```bash
python scripts/eval_robustness.py --checkpoint checkpoints/qformer_v5_heldout_best.pt --dataset kodak --n-payload 5000
```

Runtime: about 3 minutes. The impulse-noise, median-filter, and sigma=1 rows added in revision come from `scripts/verify_revision_experiments.py`. The bit-match audit under a bypassed consistency check is produced by `scripts/measure_runtime_and_fpr.py`.

## Figure 2. Kodak Pareto front

```bash
python scripts/plot_3method_pareto.py
```

Requires the Table 3 and Table 4 JSON files to be present in `results/`. Output: `results/fig_pareto.pdf`.

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

```bash
# scalar baselines for Table 10, matched parameter count (947,691)
python -m src.training.train_scalar --d-model 104 --depth 6 --epochs 50 --save-name scalar_v2_matchparams
python -m src.training.train_scalar --d-model 104 --depth 6 --epochs 80 --save-name scalar_v3_80ep
```

The 80-epoch scalar run is required for Table 10. Comparing an 80-epoch quaternion model against a 50-epoch scalar baseline inverts the accuracy conclusion, which is why both budgets are reported.

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
