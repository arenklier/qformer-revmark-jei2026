#!/usr/bin/env bash
# Run the full evaluation suite on a given checkpoint.
#   bash scripts/run_final_eval.sh checkpoints/qformer_v2_best.pt
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

CKPT="${1:-checkpoints/qformer_v2_best.pt}"
TAG="$(basename "${CKPT}" .pt)"
echo "============================================================"
echo "  Final evaluation suite for: ${CKPT}  (tag=${TAG})"
echo "============================================================"

echo
echo ">>>> [1/4]  T-sweep on Kodak"
python -m scripts.eval_capacity_curve --checkpoint "${CKPT}" --dataset kodak \
    --T-values 0,1,2,3,5,8,12,20

echo
echo ">>>> [2/4]  T-sweep on CLIC"
python -m scripts.eval_capacity_curve --checkpoint "${CKPT}" --dataset clic \
    --T-values 0,1,2,3,5,8 --batch 4

echo
echo ">>>> [3/4]  Empirical I(R;G;B) on all datasets (Lemma 1 validation)"
python -m scripts.measure_mi_errors --checkpoint "${CKPT}" \
    --datasets kodak,sipi,clic,coco,isic --batch 4

echo
echo ">>>> [4/4]  Plot PSNR-vs-bpp"
python -m scripts.plot_curves --datasets kodak,clic --out "results/fig_psnr_vs_bpp_${TAG}"

echo
echo "============================================================"
echo "  Done. Artifacts under results/"
ls -lh results/ | tail -20
echo "============================================================"
