"""
Training loop for the QFormer predictor inside QFormer-RevMark.

Self-supervised objective
-------------------------
For each training image, randomly mask one half of the checkerboard pattern
(Cross or Dot), feed the masked image to the predictor, and minimise the MSE
on the masked positions.

This trains the predictor to be a strong "inpainting" model: it learns to
predict the missing colour values from the surrounding context. The lower the
prediction error, the higher the PEE embedding capacity at a fixed PSNR.

Usage
-----
    python -m src.training.train --epochs 20 --batch 16 --image-size 256
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.datasets import ImageFolderFlat  # noqa: E402
from src.models.qformer_revmark import (
    QFormerRevMark as _QRevMark,  # noqa: E402
    QFormerRevMark,
    checkerboard_mask,
    float_to_uint8,
)


# ============================================================
# Data
# ============================================================
def build_train_loader(image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    """Combine all training datasets we have on disk (COCO + CLIC + Kodak)."""
    datasets = []
    for name in ("coco", "clic", "kodak"):
        root = ROOT / "data" / name
        try:
            ds = ImageFolderFlat(str(root), image_size=image_size, crop="random")
            datasets.append(ds)
        except (FileNotFoundError, RuntimeError):
            pass
    if not datasets:
        raise RuntimeError("no training data found under data/")
    big = ConcatDataset(datasets)
    return DataLoader(
        big,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )


def build_eval_loader(image_size: int, num_workers: int) -> DataLoader:
    """Kodak as canonical eval set (24 images, center-crop)."""
    root = ROOT / "data" / "kodak"
    ds = ImageFolderFlat(str(root), image_size=image_size, crop="center")
    return DataLoader(ds, batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=True)


# ============================================================
# Self-supervised training step
# ============================================================
def _masked_prediction_loss(
    predictor: nn.Module,
    rgb: torch.Tensor,
    cross_mask: torch.Tensor,
    dot_mask: torch.Tensor,
    pass_kind: str,
) -> torch.Tensor:
    """
    pass_kind = 'predict_dot'    : input = Cross-only, target = Dot pixels
                'predict_cross'  : input = Dot-only,   target = Cross pixels
    """
    if pass_kind == "predict_dot":
        input_mask = cross_mask
        target_mask = dot_mask
    elif pass_kind == "predict_cross":
        input_mask = dot_mask
        target_mask = cross_mask
    else:
        raise ValueError(pass_kind)

    masked_input = rgb * input_mask
    pred = predictor(masked_input)
    # MSE on target positions only, plus an integer-alignment L1 term.
    diff = (pred - rgb) * target_mask
    n = (target_mask.sum() * rgb.size(0) * rgb.size(1)).clamp_min(1.0)
    mse_term = diff.pow(2).sum() / n
    # Integer-aware: penalise distance from round(pred*255) to round(target*255)
    # using straight-through estimator so gradients flow.
    import torch as _torch
    pred255 = pred * 255.0
    rgb255 = rgb * 255.0
    pred255_ste = pred255 + (_torch.round(pred255) - pred255).detach()
    int_diff = (pred255_ste - _torch.round(rgb255)) * target_mask
    int_term = int_diff.abs().sum() / (n * 255.0)
    return mse_term + 0.5 * int_term


def train_one_epoch(
    model: QFormerRevMark,
    loader: DataLoader,
    optimizer: AdamW,
    device: str,
    log_every: int = 20,
) -> Dict[str, float]:
    model.train()
    predictor = model.predictor
    image_size = model.image_size
    cross_mask = checkerboard_mask(image_size, image_size, parity=0, device=device, dtype=torch.float32)
    dot_mask = 1.0 - cross_mask

    total_loss = 0.0
    n_batches = 0
    t0 = time.time()
    for it, (rgb, _) in enumerate(loader):
        rgb = rgb.to(device, non_blocking=True)
        # alternate which half is predicted each batch
        pass_kind = "predict_dot" if it % 2 == 0 else "predict_cross"
        loss = _masked_prediction_loss(predictor, rgb, cross_mask, dot_mask, pass_kind)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (it + 1) % log_every == 0:
            speed = (it + 1) / (time.time() - t0)
            print(
                f"  it {it + 1:4d}  loss={loss.item():.5f}  avg={total_loss / n_batches:.5f}  "
                f"{speed:.1f} it/s",
                flush=True,
            )
    return {"loss": total_loss / max(n_batches, 1)}


# ============================================================
# Evaluation (PSNR + capacity)
# ============================================================
@torch.no_grad()
def evaluate(
    model: QFormerRevMark,
    loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    model.eval()
    image_size = model.image_size

    n_images = 0
    psnr_sum = 0.0
    cap_sum = 0.0
    bit_exact_count = 0

    for rgb, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        img_uint8 = float_to_uint8(rgb)
        bits = torch.randint(0, 2, img_uint8.shape, dtype=torch.int64, device=device)

        er = model.embed(img_uint8, bits)
        xr = model.extract(er.watermarked_uint8)

        # Per-image stats
        for i in range(rgb.size(0)):
            orig = img_uint8[i]
            marked = er.watermarked_uint8[i]
            mse = ((marked - orig).float() ** 2).mean().item()
            psnr = float("inf") if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)
            psnr_sum += psnr if not math.isinf(psnr) else 100.0  # cap inf at 100 dB for avg
            cap_sum += er.capacity_bpp
            bit_exact_count += int(torch.equal(xr.recovered_uint8[i], orig))
            n_images += 1

    return {
        "psnr_mean": psnr_sum / max(n_images, 1),
        "capacity_bpp_mean": cap_sum / max(n_images, 1),
        "n_images": n_images,
        "n_bit_exact": bit_exact_count,
    }


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--d-quat", type=int, default=48)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--init-T", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--save-name", type=str, default="qformer_v1")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--limit-batches", type=int, default=0,
                    help="if > 0, stop training each epoch after this many batches (smoke test)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader = build_train_loader(args.image_size, args.batch, args.num_workers)
    eval_loader = build_eval_loader(args.image_size, args.num_workers)
    print(f"Training samples: {len(train_loader.dataset)}  "
          f"(batches per epoch: {len(train_loader)})")
    print(f"Eval samples:     {len(eval_loader.dataset)}")

    model = QFormerRevMark(
        image_size=args.image_size,
        patch_size=args.patch_size,
        d_quaternion=args.d_quat,
        depth=args.depth,
        num_heads=args.heads,
        init_T=args.init_T,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Only train the predictor parameters (PEE threshold is fixed during this stage).
    optimizer = AdamW(model.predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / f"{args.save_name}_train.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "a")

    history: List[dict] = []
    best_psnr = -1.0
    print()
    print("=== Pre-training evaluation (untrained predictor) ===")
    eval_metrics = evaluate(model, eval_loader, device)
    print(
        f"  PSNR={eval_metrics['psnr_mean']:.2f} dB  "
        f"Capacity={eval_metrics['capacity_bpp_mean']:.4f} bpp  "
        f"Bit-exact={eval_metrics['n_bit_exact']}/{eval_metrics['n_images']}"
    )

    for epoch in range(1, args.epochs + 1):
        print()
        print(f"=== Epoch {epoch}/{args.epochs} (lr={scheduler.get_last_lr()[0]:.2e}) ===")
        if args.limit_batches > 0:
            # Wrap loader so we stop after limit_batches per epoch
            def limited():
                for i, batch in enumerate(train_loader):
                    if i >= args.limit_batches:
                        break
                    yield batch
            ep_metrics = train_one_epoch(
                model, _ListIterableWrapper(limited(), len(train_loader)), optimizer, device,
            )
        else:
            ep_metrics = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        rec = {"epoch": epoch, "train_loss": ep_metrics["loss"], "lr": scheduler.get_last_lr()[0]}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print("  --- eval ---")
            eval_metrics = evaluate(model, eval_loader, device)
            rec.update(eval_metrics)
            print(
                f"  PSNR={eval_metrics['psnr_mean']:.2f} dB  "
                f"Capacity={eval_metrics['capacity_bpp_mean']:.4f} bpp  "
                f"Bit-exact={eval_metrics['n_bit_exact']}/{eval_metrics['n_images']}"
            )

            if eval_metrics["psnr_mean"] > best_psnr:
                best_psnr = eval_metrics["psnr_mean"]
                ckpt_path = ckpt_dir / f"{args.save_name}_best.pt"
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                    "metrics": eval_metrics,
                }, ckpt_path)
                print(f"  saved best -> {ckpt_path}")

        history.append(rec)
        log_fp.write(json.dumps(rec) + "\n")
        log_fp.flush()

    log_fp.close()

    # Final save
    final_path = ckpt_dir / f"{args.save_name}_final.pt"
    torch.save({
        "model": model.state_dict(),
        "epoch": args.epochs,
        "args": vars(args),
    }, final_path)
    print()
    print(f"Done. Best Kodak PSNR = {best_psnr:.2f} dB. Final model saved at {final_path}")


class _ListIterableWrapper:
    """Wraps a generator so DataLoader-like __len__ works for tqdm-free logging."""
    def __init__(self, it, length):
        self._it = it
        self._length = length
    def __iter__(self):
        return iter(self._it)
    def __len__(self):
        return self._length


if __name__ == "__main__":
    main()
