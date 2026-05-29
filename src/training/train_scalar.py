"""
Train the ScalarTransformerRevMark baseline (ablation).

Identical training schedule to train.py but uses ScalarTransformerPredictor
instead of QFormerPredictor. The PEE module and embed/extract pipeline are
identical, so the only difference is the predictor backbone.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.datasets import ImageFolderFlat  # noqa: E402
from src.models.scalar_transformer_predictor import ScalarTransformerRevMark  # noqa: E402
from src.models.qformer_revmark import (  # noqa: E402
    checkerboard_mask,
    float_to_uint8,
)
from src.training.train import (  # noqa: E402
    build_train_loader,
    build_eval_loader,
    train_one_epoch,
    evaluate,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=192,
                    help="hidden width; for matched-dim ablation use 4*d_quat (default 192 = 4*48)")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--init-T", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--save-name", type=str, default="scalar_v1")
    ap.add_argument("--eval-every", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader = build_train_loader(args.image_size, args.batch, args.num_workers)
    eval_loader = build_eval_loader(args.image_size, args.num_workers)
    print(f"Training samples: {len(train_loader.dataset)}  batches/epoch: {len(train_loader)}")
    print(f"Eval samples:     {len(eval_loader.dataset)}")

    model = ScalarTransformerRevMark(
        image_size=args.image_size,
        patch_size=args.patch_size,
        d_model=args.d_model,
        depth=args.depth,
        num_heads=args.heads,
        init_T=args.init_T,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_pred = sum(p.numel() for p in model.predictor.parameters())
    print(f"Scalar wrapper params: {n_params:,}  (predictor: {n_pred:,})")

    optimizer = AdamW(model.predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / f"{args.save_name}_train.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "a")

    print()
    print("=== Pre-training evaluation ===")
    m0 = evaluate(model, eval_loader, device)
    print(f"  PSNR={m0['psnr_mean']:.2f} dB  Cap={m0['capacity_bpp_mean']:.4f} bpp  "
          f"Bit-exact={m0['n_bit_exact']}/{m0['n_images']}")

    best_psnr = -1.0
    for epoch in range(1, args.epochs + 1):
        print()
        print(f"=== Epoch {epoch}/{args.epochs} (lr={scheduler.get_last_lr()[0]:.2e}) ===")
        ep = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        rec = {"epoch": epoch, "train_loss": ep["loss"], "lr": scheduler.get_last_lr()[0]}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print("  --- eval ---")
            em = evaluate(model, eval_loader, device)
            rec.update(em)
            print(f"  PSNR={em['psnr_mean']:.2f} dB  Cap={em['capacity_bpp_mean']:.4f} bpp  "
                  f"Bit-exact={em['n_bit_exact']}/{em['n_images']}")
            if em["psnr_mean"] > best_psnr:
                best_psnr = em["psnr_mean"]
                ckpt = ckpt_dir / f"{args.save_name}_best.pt"
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "args": vars(args), "metrics": em,
                            "model_kind": "scalar"}, ckpt)
                print(f"  saved best -> {ckpt}")

        log_fp.write(json.dumps(rec) + "\n")
        log_fp.flush()

    log_fp.close()
    final = ckpt_dir / f"{args.save_name}_final.pt"
    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                "args": vars(args), "model_kind": "scalar"}, final)
    print()
    print(f"Done. Best Kodak PSNR = {best_psnr:.2f} dB. Final model saved at {final}")


if __name__ == "__main__":
    main()
