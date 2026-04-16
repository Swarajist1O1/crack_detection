"""
train.py — YOLOv8 segmentation training for road crack detection.

Usage:
    python train.py                         # default settings
    python train.py --epochs 150 --imgsz 1024
    python train.py --resume runs/crack_seg/weights/last.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8-seg on road crack data")
    p.add_argument("--model",   default="yolov8m-seg.pt",
                   help="Backbone: yolov8n/s/m/l/x-seg.pt  (n=fastest, x=best)")
    p.add_argument("--data",    default="dataset.yaml")
    p.add_argument("--epochs",  type=int, default=100)
    p.add_argument("--imgsz",   type=int, default=640,
                   help="Input resolution (use 1024 for large-crack detail)")
    p.add_argument("--batch",   type=int, default=16,
                   help="Batch size; set -1 for auto based on GPU VRAM")
    p.add_argument("--device",  default="0",
                   help="CUDA device id (e.g. '0', '0,1') or 'cpu'")
    p.add_argument("--project", default="runs",  help="Output root directory")
    p.add_argument("--name",    default="crack_seg")
    p.add_argument("--resume",  default=None,
                   help="Resume training from a checkpoint (.pt path)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load or resume model ─────────────────────────────────────────────────
    if args.resume:
        model = YOLO(args.resume)
        print(f"[resume] continuing from {args.resume}")
    else:
        model = YOLO(args.model)
        print(f"[start]  training {args.model}")

    # ── Train ─────────────────────────────────────────────────────────────────
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,

        # ── Optimiser & LR schedule ──────────────────────────────────────────
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,

        # ── Regularisation ────────────────────────────────────────────────────
        weight_decay=5e-4,
        dropout=0.0,       # increase to 0.1–0.3 if overfitting

        # ── Augmentation (tuned for road textures) ────────────────────────────
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        flipud=0.0,        # roads are rarely photographed upside-down
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,

        # ── Segmentation head ─────────────────────────────────────────────────
        overlap_mask=True,
        mask_ratio=4,

        # ── Misc ──────────────────────────────────────────────────────────────
        patience=30,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n✓  Training complete. Best weights: {best}")
    return results


if __name__ == "__main__":
    main()
