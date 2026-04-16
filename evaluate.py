"""
evaluate.py — Run full evaluation on the test set.

Usage:
    python evaluate.py
    python evaluate.py --weights runs/crack_seg/weights/best.pt --split test
"""

import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained crack model")
    p.add_argument("--weights", default="runs/crack_seg/weights/best.pt")
    p.add_argument("--data",    default="dataset.yaml")
    p.add_argument("--split",   default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--device",  default="0")
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.6)
    return p.parse_args()


def main():
    args = parse_args()
    model  = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=True,  # COCO-format JSON
    )

    print("\n── Segmentation metrics ───────────────────────────────────")
    print(f"  mAP50      (box): {metrics.box.map50:.4f}")
    print(f"  mAP50-95   (box): {metrics.box.map:.4f}")
    print(f"  mAP50      (seg): {metrics.seg.map50:.4f}")
    print(f"  mAP50-95   (seg): {metrics.seg.map:.4f}")
    print("────────────────────────────────────────────────────────────")

    # Per-class breakdown
    names = model.names
    print("\n── Per-class mAP50 (seg) ──────────────────────────────────")
    for i, ap in enumerate(metrics.seg.ap50):
        print(f"  {names[i]:<25} {ap:.4f}")
    print("────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
