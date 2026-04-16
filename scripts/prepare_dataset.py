"""
scripts/prepare_dataset.py
Converts Label Studio JSON export to YOLO segmentation format.

Usage:
    python scripts/prepare_dataset.py \
        --ls-json  annotations/export.json \
        --images   /path/to/all/images \
        --output   data \
        --split    0.7 0.2 0.1
"""

import argparse
import json
import random
import shutil
from pathlib import Path


CLASS_MAP = {
    "longitudinal_crack": 0,
    "transverse_crack":   1,
    "alligator_crack":    2,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ls-json",  required=True)
    p.add_argument("--images",   required=True)
    p.add_argument("--output",   default="data")
    p.add_argument("--split",    nargs=3, type=float, default=[0.7, 0.2, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def polygon_to_yolo(points, img_w, img_h):
    flat = []
    for pt in points:
        flat.append(round(pt["x"] / 100, 6))
        flat.append(round(pt["y"] / 100, 6))
    return flat


def main():
    args = parse_args()
    random.seed(args.seed)

    with open(args.ls_json) as f:
        tasks = json.load(f)

    out = Path(args.output)
    for s in ["train", "val", "test"]:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    random.shuffle(tasks)
    n = len(tasks)
    tr, va = int(n * args.split[0]), int(n * (args.split[0] + args.split[1]))
    assigned = (
        [(t, "train") for t in tasks[:tr]] +
        [(t, "val")   for t in tasks[tr:va]] +
        [(t, "test")  for t in tasks[va:]]
    )

    skipped, converted = 0, 0
    for task, split in assigned:
        img_filename = Path(task["data"]["image"]).name
        img_src = Path(args.images) / img_filename
        if not img_src.exists():
            print(f"[warn] {img_src} not found")
            skipped += 1
            continue

        yolo_lines = []
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                if result["type"] != "polygonlabels":
                    continue
                label = result["value"]["polygonlabels"][0]
                cls_id = CLASS_MAP.get(label)
                if cls_id is None:
                    continue
                coords = polygon_to_yolo(result["value"]["points"],
                                         result["original_width"],
                                         result["original_height"])
                if len(coords) < 6:
                    continue
                yolo_lines.append(f"{cls_id} " + " ".join(map(str, coords)))

        shutil.copy2(img_src, out / "images" / split / img_filename)
        (out / "labels" / split / (Path(img_filename).stem + ".txt")
         ).write_text("\n".join(yolo_lines))
        converted += 1

    print(f"Converted {converted} images ({skipped} skipped)")
    print(f"Train {tr}  Val {va-tr}  Test {n-va}")


if __name__ == "__main__":
    main()
