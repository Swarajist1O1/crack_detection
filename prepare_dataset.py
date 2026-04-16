"""
Dataset Preparation for Road Crack Detection
=============================================
Downloads and organises public crack datasets into YOLO segmentation format.

Supported datasets
------------------
1. Crack500   – road crack segmentation (500 images, pixel-level masks)
2. CFD        – Crack Forest Dataset (118 images)
3. DeepCrack  – 537 images with fine-grained masks

YOLO segmentation label format (per .txt file):
    <class> <x1> <y1> <x2> <y2> … <xn> <yn>
    coordinates are normalised to [0, 1].

Usage
-----
    python prepare_dataset.py --dataset crack500 --output ./data
    python prepare_dataset.py --dataset custom   --images ./my_images --masks ./my_masks --output ./data
"""

import os
import cv2
import shutil
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ── Helpers ───────────────────────────────────────────────────────────────────

def mask_to_polygons(mask: np.ndarray, min_area: int = 50) -> list[list[float]]:
    """Convert binary mask → list of normalised YOLO polygon coordinates."""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        # Simplify to reduce point count
        eps  = 0.002 * cv2.arcLength(cnt, True)
        cnt  = cv2.approxPolyDP(cnt, eps, True)
        if len(cnt) < 3:
            continue
        pts = cnt.reshape(-1, 2)
        norm = []
        for x, y in pts:
            norm.extend([round(x / w, 6), round(y / h, 6)])
        polygons.append(norm)
    return polygons


def write_yolo_label(label_path: Path, polygons: list[list[float]], class_id: int = 0):
    with open(label_path, "w") as f:
        for poly in polygons:
            coords = " ".join(map(str, poly))
            f.write(f"{class_id} {coords}\n")


def split_dataset(items: list, train_ratio=0.75, val_ratio=0.15, seed=42):
    random.seed(seed)
    random.shuffle(items)
    n     = len(items)
    n_tr  = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return items[:n_tr], items[n_tr:n_tr + n_val], items[n_tr + n_val:]


def setup_dirs(output_root: Path):
    for split in ["train", "val", "test"]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


# ── Custom dataset converter ──────────────────────────────────────────────────

def convert_custom(images_dir: str, masks_dir: str, output_dir: str,
                   train_ratio=0.75, val_ratio=0.15, min_area=50):
    """
    Convert a custom image+mask dataset to YOLO segmentation format.

    Expected mask format: single-channel PNG where crack pixels are > 0.
    """
    images_dir = Path(images_dir)
    masks_dir  = Path(masks_dir)
    output_dir = Path(output_dir)
    setup_dirs(output_dir)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in img_exts:
            continue
        # Try matching mask by same stem with any extension
        mask_path = None
        for ext in [".png", ".jpg", ".bmp"]:
            candidate = masks_dir / (img_path.stem + ext)
            if candidate.exists():
                mask_path = candidate
                break
        if mask_path is None:
            print(f"[!] No mask for {img_path.name}, skipping.")
            continue
        pairs.append((img_path, mask_path))

    if not pairs:
        print("[!] No valid image-mask pairs found.")
        return

    train_p, val_p, test_p = split_dataset(pairs, train_ratio, val_ratio)
    splits = {"train": train_p, "val": val_p, "test": test_p}

    total_ok = 0
    for split, items in splits.items():
        for img_path, mask_path in tqdm(items, desc=f"  {split:5s}"):
            img  = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue

            # Binarise mask
            _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

            polygons = mask_to_polygons(binary, min_area=min_area)
            if not polygons:
                continue  # Skip images with no detectable cracks

            dst_img   = output_dir / "images" / split / img_path.name
            dst_label = output_dir / "labels" / split / (img_path.stem + ".txt")

            shutil.copy(img_path, dst_img)
            write_yolo_label(dst_label, polygons)
            total_ok += 1

    print(f"\n[✓] Converted {total_ok} image-mask pairs → {output_dir}")
    print(f"    train={len(train_p)}  val={len(val_p)}  test={len(test_p)}")


# ── Crack500 downloader ───────────────────────────────────────────────────────

def prepare_crack500(output_dir: str):
    """
    Crack500 dataset guide.

    The dataset is available at:
      https://github.com/fyangneil/pavement-crack-detection

    Steps to download manually (automated download requires institutional access):
    1. Visit the link above and request/download the dataset.
    2. Extract so you have:
         Crack500/
           training/   image/  mask/
           testing/    image/  mask/
           validation/ image/  mask/
    3. Then run:
         python prepare_dataset.py --dataset custom \\
           --images Crack500/training/image \\
           --masks  Crack500/training/mask  \\
           --output ./data
    """
    print("\n[Crack500] Manual download required.")
    print(__doc__.split("Crack500")[1].split("CFD")[0])


# ── Roboflow public crack dataset ─────────────────────────────────────────────

def prepare_roboflow(output_dir: str, api_key: str = None):
    """
    Download a public crack segmentation dataset from Roboflow Universe.

    Requires: pip install roboflow
    Dataset:  https://universe.roboflow.com/university-bswxt/crack-bphdr
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[!] Install roboflow: pip install roboflow")
        return

    if not api_key:
        print("[!] Provide your Roboflow API key via --api-key")
        return

    rf      = Roboflow(api_key=api_key)
    project = rf.workspace("university-bswxt").project("crack-bphdr")
    dataset = project.version(2).download("yolov8", location=output_dir)
    print(f"[✓] Downloaded to {dataset.location}")


# ── Augmentation helper ───────────────────────────────────────────────────────

def augment_dataset(images_dir: str, labels_dir: str, factor: int = 2):
    """
    Simple offline augmentation: horizontal flip + brightness jitter.
    Inline augmentation in YOLOv8 is usually sufficient; use this only
    if your dataset is very small (< 200 images).
    """
    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)
    count   = 0

    for img_path in tqdm(list(img_dir.iterdir()), desc="Augmenting"):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img  = cv2.imread(str(img_path))
        text = lbl_path.read_text()

        for i in range(factor):
            # Horizontal flip
            aug_img = cv2.flip(img, 1)
            lines   = []
            for line in text.strip().splitlines():
                parts = line.split()
                cls   = parts[0]
                coords = list(map(float, parts[1:]))
                # Flip x-coords: x_new = 1 - x
                flipped = []
                for j in range(0, len(coords), 2):
                    flipped.extend([round(1 - coords[j], 6), coords[j + 1]])
                lines.append(f"{cls} " + " ".join(map(str, flipped)))

            aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            cv2.imwrite(str(img_dir / aug_name), aug_img)
            (lbl_dir / (img_path.stem + f"_aug{i}.txt")).write_text("\n".join(lines))
            count += 1

    print(f"[✓] Created {count} augmented samples.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Crack Dataset Preparation")
    parser.add_argument("--dataset",  type=str, default="custom",
                        choices=["custom", "crack500", "roboflow"],
                        help="Dataset source")
    parser.add_argument("--images",   type=str, help="Path to images dir (custom)")
    parser.add_argument("--masks",    type=str, help="Path to masks dir  (custom)")
    parser.add_argument("--output",   type=str, default="./data",
                        help="Output directory for YOLO dataset")
    parser.add_argument("--api-key",  type=str, help="Roboflow API key")
    parser.add_argument("--augment",  action="store_true",
                        help="Apply offline augmentation after conversion")
    args = parser.parse_args()

    if args.dataset == "custom":
        if not args.images or not args.masks:
            parser.error("--images and --masks are required for custom dataset")
        convert_custom(args.images, args.masks, args.output)
        if args.augment:
            for split in ["train"]:
                augment_dataset(
                    f"{args.output}/images/{split}",
                    f"{args.output}/labels/{split}",
                    factor=2,
                )
    elif args.dataset == "crack500":
        prepare_crack500(args.output)
    elif args.dataset == "roboflow":
        prepare_roboflow(args.output, args.api_key)


if __name__ == "__main__":
    main()
