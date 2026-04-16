"""
infer.py — Run inference with a trained crack detection model.

Supports: single image, folder of images, video file, webcam stream.

Usage:
    python infer.py --source image.jpg
    python infer.py --source /path/to/images/
    python infer.py --source road_video.mp4 --save-video
    python infer.py --source 0              # webcam
    python infer.py --source image.jpg --severity
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Class config ─────────────────────────────────────────────────────────────
CLASS_NAMES  = {0: "longitudinal", 1: "transverse", 2: "alligator"}
CLASS_COLORS = {0: (0, 80, 255), 1: (0, 165, 255), 2: (0, 230, 230)}  # BGR


def parse_args():
    p = argparse.ArgumentParser(description="Crack detection inference")
    p.add_argument("--weights", default="runs/crack_seg/weights/best.pt")
    p.add_argument("--source",  required=True,
                   help="Image path, folder, video, or webcam index (0)")
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--device",  default="0")
    p.add_argument("--output",  default="output",
                   help="Folder to save annotated results")
    p.add_argument("--save-video",  action="store_true")
    p.add_argument("--severity",    action="store_true",
                   help="Print per-image severity score (mask area %)")
    p.add_argument("--no-display",  action="store_true",
                   help="Skip OpenCV preview window")
    return p.parse_args()


# ── Severity scoring ──────────────────────────────────────────────────────────
def crack_severity(masks, img_h, img_w):
    """Return total crack area as a percentage of image area."""
    if masks is None or len(masks) == 0:
        return 0.0
    combined = np.zeros((img_h, img_w), dtype=np.uint8)
    for m in masks.data.cpu().numpy().astype(np.uint8):
        # masks are already (H, W) after YOLO post-processing
        combined = np.maximum(combined, m)
    coverage = combined.sum() / (img_h * img_w) * 100
    return round(float(coverage), 2)


def severity_label(pct):
    if pct < 2:   return "None",     (80, 200, 80)
    if pct < 10:  return "Low",      (0, 200, 200)
    if pct < 25:  return "Moderate", (0, 165, 255)
    return             "Severe",     (0, 60, 255)


# ── Overlay drawing ───────────────────────────────────────────────────────────
def draw_results(frame, result, show_severity=False):
    img = frame.copy()
    h, w = img.shape[:2]

    if result.masks is not None:
        masks  = result.masks.data.cpu().numpy().astype(np.uint8)
        clsids = result.boxes.cls.cpu().numpy().astype(int)
        confs  = result.boxes.conf.cpu().numpy()

        overlay = img.copy()
        for mask, cls_id, conf in zip(masks, clsids, confs):
            color  = CLASS_COLORS.get(cls_id, (128, 128, 128))
            colored = np.zeros_like(img)
            colored[mask == 1] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored, 0.45, 0)

            # Contour outline
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

            # Label near bounding box
            x1, y1, x2, y2 = result.boxes.xyxy[
                list(clsids).index(cls_id)].cpu().numpy().astype(int)
            label = f"{CLASS_NAMES.get(cls_id, cls_id)} {conf:.2f}"
            cv2.rectangle(overlay, (x1, y1 - 18), (x1 + len(label)*8, y1),
                          color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        img = overlay

    if show_severity:
        pct   = crack_severity(result.masks, h, w)
        label, color = severity_label(pct)
        text  = f"Severity: {label}  ({pct:.1f}%)"
        cv2.rectangle(img, (8, 8), (10 + len(text)*10, 34), (30, 30, 30), -1)
        cv2.putText(img, text, (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return img


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    model.to(args.device)

    source = args.source
    is_video = False

    # Detect if source is a video or webcam
    try:
        cam_id = int(source)
        is_video = True
    except ValueError:
        if Path(source).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            is_video = True

    if is_video:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        writer = None
        if args.save_video:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = str(out_dir / "output.mp4")
            writer = cv2.VideoWriter(out_path,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (fw, fh))

        print("Running — press Q to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=args.conf, iou=args.iou,
                                    imgsz=args.imgsz, verbose=False)
            annotated = draw_results(frame, results[0], args.severity)
            if writer:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("Crack detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer:
            writer.release()
            print(f"Video saved to {out_path}")
        cv2.destroyAllWindows()

    else:
        # Image(s)
        paths = sorted(Path(source).glob("*")) if Path(source).is_dir() \
            else [Path(source)]
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        paths = [p for p in paths if p.suffix.lower() in image_exts]

        for img_path in paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"[warn] cannot read {img_path}, skipping")
                continue
            results = model.predict(frame, conf=args.conf, iou=args.iou,
                                    imgsz=args.imgsz, verbose=False)
            annotated = draw_results(frame, results[0], args.severity)

            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), annotated)

            if args.severity:
                pct = crack_severity(results[0].masks,
                                     frame.shape[0], frame.shape[1])
                lbl, _ = severity_label(pct)
                print(f"{img_path.name}: {lbl} ({pct}%)")

            if not args.no_display:
                cv2.imshow("Crack detection", annotated)
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    break

        cv2.destroyAllWindows()
        print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
