"""
Road/Pavement Crack Detection - Inference & Visualization
=========================================================
Run crack segmentation on images, video, or webcam.
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from ultralytics import YOLO


# ── Colour palette ────────────────────────────────────────────────────────────
CRACK_COLOR   = (0, 0, 255)      # Red  (BGR) – crack mask overlay
CONTOUR_COLOR = (0, 255, 255)    # Yellow – crack boundary
TEXT_BG       = (20, 20, 20)
TEXT_FG       = (255, 255, 255)
ALPHA         = 0.45             # Mask transparency


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: tuple, alpha: float) -> np.ndarray:
    """Blend a binary mask onto a frame."""
    colored = np.zeros_like(frame)
    colored[mask > 0] = color
    return cv2.addWeighted(frame, 1.0, colored, alpha, 0)


def draw_contours(frame: np.ndarray, mask: np.ndarray, color: tuple, thickness: int = 2):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, thickness)
    return frame


def compute_crack_stats(mask: np.ndarray, frame_area: int) -> dict:
    crack_px   = int(mask.sum())
    coverage   = crack_px / frame_area * 100
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_cracks = len(contours)
    return {"crack_pixels": crack_px, "coverage_pct": coverage, "num_cracks": num_cracks}


def draw_hud(frame: np.ndarray, stats: dict, fps: float, conf_thresh: float):
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"Cracks: {stats['num_cracks']}",
        f"Coverage: {stats['coverage_pct']:.2f}%",
        f"Conf: >{conf_thresh:.2f}",
    ]
    pad, lh = 8, 22
    panel_h = len(lines) * lh + pad * 2
    cv2.rectangle(frame, (0, 0), (180, panel_h), TEXT_BG, -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (pad, pad + (i + 1) * lh - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_FG, 1, cv2.LINE_AA)
    return frame


def process_image(model, img_path: str, conf: float, save_dir: Path, show: bool):
    img_path = Path(img_path)
    frame    = cv2.imread(str(img_path))
    if frame is None:
        print(f"[!] Cannot read {img_path}")
        return

    h, w  = frame.shape[:2]
    result = model(frame, conf=conf, verbose=False)[0]
    output = frame.copy()

    merged_mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is not None:
        for seg_mask in result.masks.data:
            m = seg_mask.cpu().numpy()
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            m = (m > 0.5).astype(np.uint8)
            merged_mask = np.maximum(merged_mask, m)

    output = overlay_mask(output, merged_mask, CRACK_COLOR, ALPHA)
    output = draw_contours(output, merged_mask, CONTOUR_COLOR)
    stats  = compute_crack_stats(merged_mask, h * w)
    draw_hud(output, stats, fps=0.0, conf_thresh=conf)

    # Side-by-side comparison
    comparison = np.hstack([frame, output])
    cv2.putText(comparison, "ORIGINAL", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(comparison, "CRACK DETECTION", (w + 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CRACK_COLOR, 2)

    save_path = save_dir / f"{img_path.stem}_result{img_path.suffix}"
    cv2.imwrite(str(save_path), comparison)
    print(f"[✓] Saved → {save_path}  |  {stats['num_cracks']} crack(s), {stats['coverage_pct']:.2f}% area")

    if show:
        cv2.imshow("Crack Detection", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(model, source, conf: float, save_dir: Path, show: bool):
    cap = cv2.VideoCapture(0 if source == "webcam" else source)
    if not cap.isOpened():
        print(f"[!] Cannot open source: {source}")
        return

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if source != "webcam":
        out_path = save_dir / (Path(source).stem + "_result.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    prev_t = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, conf=conf, verbose=False)[0]
        output = frame.copy()
        merged_mask = np.zeros((h, w), dtype=np.uint8)

        if result.masks is not None:
            for seg_mask in result.masks.data:
                m = seg_mask.cpu().numpy()
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                m = (m > 0.5).astype(np.uint8)
                merged_mask = np.maximum(merged_mask, m)

        output = overlay_mask(output, merged_mask, CRACK_COLOR, ALPHA)
        output = draw_contours(output, merged_mask, CONTOUR_COLOR)

        cur_t = time.time()
        cur_fps = 1.0 / max(cur_t - prev_t, 1e-6)
        prev_t = cur_t

        stats = compute_crack_stats(merged_mask, h * w)
        draw_hud(output, stats, cur_fps, conf)

        if writer:
            writer.write(output)
        if show:
            cv2.imshow("Crack Detection", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[✓] Video processing complete.")


def main():
    parser = argparse.ArgumentParser(description="Crack Detection Inference")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained YOLOv8-seg weights (.pt)")
    parser.add_argument("--source",  type=str, required=True,
                        help="Image path / video path / 'webcam'")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save-dir", type=str, default="results",
                        help="Directory to save outputs")
    parser.add_argument("--no-show", action="store_true",
                        help="Disable display window")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[→] Loading model: {args.weights}")
    model = YOLO(args.weights)

    src   = args.source
    show  = not args.no_show

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    vid_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

    if src == "webcam":
        process_video(model, "webcam", args.conf, save_dir, show)
    elif Path(src).suffix.lower() in img_exts:
        process_image(model, src, args.conf, save_dir, show)
    elif Path(src).suffix.lower() in vid_exts:
        process_video(model, src, args.conf, save_dir, show)
    elif Path(src).is_dir():
        imgs = [p for p in Path(src).iterdir() if p.suffix.lower() in img_exts]
        print(f"[→] Processing {len(imgs)} images …")
        for img in imgs:
            process_image(model, str(img), args.conf, save_dir, show=False)
    else:
        print(f"[!] Unknown source: {src}")


if __name__ == "__main__":
    main()
