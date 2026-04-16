# Road Crack Detection — YOLOv8 Segmentation

Detect and segment **longitudinal**, **transverse**, and **alligator** cracks
in road/pavement images using YOLOv8 instance segmentation.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Download a public crack dataset to bootstrap
#    e.g. CrackForest, CFD, or search Roboflow Universe for "road crack"

# 3. Annotate your images with Label Studio (polygon tool)
#    Export as JSON, then convert:
python scripts/prepare_dataset.py \
    --ls-json  annotations/export.json \
    --images   /path/to/raw/images \
    --output   data

# 4. Train
python train.py                             # 100 epochs, yolov8m-seg
python train.py --model yolov8l-seg.pt --epochs 150 --imgsz 1024

# 5. Evaluate on test set
python evaluate.py

# 6. Infer
python infer.py --source test_image.jpg --severity
python infer.py --source road_video.mp4 --save-video
python infer.py --source 0              # live webcam
```

---

## Project layout

```
crack_detection/
├── dataset.yaml              # YOLO dataset config
├── train.py                  # training script
├── infer.py                  # inference + visualisation
├── evaluate.py               # mAP / per-class evaluation
├── requirements.txt
├── scripts/
│   └── prepare_dataset.py    # Label Studio → YOLO converter
├── data/
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
└── runs/
    └── crack_seg/
        └── weights/
            ├── best.pt
            └── last.pt
```

---

## Classes

| ID | Name | Description |
|----|------|-------------|
| 0 | longitudinal_crack | Parallel to road direction |
| 1 | transverse_crack | Perpendicular to road direction |
| 2 | alligator_crack | Interconnected fatigue cracking |

---

## Model selection guide

| Model | Speed | Accuracy | Use case |
|-------|-------|----------|----------|
| yolov8n-seg | ★★★★★ | ★★ | Embedded / real-time |
| yolov8s-seg | ★★★★ | ★★★ | Edge GPU |
| yolov8m-seg | ★★★ | ★★★★ | **Default recommendation** |
| yolov8l-seg | ★★ | ★★★★★ | Server / high accuracy |
| yolov8x-seg | ★ | ★★★★★ | Maximum accuracy |

---

## Tips for road crack data

- Collect images from multiple times of day (lighting variation matters).
- Include both dry and wet pavement — cracks look very different.
- Use `--imgsz 1024` for fine hairline cracks; `640` is fine for alligator.
- `copy_paste=0.1` augmentation helps when alligator cracks are rare.
- If mAP plateaus early, increase `patience` or lower `lr0` to `5e-4`.

---

## Severity scoring (infer.py --severity)

| Label | Coverage |
|-------|----------|
| None | < 2% |
| Low | 2–10% |
| Moderate | 10–25% |
| Severe | > 25% |
