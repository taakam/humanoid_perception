"""
Purpose:
Run 2D semantic object detection on rendered images, save detections as JSON,
and create visualizations with bounding boxes.

Expected input:
- Directory of rendered images.

Expected output:
- detections.json
- visualizations/ folder with annotated images.
"""

from pathlib import Path
import argparse
import json

import cv2
from ultralytics import YOLO


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

DEFAULT_INDOOR_CLASSES = [
    "floor", "wall", "ceiling", "door", "window", "stairs",
    "table", "desk", "chair", "sofa", "bed", "shelf", "cabinet",
    "monitor", "laptop", "keyboard", "mouse", "tv", "lamp",
    "bottle", "cup", "book", "box", "bag", "remote", "phone",
    "clothes", "towel", "blanket", "pillow", "shoes",
    "plant", "picture", "mirror",
    "person", "toy",
]


AFFORDANCE_MAP = {
    "floor": ["traversable"],
    "wall": ["boundary"],
    "ceiling": ["boundary"],
    "door": ["passage", "openable"],
    "window": ["boundary", "transparent_surface"],
    "stairs": ["traversable_with_constraint"],

    "table": ["support_surface", "obstacle"],
    "desk": ["support_surface", "workspace", "obstacle"],
    "chair": ["sittable", "movable_obstacle"],
    "sofa": ["sittable", "obstacle"],
    "bed": ["rest_surface", "obstacle"],
    "shelf": ["storage_surface", "obstacle"],
    "cabinet": ["storage", "openable", "obstacle"],

    "monitor": ["electronic_object", "fragile"],
    "laptop": ["electronic_object", "fragile", "movable"],
    "keyboard": ["electronic_object", "movable"],
    "mouse": ["electronic_object", "graspable"],
    "tv": ["electronic_object", "fragile"],
    "lamp": ["illumination_source", "fragile"],

    "bottle": ["graspable", "movable"],
    "toy": ["graspable", "movable"],
    "cup": ["graspable", "movable", "container"],
    "book": ["graspable", "movable"],
    "box": ["movable", "container"],
    "bag": ["movable", "container", "deformable"],
    "remote": ["graspable", "movable"],
    "phone": ["graspable", "electronic_object", "fragile"],

    "clothes": ["deformable", "movable"],
    "towel": ["deformable", "movable"],
    "blanket": ["deformable", "movable"],
    "pillow": ["deformable", "movable"],
    "shoes": ["movable"],

    "plant": ["fragile", "movable_or_static"],
    "picture": ["decorative", "fragile"],
    "mirror": ["reflective_surface", "fragile"],

    "person": ["human_agent", "dynamic_obstacle"],
}


def parse_custom_classes(custom_classes: str | None) -> list[str]:
    if not custom_classes:
        return []

    return [
        item.strip()
        for item in custom_classes.split(",")
        if item.strip()
    ]


def detect_semantics(
    image_dir: str | Path,
    output_json: str | Path,
    model_name: str = "yolov8x-world.pt",
    confidence: float = 0.25,
    custom_classes: list[str] | None = None,
) -> Path:
    image_dir = Path(image_dir).resolve()
    output_json = Path(output_json).resolve()

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in VALID_IMAGE_EXTS
    )

    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    output_json.parent.mkdir(parents=True, exist_ok=True)

    vis_dir = output_json.parent / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    detection_classes = DEFAULT_INDOOR_CLASSES.copy()

    if custom_classes:
        detection_classes.extend(custom_classes)

    detection_classes = sorted(set(detection_classes))

    print(f"[INFO] Using {len(detection_classes)} semantic classes:")
    print(detection_classes)

    model = YOLO(model_name)
    model.set_classes(detection_classes)

    results_json = []

    for image_path in image_paths:
        print(f"[DETECT] {image_path.name}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        results = model.predict(
            source=str(image_path),
            conf=confidence,
            verbose=False,
        )

        detections = []

        for result in results:
            names = result.names

            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)

                affordances = AFFORDANCE_MAP.get(label, ["unknown_affordance"])

                detections.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "bbox_xyxy": xyxy,
                        "affordances": affordances,
                    }
                )

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    image,
                    f"{label} {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        cv2.imwrite(str(vis_dir / image_path.name), image)

        results_json.append(
            {
                "image": str(image_path),
                "detections": detections,
            }
        )

    with open(output_json, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"[OK] Saved detections to {output_json}")
    print(f"[OK] Saved visualizations to {vis_dir}")

    return output_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run semantic object detection on images."
    )

    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="yolov8x-world.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--custom-classes",
        default=None,
        help="Comma-separated extra classes, e.g. 'slippers,napkin,curtain'",
    )

    args = parser.parse_args()

    detect_semantics(
        image_dir=args.images,
        output_json=args.output,
        model_name=args.model,
        confidence=args.conf,
        custom_classes=parse_custom_classes(args.custom_classes),
    )