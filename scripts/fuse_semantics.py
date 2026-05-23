"""
Purpose:
Fuse 2D semantic detections across frames into persistent object instances.

This converts frame-by-frame detections such as:
- box
- chair
- bottle

into clean robot-facing object IDs such as:
- box_001
- chair_001
- bottle_002

Expected input:
- detections.json from semantic_detect.py

Expected output:
- semantic_objects.json
- fused_visualizations/ folder with annotated object IDs
"""

from pathlib import Path
import argparse
import json
from collections import defaultdict

import cv2


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def center_distance(bbox_a, bbox_b):
    ax, ay = bbox_center(bbox_a)
    bx, by = bbox_center(bbox_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def iou(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = bbox_area([inter_x1, inter_y1, inter_x2, inter_y2])
    union_area = bbox_area(bbox_a) + bbox_area(bbox_b) - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def assign_internal_id(label, instance_counts):
    instance_counts[label] += 1
    return f"{label}_track_{instance_counts[label]:03d}"


def renumber_robot_ids(objects):
    label_counts = defaultdict(int)

    for obj in objects:
        label = obj["label"]
        label_counts[label] += 1
        obj["track_id"] = obj["object_id"]
        obj["object_id"] = f"{label}_{label_counts[label]:03d}"

    return objects


def fuse_semantics(
    detections_json: str | Path,
    output_json: str | Path,
    min_views: int = 3,
    min_mean_confidence: float = 0.35,
    max_center_distance: float = 120.0,
    min_iou: float = 0.05,
    max_frame_gap: int = 5,
    visualize: bool = True,
) -> Path:
    detections_json = Path(detections_json).resolve()
    output_json = Path(output_json).resolve()

    if not detections_json.exists():
        raise FileNotFoundError(f"Detections file not found: {detections_json}")

    with open(detections_json, "r") as f:
        frame_detections = json.load(f)

    tracks = []
    instance_counts = defaultdict(int)

    for frame_idx, frame in enumerate(frame_detections):
        image_path = frame["image"]

        for det in frame["detections"]:
            label = det["label"]
            bbox = det["bbox_xyxy"]
            confidence = det["confidence"]
            affordances = det.get("affordances", [])

            best_track = None
            best_score = -1.0

            for track in tracks:
                if track["label"] != label:
                    continue

                if frame_idx - track["last_frame_idx"] > max_frame_gap:
                    continue

                last_bbox = track["last_bbox"]

                box_iou = iou(bbox, last_bbox)
                dist = center_distance(bbox, last_bbox)

                if box_iou < min_iou and dist > max_center_distance:
                    continue

                score = box_iou - (dist / max_center_distance) * 0.1

                if score > best_score:
                    best_score = score
                    best_track = track

            if best_track is None:
                internal_id = assign_internal_id(label, instance_counts)

                best_track = {
                    "object_id": internal_id,
                    "label": label,
                    "detections": [],
                    "last_bbox": bbox,
                    "last_frame_idx": frame_idx,
                }

                tracks.append(best_track)

            best_track["detections"].append(
                {
                    "image": image_path,
                    "frame_idx": frame_idx,
                    "confidence": confidence,
                    "bbox_xyxy": bbox,
                    "affordances": affordances,
                }
            )

            best_track["last_bbox"] = bbox
            best_track["last_frame_idx"] = frame_idx

    semantic_objects = []

    for track in tracks:
        detections = track["detections"]
        views_seen = len(set(det["image"] for det in detections))
        mean_confidence = sum(det["confidence"] for det in detections) / len(detections)

        if views_seen < min_views:
            continue

        if mean_confidence < min_mean_confidence:
            continue

        affordances = sorted(
            set(
                aff
                for det in detections
                for aff in det.get("affordances", [])
            )
        )

        semantic_objects.append(
            {
                "object_id": track["object_id"],
                "label": track["label"],
                "views_seen": views_seen,
                "num_detections": len(detections),
                "mean_confidence": mean_confidence,
                "affordances": affordances,
                "detections": detections,
            }
        )

    semantic_objects = renumber_robot_ids(semantic_objects)

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(semantic_objects, f, indent=2)

    print(f"[OK] Saved {len(semantic_objects)} object instances to {output_json}")

    if visualize:
        vis_dir = output_json.parent / "fused_visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        frame_to_detections = defaultdict(list)

        for obj in semantic_objects:
            for det in obj["detections"]:
                frame_to_detections[det["image"]].append(
                    {
                        "object_id": obj["object_id"],
                        "label": obj["label"],
                        "confidence": det["confidence"],
                        "bbox_xyxy": det["bbox_xyxy"],
                    }
                )

        for frame in frame_detections:
            image_path = Path(frame["image"])
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"[WARN] Could not read image: {image_path}")
                continue

            for det in frame_to_detections.get(str(image_path), []):
                x1, y1, x2, y2 = map(int, det["bbox_xyxy"])

                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                cv2.putText(
                    image,
                    f"{det['object_id']} {det['confidence']:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

            cv2.imwrite(str(vis_dir / image_path.name), image)

        print(f"[OK] Saved fused visualizations to {vis_dir}")

    return output_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fuse detections into persistent semantic object instances."
    )

    parser.add_argument("--detections", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--max-center-distance", type=float, default=120.0)
    parser.add_argument("--min-iou", type=float, default=0.05)
    parser.add_argument("--max-frame-gap", type=int, default=5)
    parser.add_argument("--no-visualize", action="store_true")

    args = parser.parse_args()

    fuse_semantics(
        detections_json=args.detections,
        output_json=args.output,
        min_views=args.min_views,
        min_mean_confidence=args.min_confidence,
        max_center_distance=args.max_center_distance,
        min_iou=args.min_iou,
        max_frame_gap=args.max_frame_gap,
        visualize=not args.no_visualize,
    )