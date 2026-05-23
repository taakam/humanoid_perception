"""
Purpose:
Associate fused 2D semantic object instances with COLMAP 3D points.

Method:
- Read semantic_objects.json from fuse_semantics.py
- Read COLMAP images.txt and points3D.txt
- For each detection bbox, find COLMAP 2D keypoints inside the bbox
- Use their linked 3D point IDs
- Aggregate 3D points per object_id
- Compute centroid and 3D bounding box

Expected input:
- semantic_objects.json
- COLMAP sparse model directory:
  sparse/0/

Expected output:
- semantic_objects_3d.json
"""

from pathlib import Path
import argparse
import json
import subprocess
from collections import defaultdict

import numpy as np


def run_command(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(map(str, cmd))}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")


def ensure_text_colmap_model(sparse_dir: Path, text_dir: Path) -> Path:
    """
    Converts COLMAP binary model to text format if needed.
    """

    required_text_files = [
        text_dir / "cameras.txt",
        text_dir / "images.txt",
        text_dir / "points3D.txt",
    ]

    if all(path.exists() for path in required_text_files):
        print(f"[OK] Existing COLMAP text model found at: {text_dir}")
        return text_dir

    text_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(sparse_dir),
            "--output_path",
            str(text_dir),
            "--output_type",
            "TXT",
        ]
    )

    for path in required_text_files:
        if not path.exists():
            raise RuntimeError(f"COLMAP text conversion failed. Missing: {path}")

    return text_dir


def parse_points3d(points3d_txt: Path) -> dict[int, np.ndarray]:
    points = {}

    with open(points3d_txt, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            point_id = int(parts[0])
            xyz = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])],
                dtype=float,
            )

            points[point_id] = xyz

    return points


def parse_images(images_txt: Path) -> dict[str, list[dict]]:
    """
    Parses COLMAP images.txt.

    Returns:
    {
        "frame_0001.jpg": [
            {"x": ..., "y": ..., "point3d_id": ...},
            ...
        ]
    }
    """

    image_observations = {}

    with open(images_txt, "r") as f:
        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith("#")
        ]

    i = 0

    while i < len(lines):
        image_meta = lines[i].split()
        points_line = lines[i + 1].split()

        image_name = image_meta[-1]

        observations = []

        for j in range(0, len(points_line), 3):
            x = float(points_line[j])
            y = float(points_line[j + 1])
            point3d_id = int(points_line[j + 2])

            if point3d_id == -1:
                continue

            observations.append(
                {
                    "x": x,
                    "y": y,
                    "point3d_id": point3d_id,
                }
            )

        image_observations[image_name] = observations

        i += 2

    return image_observations


def bbox_contains_point(bbox, x: float, y: float) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def associate_objects_3d(
    semantic_objects_json: str | Path,
    sparse_dir: str | Path,
    output_json: str | Path,
    min_points: int = 5,
) -> Path:
    semantic_objects_json = Path(semantic_objects_json).resolve()
    sparse_dir = Path(sparse_dir).resolve()
    output_json = Path(output_json).resolve()

    if not semantic_objects_json.exists():
        raise FileNotFoundError(f"Semantic objects file not found: {semantic_objects_json}")

    if not sparse_dir.exists():
        raise FileNotFoundError(f"COLMAP sparse directory not found: {sparse_dir}")

    text_dir = sparse_dir.parent / "text_model"
    ensure_text_colmap_model(sparse_dir, text_dir)

    points3d = parse_points3d(text_dir / "points3D.txt")
    image_observations = parse_images(text_dir / "images.txt")

    with open(semantic_objects_json, "r") as f:
        semantic_objects = json.load(f)

    output_objects = []

    for obj in semantic_objects:
        object_id = obj["object_id"]
        label = obj["label"]

        matched_point_ids = set()

        for det in obj["detections"]:
            image_name = Path(det["image"]).name
            bbox = det["bbox_xyxy"]

            if image_name not in image_observations:
                continue

            for obs in image_observations[image_name]:
                if bbox_contains_point(bbox, obs["x"], obs["y"]):
                    matched_point_ids.add(obs["point3d_id"])

        matched_points = [
            points3d[point_id]
            for point_id in matched_point_ids
            if point_id in points3d
        ]

        if len(matched_points) < min_points:
            print(
                f"[WARN] {object_id} matched only {len(matched_points)} 3D points. "
                "Skipping 3D centroid."
            )

            obj_3d = {
                **obj,
                "has_3d_association": False,
                "num_3d_points": len(matched_points),
                "centroid_xyz": None,
                "bbox_3d_min": None,
                "bbox_3d_max": None,
            }

            output_objects.append(obj_3d)
            continue

        points_array = np.vstack(matched_points)

        centroid = points_array.mean(axis=0)
        bbox_min = points_array.min(axis=0)
        bbox_max = points_array.max(axis=0)

        obj_3d = {
            **obj,
            "has_3d_association": True,
            "num_3d_points": int(len(matched_points)),
            "centroid_xyz": centroid.tolist(),
            "bbox_3d_min": bbox_min.tolist(),
            "bbox_3d_max": bbox_max.tolist(),
        }

        output_objects.append(obj_3d)

        print(
            f"[OK] {object_id}: {len(matched_points)} points, "
            f"centroid={centroid.tolist()}"
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(output_objects, f, indent=2)

    print(f"[OK] Saved 3D-associated semantic objects to {output_json}")

    return output_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Associate 2D semantic objects with COLMAP 3D points."
    )

    parser.add_argument("--objects", required=True, help="Path to semantic_objects.json")
    parser.add_argument("--sparse", required=True, help="Path to COLMAP sparse/0 directory")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--min-points", type=int, default=5)

    args = parser.parse_args()

    associate_objects_3d(
        semantic_objects_json=args.objects,
        sparse_dir=args.sparse,
        output_json=args.output,
        min_points=args.min_points,
    )