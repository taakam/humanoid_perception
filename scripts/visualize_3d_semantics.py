"""
Purpose:
Visualize COLMAP 3D points and semantic object centroids.

Expected input:
- semantic_objects_3d.json

Expected output:
- PNG 3D scatter plot.
"""

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def visualize_3d_semantics(objects_json, output_path):
    objects_json = Path(objects_json).resolve()
    output_path = Path(output_path).resolve()

    with open(objects_json, "r") as f:
        objects = json.load(f)

    centroids = []
    labels = []

    for obj in objects:
        if obj.get("has_3d_association") and obj.get("centroid_xyz") is not None:
            centroids.append(obj["centroid_xyz"])
            labels.append(obj["object_id"])

    if not centroids:
        raise RuntimeError("No objects with valid 3D centroids found.")

    centroids = np.array(centroids)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        s=60,
    )

    for point, label in zip(centroids, labels):
        ax.text(point[0], point[1], point[2], label, fontsize=8)

    ax.set_title("3D Semantic Object Centroids")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

    print(f"[OK] Saved 3D semantic visualization to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 3D semantic objects.")
    parser.add_argument("--objects", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    visualize_3d_semantics(
        objects_json=args.objects,
        output_path=args.output,
    )