"""
Purpose:
Build a basic 3D semantic scene graph from 3D-associated objects.

Expected input:
- semantic_objects_3d.json

Expected output:
- scene_graph.json
"""

from pathlib import Path
import argparse
import json
import numpy as np


def distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def build_scene_graph(
    objects_json: str | Path,
    output_json: str | Path,
    near_threshold: float = 0.75,
    vertical_threshold: float = 0.20,
) -> Path:
    objects_json = Path(objects_json).resolve()
    output_json = Path(output_json).resolve()

    with open(objects_json, "r") as f:
        objects = json.load(f)

    valid_objects = [
        obj for obj in objects
        if obj.get("has_3d_association") and obj.get("centroid_xyz") is not None
    ]

    nodes = []
    edges = []

    scene_node = {
        "id": "scene_001",
        "type": "scene",
        "label": "reconstructed_room",
    }

    nodes.append(scene_node)

    for obj in valid_objects:
        nodes.append(
            {
                "id": obj["object_id"],
                "type": "object",
                "label": obj["label"],
                "centroid_xyz": obj["centroid_xyz"],
                "bbox_3d_min": obj.get("bbox_3d_min"),
                "bbox_3d_max": obj.get("bbox_3d_max"),
                "affordances": obj.get("affordances", []),
                "mean_confidence": obj.get("mean_confidence"),
                "num_3d_points": obj.get("num_3d_points"),
            }
        )

        edges.append(
            {
                "source": "scene_001",
                "target": obj["object_id"],
                "relationship": "contains",
            }
        )

        for affordance in obj.get("affordances", []):
            affordance_id = f"affordance::{affordance}"

            if not any(node["id"] == affordance_id for node in nodes):
                nodes.append(
                    {
                        "id": affordance_id,
                        "type": "affordance",
                        "label": affordance,
                    }
                )

            edges.append(
                {
                    "source": obj["object_id"],
                    "target": affordance_id,
                    "relationship": "has_affordance",
                }
            )

    for i, obj_a in enumerate(valid_objects):
        for obj_b in valid_objects[i + 1:]:
            id_a = obj_a["object_id"]
            id_b = obj_b["object_id"]

            ca = obj_a["centroid_xyz"]
            cb = obj_b["centroid_xyz"]

            d = distance(ca, cb)

            if d <= near_threshold:
                edges.append(
                    {
                        "source": id_a,
                        "target": id_b,
                        "relationship": "near",
                        "distance": d,
                    }
                )

                edges.append(
                    {
                        "source": id_b,
                        "target": id_a,
                        "relationship": "near",
                        "distance": d,
                    }
                )

            dz = cb[2] - ca[2]

            if dz > vertical_threshold:
                edges.append(
                    {
                        "source": id_b,
                        "target": id_a,
                        "relationship": "above",
                        "delta_z": dz,
                    }
                )
                edges.append(
                    {
                        "source": id_a,
                        "target": id_b,
                        "relationship": "below",
                        "delta_z": dz,
                    }
                )

            elif dz < -vertical_threshold:
                edges.append(
                    {
                        "source": id_a,
                        "target": id_b,
                        "relationship": "above",
                        "delta_z": -dz,
                    }
                )
                edges.append(
                    {
                        "source": id_b,
                        "target": id_a,
                        "relationship": "below",
                        "delta_z": -dz,
                    }
                )

    scene_graph = {
        "metadata": {
            "num_objects": len(valid_objects),
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "near_threshold": near_threshold,
            "vertical_threshold": vertical_threshold,
        },
        "nodes": nodes,
        "edges": edges,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(scene_graph, f, indent=2)

    print(f"[OK] Scene graph saved to {output_json}")
    print(f"[INFO] Nodes: {len(nodes)}")
    print(f"[INFO] Edges: {len(edges)}")

    return output_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build semantic 3D scene graph.")
    parser.add_argument("--objects", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--near-threshold", type=float, default=0.75)
    parser.add_argument("--vertical-threshold", type=float, default=0.20)

    args = parser.parse_args()

    build_scene_graph(
        objects_json=args.objects,
        output_json=args.output,
        near_threshold=args.near_threshold,
        vertical_threshold=args.vertical_threshold,
    )