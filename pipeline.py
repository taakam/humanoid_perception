from pathlib import Path
import argparse
import shutil

from scripts.extract_frames import extract_frames
from scripts.run_colmap import run_colmap
from scripts.undistort import undistort_dataset
from scripts.train_gaussians import train_gaussians
from scripts.render_scene import render_scene
from scripts.semantic_detect import detect_semantics, parse_custom_classes
from scripts.fuse_semantics import fuse_semantics
from scripts.associate_3d import associate_objects_3d
from scripts.build_scene_graph import build_scene_graph


def export_viewer_data(gs_output_dir: Path, semantics_dir: Path, viewer_dir: Path) -> None:
    viewer_data_dir = viewer_dir / "data"
    viewer_data_dir.mkdir(parents=True, exist_ok=True)

    point_cloud_path = gs_output_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    semantic_objects_3d_path = semantics_dir / "semantic_objects_3d.json"
    scene_graph_path = semantics_dir / "scene_graph.json"

    if not point_cloud_path.exists():
        raise FileNotFoundError(f"Gaussian point cloud not found: {point_cloud_path}")

    if not semantic_objects_3d_path.exists():
        raise FileNotFoundError(f"Semantic 3D objects not found: {semantic_objects_3d_path}")

    if not scene_graph_path.exists():
        raise FileNotFoundError(f"Scene graph not found: {scene_graph_path}")

    shutil.copy(point_cloud_path, viewer_data_dir / "point_cloud.ply")
    shutil.copy(semantic_objects_3d_path, viewer_data_dir / "semantic_objects_3d.json")
    shutil.copy(scene_graph_path, viewer_data_dir / "scene_graph.json")

    print(f"[OK] Exported viewer data to: {viewer_data_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Video-to-semantic Gaussian world model pipeline."
    )

    parser.add_argument("--video", required=True)
    parser.add_argument("--name", default="test_run")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--gs-repo", default="third_party/gaussian-splatting")

    parser.add_argument("--viewer-dir", default="viewer")

    parser.add_argument("--semantic-model", default="yolov8x-world.pt")
    parser.add_argument("--semantic-conf", type=float, default=0.25)
    parser.add_argument(
        "--custom-classes",
        default=None,
        help="Comma-separated extra semantic classes, e.g. 'slippers,napkin,curtain'",
    )

    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--min-3d-points", type=int, default=5)

    parser.add_argument("--near-threshold", type=float, default=0.75)
    parser.add_argument("--vertical-threshold", type=float, default=0.20)

    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-semantics", action="store_true")
    parser.add_argument("--skip-viewer-export", action="store_true")

    args = parser.parse_args()

    run_dir = Path("runs") / args.name
    frames_dir = run_dir / "frames"
    colmap_dir = run_dir / "colmap"
    undistorted_dir = run_dir / "dataset_undistorted"
    gs_output_dir = run_dir / "gs_output"
    semantics_dir = run_dir / "semantics"

    gs_repo_dir = Path(args.gs_repo)
    viewer_dir = Path(args.viewer_dir)

    print("\n[1/10] Extracting frames")
    extract_frames(args.video, frames_dir, fps=args.fps)

    if not args.skip_semantics:
        print("\n[2/10] Running 2D semantic detection on extracted frames")
        frame_detections_json = semantics_dir / "frame_detections.json"

        detect_semantics(
            image_dir=frames_dir,
            output_json=frame_detections_json,
            model_name=args.semantic_model,
            confidence=args.semantic_conf,
            custom_classes=parse_custom_classes(args.custom_classes),
        )
    else:
        frame_detections_json = semantics_dir / "frame_detections.json"

    print("\n[3/10] Running COLMAP")
    sparse_dir = run_colmap(frames_dir, colmap_dir)

    print("\n[4/10] Undistorting dataset")
    undistort_dataset(frames_dir, sparse_dir, undistorted_dir)

    if not args.skip_training:
        print("\n[5/10] Training Gaussian Splatting model")
        train_gaussians(
            dataset_dir=undistorted_dir,
            gaussian_repo_dir=gs_repo_dir,
            output_dir=gs_output_dir,
        )
    else:
        print("\n[5/10] Skipping Gaussian training")

    if not args.skip_render:
        print("\n[6/10] Rendering trained Gaussian model")
        render_scene(
            model_dir=gs_output_dir,
            gaussian_repo_dir=gs_repo_dir,
        )
    else:
        print("\n[6/10] Skipping render")

    if not args.skip_semantics:
        print("\n[7/10] Fusing 2D detections into object instances")
        frame_semantic_objects_json = semantics_dir / "frame_semantic_objects.json"

        fuse_semantics(
            detections_json=frame_detections_json,
            output_json=frame_semantic_objects_json,
            min_views=args.min_views,
            min_mean_confidence=args.min_confidence,
            visualize=True,
        )

        print("\n[8/10] Associating semantic objects with COLMAP 3D points")
        semantic_objects_3d_json = semantics_dir / "semantic_objects_3d.json"

        associate_objects_3d(
            semantic_objects_json=frame_semantic_objects_json,
            sparse_dir=colmap_dir / "sparse" / "0",
            output_json=semantic_objects_3d_json,
            min_points=args.min_3d_points,
        )

        print("\n[9/10] Building 3D semantic scene graph")
        scene_graph_json = semantics_dir / "scene_graph.json"

        build_scene_graph(
            objects_json=semantic_objects_3d_json,
            output_json=scene_graph_json,
            near_threshold=args.near_threshold,
            vertical_threshold=args.vertical_threshold,
        )
    else:
        print("\n[7-9/10] Skipping semantic fusion, 3D association, and scene graph")

    if not args.skip_viewer_export:
        print("\n[10/10] Exporting viewer data")
        export_viewer_data(
            gs_output_dir=gs_output_dir,
            semantics_dir=semantics_dir,
            viewer_dir=viewer_dir,
        )
    else:
        print("\n[10/10] Skipping viewer data export")

    print("\n[DONE] Pipeline complete.")
    print(f"Run saved at: {run_dir}")
    print("\nTo open the viewer:")
    print("cd viewer")
    print("npm run dev")


if __name__ == "__main__":
    main()