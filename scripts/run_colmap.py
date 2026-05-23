"""
Purpose:
Run COLMAP sparse reconstruction on extracted video frames.

Expected input:
- Directory containing extracted image frames.

Expected output:
- COLMAP workspace containing sparse reconstruction:
  workspace_dir/sparse/0/cameras.bin
  workspace_dir/sparse/0/images.bin
  workspace_dir/sparse/0/points3D.bin
"""

from pathlib import Path
import shutil
import subprocess
import os


def run_command(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(map(str, cmd))}")

    env = os.environ.copy()

    # Prevent OpenCV's bundled Qt plugins from breaking COLMAP.
    env.pop("QT_PLUGIN_PATH", None)
    env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")


def check_colmap_installed() -> None:
    if shutil.which("colmap") is None:
        raise EnvironmentError(
            "COLMAP is not installed or not available in PATH. "
            "Install it with: sudo apt install colmap"
        )


def validate_image_dir(image_dir: Path) -> None:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = (
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png"))
    )

    if len(image_files) < 20:
        raise RuntimeError(
            f"Only found {len(image_files)} images in {image_dir}. "
            "COLMAP usually needs enough overlapping frames for reliable SfM."
        )


def validate_sparse_output(workspace_dir: Path) -> Path:
    sparse_model_dir = workspace_dir / "sparse" / "0"

    required_files = [
        sparse_model_dir / "cameras.bin",
        sparse_model_dir / "images.bin",
        sparse_model_dir / "points3D.bin",
    ]

    missing = [file for file in required_files if not file.exists()]

    if missing:
        raise RuntimeError(
            "COLMAP finished, but sparse reconstruction output is incomplete. "
            f"Missing files: {[str(file) for file in missing]}"
        )

    print(f"[OK] Sparse reconstruction found at: {sparse_model_dir}")
    return sparse_model_dir


def run_colmap(
    image_dir: str | Path,
    workspace_dir: str | Path,
    overwrite: bool = True,
) -> Path:
    image_dir = Path(image_dir).resolve()
    workspace_dir = Path(workspace_dir).resolve()

    check_colmap_installed()
    validate_image_dir(image_dir)

    if overwrite and workspace_dir.exists():
        print(f"[INFO] Removing existing COLMAP workspace: {workspace_dir}")
        shutil.rmtree(workspace_dir)

    workspace_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "colmap",
            "automatic_reconstructor",
            "--workspace_path",
            str(workspace_dir),
            "--image_path",
            str(image_dir),
        ]
    )

    return validate_sparse_output(workspace_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run COLMAP sparse reconstruction.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--no-overwrite", action="store_true")

    args = parser.parse_args()

    run_colmap(
        image_dir=args.images,
        workspace_dir=args.workspace,
        overwrite=not args.no_overwrite,
    )