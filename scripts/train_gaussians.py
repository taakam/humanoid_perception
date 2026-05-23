"""
Purpose:
Train a 3D Gaussian Splatting scene representation from an
undistorted COLMAP dataset.

Expected input:
- Undistorted dataset structure:
  dataset/
    images/
    sparse/0/

Expected output:
- Gaussian Splatting training output:
  output/
    point_cloud/
    cameras.json
    cfg_args
    renders/
"""

from pathlib import Path
import shutil
import subprocess


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[RUN] {' '.join(map(str, cmd))}")

    result = subprocess.run(cmd, cwd=cwd)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")


def validate_dataset(dataset_dir: Path) -> None:
    required_paths = [
        dataset_dir / "images",
        dataset_dir / "sparse" / "0" / "cameras.bin",
        dataset_dir / "sparse" / "0" / "images.bin",
        dataset_dir / "sparse" / "0" / "points3D.bin",
    ]

    missing = [path for path in required_paths if not path.exists()]

    if missing:
        raise RuntimeError(
            f"Dataset validation failed. Missing: {[str(m) for m in missing]}"
        )


def validate_output(output_dir: Path) -> None:
    if not output_dir.exists():
        raise RuntimeError("Training finished but output directory was not created.")

    point_cloud_dir = output_dir / "point_cloud"

    if not point_cloud_dir.exists():
        raise RuntimeError(
            "Training output incomplete: point_cloud directory missing."
        )

    print(f"[OK] Gaussian training output found at: {output_dir}")


def train_gaussians(
    dataset_dir: str | Path,
    gaussian_repo_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = True,
) -> Path:

    dataset_dir = Path(dataset_dir).resolve()
    gaussian_repo_dir = Path(gaussian_repo_dir).resolve()
    output_dir = Path(output_dir).resolve()

    validate_dataset(dataset_dir)

    if overwrite and output_dir.exists():
        print(f"[INFO] Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    train_script = gaussian_repo_dir / "train.py"

    if not train_script.exists():
        raise FileNotFoundError(
            f"Could not find Gaussian Splatting train.py at: {train_script}"
        )

    run_command(
        [
            "python",
            str(train_script),
            "-s",
            str(dataset_dir),
            "-m",
            str(output_dir),
        ],
        cwd=gaussian_repo_dir,
    )

    validate_output(output_dir)

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Gaussian Splatting model."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to undistorted dataset.",
    )

    parser.add_argument(
        "--repo",
        required=True,
        help="Path to Gaussian Splatting repository.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory for Gaussian training output.",
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing output directory.",
    )

    args = parser.parse_args()

    train_gaussians(
        dataset_dir=args.dataset,
        gaussian_repo_dir=args.repo,
        output_dir=args.output,
        overwrite=not args.no_overwrite,
    )