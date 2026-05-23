"""
Purpose:
Undistort COLMAP reconstruction images and convert the dataset
into a format compatible with Gaussian Splatting.

Expected input:
- Image directory containing extracted frames
- COLMAP sparse reconstruction directory:
  sparse/0/cameras.bin
  sparse/0/images.bin
  sparse/0/points3D.bin

Expected output:
- Undistorted dataset structure:
  output_dir/
    images/
    sparse/0/
"""

from pathlib import Path
import shutil
import subprocess


def run_command(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(map(str, cmd))}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")


def validate_sparse_model(sparse_dir: Path) -> None:
    required_files = [
        sparse_dir / "cameras.bin",
        sparse_dir / "images.bin",
        sparse_dir / "points3D.bin",
    ]

    missing = [file for file in required_files if not file.exists()]

    if missing:
        raise RuntimeError(
            f"Missing sparse reconstruction files: {[str(m) for m in missing]}"
        )


def fix_sparse_structure(output_dir: Path) -> None:
    sparse_dir = output_dir / "sparse"
    sparse_zero_dir = sparse_dir / "0"

    sparse_zero_dir.mkdir(parents=True, exist_ok=True)

    for file_name in ["cameras.bin", "images.bin", "points3D.bin"]:
        source = sparse_dir / file_name
        target = sparse_zero_dir / file_name

        if source.exists():
            shutil.move(str(source), str(target))

    print(f"[OK] Fixed sparse folder structure at: {sparse_zero_dir}")


def undistort_dataset(
    image_dir: str | Path,
    sparse_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = True,
) -> Path:

    image_dir = Path(image_dir).resolve()
    sparse_dir = Path(sparse_dir).resolve()
    output_dir = Path(output_dir).resolve()

    validate_sparse_model(sparse_dir)

    if overwrite and output_dir.exists():
        print(f"[INFO] Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "colmap",
            "image_undistorter",
            "--image_path",
            str(image_dir),
            "--input_path",
            str(sparse_dir),
            "--output_path",
            str(output_dir),
            "--output_type",
            "COLMAP",
        ]
    )

    fix_sparse_structure(output_dir)

    validate_sparse_model(output_dir / "sparse" / "0")

    print(f"[OK] Undistorted dataset ready at: {output_dir}")

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Undistort COLMAP dataset for Gaussian Splatting."
    )

    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing extracted frames.",
    )

    parser.add_argument(
        "--sparse",
        required=True,
        help="Path to COLMAP sparse/0 directory.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory for undistorted dataset.",
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing output directory.",
    )

    args = parser.parse_args()

    undistort_dataset(
        image_dir=args.images,
        sparse_dir=args.sparse,
        output_dir=args.output,
        overwrite=not args.no_overwrite,
    )