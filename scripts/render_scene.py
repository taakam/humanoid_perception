"""
Purpose:
Render train/test camera views from a trained 3D Gaussian Splatting model.

Expected input:
- Trained Gaussian Splatting output directory:
  gs_output/
    point_cloud/
    cameras.json
    cfg_args

Expected output:
- Rendered images saved by Gaussian Splatting, typically:
  gs_output/train/
  gs_output/test/
"""

from pathlib import Path
import subprocess


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[RUN] {' '.join(map(str, cmd))}")

    result = subprocess.run(cmd, cwd=cwd)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")


def validate_model_dir(model_dir: Path) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    point_cloud_dir = model_dir / "point_cloud"

    if not point_cloud_dir.exists():
        raise RuntimeError(
            f"Invalid Gaussian model directory: missing {point_cloud_dir}"
        )


def validate_render_output(model_dir: Path) -> None:
    train_dir = model_dir / "train"
    test_dir = model_dir / "test"

    if not train_dir.exists() and not test_dir.exists():
        raise RuntimeError(
            "Render finished, but no train/ or test/ render folders were created."
        )

    print(f"[OK] Render outputs created inside: {model_dir}")


def render_scene(
    model_dir: str | Path,
    gaussian_repo_dir: str | Path,
) -> Path:
    model_dir = Path(model_dir).resolve()
    gaussian_repo_dir = Path(gaussian_repo_dir).resolve()

    validate_model_dir(model_dir)

    render_script = gaussian_repo_dir / "render.py"

    if not render_script.exists():
        raise FileNotFoundError(
            f"Could not find Gaussian Splatting render.py at: {render_script}"
        )

    run_command(
        [
            "python",
            str(render_script),
            "-m",
            str(model_dir),
        ],
        cwd=gaussian_repo_dir,
    )

    validate_render_output(model_dir)

    return model_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render views from a trained Gaussian Splatting model."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained Gaussian Splatting model/output directory.",
    )

    parser.add_argument(
        "--repo",
        required=True,
        help="Path to Gaussian Splatting repository.",
    )

    args = parser.parse_args()

    render_scene(
        model_dir=args.model,
        gaussian_repo_dir=args.repo,
    )