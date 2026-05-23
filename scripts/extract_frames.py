"""
Purpose:
Extract image frames from an input video using FFmpeg.

Expected input:
- Supported video file (.mp4, .mov, .avi, .mkv)

Expected output:
- Directory containing extracted image frames:
  frame_0001.jpg
  frame_0002.jpg
  ...
"""

from pathlib import Path
import subprocess

VALID_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def run_command(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def validate_video(video_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if video_path.suffix.lower() not in VALID_VIDEO_EXTS:
        raise ValueError(
            f"Unsupported video format: {video_path.suffix}. "
            f"Supported formats: {sorted(VALID_VIDEO_EXTS)}"
        )


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: int = 2,
    overwrite: bool = True,
) -> Path:
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()

    validate_video(video_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for frame in output_dir.glob("*.jpg"):
            frame.unlink()

    output_pattern = output_dir / "frame_%04d.jpg"

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        str(output_pattern),
    ]

    run_command(cmd)

    frame_count = len(list(output_dir.glob("*.jpg")))

    if frame_count < 20:
        raise RuntimeError(
            f"Only extracted {frame_count} frames. "
            "This may be too few for reliable reconstruction. "
            "Try a longer video or higher FPS."
        )

    print(f"[OK] Extracted {frame_count} frames to {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from a video using ffmpeg.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--output", required=True, help="Directory to save extracted frames.")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second to extract.")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing frames.")

    args = parser.parse_args()

    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        overwrite=not args.no_overwrite,
    )