#!/usr/bin/env python3

import subprocess
from pathlib import Path

# ====== SETTINGS ======
INPUT_DIR = Path("/home/aloy/Mardi_tech/data/videos")
OUTPUT_DIR = Path("/home/aloy/Mardi_Extracted_Data")

# Set to None to extract ALL frames
# Set to a number like 1, 2, 5, 10 to extract that many frames per second
FPS = 5

IMAGE_FORMAT = "jpg"  # "png" or "jpg"
RECURSIVE = True      # True = also search subfolders

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm", ".m4v"
}
# ======================


def find_videos(folder: Path, recursive: bool = True):
    def sort_key(path: Path):
        relative_path = path.relative_to(folder)
        return (len(relative_path.parts) > 1, str(relative_path).lower())

    if recursive:
        return sorted(
            (
                p for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
            ),
            key=sort_key,
        )
    return sorted(
        (
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=sort_key,
    )


def extract_frames(video_path: Path, output_root: Path, fps, image_format: str):
    output_root.mkdir(parents=True, exist_ok=True)
    output_prefix = f"{video_path.stem}_frame_"
    output_pattern = output_root / f"{output_prefix}%06d.{image_format}"

    for existing_file in output_root.glob(f"{output_prefix}*.{image_format}"):
        existing_file.unlink()

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-progress", "pipe:1",
        "-nostats",
        "-noautorotate",
        "-i", str(video_path),
    ]

    if fps is not None:
        command += ["-vf", f"fps={fps}"]

    command.append(str(output_pattern))

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if line.startswith("progress="):
            print(f"[ffmpeg] {line}")

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    print(f"Done: {video_path} -> {output_root}")


def main():
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = find_videos(INPUT_DIR, recursive=RECURSIVE)

    if not videos:
        print(f"No video files found in {INPUT_DIR}")
        return

    print(f"Found {len(videos)} video(s) in {INPUT_DIR}")

    for index, video in enumerate(videos, start=1):
        try:
            print(f"[{index}/{len(videos)}] {video.relative_to(INPUT_DIR)}")
            extract_frames(video, OUTPUT_DIR, FPS, IMAGE_FORMAT)
        except subprocess.CalledProcessError as e:
            print(f"Failed: {video} ({e})")


if __name__ == "__main__":
    main()
