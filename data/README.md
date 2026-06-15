# Extract Frames

Use `extract_frames.py` to turn videos into image frames for Roboflow.

## What It Does

- Reads videos from `/home/aloy/Mardi_tech/data/videos`
- Also checks subfolders like `data/videos/Phone`
- Saves frames to `/home/aloy/Mardi_Extracted_Data`
- Extracts `2` frames per second
- Saves frames as `.jpg`
- Keeps phone videos in their raw landscape orientation using `-noautorotate`

## Extract Frames

From the repo root:

```bash
python data/extract_frames.py
```

The output will be saved here:

```text
/home/aloy/Mardi_Extracted_Data
```

Example output files:

```text
IMG_5735_frame_000001.jpg
IMG_5735_frame_000002.jpg
```

## Clear Old Extracted Frames

To delete all extracted frames but keep the folder:

```bash
rm -rf /home/aloy/Mardi_Extracted_Data/*
```

## Upload To Roboflow

After extracting frames, run:

```bash
cd /home/aloy/Mardi_tech/vision
python rf_upload_folder.py /home/aloy/Mardi_Extracted_Data
```

The folder path at the end is required. Without it, the upload script will look for this old default folder instead:

```text
/home/aloy/zed_extracted/MARDI_Phase_1_Pineapple_Pontian
```

The upload goes to:

```text
Workspace: rnd-kyodu
Project: pineapple_ai_system
Type: instance-segmentation
```

## Change Extraction Settings

Edit these values near the top of `extract_frames.py`:

```python
FPS = 2
IMAGE_FORMAT = "jpg"
RECURSIVE = True
```

Examples:

```python
FPS = 5      # extract 5 frames per second
FPS = None   # extract every frame
```

## Requirements

- Python 3
- ffmpeg

Check ffmpeg:

```bash
ffmpeg -version
```
