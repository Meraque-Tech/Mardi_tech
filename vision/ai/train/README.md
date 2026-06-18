# YOLOv8 Training

This folder contains a simple Python script for training a YOLOv8 object
detection model with Ultralytics.

## Install

From the repository root:

```bash
pip install ultralytics
```

## Prepare Dataset

The script expects a YOLO dataset YAML file. A simple example:

```yaml
path: /home/aloy/Mardi_Extracted_Data/dataset
train: images/train
val: images/val

names:
  0: flat
  1: missing
  2: ok
```

Expected dataset layout:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Each label file should use YOLO format:

```text
class_id x_center y_center width height
```

The box values must be normalized from `0` to `1`.

## Train

First, edit the `TRAINING_CONFIG` block at the top of `train_yolov8.py`:

```python
TRAINING_CONFIG = {
    "data": "data/pineapple.yaml",
    "model": "yolov8n.pt",
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "patience": 50,
    "save_period": -1,
    "device": None,
    "workers": 8,
    "project": "runs/detect",
    "name": "train",
    "resume": False,
}
```

Then run training without command flags:

```bash
python3 vision/ai/train/train_yolov8.py
```

You can still override any setting from the command line for one-off runs:

```bash
python3 vision/ai/train/train_yolov8.py \
  --data data/pineapple.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640
```

Train on GPU `0`:

```bash
python3 vision/ai/train/train_yolov8.py \
  --device 0
```

Use automatic batch size:

```bash
python3 vision/ai/train/train_yolov8.py \
  --batch -1
```

Stop early if validation does not improve for 20 epochs:

```bash
python3 vision/ai/train/train_yolov8.py \
  --patience 20
```

Save an extra checkpoint every 10 epochs:

```bash
python3 vision/ai/train/train_yolov8.py \
  --save-period 10
```

Resume from a checkpoint:

```bash
python3 vision/ai/train/train_yolov8.py \
  --model runs/detect/train/weights/last.pt \
  --resume
```

## Checkpoints and Resume

YOLOv8 automatically saves checkpoints during training:

```text
runs/detect/train/weights/last.pt
runs/detect/train/weights/best.pt
```

Use `last.pt` to continue a stopped run. Use `best.pt` for inference or
deployment.

To stop training, press `Ctrl+C` in the terminal. Then resume later with:

```bash
python3 vision/ai/train/train_yolov8.py \
  --model runs/detect/train/weights/last.pt \
  --resume
```

If you use a custom run name, match that path when resuming:

```bash
python3 vision/ai/train/train_yolov8.py \
  --model runs/detect/pineapple_run/weights/last.pt \
  --resume
```

## Outputs

Training results are saved by default under:

```text
runs/detect/train/
```

The most useful files are usually:

```text
runs/detect/train/weights/best.pt
runs/detect/train/weights/last.pt
```

Use `best.pt` for inference or later conversion to TensorRT.

## Options

These flags are optional. They override the values in `TRAINING_CONFIG`.

```text
--data      Path to dataset YAML file.
--model     Base model or checkpoint. Default: yolov8n.pt
--epochs    Number of epochs. Default: 100
--imgsz     Image size. Default: 640
--batch     Batch size. Use -1 for auto-batch. Default: 16
--patience  Early stopping patience in epochs. Default: 50
--save-period Save an extra checkpoint every N epochs. Default: -1
--device    Device, for example 0, 0,1, cpu, or mps.
--workers   Dataloader workers. Default: 8
--project   Output project directory. Default: runs/detect
--name      Training run name. Default: train
--resume    Resume training from checkpoint.
```
