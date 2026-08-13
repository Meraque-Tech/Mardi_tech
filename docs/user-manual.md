# User Manual — Sensor Fusion & Bed Detection Stack

This system runs on the Jetson (Orin Nano) and combines a GNSS receiver, an IMU, and a camera to detect and count objects ("beds") while tracking the vehicle's motion and position. It exposes a web dashboard for control and monitoring.

## What it does

1. Reads position from a GNSS receiver and motion from an IMU, fuses them into a smooth, reliable estimate of where the vehicle is and whether it's moving forward, backward, or idle.
2. Runs a camera-based detector that counts objects in view.
3. Combines both: only auto-saves a detection snapshot when the vehicle is actually moving forward and has traveled far enough since the last save — avoiding duplicate saves while stationary or reversing.
4. Provides a web dashboard to start/stop detection, view live counts, browse saved images, tune settings, and download QA reports.

## Starting the system

On the Jetson:

```bash
cd ~/Mardi_tech
./run_ai.sh
```

This detects the CPU architecture and starts all three required containers together. It also force-stops any previously running containers first.

To check everything is up:

```bash
docker compose ps
docker compose logs -f imu_gnss_raw-aarch64
```

## The dashboard

Open a browser to:

```
http://<jetson-ip>:8090/
```

(Replace `<jetson-ip>` with the Jetson's actual IP address on your network.)

The raw camera stream (no controls, just video) is also available directly at:

```
http://<jetson-ip>:8080/
```

### Dashboard features

- **Live count** — current detected object count per class, updated in real time.
- **Confidence / status** — whether a detection above the configured confidence threshold is currently in view.
- **Start / Stop detection** — turns the detector on or off. Detection does not run automatically on boot; you must press Start (or it will auto-start if the dashboard was left running detection previously — check status first).
- **GNSS position & direction** — live latitude/longitude and whether the vehicle is moving forward, backward, or idle, shown alongside a recent path trace.
- **Auto-save toggle** — when enabled, the system automatically saves a snapshot (image + counts + position) whenever the vehicle is moving forward and has traveled at least the configured distance since the last save. This prevents saving duplicate images while parked or reversing.
- **Motion deadband** — the minimum distance (meters) the vehicle must travel before auto-save considers it "new". Adjustable live from the dashboard; default is 0.3 m.
- **Saved images gallery** — browse, download, or delete previously saved snapshots.
- **Model management** — upload new detection models, convert them, and switch which one is active, without restarting the container.
- **QA report** — downloadable HTML report flagging saved frames with low confidence, missing GNSS data, near-duplicate positions, or long gaps with no detections.
- **Shutdown** — safely powers off the physical Jetson from the dashboard.

## Typical workflow

1. Power on the Jetson and confirm `./run_ai.sh` has started all three services.
2. Open the dashboard and confirm GNSS position is showing (green/valid indicator) and the camera preview is live.
3. Press **Start** to begin detection.
4. Enable **Auto-save** if you want the system to automatically log snapshots as the vehicle moves.
5. Drive the route.
6. When finished, press **Stop**, then use the **Saved images** gallery or **Download report** to review results.

## Cautions

- **Auto-save can fill storage quickly** if left running continuously without a moving vehicle — with the shortest save interval it can generate a very large number of images per day. Check the images gallery periodically and clear old data if storage is a concern (the **Delete all data** action is destructive and cannot be undone).
- **GNSS position must be valid before auto-save is meaningful.** If the "GNSS valid" indicator is red/stale, auto-save based on position will not trigger correctly.
- **Do not physically move the GNSS receiver or camera cables to a different USB port** without notifying whoever manages the system config — the GNSS receiver is pinned to a specific physical port so it's always found reliably. Moving it to a different port will break auto-detection until the configuration is updated.
- **Shutdown from the dashboard powers off the whole computer**, not just the software — use it only when you're done with the session.

## Troubleshooting

| Symptom | Likely cause | What to try |
|---|---|---|
| Dashboard won't load | Container not running or wrong IP | `docker compose ps` on the Jetson; confirm you're using the Jetson's IP, not `localhost`, from another machine |
| No camera image | Camera not detected or in use elsewhere | Check the camera is plugged in; restart the container |
| GNSS position stuck / not updating | Receiver unplugged, wrong port, or no satellite fix (e.g. indoors) | Check `/dev/gnss_rtk` exists on the host; move to open sky; see developer manual for udev rule details |
| Direction always shows "idle" | Vehicle hasn't moved beyond the configured deadband distance yet | This is expected at low speed/short distances — lower the deadband if needed |
| Detection won't start | Model/engine not loaded yet | Use the Model management panel to select or build an engine first |
| Auto-save not saving anything | Direction not "forward", GNSS stale, or distance threshold not yet met | Check the GNSS/direction indicators are valid before expecting saves |

For deeper technical detail (topics, parameters, container internals), see the [Developer Manual](developer-manual.md) and [API Reference](api-reference.md).
