# Hailo-8 Model Benchmarks

Complete benchmark results for every YOLO model tested on our Hailo-8 hardware. These numbers are from actual hardware testing, not theoretical estimates.

## Test Hardware

- **Accelerator:** Hailo-8 M.2 M-Key (HM218B1C2FAE "EXT TEMP" variant)
- **Firmware:** 4.20.0
- **Driver:** 4.20.0 (compiled from source, DKMS)
- **Clock:** Sustained 400 MHz (copper heatsink required — without it, chip hits 104°C and throttles to 350→300 MHz)
- **Host:** Minisforum Venus, Intel i9-12900H, 32GB RAM
- **GPU:** Intel Iris Xe (VAAPI decode only)
- **NVR Software:** Frigate (dev build 678ae87, HailoRT 4.20.0)
- **Cameras:** 8x ONWOTE YM400L_AF, 2592x1944 (5MP), H.265, 15fps
- **Detect resolution:** 1920x1440 (4:3)

## Test Method

### Hardware Benchmark (raw inference speed)

Used `hailortcli benchmark` inside the Frigate Docker container with Frigate stopped (exclusive Hailo access):

```bash
# Stop Frigate to free the Hailo device
cd /opt/frigate && sudo docker compose stop

# Run benchmark (10 seconds per test)
sudo docker run --rm --entrypoint hailortcli \
    --device /dev/hailo0 \
    -v /opt/frigate/config/model_cache:/models \
    ghcr.io/blakeblackshear/frigate:678ae87 \
    benchmark /models/<model>.hef -t 10 --power-mode performance
```

This measures three things:
1. **HW-only FPS** — raw inference throughput (no data transfer overhead)
2. **Streaming FPS** — realistic throughput including data transfer to/from the chip
3. **HW Latency** — time for a single frame to pass through the model

All models were tested sequentially with 3-5 second cooling breaks between tests to prevent thermal carryover. The heatsink keeps the chip at sustained 400 MHz throughout.

### Live Frigate Testing (real-world performance)

With Frigate running and all 8 cameras active:

```bash
curl -s http://127.0.0.1:5000/api/stats | python3 -c "
import json, sys
d = json.load(sys.stdin)
for k, v in d['cameras'].items():
    print(f'{k}: cam={v[\"camera_fps\"]:.1f} det={v[\"detection_fps\"]:.1f} skip={v[\"skipped_fps\"]:.1f}')
print(f'Inference: {d[\"detectors\"][\"hailo\"][\"inference_speed\"]:.1f}ms')
"
```

Real-world inference is slower than raw benchmark due to:
- Frame decode (VAAPI) overhead
- Data transfer between CPU and Hailo
- Frigate's pre/post processing pipeline
- Multiple cameras competing for detector time

### Why "Per cam / 8"

The Hailo-8 has a single inference pipeline. All 8 cameras share it. "Per cam / 8" divides total FPS by 8 to show what each camera gets when all are active simultaneously. In practice, cameras with no motion get fewer frames sent to the detector, so active cameras get more than the equal-share number.

### What These HEF Files Are

HEF (Hailo Executable Format) files are pre-compiled neural networks optimized for Hailo hardware. Each model is:
1. Trained on COCO 2017 (80 classes) by Ultralytics
2. Exported to ONNX
3. Quantized from float32 to int8 using the Hailo Dataflow Compiler
4. Compiled to target Hailo-8's on-chip SRAM and dataflow architecture

The quantization step (float32 → int8) causes a small accuracy loss, shown as "Hardware mAP" vs "Float mAP" in the results.

## Results

### Raw Hardware Benchmarks

All models from the Hailo Model Zoo v2.14.0, compiled for Hailo-8.

| Model | Params | FPS | Latency | Per cam / 8 | HW mAP | Float mAP |
|---|---|---|---|---|---|---|
| **YOLOv8n** | 3.2M | **1036** | **2.5ms** | **130** | 36.4 | 37.0 |
| **YOLOv8s** | 11.2M | **398** | **6.6ms** | **50** | 44.0 | 44.6 |
| **YOLOv11n** | 2.6M | **155** | **5.6ms** | **19** | 37.8 | 39.0 |
| **YOLOv11s** | 9.4M | **89** | **10ms** | **11** | 45.5 | 46.3 |
| YOLOv8m | 25.9M | 66 | 14ms | 8.2 | 49.2 | 49.9 |
| YOLOv11m | 20.1M | 45 | 19ms | 5.6 | 49.8 | 51.5 |
| YOLOv8l | 43.7M | 28 | 35ms | 3.5 | 51.8 | 52.4 |
| YOLOv11l | 25.3M | 21 | 46ms | 2.6 | 52.0 | 52.8 |
| YOLOv8x | 68.2M | 18 | 54ms | 2.3 | 52.9 | 53.5 |
| YOLOv11x | 56.9M | 14 | 69ms | 1.8 | 53.1 | 54.1 |
| IDCAelephants v1 (v11x, 8-class) | 56.9M | 14.2 | 69ms | 1.8 | — | — |

### Full-Frame Detection Threshold

For 8 cameras at 15fps each, you need **120 FPS** from the detector to process every frame. Models above this line can theoretically detect on every frame from every camera:

| Model | FPS | Headroom | Accuracy tradeoff |
|---|---|---|---|
| YOLOv8n | 1036 | 8.6x | Lowest accuracy (36.4 mAP) |
| YOLOv8s | 398 | 3.3x | Good accuracy (44.0 mAP), massive headroom |
| YOLOv11n | 155 | 1.3x | Low accuracy (37.8 mAP), tight headroom |
| YOLOv11s | 89 | 0.74x | Best accuracy in fast tier (45.5 mAP), drops some frames |

### Live Frigate Results (8 cameras, 1920x1440 detect, 15fps)

| Model | Real-world inference | Active cam det FPS | Frame coverage |
|---|---|---|---|
| YOLOv11s | 13ms | 15-27 FPS | Near-complete on active cameras |
| YOLOv11m | 22ms | 10-21 FPS | Good, some skipping under load |
| YOLOv11l | 47ms | 3 FPS | Poor, drops 80% of frames |
| IDCAelephants v1 (v11x) | 75ms | ~4 FPS | Poor, too slow for multi-camera |

*Note: YOLOv8s and YOLOv8n were not tested live in Frigate yet. Based on raw benchmarks, YOLOv8s should deliver 30-40+ FPS per active camera in Frigate.*

### YOLO26m Note

YOLO26m (48.3 FPS, 20ms latency, 50.0 HW mAP) was benchmarked but **failed in live Frigate testing** — it's an NMS-free architecture whose output tensor format is incompatible with Frigate's Hailo plugin (hailo8l.py), which expects YOLOv8-style NMS outputs. Some classes detected (car, cat) but person detection was unreliable. Not recommended for Frigate until the plugin is updated.

## Analysis

### Speed vs Accuracy Tradeoff

```
mAP
 53 |                                              v8x  v11x
 52 |                                    v8l  v11l
 50 |                          v8m  v11m
 45 |              v8s  v11s
 38 |  v8n  v11n
 36 |
    +----+----+----+----+----+----+----+----+----+-----> FPS
    0   50  100  150  200  300  400  500       1000
```

There's a clear knee in the curve around the "s" models (YOLOv8s at 398 FPS / 44.0 mAP, YOLOv11s at 89 FPS / 45.5 mAP). Going larger than "m" gives diminishing accuracy returns with severe FPS penalties.

### Recommendation for 8-Camera Security Setup

**For maximum coverage (every frame, every camera):** YOLOv8s — 398 FPS gives 50 FPS per camera with 44.0 mAP accuracy. You'll never miss a frame.

**For best balance (current setup):** YOLOv11s — 89 FPS gives 11 FPS per camera with 45.5 mAP. Catches most frames on active cameras.

**For maximum accuracy (single camera or powerful hardware):** YOLOv11m — 45 FPS, 49.8 mAP. Only viable for 4 or fewer cameras.

### v2.0 Training Target

Based on these benchmarks, the v2.0 IDCAelephants model will be trained on the **YOLOv8s architecture** — Frigate compatible, 398 FPS on Hailo-8, and the 8-class focused training should push its 44.0 mAP significantly higher while maintaining the speed advantage.

## Reproducing These Results

All HEF files are from the [Hailo Model Zoo v2.14.0](https://github.com/hailo-ai/hailo_model_zoo):

```bash
# Download any model
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/<model_name>.hef

# Benchmark
hailortcli benchmark <model_name>.hef -t 10 --power-mode performance
```

Results will vary based on:
- Hailo-8 vs Hailo-8L (8L is ~50% slower)
- Clock speed (thermal throttling reduces FPS significantly)
- HEF compilation version (v2.14 vs v2.18 may differ slightly)
- Driver/firmware version matching
