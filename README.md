# IDCAelephants

### I Don't Care About Elephants

A security camera object detection project — because your NVR doesn't need to know what a giraffe looks like.

---

Your security camera doesn't need to know what a giraffe looks like. It doesn't need to identify broccoli. It has never once needed to detect a parking meter, and if a toaster shows up in your driveway at 3am, you have larger problems than object detection.

Stock YOLO models ship with **80 classes**. Your security camera cares about maybe **8**. The detection head's capacity is spread across hot dogs, tennis rackets, and yes — elephants.

**IDCAelephants** fixes this. We trained YOLO models exclusively on security-relevant classes and tested them end-to-end on real hardware.

## What It Detects

| # | Class | Why |
|---|---|---|
| 0 | person | The whole reason you have cameras |
| 1 | bicycle | The thing that keeps disappearing from your garage |
| 2 | car | Is that your car? No? Then we have a problem |
| 3 | motorcycle | See above, but louder |
| 4 | truck | Delivery, or someone backing into your mailbox |
| 5 | bird | Not a threat, but responsible for 40% of your false alarms |
| 6 | cat | Yours, or the neighborhood's. Either way, it's judging you |
| 7 | dog | Good boy, but bad dog. The model doesn't judge |

## What It Doesn't Detect

Elephants. Also: airplanes, bananas, baseball gloves, bears, beds, benches, books, bottles, bowls, broccoli, buses, cakes, cell phones, chairs, clocks, couches, cows, cups, dining tables, donuts, fire hydrants, forks, frisbees, hair dryers, handbags, horses, hot dogs, keyboards, kites, knives, laptops, mice, microwaves, oranges, ovens, parking meters, pizza, potted plants, refrigerators, remotes, sandwiches, scissors, sheep, sinks, skateboards, skis, snowboards, spoons, sports balls, stop signs, suitcases, surfboards, teddy bears, ties, toasters, toilets, toothbrushes, traffic lights, trains, TVs, umbrellas, vases, wine glasses, and zebras.

You're welcome.

## Status

**v1.0** — YOLOv11x trained on 8 classes. Proof of concept complete. Available as .pt and .onnx.

**v2.0 (planned)** — YOLO26m trained on 8 classes with combined security camera datasets (COCO + FLIR thermal + VisDrone + ExDark + more). Optimized for real-world edge deployment.

## v1.0 — What We Built and Learned

### Training

- **Architecture:** YOLOv11x (56.9M parameters, 195.5 GFLOPs)
- **Dataset:** COCO 2017 filtered to 8 classes — 77,065 training images, 3,258 validation
- **Input resolution:** 1280x1280
- **Training hardware:** 8x NVIDIA H100 80GB HBM3 (RunPod, $21.54/hr)
- **Batch size:** 128 (16 per GPU, DDP)
- **Optimizer:** MuSGD with cosine LR annealing
- **Augmentation:** Mosaic, mixup, random augment, horizontal flip
- **Pretrained:** Started from COCO-80 YOLOv11x weights, refined to 8 classes
- **Epochs completed:** 38 (early stopping plateau)
- **Final mAP50-95:** 0.569 (validation)
- **Final mAP50:** 0.778
- **Precision:** 0.812
- **Recall:** 0.697

### v1.0 Training Results

| Metric | IDCAelephants v1 | Stock YOLOv11x (80-class) | Stock YOLOv11s (80-class) |
|---|---|---|---|
| mAP50-95 | **0.569** | 0.541 | 0.455 |
| mAP50 | **0.778** | — | — |
| Precision | **0.812** | — | — |
| Parameters | 56.9M | 56.9M | 9.4M |
| Classes | 8 | 80 | 80 |

### Hailo-8 Hardware Benchmarks

Tested on Hailo-8 M.2 (HM218B1C2FAE "EXT TEMP") with heatsink, sustained 400 MHz.

| Model | FPS | Latency | Per cam / 8 | Hardware mAP |
|---|---|---|---|---|
| YOLOv11s (stock 80-class) | 89 | 10ms | 11.1 | 45.5 |
| **YOLO26m (stock 80-class)** | **48.3** | **20ms** | **6.0** | **50.0** |
| YOLOv11m (stock 80-class) | 45 | 19ms | 5.6 | 49.8 |
| YOLOv11l (stock 80-class) | 21 | 46ms | 2.6 | 52.0 |
| YOLOv11x (stock 80-class) | 14 | 70ms | 1.8 | 53.1 |
| IDCAelephants v1 (8-class v11x) | 14.2 | 69ms | 1.8 | — |

### Live Frigate Testing (8 cameras, 1920x1440 detect, 15fps)

| Model | Real-world inference | Active camera det FPS | Verdict |
|---|---|---|---|
| YOLOv11s (stock) | 13ms | 15-27 FPS | Fast, catches every frame |
| YOLO26m (stock) | 25ms | ~6 FPS | Good balance, better accuracy |
| YOLOv11m (stock) | 22ms | 10-21 FPS | Solid middle ground |
| YOLOv11l (stock) | 47ms | 3 FPS | Too slow, drops 80% of frames |
| IDCAelephants v1 (v11x) | 75ms | ~4 FPS | Highest accuracy, too slow for 8 cameras |

### What We Learned (the hard way)

1. **Model size matters more than class count for edge deployment.** Training YOLOv11x on 8 classes doesn't make it faster — it's still a 56.9M parameter model running at 14 FPS on Hailo-8. The architecture determines inference speed, not the number of classes.

2. **The accuracy gain from focused training is real but the speed penalty killed it.** IDCAelephants v1 achieved 0.569 mAP50-95 (beating every stock YOLO model) but at 4 FPS per camera, it misses events that the faster YOLOv11s catches at 15+ FPS.

3. **For security cameras, frame rate beats marginal accuracy.** A person walks past in 1 second. At 4 FPS you get 4 chances to detect them. At 15 FPS you get 15. More frames = more chances = better real-world detection.

4. **The right approach: focused training on a medium-sized model.** A YOLO26m (20M params, 48 FPS) trained on 8 classes would combine the accuracy benefits of focused training with the speed needed for real-time multi-camera detection.

5. **Test the full pipeline before renting GPUs.** We spent hours fighting Hailo DFC compilation issues (dependency hell, version mismatches, firmware updates) that could have been identified with a 5-minute test on a dummy model.

6. **Backbone compute is class-agnostic.** The backbone and neck (~90% of inference FLOPs) run identically regardless of class count. The gains from focused training are in detection head accuracy and NMS post-processing speed, not backbone inference.

### Hailo-8 Deployment Notes

- **Thermal throttling:** The Hailo-8 M.2 module WILL thermal throttle (104°C, drops from 400→350→300 MHz) without a heatsink. A copper M.2 heatsink is mandatory.
- **Firmware loading:** PCIe M.2 modules load firmware from `/lib/firmware/hailo/hailo8_fw.bin` at boot. `hailortcli fw-update` is for ethernet devices only — don't use it on M.2.
- **Version matching:** Driver, firmware, and library versions must match exactly. Frigate v0.17.1 = HailoRT 4.21.0. The dev build `678ae87` = HailoRT 4.20.0.
- **DFC compilation:** The Hailo AI Software Suite Docker image is the only reliable way to compile HEFs. pip-installing the DFC wheel leads to dependency hell.
- **VAAPI decode limit:** 1920x1440 works for detect resolution. Native 5MP (2592x1944) causes VAAPI filter conversion errors on Intel Iris Xe.

## v2.0 Plan

**Architecture:** YOLO26m (20.4M params, 48.3 FPS on Hailo-8, NMS-free)

**Why YOLO26m:**
- Newest YOLO architecture (Jan 2026)
- NMS-free — no DFL layer, cleaner Hailo compilation
- 48 FPS on Hailo-8 = 6 FPS per camera across 8 cameras
- Higher hardware mAP than YOLOv11m at the same parameter count

**Training data (mega-dataset, ~470K images):**

| Dataset | Images | What it teaches |
|---|---|---|
| COCO 8-class | 77K | General object appearance |
| FLIR ADAS v2 | 11K | Thermal/IR day+night |
| VisDrone | 10K | Elevated/drone camera angle |
| ExDark | 7K | Low-light visible |
| OD-VIRAT | 20K | Surveillance camera perspective |
| LLVIP | 15K | IR + visible paired |
| KAIST | 95K | Thermal + visible multispectral |
| HIT-UAV | 3K | Thermal from elevation |
| Roboflow cameras | 3.7K | Real CCTV footage |

**Method:** Fine-tune from COCO-80 pretrained YOLO26m weights. The backbone already knows how to see — we teach the detection head to focus on 8 classes using security-specific data.

## Downloads

### v1.0 (proof of concept)

| File | Format | Size | Use Case |
|---|---|---|---|
| `IDCAelephants.pt` | PyTorch | 327MB | Fine-tuning, conversion to any format |
| `IDCAelephants-640.onnx` | ONNX | 218MB | OpenVINO, TensorRT, CoreML, ONNX Runtime |
| `IDCAelephants-1280.onnx` | ONNX | 218MB | High-resolution variant |
| `IDCAelephants-hailo8-640.hef` | Hailo HEF | 54MB | Hailo-8 (compiled with DFC 3.33.0, needs HailoRT 4.21+) |

*Note: v1.0 is a YOLOv11x model (14 FPS on Hailo-8). It works but is too slow for multi-camera setups. Recommended for single-camera or powerful GPU deployments. v2.0 will use YOLO26m for proper edge deployment.*

## Usage with Frigate

```yaml
detectors:
  hailo:
    type: hailo8l
    device: PCIe

model:
  path: /config/model_cache/IDCAelephants-hailo8-640.hef
```

### Class mapping for Frigate

```yaml
objects:
  track: [person, bicycle, car, motorcycle, truck, bird, cat, dog]
```

## Why Not Just Filter Detections?

"Can't you just set `objects.track: [person, car, dog]` in Frigate and ignore the rest?"

Sure. But:

1. **Higher accuracy on what matters** — during training, the model's learned representations are optimized for 8 classes instead of spread across 80. The detection head focuses its capacity on distinguishing the objects you actually care about. This is measurable — our mAP50-95 on security-relevant classes exceeds the stock 80-class model on the same architecture.

2. **Faster post-processing** — NMS runs per-class. Processing 8 classes instead of 80 is genuinely faster, especially on edge devices where NMS runs on CPU. The output tensor is also smaller (8 class predictions per anchor vs 80).

3. **Note on backbone compute** — to be clear, the backbone and neck (where ~90% of inference FLOPs happen) run identically regardless of class count. The gains are in accuracy from focused training and speed from reduced post-processing, not from the convolution layers running faster.

4. **No elephants** — this one speaks for itself.

## FAQ

**Q: What if an elephant actually shows up on my camera?**
A: You will not receive a detection alert. You will, however, probably notice the elephant.

**Q: Can I add more classes?**
A: Fork the repo, add classes to the training config, rent some GPUs, retrain. Or just accept that your camera doesn't need to know what a kite looks like.

**Q: Why did v1.0 use YOLOv11x?**
A: Hubris. We wanted the biggest model and the best mAP number. It worked — we beat every published YOLO model on security-relevant accuracy. But 14 FPS on Hailo-8 isn't practical for 8 cameras. v2.0 uses YOLO26m because we learned that frame rate matters more than marginal accuracy gains.

**Q: Why is it called IDCAelephants?**
A: I Don't Care About Elephants. Because we don't. And neither does your security camera.

**Q: Isn't Frigate+ doing the same thing?**
A: [Blake Blackshear proposed this exact idea in 2021](https://github.com/blakeblackshear/frigate/discussions/1043) — a custom model trained on security camera data. Five years later, Frigate+ offers it as a paid subscription ($5/month) where your camera footage goes to their cloud for training. IDCAelephants is the open source alternative: no subscription, no cloud, no sending your family's camera footage to anyone. Train it yourself or use our weights.

## License

Released under the [AGPL-3.0 License](LICENSE). The COCO dataset is licensed under [CC BY 4.0](https://cocodataset.org/#termsofuse). YOLO is by [Ultralytics](https://github.com/ultralytics/ultralytics) under AGPL-3.0.

## Acknowledgments

- **COCO dataset** — for providing 80 classes when we only needed 8
- **Ultralytics** — for making YOLO training stupidly easy
- **Hailo** — for making the compilation process just hard enough to be interesting
- **RunPod** — for renting us 8x H100s so we could train a model that ignores elephants
- **FLIR/Teledyne** — for the thermal dataset
- **Every elephant** — sorry, nothing personal

---

*Built by [Flukethoughts](https://github.com/Flukethoughts) with help from Claude. No elephants were detected in the making of this model.*
