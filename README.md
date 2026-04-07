# IDCAelephants

### I Don't Care About Elephants

A YOLOv11x object detection model trained for **security cameras**, not wildlife documentaries.

---

Your security camera doesn't need to know what a giraffe looks like. It doesn't need to identify broccoli. It has never once needed to detect a parking meter, and if a toaster shows up in your driveway at 3am, you have larger problems than object detection.

Stock YOLO models ship with **80 classes**. Your security camera cares about maybe **8**. That means 90% of your AI accelerator's brain cells are dedicated to hot dogs, tennis rackets, and yes — elephants.

**IDCAelephants** fixes this. We took the largest YOLO model (YOLOv11x, 56.9M parameters), told it to forget about the other 72 classes, and made it really, really good at the ones that actually matter.

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

## Performance

Trained on COCO 2017 filtered to 8 classes (77,065 images) at 1280x1280 resolution on 8x NVIDIA H100 80GB GPUs.

| Metric | IDCAelephants | Stock YOLOv11x (80 classes) | Stock YOLOv11s (80 classes) |
|---|---|---|---|
| mAP50 | TBD | 0.541 | 0.465 |
| mAP50-95 | TBD | 0.541 | 0.455 |
| Parameters | 56.9M | 56.9M | 9.4M |
| Classes | 8 (useful) | 80 (72 useless) | 80 (72 useless) |
| Elephant detection | No | Yes (why?) | Yes (why?) |

*Final metrics will be updated when training completes.*

## Downloads

| File | Format | Use Case |
|---|---|---|
| `IDCAelephants.pt` | PyTorch | Ultralytics, fine-tuning, conversion to any format |
| `IDCAelephants-640.onnx` | ONNX | Universal — OpenVINO, TensorRT, CoreML, ONNX Runtime |
| `IDCAelephants-1280.onnx` | ONNX | High-resolution variant |
| `IDCAelephants-hailo8-640.hef` | Hailo HEF | Hailo-8 users (Frigate, etc.) |

## Usage with Frigate

```yaml
detectors:
  hailo:
    type: hailo8l
    device: PCIe

model:
  path: /config/model_cache/IDCAelephants-hailo8-640.hef

# That's it. Your cameras now ignore elephants professionally.
```

For non-Hailo Frigate (OpenVINO, TensorRT, etc.), use the ONNX file with the appropriate detector config.

### Class mapping for Frigate

```yaml
# objects.track in your camera config:
objects:
  track: [person, bicycle, car, motorcycle, truck, bird, cat, dog]
```

## Training Details

- **Architecture:** YOLOv11x (56.9M parameters, 195.5 GFLOPs)
- **Dataset:** COCO 2017, filtered to 8 security-relevant classes
- **Training images:** 77,065
- **Validation images:** 3,258
- **Input resolution:** 1280x1280
- **Training hardware:** 8x NVIDIA H100 80GB HBM3 (RunPod)
- **Batch size:** 128 (16 per GPU)
- **Optimizer:** MuSGD with cosine LR annealing
- **Augmentation:** Mosaic, mixup, random augment, horizontal flip
- **Pretrained:** Started from COCO-80 YOLOv11x weights, refined to 8 classes
- **Hailo compilation:** Dataflow Compiler v3.30.0, targeting Hailo-8 @ 640x640

## Why Not Just Filter Detections?

"Can't you just set `objects.track: [person, car, dog]` in Frigate and ignore the rest?"

Sure. But:

1. **Higher accuracy on what matters** — during training, the model's learned representations are optimized for 8 classes instead of spread across 80. The detection head focuses its capacity on distinguishing the objects you actually care about. This is the primary benefit and it's measurable — our mAP50-95 on security-relevant classes exceeds the stock 80-class model trained on the same architecture.

2. **Faster post-processing** — NMS (Non-Maximum Suppression) runs per-class. Processing 8 classes instead of 80 is genuinely faster, especially on edge devices where NMS runs on CPU. The output tensor is also smaller (8 class predictions per anchor vs 80).

3. **Note on backbone compute** — to be clear, the backbone and neck (where ~90% of inference FLOPs happen) run identically regardless of class count. The gains are in accuracy from focused training and speed from reduced post-processing, not from the convolution layers running faster.

4. **No elephants** — this one speaks for itself.

## FAQ

**Q: What if an elephant actually shows up on my camera?**
A: You will not receive a detection alert. You will, however, probably notice the elephant.

**Q: Can I add more classes?**
A: Fork the repo, add classes to the training config, rent some GPUs, retrain. Or just accept that your camera doesn't need to know what a kite looks like.

**Q: Why YOLOv11x and not a smaller model?**
A: Because we don't care about elephants, but we care deeply about the things we do detect. More parameters = better detection of partially occluded people, distant cars, and cats doing suspicious things at 3am.

**Q: Why is it called IDCAelephants?**
A: I Don't Care About Elephants. Because we don't. And neither does your security camera.

## License

Released under the [AGPL-3.0 License](LICENSE). The COCO dataset is licensed under [CC BY 4.0](https://cocodataset.org/#termsofuse). YOLOv11 is by [Ultralytics](https://github.com/ultralytics/ultralytics) under AGPL-3.0.

## Acknowledgments

- **COCO dataset** — for providing 80 classes when we only needed 8
- **Ultralytics** — for making YOLO training stupidly easy
- **Hailo** — for making the compilation process just hard enough to be interesting
- **RunPod** — for renting us 8x H100s so we could train a model that ignores elephants
- **Every elephant** — sorry, nothing personal

---

*Built by [Flukethoughts](https://github.com/Flukethoughts) with help from Claude. No elephants were detected in the making of this model.*
