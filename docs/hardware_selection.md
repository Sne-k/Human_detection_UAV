# Hardware Selection Matrix - Human Detection Payload

The payload's share of the Phase-2 hardware trade study. Covers the companion
computer, thermal camera and RGB camera. Flight-control hardware is selected
separately.

Every requirement here is derived from a measurement in this repository rather
than from a specification sheet, and each row names the script that produced
it.

**Nothing has been purchased.** This is the basis for that decision.

---

## 1. Derived Requirements

| # | Requirement | Value | Derived by |
| - | ----------- | ----- | ---------- |
| R1 | Detection frame rate | >= 6.7 FPS at 60 m, >= 4.0 at 100 m | `coverage_requirements.py` |
| R2 | Thermal sensor resolution | >= 384 x 288 | `sensor_resolution_study.py` |
| R3 | Memory for inference | >= 512 MB free | `benchmark_edge_cpu.py` |
| R4 | Model input | 640 x 512, single class | MT-005 configuration |
| R5 | Sustained thermal headroom | No throttling under continuous load | Pi 5 characteristic |
| R6 | Electrical isolation from flight control | Separate regulator | Hadi et al. [5] |

### On R1

Not a comfort target. The binding case is movement classification at 60 m and
20 m/s cruise, where the aircraft crosses its own along-track footprint in
2.24 s. Coverage alone needs only 0.64 FPS; a reflex 30 FPS target would
oversize the board by roughly an order of magnitude.

Under the detect-and-report concept of operations adopted in
`training_log.md` section 28, the search-phase requirement falls to 1.34 FPS
and movement classification moves to a loiter phase, where the ground speed is
lower and the requirement falls with it.

---

## 2. Companion Computer

### Candidates

| Criterion | Raspberry Pi 5 4 GB | Pi 5 + AI HAT+ (26 TOPS) | Jetson Orin Nano Super |
| --------- | ------------------- | ------------------------ | ---------------------- |
| Compute | 4x Cortex-A76 2.4 GHz | + Hailo-8 NPU | 1024 CUDA, 67 TOPS |
| Memory | 4 GB | 4 GB | 8 GB |
| Est. FPS, MT-005 | 5.4 - 9.0 | Well above R1 | Well above R1 |
| Meets R1 at 60 m | Marginal | Yes | Yes |
| Meets R1 at 100 m | Yes | Yes | Yes |
| Meets R3 | Yes, 25x headroom | Yes | Yes |
| Two-model ensemble | 1.7 - 2.9 FPS | Yes | Yes |
| Larger model (YOLO26s) | No, 1.7 - 2.8 FPS | Yes | Yes |
| Power | ~7-12 W | ~12-17 W | 7-25 W |
| Toolchain | ONNX Runtime | Hailo SDK, recompile | CUDA / TensorRT |
| Cost | Lowest | Middle | Highest |

### Measured basis

CPU inference, ONNX Runtime, 4 threads, 512 x 640, from
`benchmark_edge_cpu.py`. Development figures are x86 on a Ryzen 7 7435HS;
target estimates apply a 3-5x per-core slowdown. **That is an engineering
estimate, not a conversion**, and must be confirmed on real hardware.

| Model            | Dev CPU  | Target estimate | Target FPS |
| ---------------- | -------: | --------------- | ---------- |
| MT-005 (YOLO26n) |  37.1 ms | 111 - 186 ms    | 5.4 - 9.0  |
| MT-006 (960 px)  |  77.4 ms | 232 - 387 ms    | 2.6 - 4.3  |
| YOLO26s          | 117.3 ms | 352 - 586 ms    | 1.7 - 2.8  |
| MT-004 RGB 1280  | 185.9 ms | 558 - 930 ms    | 1.1 - 1.8  |

Memory, measured directly:

| Configuration | Resident |
| ------------- | -------: |
| Single model + opencv + frame buffers | 150 MB |
| Three models resident | 383 MB |

**R3 is satisfied with roughly 25x headroom on a single model.** Memory is not
the constraint; a 2 GB board would also run it. 4 GB is chosen for margin, not
necessity.

### Recommendation: Raspberry Pi 5, 4 GB

With the Active Cooler, and with the detect-and-report concept of operations.

**Reasoning.** It meets R1 at 100 m outright and marginally at 60 m, meets R3
with large margin, is the lowest cost and lowest power, and uses the ONNX
export already built and verified. Flying the search pattern at 100 m - well
within the 60-130 m band the detector was trained on - removes the marginal
case entirely.

**Against it.** Both reference systems in the literature chose Jetson.
Lygouras et al. [4] used a Raspberry Pi 3 for autonomous landing, judged it
insufficient for swimmer detection, and moved to a Jetson TX1 at 12 fps. That
precedent deserves weight. Two things reduce it: a Pi 3 to Pi 5 is roughly
8-10x, and their 12 fps was needed to hover over a *drifting* swimmer, where a
fixed-wing search pattern needs 6.7.

**The condition.** If movement classification is required during the search
phase at 60 m, or if a two-model ensemble is wanted in flight, the Pi 5 CPU
does not suffice and the AI HAT+ becomes necessary. That is the decision point
to hold open until the concept of operations is fixed.

### Risks

| Risk | Severity | Mitigation |
| ---- | -------- | ---------- |
| Thermal throttling in a sealed fuselage | High | Active Cooler, ducted airflow, thermal soak test |
| ARM slower than the 3-5x estimate | Medium | Benchmark on real hardware before committing |
| Power draw exceeds budget | Medium | Measure under sustained load; dedicated regulator |
| microSD failure under logging | Low | NVMe via M.2 HAT |

---

## 3. Thermal Camera

**The decision with the largest effect on mission performance, and the most
expensive component.** The pressure to economise lands exactly where it does
most damage.

| Sensor class | Resolution | Recall | vs native | Extra missed | Verdict |
| ------------ | ---------- | ------ | --------- | -----------: | ------- |
| 640 x 512 module | 640 x 512 | 0.9288 | 100.0% | +0 | **Preferred** |
| Mid-range LWIR | 384 x 288 | 0.8905 | 95.9% | +100 | **Acceptable floor** |
| Budget USB module | 256 x 192 | 0.8407 | 90.5% | +230 | Marginal |
| FLIR Lepton 3.5 | 160 x 120 | 0.6254 | 67.3% | **+792** | **Reject** |

From `sensor_resolution_study.py`: each test image is degraded to the
candidate resolution and resampled back, reproducing the information the
sensor would actually have captured. Ground truth is unchanged.

### Why the Lepton class is rejected

It loses one person in three. Images where a person is present and nothing at
all is reported rise from 8 to 72, a nine-fold increase. For a search payload
that is a different capability, not a degraded one.

### Caveat

The detector was trained at 640 x 512, so part of the loss is domain mismatch
rather than pure information loss. Retraining at the target resolution would
recover some of it, but cannot recover what the sensor never captured - at
160 x 120 a person is 3 x 5 px. **If a sensor below 640 x 512 is chosen,
retraining at that resolution is mandatory and the recall figures must be
restated.**

### Recommendation

**640 x 512 if the budget allows; 384 x 288 as the floor.** Spend here rather
than on the compute board. A better camera with a Pi 5 outperforms a worse
camera with a Jetson, because no amount of compute recovers information the
sensor did not capture.

---

## 4. RGB Camera

| Criterion | Requirement |
| --------- | ----------- |
| Resolution | 1920 x 1080 sufficient |
| Frame rate | >= 10 fps |
| Interface | USB 3.0 or MIPI CSI |

### An unresolved tension

MT-004 is the selected RGB reference because resolution improved its metrics
at 1280 px. At an estimated 558-930 ms per frame it **cannot run on a CPU-only
board**. An RGB detector for this aircraft would have to drop to 640 px, which
returns it to roughly MT-002 accuracy - mAP@50 about 0.50 against MT-004's
0.654.

Three options:

1. **RGB without AI.** Revert the RGB camera to its originally planned role:
   delivery-zone confirmation, payload-release monitoring and recording. Run
   detection on thermal only. Lowest risk, and matches the original camera
   architecture.
2. **RGB detection at 640 px.** Accept roughly MT-002 accuracy for a daylight
   detection capability.
3. **AI accelerator.** Makes MT-004 viable and resolves the tension, at cost.

**Recommendation: option 1** for the current phase. Thermal is the stronger
search modality, works at night when search and rescue actually happens, and
already meets the frame-rate requirement. RGB earns its place on the aircraft
for delivery confirmation regardless of whether it runs a detector.

---

## 5. Summary

| Component | Selection | Confidence | Blocking |
| --------- | --------- | ---------- | -------- |
| Companion computer | Raspberry Pi 5 4 GB + Active Cooler | Medium | Confirm on hardware |
| Thermal camera | >= 384 x 288, prefer 640 x 512 | **High** | Budget |
| RGB camera | Any 1080p, no AI in this phase | High | None |
| Storage | NVMe via M.2 HAT | High | None |
| AI accelerator | Defer | Medium | Concept of operations |

### What would change the recommendation

- **Movement classification required during search at 60 m** - the Pi 5 CPU
  does not suffice; add the AI HAT+.
- **Real Pi 5 benchmarks worse than the 3-5x estimate** - same conclusion.
- **RGB detection becomes a requirement** - same conclusion.
- **Thermal budget forces below 384 x 288** - retraining becomes mandatory and
  the mission recall target must be lowered explicitly.

The single highest-value next measurement is a real Pi 5 running the exported
MT-005 ONNX graph. Everything in section 2 is an extrapolation from x86 until
that exists.
