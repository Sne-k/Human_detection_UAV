# Deployment Target and Project Context

This page records where the human-detection subsystem sits in the wider UAV
project, what hardware it must eventually run on, and what that hardware
implies for the models developed so far.

It exists for two reasons. The detection work was developed entirely on an
RTX 4050 laptop, and several optimisation results measured there **do not
transfer to a CPU-only companion computer at all** - better stated plainly now
than discovered during integration. And the board is still undecided, so the
useful output is a requirement to select against rather than a tuning target.

---

## 1. Project Context

**Project:** Development of a Fixed-Wing eVTOL UAV for Emergency Logistics

| Phase              | Scope                                                |
| ------------------ | ---------------------------------------------------- |
| Previous semester  | Conceptual and preliminary aircraft design           |
| **This semester**  | **Flight control system development**                |
| Later              | Simulation, HIL, flight testing                      |

Airframe established in the previous phase: approximately 5 kg MTOW, 20 m/s
cruise, 20 km range, ~1 hour endurance, 4+1 propulsion (four lift motors plus
one pusher), 1.7 m wingspan, S7075 airfoil, 15.66% static margin.

### Where human detection sits

The formal objectives for this semester are all flight-control-system
objectives: architecture, requirements traceability, hardware trade study,
communication and power architecture, control algorithms, simulation,
integration and validation.

**Human detection is not one of them.** It is a parallel workstream, started
early specifically because the camera and companion computer had not been
purchased yet, so the algorithm could be developed on a laptop and be ready
when the hardware arrives.

This has two consequences worth being explicit about:

1. The detection subsystem is a **payload**, not part of the flight-control
   loop. A failure in the vision computer must not affect flight control. The
   two communicate over MAVLink; the flight controller remains solely
   responsible for flight-critical control.
2. Detection accuracy beyond what the mission needs has **no project value**.
   The deliverable is a working payload on the target hardware, not a
   state-of-the-art detector.

### Intended system architecture

```text
   Sensors ------> Flight Controller ------> Motors / Servos
   (IMU, GPS,      (Pixhawk class)
    baro, airspeed)          |
                             | MAVLink
                             |
   RGB camera ----+          |
                  +--> Companion Computer --> Detection output
   Thermal LWIR --+          (Raspberry Pi 5)         |
                                                      v
                                            Telemetry / Ground Station
```

The detection output feeds the ground station, tagged with UAV position. It
does not command the aircraft.

---

## 2. Hardware Candidates

**The companion computer is not yet decided.** A Raspberry Pi 5 class board is
the leading candidate under budget constraints, with a Jetson considered and
set aside on cost, but nothing is committed and no hardware has been bought.

| Component          | Status                                          |
| ------------------ | ----------------------------------------------- |
| Flight controller  | Pixhawk-class, separate from vision             |
| Companion computer | **Undecided** - Pi 5 class is the front-runner  |
| AI accelerator     | Undecided, needed only if the board falls short |
| RGB camera         | Not purchased                                   |
| Thermal camera     | Low-resolution LWIR, not purchased              |
| Link               | MAVLink to flight controller                    |

Because the board is open, the useful thing is not to optimise for one
candidate but to state the **requirement** the board has to meet. Section 3
derives that from the mission, and section 4 says what clears it.

### What this invalidates

If the board is CPU-only - which every candidate under consideration is,
absent an accelerator - then three results measured during development do not
apply:

| Development result                | Applies to a CPU-only board? |
| --------------------------------- | ---------------------------- |
| CUDA-graph speedup, 16x on MT-005 | **No** - no CUDA device |
| Launch-bound diagnosis            | **No** - no kernel launches to remove |
| Batch-8 throughput figures        | **No** - a live camera delivers one frame at a time |
| ONNX export and its verification  | **Yes** - this is the part that transfers |

The launch-bound analysis was correct about the development machine and is
still the reason single-frame GPU timings there looked flat across
resolutions. It simply describes a bottleneck the target does not have.

---

## 3. Measured CPU Performance

Measured with `scripts/benchmark_edge_cpu.py`: exported ONNX graphs run
through ONNX Runtime on CPU, threads restricted to 4 to match the target's
core count.

**These are x86 measurements on a Ryzen 7 7435HS, used as a proxy.** The
target estimate applies a 3-5x per-core slowdown for a Cortex-A76 at 2.4 GHz.
That is an engineering estimate, not a conversion: NEON and AVX2 do not scale
identically across operator types. Confirm on real hardware before any
purchase decision rests on these numbers.

| Model  | Input      | Dev CPU  | Pi 5 estimate | Estimated FPS | Verdict      |
| ------ | ---------- | -------: | ------------- | ------------- | ------------ |
| MT-005 | 512 x 640  |  37.1 ms | 111 - 186 ms  | 5.4 - 9.0     | Viable       |
| MT-007 | 512 x 640  |  34.9 ms | 105 - 174 ms  | 5.8 - 9.5     | Viable       |
| MT-006 | 768 x 960  |  77.4 ms | 232 - 387 ms  | 2.6 - 4.3     | Marginal     |
| MT-004 | 1280 x 1280| 185.9 ms | 558 - 930 ms  | 1.1 - 1.8     | Not viable   |
| 3-model ensemble | - | 149.3 ms | 448 - 747 ms | 1.3 - 2.2    | Not viable   |

### What this means

**A single 640 px model is deployable on a CPU-only board; the ensemble is
not, at least not with movement classification.**

This inverts the conclusion reached from GPU measurements. On the laptop, the
three-model ensemble under CUDA graphs cost 9.4 ms per frame and was
comfortably affordable. On the actual target it costs an estimated 0.45-0.75
seconds per frame, which no amount of graph optimisation recovers, because the
cost is arithmetic rather than dispatch overhead.

The ensemble's accuracy advantage is real - 51 more persons found, at 1.6 false
positives each - but it costs roughly 4x the compute of a single model. That
trade is affordable on a GPU and not affordable on this board.

**The RGB path needs rethinking.** MT-004 at 1280 px is the selected RGB
reference precisely because resolution improved its metrics, but at an
estimated 0.56-0.93 s per frame it cannot run on the target. An RGB detector
for this aircraft would have to drop to 640 px, which returns it to roughly
MT-002 performance (mAP@50 about 0.50). That is a real and unresolved tension
between the RGB accuracy work and the deployment constraint.

---

## 3b. How Much Frame Rate Does the Mission Actually Need?

Asking a board to hit 30 FPS is a reflex from video, not a requirement. A
search payload needs two things, and both are set by ground speed and the
camera's along-track footprint. Derived by
`scripts/coverage_requirements.py` at 20 m/s cruise with a 640 x 512 sensor
and a 50 degree horizontal field of view:

| Altitude | Swath  | Along-track | Dwell  | Coverage | Confirm | Tracking |
| -------: | -----: | ----------: | -----: | -------: | ------: | -------: |
|     60 m |  56.0 m|      44.8 m | 2.24 s | 0.64 FPS | 1.34 FPS| 6.70 FPS |
|     80 m |  74.6 m|      59.7 m | 2.98 s | 0.48 FPS | 1.01 FPS| 5.03 FPS |
|    100 m |  93.3 m|      74.6 m | 3.73 s | 0.38 FPS | 0.80 FPS| 4.02 FPS |
|    130 m | 121.2 m|      97.0 m | 4.85 s | 0.29 FPS | 0.62 FPS| 3.09 FPS |

- **Coverage** - every point of ground passes through at least one processed
  frame, with 30% along-track overlap so nothing is lost at a frame edge.
- **Confirm** - three frames on the same target, so a detection is not a
  single-frame artefact.
- **Tracking** - the 15 frames the movement classifier needs before it will
  commit to moving or stationary.

**The binding requirement is 6.7 FPS**, at the lowest altitude, and it comes
from movement detection rather than from coverage. Coverage alone is satisfied
below 1 FPS, because the aircraft takes 2-5 seconds to traverse its own
footprint. Area search rate at 60 m is about 4.0 km2/h.

### Matching that against the measurements

| Configuration          | Estimated FPS | 6.7 FPS (tracking) | 1.34 FPS (report only) |
| ---------------------- | ------------- | ------------------ | ---------------------- |
| MT-005 single, CPU     | 5.4 - 9.0     | Marginal at 60 m, clears it at 80 m+ | Yes, comfortably |
| MT-007 single, CPU     | 5.8 - 9.5     | Same               | Yes                    |
| MT-006 single, CPU     | 2.6 - 4.3     | No                 | Yes                    |
| 3-model ensemble, CPU  | 1.3 - 2.2     | No                 | Marginal               |
| MT-004 RGB 1280, CPU   | 1.1 - 1.8     | No                 | Marginal               |

This is a more useful answer than "the ensemble is not viable". It is not
viable *for movement classification*. It is close to sufficient for the
detect-and-report mission, which is the part that actually saves lives.

Three ways to buy margin without new hardware, in order of preference:

1. **Fly higher.** At 100 m the tracking requirement falls to 4.0 FPS and a
   single CPU model clears it outright. The detector was trained on imagery
   from 60-130 m, so this costs nothing in accuracy.
2. **Relax movement classification.** Reporting *where* people are, without
   classifying movement, drops the requirement to 1.34 FPS and puts even the
   ensemble in range.
3. **Shorten the movement history window.** 15 frames is a tuning choice, not
   a physical constant; 8 frames would halve the requirement, at some cost in
   movement-classification stability.

### The specification to shop with

Any companion computer that sustains **about 7 FPS on a 640 x 512 YOLO26n**
satisfies the full mission at every trained altitude. That is the number to
take to a hardware trade study - considerably less demanding than a reflex
30 FPS target, which would oversize the board by roughly an order of magnitude
on an airframe where weight and power are already constrained.

---

## 3c. Memory Footprint

Measured directly, not estimated, with ONNX Runtime on CPU:

| Stack                                      | Resident |
| ------------------------------------------ | -------: |
| python + numpy + opencv + onnxruntime      |    58 MB |
| **+ MT-005 session + frame buffers**       | **150 MB** |
| All three models resident (ensemble)       |   383 MB |

Against a 4 GB board running a headless OS (~200-300 MB), that leaves over
3.5 GB free. **Memory is not a constraint for this workload** - there is
roughly 25x headroom on a single model. Even a 2 GB board would run it; 4 GB
simply buys margin for the OS, video buffers and logging.

The binding constraints are compute, thermal headroom and sensor resolution -
not RAM.

---

## 3d. Thermal Sensor Resolution - the Binding Purchase Decision

Every accuracy figure in this project assumes a **640 x 512** thermal sensor,
because that is what HIT-UAV was captured at. Budget LWIR modules are much
coarser, and the failure analysis already established that the remaining
misses cluster around targets of 12 x 19 px. A coarser sensor pushes the whole
population below that.

`scripts/sensor_resolution_study.py` measures the effect by downscaling each
test image to a candidate sensor resolution and resampling back to the model
input, which reproduces the information the camera would actually have
captured. Ground truth is unchanged, so recall is directly comparable.

MT-005, 579 test images, 2,611 persons, confidence 0.25:

| Sensor      | Person size | Matched | Recall | vs native | Extra missed | Blind images |
| ----------- | ----------- | ------: | ------ | --------- | -----------: | -----------: |
| 640 x 512   | 12.0 x 19.0 |   2,425 | 0.9288 | 100.0%    |           +0 |            8 |
| 384 x 288   |  7.2 x 11.4 |   2,325 | 0.8905 | 95.9%     |         +100 |           17 |
| 320 x 256   |  6.0 x 9.5  |   2,291 | 0.8774 | 94.5%     |         +134 |           18 |
| 256 x 192   |  4.8 x 7.6  |   2,195 | 0.8407 | 90.5%     |         +230 |           28 |
| 160 x 120   |  3.0 x 4.8  |   1,633 | 0.6254 | 67.3%     |         +792 |           72 |

### Reading this

**A 160 x 120 sensor - the FLIR Lepton class, and the usual budget choice -
loses one person in three.** Recall falls from 0.929 to 0.625, and images where
a person is present but nothing at all is reported rise from 8 to 72, a 9x
increase. That is not a degraded search; it is a different capability.

A 256 x 192 USB module costs 230 people, 8.8 points of absolute recall. 
384 x 288 costs 100 people, 3.8 points, which is defensible.

**The thermal camera, not the companion computer, is the decision that
determines whether this payload works.** It is also the expensive component,
so the temptation to economise lands exactly where it does the most damage.

### One caveat, and an option

These numbers use a detector *trained* at 640 x 512 and then shown coarser
imagery, so part of the loss is domain mismatch rather than pure information
loss. Retraining at the target sensor resolution would recover some of it -
the model would at least learn what a 5 x 8 px person looks like.

It cannot recover all of it. Information the sensor never captured is gone,
and at 160 x 120 a person is 3 x 5 px, which is close to the limit of what any
detector can localise to IoU 0.50. If a coarse sensor is chosen for cost
reasons, retraining at that resolution should be treated as mandatory rather
than optional, and the recall target revised accordingly.

---

## 3e. A Precedent Worth Weighing

Lygouras et al. (2019), reference [4] in the literature survey, built a
comparable onboard-detection rescue UAV. They used a Raspberry Pi 3 with
NNPACK for autonomous landing on a fixed target, then concluded that swimmer
detection needed "a higher resolution as well as a higher frame rate" and a
"powerful GPU embedded system", and moved to an Nvidia Jetson TX1 running at
12 fps. Rizk et al. (2021), reference [3], likewise chose a Jetson Xavier NX.

Both reference systems chose Jetson-class hardware. That is a real signal and
should not be waved away.

Two things make it weaker than it looks here:

1. A Pi 3 is not a Pi 5 - Cortex-A53 at 1.2 GHz against Cortex-A76 at 2.4 GHz,
   roughly 8-10x in practice.
2. Their frame-rate requirement was higher. They hover over a drifting swimmer
   and release apparatus onto them, so position must update quickly. A
   fixed-wing aircraft flying a search pattern needs 6.7 FPS by the derivation
   in section 3b, not 12.

See [`literature_comparison.md`](literature_comparison.md) for the full
comparison.

---

## 3f. The Architecture Budget Gate

Everything above measures models that already exist. The complementary
question is whether a *proposed* architecture can fit before a training run is
spent on it, and `scripts/architecture_budget.py` answers it: build the
candidate from its YAML at the deployment class count, export to ONNX at the
deployment shape, time it through the same CPU harness.

Weights are random, because latency depends on the graph and not on weight
values. **The cost is knowable before the accuracy is.**

| Architecture | Params | GFLOPs | Dev CPU | Pi 5 estimate | Pi 5 FPS |
|--------------|-------:|-------:|--------:|---------------|----------|
| YOLO26n (deployed) | 2.50 M | 4.7 | 38.8 ms | 117 - 194 ms | 5.2 - 8.6 |
| YOLO26n-p2 | 2.52 M | 6.1 | 50.2 ms | 150 - 251 ms | 4.0 - 6.6 |
| YOLO26s | 9.95 M | 18.2 | 114.2 ms | 343 - 571 ms | 1.8 - 2.9 |

| Architecture | 6.7 FPS @ 60 m | 4.0 FPS @ 100 m | 1.34 FPS report-only |
|--------------|----------------|-----------------|----------------------|
| YOLO26n | marginal | clears | clears |
| YOLO26n-p2 | **FAILS** | marginal | clears |
| YOLO26s | FAILS | FAILS | clears |

This cancelled MT-012, the P2 small-object head recommended by the literature
survey, before it was trained - its optimistic end still falls below the
binding requirement.

**GFLOPs is not a safe proxy for cost here.** A P2 head costs 1.3-1.5x latency
for 1.30x arithmetic because a stride-4 feature map is bandwidth-bound, and
bandwidth is where a Pi 5 is weakest relative to x86. A wider backbone runs
the other way: YOLO26s costs 2.9-3.0x for 3.87x arithmetic, because dense
matrix work is what SIMD and cache handle well.

The rule: **changes that touch the network go through this gate before they
are trained; changes that touch only training - losses, augmentation,
datasets, schedules - are free at inference and do not.** Every experiment
from MT-005 to MT-010 was the second kind, which is why the deployed cost has
never moved.

---

## 4. Options If More Performance Is Needed

| Option                       | Effect                                    | Cost |
| ---------------------------- | ----------------------------------------- | ---- |
| Keep MT-005 at 640, CPU only | 5-9 FPS, no extra hardware                | None |
| Add an AI accelerator (Hailo)| Large speedup, enables ensemble           | Adds cost |
| INT8 quantisation            | Typically 2-3x on CPU, needs re-verification | None, but accuracy must be re-checked |
| NCNN / TFLite instead of ORT | Often faster than ORT on ARM              | Effort only |
| Reduce input resolution      | Direct saving, costs recall               | None |

INT8 quantisation is the most promising untried option, because it costs
nothing but effort and the verification harness to check it already exists.
It has not been attempted, and it would change the detection numbers, so it
would need a full re-run of the accuracy comparison.

---

## 5. Consequences for the Work Plan

1. **Candidate B (the three-model ensemble) is deprioritised for deployment.**
   It remains the best-performing configuration and is worth reporting as
   such, but it is not the deployment candidate on current hardware.

2. **The A / B / C comparison needs a hardware column.** Accuracy alone would
   select the ensemble; adding the target-latency column changes the answer.

3. **MT-008 matters more than it did.** If mosaic ablation improves a single
   640 px model, that improvement is directly deployable, whereas the
   ensemble's is not.

4. **Real-hardware confirmation is now the highest-value missing measurement.**
   Everything above is an extrapolation from x86.

5. **The RGB/thermal split should be revisited.** If only one detector can run,
   the thermal model is the better choice for the search mission, and the RGB
   camera reverts to its originally planned role: delivery-zone confirmation
   and recording, without AI detection.
