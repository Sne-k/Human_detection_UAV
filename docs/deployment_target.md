# Deployment Target and Project Context

This page records where the human-detection subsystem sits in the wider UAV
project, what hardware it must eventually run on, and what that hardware
implies for the models developed so far.

It exists because the detection work was developed entirely on an RTX 4050
laptop, while the aircraft will carry a Raspberry Pi class board. Several
optimisation results measured during development **do not transfer to the
target at all**, and that needs to be stated plainly rather than discovered
during integration.

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

## 2. Hardware Target

Selected under explicit budget constraints. A Jetson was considered and
**rejected on cost**.

| Component         | Selection                                      |
| ----------------- | ---------------------------------------------- |
| Flight controller | Pixhawk-class (separate from vision)           |
| Companion computer| Raspberry Pi 5, 2 GB                           |
| AI accelerator    | Optional, only if required                     |
| RGB camera        | USB/CSI, not yet purchased                     |
| Thermal camera    | Low-resolution LWIR module, not yet purchased  |
| Link              | MAVLink to flight controller                   |

Raspberry Pi 5: 4x Cortex-A76 at 2.4 GHz, no CUDA device.

### What this invalidates

Three results measured during development do not apply to this target:

| Development result                     | Applies to the Pi 5? |
| -------------------------------------- | -------------------- |
| CUDA-graph speedup, 16x on MT-005      | **No** - no CUDA device |
| Launch-bound diagnosis                 | **No** - no kernel launches to remove |
| Batch-8 throughput figures             | **No** - a live camera delivers one frame at a time |
| ONNX export and its verification       | **Yes** - this is the part that transfers |

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

**MT-005 is deployable on the budget hardware; the ensemble is not.**

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

### Is 5-9 FPS enough?

Probably yes, for this mission. At 20 m/s cruise, 5 FPS gives a detection
opportunity every 4 m of ground track. Search and rescue needs coverage of the
search area, not high frame rate; a person is typically in view across many
consecutive frames. The tracking and movement-detection stages already assume
a modest frame rate.

This should still be verified against the camera's field of view and the
planned search altitude, which are not yet fixed.

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
