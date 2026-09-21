# Comparison with the Reference Systems

Two papers in the project's literature survey build the same thing this
subsystem builds - onboard human detection on a search-and-rescue UAV. They are
the right systems to position this work against:

- **Rizk, Slim and Charara (2021)**, reference [3] - *Toward AI-Assisted UAV
  for Human Detection in Search and Rescue Missions*
- **Lygouras et al. (2019)**, reference [4] - *Unsupervised Human Detection
  with an Embedded Vision System on a Fully Autonomous UAV for Search and
  Rescue Operations*

---

## 1. The Comparison That Must Not Be Made

It is tempting to put this next to them:

| System     | Headline number |
| ---------- | --------------- |
| Rizk       | 82.21%          |
| Lygouras   | 67% mAP         |
| MT-005     | 93.3% mAP@50    |

**That table is wrong and should not appear in the report.** Three separate
problems make those numbers non-comparable:

1. **Different metrics.** Rizk reports an "average accuracy" over test images,
   plus a per-image detection rate of 88.88%. That is not mean Average
   Precision. Lygouras reports 67% mAP at an inference resolution of 416 x 416.
   MT-005 reports mAP@50 computed by the Ultralytics validator. Three
   quantities, one column.

2. **Radically different altitudes.** This is the important one.

   | System   | Operating altitude          | Person size in frame |
   | -------- | --------------------------- | -------------------- |
   | Lygouras | ~5.75 m minimum, hovering   | Very large           |
   | Rizk     | 14 m for the live video test| Large                |
   | MT-005   | **60 - 130 m**              | ~12 x 19 px          |

   A swimmer filmed from 6 m and a person imaged from 100 m are not the same
   detection problem. The altitude difference alone is an order of magnitude,
   and the failure analysis in this project showed that performance is
   dominated by target scale.

3. **Different test protocols.** Rizk evaluates on 940 images. Lygouras keeps
   10% of 9,000 images as validation and reports on that. This project holds
   out a 579-image test split that is never used for training or model
   selection, and reports on it separately from validation.

---

## 2. The Comparison That Is Fair

Where the systems can be compared is in **method and rigour**, not in a single
accuracy figure.

| Dimension        | Rizk (2021)               | Lygouras (2019)            | This project                     |
| ---------------- | ------------------------- | -------------------------- | -------------------------------- |
| Detector         | YOLOv3                    | Tiny YOLOv3 @ 416          | YOLO26n @ 640                    |
| Compute          | Jetson Xavier NX          | Jetson TX1, 12 fps         | Undecided, Pi 5 class candidate  |
| Training data    | 3,500 internet images     | 9,000 (4,500 + negatives)  | VisDrone 6,471 + HIT-UAV 2,029   |
| Data source      | Internet, non-aerial      | Own GoPro drone footage    | Two published aerial benchmarks  |
| Modality         | RGB only                  | RGB only                   | RGB **and thermal**              |
| Altitude         | 14 m live test            | ~6 m, hovering             | 60 - 130 m                       |
| Held-out test    | 940 images                | 10% validation split       | 579-image held-out split         |
| Error analysis   | Not reported              | False-positive driven      | Per-person failure mechanisms    |
| Geo-referencing  | Yes, GPS + RFD900         | Yes, GNSS + landing        | Yes, `scripts/geolocate.py`      |
| Tracking         | Not reported              | Not reported               | ByteTrack + movement state       |

### Where this project is genuinely stronger

- **Aerial training data.** Rizk's 3,500 images were "collected from internet
  sources" showing full or partial human bodies - not aerial imagery. A person
  seen from above at 100 m looks nothing like a person in a web photograph.
  This project trains on two published UAV benchmarks throughout.
- **Thermal.** Both papers are RGB only, although Rizk cites thermal work in
  their related-work section. The thermal branch here is a real capability
  difference for night and low-light operation, which is exactly when search
  and rescue happens.
- **Altitude.** Operating at 60-130 m rather than 6-14 m covers far more ground
  per pass, which is the point of using an aircraft.
- **Error analysis.** Neither paper decomposes its failures. This project
  identifies the mechanisms - merged boxes in crowds, undersized boxes on
  partial thermal signatures, daylight contrast failures - which is what makes
  the next experiment designed rather than guessed.

### Where they are ahead

- **Both flew.** Both systems are integrated on real aircraft with real
  cameras and demonstrated end to end. This project has no hardware yet, and
  every deployment figure here is an extrapolation.
- **Both close the loop.** Rizk transmits a notification packet over a 40 km
  RFD900 link. Lygouras autonomously navigates to the detected swimmer and
  releases rescue apparatus. Neither telemetry nor any flight-controller
  interface exists here yet.
- **Lygouras handles the false-positive problem deliberately.** See below.

---

## 3. Two Findings Worth Acting On

### Lygouras's negative-mining method

Their first training run produced many false positives - boats, mostly. Rather
than adjusting thresholds, they collected a second round of images
*specifically containing the objects that were being falsely detected*, at a
1:1 ratio with the positives, doubling the dataset to 9,000. They report this
raised both recall and mAP.

This is directly applicable. MT-005 produces **638 unmatched prediction boxes**
at confidence 0.25 against 2,425 matched persons - a custom precision of 0.79.
Nothing in this project has yet attempted targeted hard-negative mining. HIT-UAV
already contains 864 background-only training images, but they are whatever the
dataset happened to include, not images chosen because the model fails on them.

Mining the actual false positives from the test predictions and adding
comparable imagery to training is an untried, cheap experiment with a
precedent in the literature.

### Lygouras rejected a Raspberry Pi

Worth recording honestly, since a Pi 5 is the leading candidate here. They used
a Raspberry Pi 3 with NNPACK for autonomous landing on a fixed target, then
concluded that for swimmer detection "a higher resolution as well as a higher
frame rate is needed... a powerful GPU embedded system with low consumption
would be required", and moved to a Jetson TX1 at 12 fps.

Two things make that precedent weaker than it first appears:

1. **A Pi 3 is not a Pi 5.** Cortex-A53 at 1.2 GHz versus Cortex-A76 at
   2.4 GHz, roughly an 8-10x difference in practice.
2. **Their frame-rate requirement was higher.** They hover over a drifting
   swimmer and release apparatus onto them, so position must update fast. A
   fixed-wing aircraft flying a search pattern needs 6.7 FPS by the derivation
   in `deployment_target.md`, not 12.

It remains a genuine caution: an independent team building a comparable system
found Pi-class hardware insufficient and paid for a Jetson. That should be
weighed against the measurements here rather than dismissed.

---

## 4. How to Present This in the Report

State the comparison qualitatively, and give the altitude difference
explicitly. For example:

> The proposed subsystem detects people from 60-130 m using both RGB and
> thermal imagery, achieving 0.933 mAP@50 and 0.892 recall on a held-out
> 579-image thermal test split. Direct numerical comparison with Rizk et al.
> [3] and Lygouras et al. [4] is not meaningful: those systems operate at 14 m
> and approximately 6 m respectively, report different metrics, and use RGB
> only. The present work addresses a substantially longer detection range and
> adds a thermal branch for night operation, at the cost of not yet having been
> demonstrated on flight hardware.

That last clause matters. Both reference systems flew; this one has not. Saying
so is more credible than omitting it, and it frames the remaining hardware work
as the obvious next phase rather than a gap.
