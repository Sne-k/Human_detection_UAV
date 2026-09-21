# Related Work and Techniques Worth Adopting

A survey of comparable systems and published work on the same dataset, done to
find techniques worth adapting after seven single-model training directions
had failed.

Two categories are distinguished throughout: work on **HIT-UAV**, whose numbers
are directly comparable to this project's, and work on other datasets, which
is not.

---

## 1. The Benchmark That Is Actually Comparable

The HIT-UAV dataset paper (Suo et al., *Scientific Data*, 2023) publishes
baselines on the same data and splits this project uses. These are the only
external numbers that can be compared without caveat.

| Model | Person AP@0.50 |
|-------|---------------:|
| YOLOv4 | 89.88% |
| SSD-512 | 85.6% |
| Faster-RCNN | 75.5% |
| YOLOv4-tiny | 16.86% |
| **MT-005 (this project)** | **93.3%** |

MT-005 exceeds the strongest published baseline by **3.4 points**.

### Caveats that must accompany that claim

1. **Single-class versus four-class.** The paper trains on Person, Car,
   Bicycle and OtherVehicle; this project trains person-only. Removing
   inter-class confusion makes the task easier, so this is not a like-for-like
   architecture comparison.
2. **Six years of architecture progress.** YOLOv4 is 2020, YOLO26n is 2026.
   Beating it is expected, not remarkable.
3. What the comparison does establish is that the pipeline, conversion and
   training in this project produce a result in the right range on a
   recognised benchmark - which is worth stating precisely because the
   comparisons against Rizk, Lygouras and AlienSight cannot be made at all.

---

## 2. Comparable Systems

### AlienSight (Ghitri et al., 2025)

The strongest comparable system found. YOLOv8n with two-stage fine-tuning,
monocular geolocation from telemetry, deployed on a Qualcomm RB3 Gen 2.

| Aspect | AlienSight | This project |
|--------|------------|--------------|
| Flew a real aircraft | Yes, DJI Air 3S | No |
| Geolocation validated | **60 surveyed points, 1.01 m mean error** | Geometry only |
| Edge deployment measured | 37.3 ms / 26.8 FPS on real NPU | Extrapolated from x86 |
| Modality | RGB only | **Thermal and RGB** |
| Altitude | 15 - 30 m | **60 - 130 m** |
| Published | Preprint with DOI | No |

**They validated on hardware and this project has not.** That is the honest
gap, and it does not close with more experiments - only with a camera and an
airframe.

**Where this project leads:** AlienSight is daylight-only. It cannot operate at
night. This project's thermal branch reaches 0.942 recall in darkness, and
operates at two to eight times the altitude.

**Their transferable finding:** two-stage sequential fine-tuning.

| Training | mAP@50 |
|----------|-------:|
| VisDrone only | 0.597 |
| HERIDAL only | 0.941 |
| **VisDrone -> HERIDAL** | **0.965** |

An aerial-domain pretraining stage before the target dataset gave a
substantial gain. Every thermal model here went straight from COCO weights to
HIT-UAV; an intermediate aerial stage has never been tested, and both datasets
are already on disk.

### Other repositories reviewed

Three further public repositories were examined and none provides a usable
benchmark:

- **YOLOv3 UAV detection** - stock COCO weights, no training, no metrics.
- **YOLOv11 on C2A** - reports 90% accuracy, 83% precision, 82% recall with no
  stated evaluation protocol, and cites two journals that do not appear to
  exist.
- **SSD-based SAR detection** - no published metrics.

They are recorded here only so the survey is complete.

---

## 3. Published Work on HIT-UAV Person Detection

Several 2024-2025 papers target exactly this problem. Their architectural
choices converge, and the convergence is informative.

### IPD-YOLO (2025) - the most relevant

Improved YOLO11 for infrared person detection from UAVs, trained on HIT-UAV
plus a proprietary DJI dataset. Reports gains of 4.7 to 7.4 mAP@50 points over
YOLOv5n, YOLOv8n, YOLOv10n and YOLO11n.

Components:

| Component | Purpose |
|-----------|---------|
| **Small-object detection layer** | Detection head specialised for tiny targets |
| **NWD-Inner CIoU** | Normalized Wasserstein Distance localisation loss |
| LQEHead | Localisation quality estimation |
| MASRCNet | Multi-scale attention backbone module |

### Why NWD matters here specifically

Normalized Wasserstein Distance (Wang et al., 2021) models a box as a 2D
Gaussian and measures similarity by Wasserstein distance rather than IoU.
For tiny objects IoU is pathologically unstable: a few pixels of displacement
collapses it, and the loss gradient becomes uninformative.

That is precisely the failure mechanism measured in this project.
`training_log.md` section 24 established that **56% of unmatched predictions
are near-miss boxes on real people**, and section 32 showed that for a
12 x 19 px target a **4-pixel** displacement is enough to fall below IoU 0.50.

NWD degrades smoothly with displacement instead of falling off a cliff. It is
the only technique found that is aimed directly at the dominant measured
failure mode rather than at target scale, which three experiments have already
ruled out.

### Related work using the same ingredients

- **YOFIR** - YOLO plus FasterNet, +4 points mAP@50 on HIT-UAV.
- **YOLO-TSL** - Triplet attention and Slim-neck for UAV infrared.
- **IFD-YOLO**, **ISTD-YOLO** - lightweight infrared small-target detectors.

The recurring pattern across all of them is **a higher-resolution detection
head plus a tiny-object-aware localisation loss**. Neither has been tried here.

---

## 4. What to Adopt

Ranked by evidence and by fit to the measured failure profile.

### A. Small-object detection head (P2)

`yolo26-p2.yaml` ships with Ultralytics, so this is a configuration change
rather than an implementation.

| Config | Params | GFLOPs @ 640 | Heads |
|--------|-------:|-------------:|-------|
| YOLO26n (current) | 2.57 M | 6.2 | P3, P4, P5 |
| **YOLO26n-p2** | **2.66 M** | **9.6** | P2, P3, P4, P5 |
| YOLO26s (rejected on compute) | 10.0 M | 23.1 | P3, P4, P5 |

At 640 input, P3 has stride 8 and P2 stride 4. A 12 x 19 px person spans about
1.5 x 2.4 cells at P3 but 3.0 x 4.8 at P2 - considerably more resolution
exactly where every target in this dataset sits.

Cost is 1.55x compute for +0.09 M parameters, against the 3.7x that ruled out
YOLO26s. Estimated 3.5 - 5.8 FPS on the target: clears the 100 m requirement,
marginal at 60 m.

### B. Two-stage aerial pretraining

VisDrone-pretrained weights, then fine-tune on HIT-UAV. Costs nothing at
inference. Published evidence from AlienSight within-modality; cross-modality
RGB to thermal is a weaker prior, since appearance differs even though aerial
human shape and scale transfer.

### C. NWD localisation loss

The best-motivated technique and the most invasive. Ultralytics does not
expose it, so it requires modifying the loss function rather than passing a
flag. Directly attacks the mechanism behind 56% of failures.

### D. Alerting

AlienSight emails the rescue team a GPS coordinate, a Maps link and an
annotated snapshot. The JSONL stream here already carries latitude, longitude
and a position error, so this is an interface rather than a capability.

---

## 5. Honest Position

On the one benchmark where comparison is valid, this project's detector
exceeds the published baselines. On system validation it is behind
AlienSight, which flew and measured; that gap is procurement, not research.

The analytical work here - failure mechanisms, derived frame-rate
requirements, sensor-resolution sensitivity, the operational matching
criterion, and seven documented negative results - has no counterpart in any
of the systems surveyed. Most report a single accuracy figure and no failure
analysis at all.

---

## Sources

- [HIT-UAV dataset paper (arXiv)](https://arxiv.org/pdf/2204.03245) ·
  [Scientific Data](https://www.nature.com/articles/s41597-023-02066-6)
- [IPD-YOLO](https://www.sciencedirect.com/science/article/abs/pii/S1051200425004919)
- [YOFIR](https://www.sciencedirect.com/science/article/abs/pii/S1350449524005115)
- [YOLO-TSL](https://www.sciencedirect.com/science/article/abs/pii/S1350449524003712)
- [IFD-YOLO](https://doi.org/10.3390/s25247449)
- [ISTD-YOLO](https://arxiv.org/pdf/2504.14289)
- [AlienSight](https://github.com/7amzaGH/UAV-SAR-Human-Detection-and-Geolocation) ·
  [preprint](https://doi.org/10.31224/7303)
