# References

Every external source this project draws on, with its provenance stated.

**Why provenance is recorded separately.** Sources here carry very different
evidential weight. A peer-reviewed paper reporting a measured result on a
named dataset is not the same kind of claim as a README asserting an accuracy
figure with no stated protocol, and a documentation page is not a claim at
all. Mixing them in one list would flatten that distinction, so each entry is
tagged, and section F says plainly which sources were found unreliable.

| Tag | Meaning |
|-----|---------|
| **PDF** | Peer-reviewed paper or preprint, held locally in the project reference folder |
| **WEB-PAPER** | Peer-reviewed paper or preprint, read online, not held locally |
| **WEB-CODE** | Public source repository |
| **WEB-DOC** | Documentation, dataset page or tool reference |
| **DATA** | Dataset used for training or evaluation |
| **TOOL** | Software dependency, with the exact version used |

---

## A. Papers Held Locally

These are the PDFs in the project reference folder. Most belong to the wider
aircraft programme rather than to the detection payload; both are listed, with
a column saying which part of the work each one bears on.

### A.1 Human detection and search-and-rescue

| # | Reference | Bears on |
|---|-----------|----------|
| [1] | **Rizk, M., Slim, F., Charara, J.** (2021). *Toward AI-Assisted UAV for Human Detection in Search and Rescue Missions.* Lebanese University / Lebanese International University / IMT-Atlantique, CNRS Lab-STICC. 6 pp. **[PDF]** | Detection payload; reference system [3] in `literature_comparison.md` |
| [2] | **Lygouras, E., Santavas, N., Taitzoglou, A., Tarchanidis, K., Mitropoulos, A., Gasteratos, A.** (2019). *Unsupervised Human Detection with an Embedded Vision System on a Fully Autonomous UAV for Search and Rescue Operations.* **Sensors** 19(16), 3542. Democritus University of Thrace. **[PDF]** · [MDPI](https://www.mdpi.com/1424-8220/19/16/3542) | Detection payload; reference system [4]. Source of the negative-mining method and of the Raspberry Pi rejection discussed in `deployment_target.md` section 3e |

### A.2 Flight control, VTOL and payload delivery

| # | Reference | Bears on |
|---|-----------|----------|
| [3] | **Ducard, G. J. J., Allenspach, M.** (2021). *Review of designs and flight control techniques of hybrid and convertible VTOL UAVs.* **Aerospace Science and Technology** 118, 107035. Université Côte d'Azur / ETH Zürich. **[PDF]** | Airframe configuration and transition control |
| [4] | **Kai, J.-M.** *Full-Envelope Flight Control and a Transition Strategy for Compound eVTOL Aircrafts.* Safran Tech. arXiv:2312.09629v1. 39 pp. **[PDF]** | Transition-phase control law |
| [5] | **Kai, J.-M.** *Full-Envelope Flight Control for Compound Vertical Takeoff and Landing Aircraft.* Safran Tech. 43 pp. **[PDF]** (`paper 3.pdf`) | Extended version of [4] |
| [6] | **Comer, A., Chakraborty, I., Putra, S. H., Bhandari, R., Kunwar, B., Davis, B.** *Flight Testing a Trajectory Control System on a Subscale Transitioning VTOL Aircraft.* Oklahoma State University / Auburn University. [doi:10.2514/1.G009060](https://doi.org/10.2514/1.G009060). **[PDF]** (`paper 2.pdf`) | Middle-loop trajectory control, configuration-independent architecture |
| [7] | **Ion Guta, D. D., Gheorma, C.-T., Pascale, C., Berceanu, R., Neagu, M.** (2026). *Flight Control System for Ultra-Light Aircraft Conversion to VTOL Unmanned Aircraft Vehicle.* Preprints.org, [doi:10.20944/preprints202601.0351.v1](https://doi.org/10.20944/preprints202601.0351.v1). **Not peer-reviewed.** **[PDF]** (`paper 1.pdf`) | Flight control architecture |
| [8] | **Hakim, M. L., et al.** (2021). *Development of Unmanned Aerial Vehicle (UAV) Fixed-Wing for Monitoring, Mapping and Dropping applications on agricultural land.* **J. Phys.: Conf. Ser.** 2111, 012051. **[PDF]** | Fixed-wing survey and drop mission profile |
| [9] | **Hadi, G. S., Varianto, R., Riyanto T., B., Budiyono, A.** *Autonomous UAV System Development for Payload Dropping Mission.* **The Journal of Instrumentation, Automation and Systems.** Institut Teknologi Bandung / Konkuk University. **[PDF]** | Payload and power architecture |
| [10] | **Mardiyanto, R., Pujiantara, M., Suryoatmojo, H., Dikairono, R., Irfansyah, A. N.** (2019). *Development of Unmanned Aerial Vehicle (UAV) for Dropping Object Accurately Based on Global Positioning System.* Institut Teknologi Sepuluh Nopember. **[PDF]** | GPS-guided delivery accuracy |
| [11] | **Arangala, W. C. S.** (2021). *A Model for Precision Aerial Drops from UAVs.* University of Colombo School of Computing. 56 pp. **[PDF]** (`2016 MCS 007.pdf`) | Ballistics of the delivery phase |

---

## B. Papers Consulted Online

Not held locally. Read during the related-work survey to find techniques worth
adapting after seven training directions had failed.

### B.1 The benchmark that is directly comparable

| # | Reference | Why it matters |
|---|-----------|----------------|
| [12] | **Suo, J., Wang, T., Zhang, X., Chen, H., Zhou, W., Shi, W.** (2023). *HIT-UAV: A high-altitude infrared thermal dataset for Unmanned Aerial Vehicle-based object detection.* **Scientific Data** 10, 227. [Nature](https://www.nature.com/articles/s41597-023-02066-6) · [arXiv:2204.03245](https://arxiv.org/pdf/2204.03245). **[WEB-PAPER]** | **The only externally comparable numbers.** Same dataset, same splits. Publishes YOLOv4 89.88%, SSD-512 85.6%, Faster-RCNN 75.5%, YOLOv4-tiny 16.86% Person AP@0.50 against this project's MT-005 at 93.3% |

### B.2 The technique adopted

| # | Reference | Why it matters |
|---|-----------|----------------|
| [13] | **Wang, J., Xu, C., Yang, W., Yu, L.** (2021). *A Normalized Gaussian Wasserstein Distance for Tiny Object Detection.* [arXiv:2110.13389](https://arxiv.org/abs/2110.13389) · [code](https://github.com/jwwangchn/NWD). **[WEB-PAPER]** | **Implemented as `scripts/nwd_loss.py` for MT-013.** Models a box as a 2D Gaussian so similarity is defined for non-overlapping boxes and degrades smoothly. Directly attacks the dominant measured failure: 56% of unmatched predictions are near-miss boxes, and a 4 px offset on a 12 x 19 px person sits exactly at IoU 0.50 |

### B.3 Infrared UAV person detection, 2024-2025

Surveyed for architectural convergence. All four independently combine a
higher-resolution detection head with a tiny-object-aware localisation loss,
which is what pointed to [13].

| # | Reference | Note |
|---|-----------|------|
| [14] | **IPD-YOLO** (2025). Improved YOLO11 for infrared person detection from UAVs. **Digital Signal Processing.** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1051200425004919). **[WEB-PAPER]** | The most relevant. Trained on HIT-UAV plus a proprietary DJI set. Combines a small-object layer, NWD-Inner CIoU, LQEHead and MASRCNet |
| [15] | **YOFIR.** YOLO with FasterNet for UAV infrared. **Infrared Physics & Technology.** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1350449524005115). **[WEB-PAPER]** | +4 points mAP@50 reported on HIT-UAV |
| [16] | **YOLO-TSL.** Triplet attention and Slim-neck for UAV infrared. **Infrared Physics & Technology.** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1350449524003712). **[WEB-PAPER]** | Same two ingredients |
| [17] | **IFD-YOLO.** Lightweight infrared small-target detection. **Sensors.** [doi:10.3390/s25247449](https://doi.org/10.3390/s25247449). **[WEB-PAPER]** | Same two ingredients |
| [18] | **ISTD-YOLO.** Infrared small-target detection. [arXiv:2504.14289](https://arxiv.org/pdf/2504.14289). **[WEB-PAPER]** | Same two ingredients |

---

## C. Public Code Repositories

Found by searching for comparable UAV human-detection systems. Section F
records what each one could and could not support.

| # | Repository | Assessment |
|---|-----------|------------|
| [19] | **AlienSight** - [7amzaGH/UAV-SAR-Human-Detection-and-Geolocation](https://github.com/7amzaGH/UAV-SAR-Human-Detection-and-Geolocation) · preprint [doi:10.31224/7303](https://doi.org/10.31224/7303). **[WEB-CODE]** + **[WEB-PAPER]** | **The strongest comparable system.** YOLOv8n, two-stage fine-tuning, monocular geolocation, Qualcomm RB3 Gen 2. Validated geolocation against 60 surveyed points at 1.01 m mean error and measured 37.3 ms on real hardware - both things this project has not done. Source of the MT-011 two-stage transfer hypothesis |
| [20] | **[Yamunaasri/Human-Detection-in-Disaster-Scenarios-using-UAV-images](https://github.com/Yamunaasri/Human-Detection-in-Disaster-Scenarios-using-UAV-images)**. **[WEB-CODE]** | YOLOv11 on C2A. Reports 90% accuracy / 83% precision / 82% recall with **no stated evaluation protocol**, and cites journals that could not be found. Not usable as a benchmark - see section F |
| [21] | **[Prithivraj22/Uav_human_detection_yolo](https://github.com/Prithivraj22/Uav_human_detection_yolo)**. **[WEB-CODE]** | YOLOv3 with stock COCO weights. No training, no metrics |
| [22] | **[LeadingIndiaAI/Human-detection-for-Search-and-Rescue-operation-in-UAV-s-using-SSD](https://github.com/LeadingIndiaAI/Human-detection-for-Search-and-Rescue-operation-in-UAV-s-using-SSD)**. **[WEB-CODE]** | SSD-based. No published metrics |

---

## D. Datasets

| # | Dataset | Use here |
|---|---------|----------|
| [23] | **HIT-UAV** - high-altitude infrared thermal UAV dataset. 2,029 train / 290 val / 579 test, 640 x 512, 60-130 m altitude, day and night encoded in the filename. See [12]. **[DATA]** | **The thermal detector's entire training and evaluation basis.** MT-005 to MT-013 |
| [24] | **VisDrone-DET** - drone-captured RGB object detection benchmark. 6,471 train / 548 val after conversion to single-class person. [Project page](https://github.com/VisDrone/VisDrone-Dataset). **[DATA]** | RGB experiments MT-001 to MT-004; the aerial pretraining stage for MT-011 |

---

## E. Software

Exact versions, because several results in this project are runtime-specific -
the INT8 finding in particular is a property of ONNX Runtime's kernels rather
than of the model.

| Component | Version | Note |
|-----------|---------|------|
| Ultralytics | 8.4.152 | YOLO26n. `BboxLoss.forward` is patched by `nwd_loss.py` |
| PyTorch | 2.14.0+cu132 | CUDA 13.2 |
| ONNX | 1.23.0 | opset 17 |
| ONNX Runtime | 1.30.0 | CPU provider; the Raspberry Pi 5 proxy measurements |
| OpenCV | 5.0.0 | Preprocessing, optical flow for ego-motion |
| NumPy | 2.5.3 | |
| pypdf | 6.19.0 | Reference extraction only |

**Development hardware:** AMD Ryzen 7 7435HS, NVIDIA RTX 4050 Laptop GPU
(6,140 MiB).
**Deployment target:** Raspberry Pi 5 class, 4x Cortex-A76 @ 2.4 GHz, 4 GB.
Nothing procured; all target figures are extrapolated from x86.

---

## F. Sources Judged Unreliable

Recorded so the judgement is visible rather than silent.

**[20] YOLOv11 on C2A.** Reports 90% accuracy, 83% precision and 82% recall
with no stated evaluation protocol - no split, no IoU threshold, no confidence
threshold - and cites journals that could not be located. Accuracy is not a
meaningful detection metric in any case, since it depends on how background is
counted. **Excluded from all comparisons.**

**[21], [22].** No metrics published at all. Recorded only so the survey is
complete.

**A note on [1] and [2].** Both are sound papers and both are cited
throughout. What cannot be done is compare their numbers to this project's:
they operate at 14 m and ~5.75 m against this project's 60-130 m, report
different metrics under different protocols, and one hovers over a single
target rather than searching. An earlier draft of this project's
documentation did draw that comparison; it was wrong and is corrected in
`literature_comparison.md`.

---

## G. Where Each Source Is Used

| Document | Draws on |
|----------|----------|
| `literature_comparison.md` | [1], [2] |
| `related_work.md` | [12]-[22] |
| `deployment_target.md` | [2] (Pi rejection precedent), [12] (sensor resolution) |
| `training_log.md` s.33 | [14]-[18] (P2 head, vetoed on compute) |
| `training_log.md` s.34 | [13] (NWD, adopted as MT-013) |
| `thermal_baseline.md` | [12], [23] |
| `hardware_selection.md` | [1], [2] |
| `scripts/nwd_loss.py` | [13] |
| `scripts/mine_hard_negatives.py` | [2] (negative-mining method) |
| MT-011 | [19] (two-stage transfer) |

---

## Citation Format

Numbered references [1]-[24] are stable; new sources append rather than
renumber, so citations in other documents stay valid.
