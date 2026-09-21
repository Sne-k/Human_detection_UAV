# Thermal Baseline - MT-005 (Frozen)

**MT-005 is the frozen single-model thermal baseline.** Every thermal
experiment from this point compares against the figures on this page, and these
figures do not change.

The purpose of freezing it is to stop the reference drifting. Earlier
comparisons in this project used slightly different evaluation settings, which
made some experiments hard to compare after the fact. The numbers here were all
produced by scripts in this repository under stated conditions and can be
regenerated.

---

## 1. Model

| Property       | Value                            |
| -------------- | -------------------------------- |
| Experiment ID  | MT-005                           |
| Architecture   | YOLO26n                          |
| Dataset        | HIT-UAV Person (thermal)         |
| Input size     | 640 x 640                        |
| Epochs         | 50                               |
| Batch size     | 8                                |
| Best epoch     | 47                               |
| Training time  | 1.599 hours                      |
| Weights        | `runs/detect/results/training/MT-005/weights/best.pt` |

Training configuration is recorded in full in
[`training_log.md`](training_log.md) section 9, and the run's own `args.yaml`
is the authoritative copy.

---

## 2. Standard Metrics - Held-Out Test Split

HIT-UAV test split: **579 images, 2,611 person instances.** This split is used
only for final evaluation.

| Metric    | Result |
| --------- | ------ |
| Precision | 0.898  |
| Recall    | 0.892  |
| mAP@50    | 0.933  |
| mAP@50-95 | 0.510  |

Reproduced by the unified benchmark under uniform settings (confidence 0.001,
NMS IoU 0.70, batch 1) as 0.897 / 0.891 / 0.933 / 0.510. The small differences
come from the evaluation settings, not the model.

```bash
python scripts/benchmark_models.py --models MT-005
```

### Validation split (290 images, 1,168 instances)

| Metric    | Result |
| --------- | ------ |
| Precision | 0.890  |
| Recall    | 0.875  |
| mAP@50    | 0.919  |
| mAP@50-95 | 0.500  |

---

## 3. Custom Error Analysis - Held-Out Test Split

Operational confidence **0.25**, one-to-one greedy matching at **IoU >= 0.50**.
This answers the operational question - how many actual people were found at
the deployment threshold - rather than an AP question.

| Measurement                        | Result |
| ---------------------------------- | ------ |
| Ground-truth persons               | 2,611  |
| Predicted boxes                    | 3,063  |
| Matched persons                    | 2,425  |
| Missed persons                     | 186    |
| Unmatched prediction boxes         | 638    |
| Matched ratio (recall)             | 0.9288 |
| Matched / predicted (precision)    | 0.7917 |
| Images producing predictions       | 369    |
| Zero-prediction images             | 210    |
| Zero-prediction images with people | 8      |
| Persons in those images            | 9      |

```bash
python scripts/compare_mt005_vs_mt007_conf010.py --conf 0.25
```

### Size breakdown

| Size   |    GT | Matched | Recall |
| ------ | ----: | ------: | ------ |
| Small  | 2,606 |   2,422 | 0.9294 |
| Medium |     5 |       3 | 0.6000 |
| Large  |     0 |       0 | N/A    |

99.8% of test instances are small under the project convention
(width < 32 px OR height < 32 px).

---

## 4. Failure Profile

Established in [`training_log.md`](training_log.md) sections 21 and 24. These
are the mechanisms any future experiment must move.

Of the 87 persons missed by both MT-005 and MT-007 at confidence 0.10, among
the 68 that had an overlapping prediction:

| Mechanism             | Count | Share |
| --------------------- | ----: | ----- |
| Merged, 2+ people     |    43 | 63.2% |
| Undersized box        |    20 | 29.4% |
| Ordinary displacement |     3 | 4.4%  |
| Claimed by neighbour  |     2 | 2.9%  |

Plus 19 with no overlapping prediction at all, 57.9% of which are daylight
images.

**Reference merged-box count: 43.** This is the primary target metric for
MT-008.

Control: only **4.46%** of the 2,397 successfully detected persons involve a
multi-person prediction box.

```bash
python scripts/verify_crowding_hypothesis.py --conf 0.10
```

---

## 5. Latency

Measured on the development machine (RTX 4050 Laptop GPU, Ryzen 7 7435HS),
FP32, batch 1, at the native sensor shape 512 x 640.

| Measurement                      | Value    |
| -------------------------------- | -------- |
| Eager forward, median            | 39.07 ms |
| CUDA-graph forward, median       | 2.43 ms  |
| Speedup from removing launches   | 16.08x   |
| Validator throughput             | 25.4 ms  |

The historic figure of 4.2 ms/image quoted in earlier sections came from a
batched validator run and is **not** a single-frame latency. Use 2.43 ms
(graphed) or 39.07 ms (eager) depending on which pipeline is meant.

These are development-hardware numbers. They characterise this laptop's
dispatch rate as much as the model, and must be re-measured on any candidate
companion computer.

```bash
python scripts/benchmark_latency.py --models MT-005
```

---

## 6. Comparison Table for Future Experiments

Any new thermal experiment should report this table.

| Metric                    | MT-005 baseline | New experiment |
| ------------------------- | --------------- | -------------- |
| Precision                 | 0.898           | -              |
| Recall                    | 0.892           | -              |
| mAP@50                    | 0.933           | -              |
| mAP@50-95                 | 0.510           | -              |
| Matched persons @ 0.25    | 2,425           | -              |
| Missed persons @ 0.25     | 186             | -              |
| Unmatched boxes @ 0.25    | 638             | -              |
| Custom precision @ 0.25   | 0.7917          | -              |
| Zero-prediction images with people | 8      | -              |
| **Merged-box failures**   | **43**          | -              |
| Small-person recall       | 0.9294          | -              |
| Forward latency (graphed) | 2.43 ms         | -              |

A new model is only a replacement if it moves the failure mechanism it was
designed to move, without a large regression elsewhere. mAP alone is not
sufficient evidence - MT-006 improved mAP@50-95 while losing recall and
producing more complete-failure images.

---

## 7. Candidate Systems

Three candidates are under consideration for deployment. This table is filled
in as evidence arrives.

| Candidate | Description                        | Status                      |
| --------- | ---------------------------------- | --------------------------- |
| A         | MT-005 single model                | Frozen baseline             |
| B         | MT-005 + MT-006 + MT-007, WBF      | Measured, see section 22    |
| C         | MT-008 single model (mosaic = 0)   | Training                    |

Candidate B currently records 2,476 matched / 135 missed / 722 unmatched at
confidence 0.25 - better recall than A, but **lower custom precision** (0.7742
against 0.7917). It trades false positives for found people, which is the right
direction for search and rescue but is a trade, not a free improvement.
