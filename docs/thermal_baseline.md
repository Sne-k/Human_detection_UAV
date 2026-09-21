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

Control: only **4.46%** of the 2,397 successfully detected persons involve a
multi-person prediction box.

```bash
python scripts/verify_crowding_hypothesis.py --conf 0.10
```

### Single-model merged-box reference

The 43 above is an **intersection** figure: persons missed by MT-005 *and*
MT-007 together, at confidence 0.10. It is not comparable to a single model's
count, and using it as the MT-008 baseline would be an apples-to-oranges
comparison.

The self-contained per-model metric is: of the persons this model misses at
confidence 0.25, how many have a best-overlapping prediction that spans 2+
annotated people and is at least 1.6x the target area.

| Model  | Missed | merged | undersized | offset | stolen | no_overlap |
| ------ | -----: | -----: | ---------: | -----: | -----: | ---------: |
| MT-005 |    186 | **57** |         33 |     12 |     19 |         65 |
| MT-006 |    207 |     56 |         34 |      8 |     15 |         94 |
| MT-007 |    229 |     64 |         52 |      7 |     16 |         90 |

**Reference merged-box count for MT-005: 57.** This is the primary target
metric for MT-008.

Note that MT-006 (56) and MT-007 (64) sit essentially level with MT-005 (57).
Neither higher resolution nor crop augmentation changed merged-box behaviour at
all, which is further evidence that the mechanism is untouched by scale.

```bash
python scripts/evaluate_experiment.py --labels MT-005-test-analysis
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
| **Merged-box failures**   | **57**          | -              |
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
| C         | MT-008 single model (mosaic = 0)   | Completed - rejected         |
| E         | MT-009 single model (hard negatives) | Completed - rejected       |
| D         | MT-005 + MT-006, WBF fusion        | **Best measured**            |

| Candidate | Matched | Missed | Unmatched | Precision | Blind | Compute |
| --------- | ------: | -----: | --------: | --------- | ----: | ------- |
| A MT-005  |   2,425 |    186 |       638 | 0.7917    |     8 | 1x      |
| B 3-model |   2,476 |    135 |       722 | 0.7742    |     4 | 3x      |
| C MT-008  |   2,424 |    187 |       717 | 0.7717    |    11 | 1x      |
| E MT-009  |   2,428 |    183 |       637 | 0.7922    |    12 | 1x      |
| **D MT-005+MT-006** | **2,456** | **155** | **617** | **0.7992** | **5** | **2x** |

Candidate C is rejected: MT-008 matched one fewer person than the baseline and
raised merged-box failures from 57 to 70, refuting the mosaic hypothesis.

Candidate E is rejected: MT-009 reduced unmatched boxes by one out of 638,
which is no improvement, and raised blind images from 8 to 12 - the error mode
that matters most for a search payload.

Candidate B improves recall but at the cost of precision (0.7742 against
0.7917) and 84 extra unmatched boxes - a trade, not a free improvement.

**Candidate D is better than the baseline on every axis at once**: 31 more
people found, 31 fewer missed, 21 *fewer* unmatched boxes, higher precision and
blind images down from 8 to 5. It is the only configuration measured so far
that improves on MT-005 without a compensating regression. Its cost is 2x
inference, which rules it out where movement classification is required at
60 m but not for detect-and-report.

---

## 8. All-Conditions Performance - Day vs Night

A search payload must detect people regardless of visible light. That is the
core reason for a thermal branch, and it is worth measuring rather than
asserting.

HIT-UAV encodes lighting in the filename (field 0: 0 = day, 1 = night), so
the held-out test split can be partitioned directly. MT-005, confidence 0.25,
IoU >= 0.50:

| Condition | Images | Persons | Matched | Recall | Precision | Blind images |
| --------- | -----: | ------: | ------: | ------ | --------- | -----------: |
| **Night** |    396 |   2,059 |   1,940 | **0.9422** | **0.8234** | 3 (0.8%) |
| Day       |    183 |     552 |     485 | 0.8786 | 0.6860    | 5 (2.7%)     |
| Combined  |    579 |   2,611 |   2,425 | 0.9288 | 0.7917    | 8            |

### The detector performs better in darkness

Night recall exceeds day recall by **6.4 points**, and night precision by
**13.7 points**. Images where a person is present and nothing at all is
reported are 3.6 times rarer at night.

This is a physical result, not a quirk of the data. At night the background is
cool and a human body at roughly 37 C is a high-contrast signature. In
daylight, sun-heated roads, rooftops, vehicles and bare ground approach or
exceed body temperature, and the thermal contrast that the detector depends on
collapses.

It also confirms the section 21 finding from a different direction: recognition
failures - persons with no overlapping prediction at all - were 2.9x
over-represented in daylight.

### Consequence

**The requirement to detect people irrespective of visible light is satisfied
by the thermal branch.** Thermal sensing does not use visible light at all, so
performance in complete darkness is not merely acceptable, it is the
detector's best condition.

The weak case is bright daylight, which is the opposite of the usual
intuition and should be stated explicitly in any write-up. If daylight search
performance matters for the mission, that is where further effort belongs -
not darkness.

| Condition            | Coverage                                    |
| -------------------- | ------------------------------------------- |
| Night, total darkness| Thermal, recall 0.942                       |
| Smoke, dust, haze    | Thermal - inferred from LWIR physics, untested |
| Bright daylight      | Thermal, recall 0.879 - weakest case        |
| Visible light present| RGB could supplement, but MT-004 at 1280 px does not fit the compute budget |

Two caveats. This is HIT-UAV's day/night split; no dataset in this project
contains smoke or fog, so obscurant penetration is inferred from the physics of
long-wave infrared rather than measured. And day performance, while good, is
the genuine weak point at 0.879.
