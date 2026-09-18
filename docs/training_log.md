@'

# Training Log

This document records the human-detection model training experiments for the eVTOL UAV human-detection subsystem.

## 1. Development Environment

| Parameter    | Value                              |
| ------------ | ---------------------------------- |
| OS           | Windows                            |
| Python       | 3.12.5                             |
| Ultralytics  | 8.4.152                            |
| PyTorch      | 2.14.0+cu132                       |
| CUDA         | 13.2                               |
| GPU          | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM         | 6140 MiB                           |
| Model family | YOLO26n                            |

The development environment uses the NVIDIA GPU for accelerated training and inference.

---

# 2. RGB Detection Experiments - VisDrone

## Dataset

**Dataset:** VisDrone DET

**Task:** Single-class person detection

The original VisDrone pedestrian and people categories were combined into one class:

```text
0: person
Dataset statistics
Split	Images	Images with persons	Person annotations
Train	6471	5684	106396
Validation	548	531	13969

The converted dataset was visually inspected using bounding-box visualization before training.

MT-001 - RGB Pilot Training

A short 3-epoch pilot run was performed to verify the complete training pipeline.

Configuration
Parameter	Value
Model	YOLO26n
Dataset	VisDrone Person
Input size	640 x 640
Epochs	3
Batch size	8
GPU	NVIDIA RTX 4050 Laptop GPU
AMP	Enabled
Results
Metric	Result
Precision	0.498
Recall	0.358
mAP@50	0.366
mAP@50-95	0.132

Status: Completed - pilot only.

The run confirmed that the dataset, model, GPU and training pipeline were functioning correctly. It was not treated as a final benchmark because of the very short training duration.

MT-002 - RGB Baseline Training

A 50-epoch baseline was trained at 640 x 640 input resolution.

Configuration
Parameter	Value
Model	YOLO26n
Dataset	VisDrone Person
Input size	640 x 640
Epochs	50
Batch size	8
GPU	NVIDIA RTX 4050 Laptop GPU
AMP	Enabled
Results

Approximate final validation performance:

Metric	Result
Precision	~0.63
Recall	~0.44
mAP@50	~0.50
mAP@50-95	~0.198

Status: Completed.

Error analysis

The confusion matrix indicated a significant number of missed person detections.

The principal limitation identified was low recall, particularly for small aerial persons.

This motivated controlled experiments using increased input resolution.

3. MT-003 - 960-Pixel RGB Human Detection

YOLO26n was trained on the VisDrone person-only dataset for 50 epochs at 960 x 960 input resolution with a batch size of 4.

Configuration
Parameter	Value
Model	YOLO26n
Dataset	VisDrone Person
Input size	960 x 960
Epochs	50
Batch size	4
GPU	NVIDIA RTX 4050 Laptop GPU
AMP	Enabled
Validation Results
Metric	Result
Precision	0.716
Recall	0.548
mAP@50	0.618
mAP@50-95	0.274

Validation set: 548 images
Person instances: 13,969
Training time: 13.247 hours
Inference speed: 5.7 ms/image during validation

Status: Completed.

Observation

Increasing the training resolution from 640 x 640 to 960 x 960 improved validation precision, recall, mAP@50 and mAP@50-95.

However, recall remained the main limitation, particularly for small aerial persons.

4. MT-004 - 1280-Pixel RGB Human Detection

A controlled higher-resolution experiment was performed using the same VisDrone split and YOLO26n model.

Configuration
Parameter	Value
Model	YOLO26n
Dataset	VisDrone Person
Input size	1280 x 1280
Epochs	50
Batch size	2
GPU	NVIDIA RTX 4050 Laptop GPU
AMP	Enabled
Validation Results
Metric	Result
Precision	0.738
Recall	0.563
mAP@50	0.653
mAP@50-95	0.297

Validation set: 548 images
Person instances: 13,969

Status: Completed.

Comparison with MT-003
Metric	MT-003 (960)	MT-004 (1280)	Change
Precision	0.716	0.738	+0.022
Recall	0.548	0.563	+0.015
mAP@50	0.618	0.653	+0.035
mAP@50-95	0.274	0.297	+0.023
Observation

Increasing input resolution from 960 x 960 to 1280 x 1280 improved all four reported validation metrics.

The recall improvement was relatively modest, indicating that increased resolution alone does not completely solve the small-object detection problem.

5. RGB Confidence Threshold Study

The MT-004 model was evaluated on the same VisDrone validation set using different confidence thresholds.

Confidence	Precision	Recall	mAP@50	mAP@50-95
0.15	0.744	0.569	0.596	0.278
0.25	0.714	0.585	0.540	0.258
0.35	0.838	0.505	0.477	0.234
0.50	0.935	0.342	0.334	0.176
Observation

Increasing the confidence threshold reduced the number of accepted detections and generally increased precision while reducing recall.

The confidence threshold is therefore an operational/deployment parameter that must be selected according to the required balance between missed persons and false detections.

The reported mAP values in this threshold study should not be interpreted as conventional confidence-threshold-independent AP comparisons.

6. RGB Small-Object Error Analysis

Post-training analysis was performed on the VisDrone validation predictions using an IoU threshold of 0.50.

An area-based analysis was previously performed using:

Small: area < 32 x 32 pixels
Medium: 32 x 32 <= area < 96 x 96 pixels
Large: area >= 96 x 96 pixels
MT-003 analysis
Size	Ground Truth	Matched	Matched Ratio
Small	12,377	6,723	0.543
Medium	1,564	1,249	0.799
Large	28	26	0.929

Total ground-truth persons: 13,969
Matched detections: 7,998
Overall matched ratio: 0.573

The analysis showed substantially lower detection performance for smaller person bounding boxes.

7. MT-004 Missed-Person Analysis

The final MT-004 validation predictions were examined for images without prediction label files.

There were 21 validation images without prediction label files. Of these:

14 contained no ground-truth persons.
7 contained a total of 14 ground-truth persons.
Missed-person bounding-box characteristics
Measurement	Value
Average width	10.0 px
Average height	22.9 px
Minimum width	5 px
Minimum height	7 px
Maximum width	26 px
Maximum height	54 px

11 of the 14 missed persons had bounding-box widths of 10 pixels or less.

The missed detections were predominantly associated with very small persons in aerial imagery.

8. RGB Tiled-Inference Experiment

A custom tiled-inference pipeline was tested with MT-004 to investigate whether processing smaller image regions could improve small-person detection.

The NMS implementation was corrected before the final comparison because OpenCV NMS requires boxes in [x, y, width, height] format.

Standard vs Tiled

Both methods were evaluated on the 548-image VisDrone validation set using IoU >= 0.50.

Metric	Standard	Tiled
Ground-truth persons	13,969	13,969
Predicted boxes	11,509	11,024
Matched persons	8,199	8,112
Overall matched ratio	0.5869	0.5807
Zero-detection images	21	21
Size-wise results
Size	Standard Recall	Tiled Recall
Small	0.5809	0.5769
Medium	0.7724	0.6989
Large	0.8000	0.6000

The tested tiled configuration produced 87 fewer matched persons and 485 fewer predicted boxes.

One of the 14 previously identified missed persons was recovered during tiled inference, but the overall validation comparison did not show an improvement.

Conclusion: The tested tiled configuration was not selected as the next optimization direction.

9. HIT-UAV Thermal Dataset Integration

A separate thermal detection baseline was introduced using the HIT-UAV Infrared Thermal Dataset.

The dataset contains UAV thermal imagery covering day and night conditions and multiple object categories.

For this project, only the Person category was retained.

Original HIT-UAV category mapping
ID	Category
0	Person
1	Car
2	Bicycle
3	OtherVehicle
4	DontCare

The Person category is therefore mapped to:

0: person
Converted dataset statistics
Split	Images	Images with persons	Images without persons	Person annotations
Train	2,029	1,165	864	8,533
Validation	290	171	119	1,168
Test	579	355	224	2,611
Total	2,898	1,691	1,207	12,312

The COCO-style bounding boxes were converted to YOLO format.

The converted annotations were visually inspected using thermal-image bounding-box visualization. The sampled annotations showed correct placement over visible human thermal signatures.

10. MT-005 - Thermal Baseline

YOLO26n was trained on the converted HIT-UAV Person dataset.

Configuration
Parameter	Value
Model	YOLO26n
Dataset	HIT-UAV Person
Input size	640 x 640
Epochs	50
Batch size	8
GPU	NVIDIA RTX 4050 Laptop GPU
AMP	Enabled
Validation Results
Metric	Result
Precision	0.890
Recall	0.875
mAP@50	0.919
mAP@50-95	0.500

Validation set: 290 images
Person instances: 1,168
Training time: 1.599 hours

Status: Completed.

11. MT-005 Held-Out Test Evaluation

The completed MT-005 model was evaluated on the separate HIT-UAV test split.

Test Results
Metric	Result
Precision	0.898
Recall	0.892
mAP@50	0.933
mAP@50-95	0.510

Test set: 579 images
Person instances: 2,611

Inference speed: approximately 4.2 ms/image during evaluation

Status: Completed.

This held-out test result is the primary test-set benchmark for the MT-005 thermal baseline.

The thermal and RGB models should not be directly interpreted as one modality being inherently superior because they were trained and evaluated on different datasets.

12. MT-005 Thermal Test Error Analysis

The MT-005 model was evaluated on the HIT-UAV test images at confidence threshold 0.25 for detailed error analysis.

A one-to-one greedy matching procedure with IoU >= 0.50 was used for this custom analysis.

Confidence 0.25
Measurement	Result
Test images	579
Images with GT persons	355
Images with predictions	369
Zero-prediction images	210
Ground-truth persons	2,611
Predicted boxes	3,063
Matched persons	2,425
Missed GT persons	186
Unmatched prediction boxes	638
Overall matched ratio	0.9288

Of the 210 zero-prediction images:

202 contained no persons.
8 contained persons.

The 8 complete-failure images contained a total of 9 ground-truth persons.

Size analysis

Using the project-wide small-object definition of width <32 px OR height <32 px:

Size	GT	Matched	Recall
Small	2,606	2,422	0.9294
Medium	5	3	0.6000
Large	0	0	N/A

99.8% of the HIT-UAV test person instances were classified as small under this definition.

13. MT-005 Confidence 0.10 Diagnostic

A second prediction run was performed at confidence threshold 0.10 to investigate whether low-confidence detections were responsible for some of the missed persons.

Confidence 0.10
Measurement	Result
Test images	579
Images with predictions	384
Zero-prediction images	195
Predicted boxes	3,768
Matched persons	2,468
Missed GT persons	143
Unmatched prediction boxes	1,300
Overall matched ratio	0.9452

Compared with confidence 0.25:

705 additional predicted boxes were generated.
43 additional persons were matched.
43 fewer persons were missed.
662 additional prediction boxes remained unmatched.
Zero-prediction images decreased by 15.
Zero-prediction images containing persons decreased by 4.

The 43 recovered persons had detection confidences between approximately 0.1034 and 0.2490, with an average of approximately 0.1747.

This indicates that some valid thermal-person detections are produced at relatively low confidence.

14. MT-005 Remaining-Miss Analysis

The remaining misses after the confidence 0.10 diagnostic were examined further.

The traceable per-person diagnostic identified 142 remaining missed persons. The small difference from the aggregate 143-miss result is due to the matching procedure used by the diagnostic.

Remaining-miss categories
Category	Count	Percentage
No overlapping prediction at confidence 0.10	35	~24.6%
Prediction exists but IoU < 0.50	107	~75.4%
Total traceable misses	142	100%
Remaining-miss size
Measurement	Value
Average width	13.30 px
Average height	18.25 px
Minimum width	4 px
Minimum height	8 px
Maximum width	87 px
Maximum height	74 px

The majority of the remaining missed persons had some overlapping prediction but insufficient IoU for a match.

This suggests that the dominant remaining limitation is localization of very small thermal targets, with a secondary issue of complete recognition failures.

Lowering the confidence threshold alone therefore does not fully solve the problem: it recovered 43 of the 186 misses while also adding 662 unmatched prediction boxes.

15. MT-006 - Thermal Small-Object Resolution Experiment

MT-006 is the current controlled experiment.

The purpose is to determine whether increasing input resolution from 640 x 640 to 960 x 960 improves thermal person detection while keeping the other major training conditions unchanged.

Configuration
Parameter	Value
Model	YOLO26n
Dataset	HIT-UAV Person
Input size	960 x 960
Epochs	50
Batch size	4
GPU	NVIDIA RTX 4050 Laptop GPU
AMP	Enabled
Dataset split	Same as MT-005
Training objective	Small-object resolution experiment

Status: RUNNING

Final validation and test metrics will be added after MT-006 completes.

No conclusions should be drawn from MT-006 until the run has completed and the results have been evaluated.
```
