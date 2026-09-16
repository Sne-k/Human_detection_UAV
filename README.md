# Human Detection System for Fixed-Wing eVTOL UAV

## Project

**Development of a Fixed Wing eVTOL UAV for Emergency Logistics**

This repository contains the development work for the human-detection subsystem of the UAV project.

The detection subsystem is intended to support identification of people from an aerial platform during emergency and logistics operations. The development is being carried out in stages, beginning with RGB aerial imagery and later extending toward thermal/infrared sensing, movement detection, and embedded deployment.

---

## Current Development Stage

The current work focuses on developing and evaluating a deep-learning-based human detection model using aerial imagery.

### Current pipeline

```text
Aerial Image / Video
        ↓
Image Preprocessing
        ↓
YOLO Human Detector
        ↓
Person Bounding Boxes
        ↓
Confidence Filtering
        ↓
Future: Tracking / Movement Detection
        ↓
Future: UAV Payload / Ground Station Integration
```
