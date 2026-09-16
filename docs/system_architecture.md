## Purpose

The human-detection subsystem is being developed as an AI-based perception payload for the fixed-wing eVTOL UAV designed for emergency logistics.

The subsystem is intended to detect people from aerial imagery and can later support movement detection and situational awareness.

---

## Current Development Architecture

```text
             RGB Aerial Image / Video
                       |
                       v
              Image Preprocessing
                       |
                       v
                 YOLO Detector
                       |
                       v
             Person Detection
                       |
              +--------+--------+
              |                 |
              v                 v
       Bounding Boxes      Confidence
                            Filtering
              |                 |
              +--------+--------+
                       |
                       v
              Detection Output
                       |
                       v
          Future Tracking / Movement
                       |
                       v
             Ground Station /
             Payload Interface
```
