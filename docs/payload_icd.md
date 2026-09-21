# Interface Control Document - Human Detection Payload

Defines every interface across the boundary of the human-detection payload.

This is the payload's share of the project-level ICD listed as a Phase-2
deliverable. It does not cover flight-control interfaces; it specifies only
what crosses into and out of the vision subsystem, so it can be dropped into
the larger document without overlap.

**Status:** interfaces specified, cameras and companion computer not yet
procured. Electrical entries are requirements to verify at integration, not
measurements.

---

## 1. Scope and Boundary

```text
                        PAYLOAD BOUNDARY
   +--------------------------------------------------------+
   |                                                        |
   |   IF-1 RGB camera ---->+                               |
   |                        |                               |
   |   IF-2 Thermal cam --->+---> Companion Computer        |
   |                        |     detection, tracking,      |
   |   IF-3 Power --------->+     geolocation               |
   |                        |            |                  |
   |                        |            +---> IF-5 Storage |
   +------------------------|------------|-------------------+
                            |            |
              IF-4 MAVLink  v            v  IF-6 Detection stream
              (from flight controller)      (to ground station)
```

The payload is **not** in the flight-control loop. It consumes aircraft state
and emits detections. It issues no actuator or navigation commands, and its
failure must not affect flight control.

---

## 2. Interface Register

| ID   | Interface          | Direction | Medium        | Criticality |
| ---- | ------------------ | --------- | ------------- | ----------- |
| IF-1 | RGB camera         | In        | USB 3.0 / CSI | Mission     |
| IF-2 | Thermal camera     | In        | USB / CSI / SPI | Mission   |
| IF-3 | Power              | In        | Regulated 5 V | Mission     |
| IF-4 | Aircraft state     | In        | UART, MAVLink | Mission     |
| IF-5 | Onboard storage    | Internal  | NVMe / microSD| Low         |
| IF-6 | Detection stream   | Out       | Telemetry link| Mission     |

No interface is flight-critical. Loss of any one degrades or disables the
search function without affecting controllability.

---

## 3. IF-1 / IF-2 - Camera Inputs

| Parameter        | RGB (IF-1)          | Thermal (IF-2)              |
| ---------------- | ------------------- | --------------------------- |
| Resolution       | 1920 x 1080 or lower| **384 x 288 minimum**       |
| Frame rate       | >= 10 fps           | >= 10 fps                   |
| Interface        | USB 3.0 or MIPI CSI | USB, MIPI CSI or SPI        |
| Pixel format     | MJPEG or raw        | Radiometric or 8-bit grey   |
| Field of view    | To be fixed         | To be fixed, ~50 deg assumed|
| Mounting         | Downward            | Downward, co-aligned        |

### Thermal resolution is a hard requirement, not a preference

Measured in `sensor_resolution_study.py`, with MT-005 on the held-out split:

| Sensor    | Recall | vs native | Extra people missed |
| --------- | ------ | --------- | ------------------: |
| 640 x 512 | 0.9288 | 100.0%    |                  +0 |
| 384 x 288 | 0.8905 | 95.9%     |                +100 |
| 256 x 192 | 0.8407 | 90.5%     |                +230 |
| 160 x 120 | 0.6254 | 67.3%     |                **+792** |

A 160 x 120 module loses one person in three and raises images where a person
is present but nothing is reported from 8 to 72. **384 x 288 is the floor.**

If a sensor below 640 x 512 is selected, retraining the detector at that
resolution becomes mandatory rather than optional, and the recall figures in
this project must be restated.

### Co-alignment

RGB and thermal are treated as independent detectors, not a fused pair. No
registration between them is required by the current design. If fusion is
added later it will need a calibrated transform, which is not specified here.

---

## 4. IF-3 - Power

| Parameter         | Requirement                          |
| ----------------- | ------------------------------------ |
| Supply voltage    | 5 V DC regulated, +/- 5%             |
| Current, steady   | To be measured; budget 3 A           |
| Current, peak     | Budget 5 A                           |
| Isolation         | Separate regulator from flight control |
| Protection        | Over-current, reverse polarity        |
| Inrush            | Soft-start required                  |

### Isolation is a requirement, not a preference

Hadi et al. (reference [5]) lost control of their aircraft during integrated
flight testing and attributed it to wiring, electromagnetic interference and
inadequate electrical isolation. Their recommendation - improved wiring,
component placement, shielding and opto-isolation - applies directly.

The payload draws a large, rapidly varying current under inference load. It
must not share a regulator with the flight controller.

### Thermal

A CPU-only companion computer running continuous inference sits at or near
100% on all cores. Active cooling is required. Throttling is a mission risk,
not just a performance one: it reduces frame rate, which reduces the ground
covered per pass.

---

## 5. IF-4 - Aircraft State In

The payload needs aircraft state to convert pixel detections into ground
coordinates. Consumed over MAVLink from the flight controller.

| Field          | Source message      | Rate     | Used for            |
| -------------- | ------------------- | -------- | ------------------- |
| Latitude       | GLOBAL_POSITION_INT | >= 5 Hz  | Geolocation origin  |
| Longitude      | GLOBAL_POSITION_INT | >= 5 Hz  | Geolocation origin  |
| Altitude AGL   | GLOBAL_POSITION_INT / DISTANCE_SENSOR | >= 5 Hz | Ground-plane range |
| Roll           | ATTITUDE            | >= 10 Hz | Ray direction       |
| Pitch          | ATTITUDE            | >= 10 Hz | Ray direction       |
| Yaw            | ATTITUDE            | >= 10 Hz | Ray direction       |
| Timestamp      | Any                 | -        | Frame association   |

### Why the rates

Attitude dominates geolocation error. At 100 m, one degree of attitude error
moves the ground fix by about 1.75 m. State must be sampled close enough in
time to the frame that aircraft rotation between them is small; at 10 Hz and
typical rates that contribution stays below the GPS term.

**Altitude above ground, not above sea level.** The geolocation model
intersects a horizontal ground plane. Feeding it MSL altitude over terrain of
unknown elevation produces a proportional range error.

This is a read-only interface. The payload never writes to the flight
controller.

---

## 6. IF-5 - Onboard Storage

| Parameter   | Requirement                                     |
| ----------- | ----------------------------------------------- |
| Medium      | NVMe preferred, microSD acceptable              |
| Contents    | Detection stream, annotated frames, run summary |
| Retention   | Full mission                                    |
| Behaviour   | Storage failure must not stop detection or telemetry |

Recording is a post-mission evidence function. It is explicitly lower priority
than IF-6: if storage fails, the payload continues detecting and transmitting.

---

## 7. IF-6 - Detection Stream Out

The mission-critical output. One JSON record per processed frame, newline
delimited, produced by `scripts/realtime_detect.py` and enriched by
`scripts/geolocate.py`.

```json
{
  "frame": 42,
  "timestamp_s": 2.8,
  "persons": 1,
  "uav": {
    "lat": 12.9716,
    "lon": 77.5946,
    "altitude_m": 100.0
  },
  "detections": [
    {
      "track_id": 1,
      "box": [79.4, 203.1, 91.2, 221.6],
      "confidence": 0.6412,
      "state": "moving",
      "displacement_px": 22.7,
      "ground": {
        "lat": 12.9718241,
        "lon": 77.5949832,
        "east_m": 41.6,
        "north_m": 24.9,
        "slant_range_m": 111.4,
        "position_error_m": 3.9
      }
    }
  ]
}
```

### Field definitions

| Field              | Type    | Meaning                                    |
| ------------------ | ------- | ------------------------------------------ |
| `frame`            | int     | Monotonic frame counter                    |
| `timestamp_s`      | float   | Seconds since stream start                 |
| `persons`          | int     | Detections in this frame                   |
| `uav`              | object  | Aircraft state used for this frame's fixes |
| `track_id`         | int     | Persistent identity; -1 if unassociated    |
| `box`              | float[4]| Pixel box, x1 y1 x2 y2                     |
| `confidence`       | float   | Detector confidence                        |
| `state`            | enum    | `moving`, `stationary`, `edge`, `unknown`  |
| `ground.lat/lon`   | float   | Estimated ground position                  |
| `position_error_m` | float   | 1-sigma position uncertainty               |

### `position_error_m` is part of the contract

A coordinate handed to a rescue team without an uncertainty invites more
confidence than the measurement supports. The error estimate combines attitude,
altitude and GPS terms and grows with slant range. The ground station must
display it.

### `state` semantics

`edge` and `unknown` mean the movement state is **withheld**, not that the
person is stationary. A person is reported regardless of movement state. See
`realtime_pipeline.md` section 2.

### Bandwidth

A frame with no detections is roughly 90 bytes; each detection adds roughly
220. At 5 FPS with 3 detections per frame this is about 3.8 kbit/s, which is
negligible for any telemetry radio under consideration. Annotated video is
**not** transmitted; it is recorded to IF-5.

---

## 8. Verification

| ID   | Method                                             |
| ---- | -------------------------------------------------- |
| IF-1 | Capture at rated resolution and frame rate for 10 min |
| IF-2 | As IF-1, plus confirm sensor resolution against section 3 |
| IF-3 | Measure steady and peak current under sustained inference; thermal soak to confirm no throttling |
| IF-4 | Log MAVLink for 10 min; confirm field presence, rates, and that altitude is AGL |
| IF-5 | Fill storage to capacity; confirm detection and IF-6 continue |
| IF-6 | Schema-validate every record of a full run; confirm ground fixes against surveyed points |

### Geolocation acceptance

The only interface needing quantitative acceptance beyond presence and rate.
Place markers at surveyed positions, overfly at 60, 100 and 130 m, and compare
reported `ground.lat/lon` against survey.

**Acceptance:** median error within the reported `position_error_m`, with no
systematic bias in along-track or across-track direction. A systematic bias
indicates a mounting-angle or frame-convention error rather than noise.

---

## 9. Open Items

| Item                        | Blocks                        |
| --------------------------- | ----------------------------- |
| Thermal camera selection    | IF-2, and the recall figures  |
| RGB camera selection        | IF-1                          |
| Companion computer selection| IF-3 current budget           |
| Camera field of view        | Geolocation, coverage rate    |
| Camera mount angle          | Geolocation                   |
| Telemetry radio             | IF-6 link budget              |
| Altitude source (AGL)       | Geolocation accuracy          |

Every open item is a procurement or configuration decision. None requires
further detector development.
