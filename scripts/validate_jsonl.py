"""
Check a detection stream against its contract - interface IF-6.

docs/payload_icd.md specifies the payload's output record by record: field
names, types, ranges and meanings. Nothing checked that the payload actually
produces it. This does, and exits non-zero on any violation, so it can be the
acceptance check at integration and a test in the suite.

Errors break the contract. Warnings are fields the contract does not define;
they are reported rather than rejected, so the ICD can be extended
deliberately rather than by accident.

Usage:

    python scripts/validate_jsonl.py detections.jsonl
    python scripts/validate_jsonl.py detections.jsonl --quiet
"""

import argparse
import json
import math
import sys

STATES = {"moving", "stationary", "edge", "unknown"}

FRAME_FIELDS = {"frame", "timestamp_s", "persons", "detections", "uav",
                "ego_motion"}
EGO_STATES = {"init", "ok", "telemetry", "predicted", "lost"}
DETECTION_FIELDS = {"track_id", "box", "confidence", "state",
                    "displacement_px", "ground"}
UAV_FIELDS = {"lat", "lon", "altitude_m", "state_age_s"}
GROUND_FIELDS = {"lat", "lon", "east_m", "north_m", "slant_range_m",
                 "position_error_m"}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def validate_record(record, previous=None):
    """
    (errors, warnings) for one record.

    `previous` is the preceding record, for the checks that span records:
    frame numbers increase and timestamps do not go backwards.
    """

    errors, warnings = [], []

    def err(message):
        errors.append(message)

    if not isinstance(record, dict):
        return ["record is not a JSON object"], []

    for field in ("frame", "timestamp_s", "persons", "detections"):
        if field not in record:
            err(f"missing required field '{field}'")

    for field in set(record) - FRAME_FIELDS:
        warnings.append(f"field '{field}' is not defined by the ICD")

    frame = record.get("frame")

    if "frame" in record and (not isinstance(frame, int) or isinstance(frame, bool) or frame < 0):
        err(f"frame must be a non-negative integer, got {frame!r}")

    timestamp = record.get("timestamp_s")

    if "timestamp_s" in record and (not _finite(timestamp) or timestamp < 0):
        err(f"timestamp_s must be a finite non-negative number, got {timestamp!r}")

    if previous is not None and isinstance(frame, int):
        if isinstance(previous.get("frame"), int) and frame <= previous["frame"]:
            err(f"frame {frame} does not increase after {previous['frame']}")

        if _finite(timestamp) and _finite(previous.get("timestamp_s")) \
                and timestamp < previous["timestamp_s"]:
            err(f"timestamp_s {timestamp} goes backwards from "
                f"{previous['timestamp_s']}")

    if "ego_motion" in record and record["ego_motion"] not in EGO_STATES:
        err(f"ego_motion {record['ego_motion']!r} is not one of {sorted(EGO_STATES)}")

    detections = record.get("detections")

    if "detections" in record and not isinstance(detections, list):
        err("detections must be a list")
        detections = []

    detections = detections or []
    persons = record.get("persons")

    if "persons" in record:
        if not isinstance(persons, int) or isinstance(persons, bool):
            err(f"persons must be an integer, got {persons!r}")
        elif persons != len(detections):
            err(f"persons is {persons} but there are {len(detections)} detections")

    # uav: an object, or null when telemetry is configured but unusable.
    if "uav" in record and record["uav"] is not None:
        uav = record["uav"]

        if not isinstance(uav, dict):
            err("uav must be an object or null")
        else:
            for field in ("lat", "lon", "altitude_m"):
                if not _finite(uav.get(field)):
                    err(f"uav.{field} missing or not finite")

            if _finite(uav.get("lat")) and not -90 <= uav["lat"] <= 90:
                err(f"uav.lat {uav['lat']} out of range")

            if _finite(uav.get("lon")) and not -180 <= uav["lon"] <= 180:
                err(f"uav.lon {uav['lon']} out of range")

            if "state_age_s" in uav and (not _finite(uav["state_age_s"])
                                         or uav["state_age_s"] < 0):
                err(f"uav.state_age_s must be finite and non-negative")

            for field in set(uav) - UAV_FIELDS:
                warnings.append(f"uav field '{field}' is not defined by the ICD")

    seen_tracks = set()

    for index, det in enumerate(detections):
        where = f"detections[{index}]"

        if not isinstance(det, dict):
            err(f"{where} is not an object")
            continue

        for field in ("track_id", "box", "confidence", "state"):
            if field not in det:
                err(f"{where} missing required field '{field}'")

        for field in set(det) - DETECTION_FIELDS:
            warnings.append(f"{where} field '{field}' is not defined by the ICD")

        track = det.get("track_id")

        if "track_id" in det:
            if not isinstance(track, int) or isinstance(track, bool) or (track < 1 and track != -1):
                err(f"{where}.track_id must be a positive integer or -1, got {track!r}")
            elif track != -1:
                if track in seen_tracks:
                    err(f"{where}.track_id {track} appears twice in one frame")
                seen_tracks.add(track)

        box = det.get("box")

        if "box" in det:
            if (not isinstance(box, list) or len(box) != 4
                    or not all(_finite(v) for v in box)):
                err(f"{where}.box must be four finite numbers")
            elif not (box[0] < box[2] and box[1] < box[3]):
                err(f"{where}.box {box} is not x1 < x2, y1 < y2")

        confidence = det.get("confidence")

        if "confidence" in det and (not _finite(confidence) or not 0 < confidence <= 1):
            err(f"{where}.confidence must be in (0, 1], got {confidence!r}")

        if "state" in det and det["state"] not in STATES:
            err(f"{where}.state {det['state']!r} is not one of {sorted(STATES)}")

        if "displacement_px" in det and (not _finite(det["displacement_px"])
                                         or det["displacement_px"] < 0):
            err(f"{where}.displacement_px must be finite and non-negative")

        ground = det.get("ground")

        if ground is not None and "ground" in det:
            if not isinstance(ground, dict):
                err(f"{where}.ground must be an object or null")
                continue

            for field in GROUND_FIELDS:
                if not _finite(ground.get(field)):
                    err(f"{where}.ground.{field} missing or not finite")

            if _finite(ground.get("lat")) and not -90 <= ground["lat"] <= 90:
                err(f"{where}.ground.lat out of range")

            if _finite(ground.get("lon")) and not -180 <= ground["lon"] <= 180:
                err(f"{where}.ground.lon out of range")

            # The ICD makes the uncertainty part of the contract: a
            # coordinate without one is a contract violation, not a detail.
            if _finite(ground.get("position_error_m")) and ground["position_error_m"] < 0:
                err(f"{where}.ground.position_error_m is negative")

            if record.get("uav") is None:
                err(f"{where} has a ground fix but the frame has no aircraft state")

    return errors, warnings


def validate_stream(lines):
    """(errors, warnings, records) over an iterable of JSONL lines."""

    errors, warnings = [], []
    previous = None
    count = 0

    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {number}: not valid JSON ({exc.msg})")
            continue

        count += 1
        e, w = validate_record(record, previous)
        errors.extend(f"line {number}: {m}" for m in e)
        warnings.extend(f"line {number}: {m}" for m in w)

        if isinstance(record, dict):
            previous = record

    return errors, warnings, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    with open(args.path, encoding="utf-8") as f:
        errors, warnings, count = validate_stream(f)

    if not args.quiet:
        for message in errors[:50]:
            print(f"ERROR   {message}")

        unique = sorted({w.split(": ", 1)[1] for w in warnings})
        for message in unique[:20]:
            print(f"warning {message}")

    print(f"{count} records, {len(errors)} errors, {len(warnings)} warnings")
    sys.exit(1 if errors or count == 0 else 0)


if __name__ == "__main__":
    main()
