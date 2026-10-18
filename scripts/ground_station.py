"""
Ground station: where the payload's output is read by a person - task D1.

Receives the detection stream (interface IF-6) - live over UDP from
`payload.py --udp`, or by following a JSONL file - and shows what a rescue
coordinator needs:

  - a map, in metres east and north of where the stream started, with the
    aircraft's track and every person's ground fix drawn with its
    uncertainty circle. The ICD makes position_error_m part of the contract
    and requires the ground station to display it: a coordinate without its
    uncertainty invites more confidence than the measurement supports.
  - a target list, one row per track: latest position and its uncertainty,
    highest confidence seen, movement state, and how many times it was seen.

On exit it writes a mission log: every target with its best fix.

Detections reported without a track (track_id -1) are drawn hollow and
listed separately as unconfirmed - seen, but not yet seen twice.

Nothing here talks back to the aircraft. Like the payload's MAVLink link,
it is read-only.

Usage:

    python scripts/ground_station.py --udp 0.0.0.0:14660
    python scripts/ground_station.py --follow detections.jsonl
    python scripts/ground_station.py --replay detections.jsonl --snapshot map.png --no-show
"""

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EARTH_RADIUS_M = 6_378_137.0

WINDOW = (1100, 700)
MAP_W = 700

STATE_COLOURS = {
    "moving": (0, 0, 255),
    "stationary": (0, 200, 0),
    "edge": (160, 160, 160),
    "unknown": (0, 200, 220),
}


class Picture:
    """Everything known so far, built up record by record."""

    def __init__(self):
        self.origin = None
        self.track = []
        self.targets = {}
        self.unconfirmed = []
        self.records = 0
        self.last_record = None
        self.ego_status = None
        self.stale_frames = 0

    def local(self, lat, lon):
        lat0, lon0 = self.origin
        north = math.radians(lat - lat0) * EARTH_RADIUS_M
        east = math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
        return east, north

    def add(self, record):
        self.records += 1
        self.last_record = time.monotonic()
        self.ego_status = record.get("ego_motion", self.ego_status)

        uav = record.get("uav")

        if uav is None:
            if "uav" in record:
                self.stale_frames += 1
            return

        if self.origin is None:
            self.origin = (uav["lat"], uav["lon"])

        self.track.append(self.local(uav["lat"], uav["lon"]))

        for det in record.get("detections", []):
            ground = det.get("ground")

            if not ground:
                continue

            east, north = self.local(ground["lat"], ground["lon"])
            fix = {
                "east": east, "north": north,
                "lat": ground["lat"], "lon": ground["lon"],
                "error_m": ground["position_error_m"],
                "confidence": det["confidence"],
                "state": det["state"],
                "frame": record["frame"],
            }

            if det["track_id"] == -1:
                self.unconfirmed.append(fix)
                continue

            target = self.targets.setdefault(det["track_id"], {
                "sightings": 0, "best_confidence": 0.0, "fixes": [],
            })
            target["sightings"] += 1
            target["best_confidence"] = max(target["best_confidence"], det["confidence"])
            target["latest"] = fix
            target["fixes"].append(fix)

    def best_fix(self, target):
        """
        Where to send someone. For a person seen moving, the latest fix - an
        older one is where they were.

        Otherwise every fix, weighted by inverse variance. Picking the single
        fix with the smallest reported uncertainty was biased: uncertainty
        grows with the altitude reading, so the pick favoured fixes whose
        altitude happened to read low, which pulls positions toward the
        aircraft - 0.6 m on the simulated mission, 3.4 standard errors
        (training_log section 53).

        The uncertainty reported is the smallest single fix's, not the
        combined one. Averaging N fixes divides it by sqrt(N) only if their
        errors are independent, and GPS and attitude errors drift slowly:
        within one pass they are largely the same error repeated.
        """

        if target["latest"]["state"] == "moving":
            return target["latest"]

        fixes = target["fixes"]
        weights = [1.0 / max(f["error_m"], 1e-3) ** 2 for f in fixes]
        total = sum(weights)

        return {
            **min(fixes, key=lambda f: f["error_m"]),
            "lat": sum(w * f["lat"] for w, f in zip(weights, fixes)) / total,
            "lon": sum(w * f["lon"] for w, f in zip(weights, fixes)) / total,
            "fixes_combined": len(fixes),
        }


def render(picture):
    canvas = np.full((WINDOW[1], WINDOW[0], 3), 24, np.uint8)
    cv2.line(canvas, (MAP_W, 0), (MAP_W, WINDOW[1]), (70, 70, 70), 1)

    points = list(picture.track)
    for target in picture.targets.values():
        points += [(f["east"], f["north"]) for f in target["fixes"]]
    points += [(f["east"], f["north"]) for f in picture.unconfirmed]

    if not points:
        cv2.putText(canvas, "waiting for records with aircraft state",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return canvas

    xs, ys = zip(*points)
    span = max(max(xs) - min(xs), max(ys) - min(ys), 50.0) * 1.15
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    scale = (MAP_W - 40) / span

    def px(east, north):
        return (int(MAP_W / 2 + (east - cx) * scale),
                int(WINDOW[1] / 2 - (north - cy) * scale))

    # 10 m scale bar
    cv2.line(canvas, (20, WINDOW[1] - 20), (20 + int(10 * scale), WINDOW[1] - 20),
             (220, 220, 220), 2)
    cv2.putText(canvas, "10 m", (20, WINDOW[1] - 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (220, 220, 220), 1)
    cv2.putText(canvas, "N", (MAP_W - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (220, 220, 220), 1)

    for a, b in zip(picture.track, picture.track[1:]):
        cv2.line(canvas, px(*a), px(*b), (200, 140, 60), 1)

    if picture.track:
        cv2.circle(canvas, px(*picture.track[-1]), 6, (255, 180, 80), -1)

    for fix in picture.unconfirmed[-200:]:
        cv2.circle(canvas, px(fix["east"], fix["north"]), 3, (120, 120, 120), 1)

    for track_id, target in picture.targets.items():
        fix = target["latest"]
        centre = px(fix["east"], fix["north"])
        colour = STATE_COLOURS.get(fix["state"], (255, 255, 255))
        cv2.circle(canvas, centre, max(2, int(fix["error_m"] * scale)), colour, 1)
        cv2.circle(canvas, centre, 3, colour, -1)
        cv2.putText(canvas, f"#{track_id}", (centre[0] + 5, centre[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

    # target list
    x0, y = MAP_W + 15, 28
    cv2.putText(canvas, f"targets {len(picture.targets)}   records {picture.records}",
                (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    y += 20
    status = picture.ego_status or "-"
    cv2.putText(canvas, f"motion estimate: {status}   stale state: {picture.stale_frames}",
                (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 200, 0) if status == "ok" else (0, 165, 255), 1)
    y += 26
    cv2.putText(canvas, "id    conf  state       +/-m   seen", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    ranked = sorted(picture.targets.items(), key=lambda kv: -kv[1]["best_confidence"])

    for track_id, target in ranked[:26]:
        y += 22
        fix = target["latest"]
        colour = STATE_COLOURS.get(fix["state"], (255, 255, 255))
        cv2.putText(canvas,
                    f"#{track_id:<4d} {target['best_confidence']:.2f}  "
                    f"{fix['state']:<10s} {fix['error_m']:5.1f}   {target['sightings']}",
                    (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1)

    if picture.unconfirmed:
        y += 30
        cv2.putText(canvas, f"unconfirmed detections: {len(picture.unconfirmed)}",
                    (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

    return canvas


def mission_log(picture, path):
    targets = []

    for track_id, target in sorted(picture.targets.items()):
        best = picture.best_fix(target)
        targets.append({
            "track_id": track_id,
            "lat": best["lat"], "lon": best["lon"],
            "position_error_m": best["error_m"],
            "best_confidence": round(target["best_confidence"], 4),
            "state": target["latest"]["state"],
            "sightings": target["sightings"],
            "first_frame": target["fixes"][0]["frame"],
            "last_frame": target["fixes"][-1]["frame"],
        })

    Path(path).write_text(json.dumps({
        "records": picture.records,
        "targets": targets,
        "unconfirmed_detections": len(picture.unconfirmed),
        "frames_without_aircraft_state": picture.stale_frames,
    }, indent=2), encoding="utf-8")


def udp_socket(address):
    host, _, port = address.rpartition(":")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host or "0.0.0.0", int(port)))
    sock.settimeout(0.1)
    return sock


def records_from_udp(sock):
    """Yields records, or None when nothing arrived - so the display keeps
    redrawing over a silent link."""

    while True:
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            yield None
            continue
        except ConnectionResetError:
            continue  # Windows reports an earlier ICMP unreachable here

        try:
            yield json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue  # a corrupt datagram is dropped, not fatal


def records_from_file(path, follow):
    with open(path, encoding="utf-8") as f:
        while True:
            line = f.readline()

            if line:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
            elif follow:
                yield None
                time.sleep(0.1)
            else:
                return


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--udp", metavar="HOST:PORT")
    source.add_argument("--follow", metavar="JSONL", help="follow a growing file")
    source.add_argument("--replay", metavar="JSONL", help="read a finished file")
    parser.add_argument("--log", default="mission_log.json")
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.udp:
        records = records_from_udp(udp_socket(args.udp))
    else:
        records = records_from_file(args.follow or args.replay, follow=bool(args.follow))

    picture = Picture()
    last_draw = 0.0

    try:
        for record in records:
            if record is not None:
                picture.add(record)

            now = time.monotonic()

            if not args.no_show and now - last_draw > 0.1:
                cv2.imshow("ground station", render(picture))
                last_draw = now

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass

    mission_log(picture, args.log)

    if args.snapshot:
        cv2.imwrite(args.snapshot, render(picture))

    if not args.no_show:
        cv2.destroyAllWindows()

    print(f"{picture.records} records, {len(picture.targets)} targets, "
          f"{len(picture.unconfirmed)} unconfirmed detections")
    print(f"Mission log: {args.log}")


if __name__ == "__main__":
    main()
