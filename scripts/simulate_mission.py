"""
A simulated search mission, end to end, scored - tasks D4 and B6.

Every piece of the payload has been tested on its own. This flies the whole
chain over ground where the answer is known, and scores what reaches the
rescue coordinator: not detections per frame, but people found, where, how
sure, and how many targets were false.

The ground
----------

A strip of real HIT-UAV thermal test images, stacked along track, rendered
at one ground sampling distance. Only steep views (camera pitch 80-90 deg,
from the file name) are used, so treating each as nadir imagery is honest,
and only images with people in them. Every labelled person becomes a fixed
point on the ground with a known latitude and longitude: the centre of their
box, since from overhead the centre of the box is the person.

The flight
----------

Level, nadir, heading north at 20 m/s and 100 m, at 7 FPS - the design
point. Frames are crops of the strip, which for a level nadir pinhole camera
is exactly what the camera would see: ground pixel size is uniform across
the frame. Telemetry is the true aircraft state plus measurement noise at
the sigmas geolocate.py assumes (attitude 1 deg, altitude 2 m, GPS 2.5 m) -
so the scene is true and the instruments are not, as in flight. The noise
drifts rather than jumping: each error is a first-order Gauss-Markov process,
correlated over 30 s for GPS and barometric altitude and 2 s for attitude.
Independent noise per sample would let averaging a target's fixes look far
better than it can be in flight, where one pass mostly repeats one error.

Scored
------

The payload's stream is read the way the ground station reads it
(ground_station.Picture). Identity is settled in pixels: each track is the
person its boxes sat on in most frames (see score()).

  found              people with a confirmed target on the map
  found incl. -1     ... or under any reported detection, unconfirmed too
  second identity    extra targets for a person already found
  no labelled person targets on nobody in the labels: a false positive,
                     or a person HIT-UAV did not label
  position error     each target's reported fix against its person;
                     the ICD criterion is median error <= reported sigma,
                     with no systematic bias

--monte-carlo N re-geolocates the recorded flight under N independent
instrument-error draws. One flight's GPS and altitude errors drift slowly
and every target shares them, so one flight can neither show a bias nor
settle whether the reported sigma is honest; many draws can.

What this is not: real flight. The scene is static apart from camera
motion, there is no wind, roll or terrain, and images are joined at seams
no real ground has.

Usage:

    python scripts/simulate_mission.py
    python scripts/simulate_mission.py --tiles 40 --seed 2
    python scripts/simulate_mission.py --score detections.jsonl
"""

import argparse
import contextlib
import io
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)
DATASET = PROJECT_ROOT / "dataset" / "hit_uav_person"
OUTPUT_DIR = PROJECT_ROOT / "results" / "mission"

WIDTH, HEIGHT = 640, 512
HFOV_DEG = 50.0
ALTITUDE_M = 100.0
SPEED_MS = 20.0
FPS = 7.0
ORIGIN = (12.9716, 77.5946)
MIN_PITCH_DEG = 80

# Correlation times of the telemetry errors: GPS and barometric altitude
# wander over tens of seconds, attitude estimates over a few.
GPS_TAU_S = 30.0
ATTITUDE_TAU_S = 2.0

EARTH_RADIUS_M = 6_378_137.0


def gsd():
    return 2.0 * ALTITUDE_M * math.tan(math.radians(HFOV_DEG) / 2.0) / WIDTH


def to_latlon(east_m, north_m):
    lat = ORIGIN[0] + math.degrees(north_m / EARTH_RADIUS_M)
    lon = ORIGIN[1] + math.degrees(
        east_m / (EARTH_RADIUS_M * math.cos(math.radians(ORIGIN[0]))))
    return lat, lon


def to_local(lat, lon):
    north = math.radians(lat - ORIGIN[0]) * EARTH_RADIUS_M
    east = math.radians(lon - ORIGIN[1]) * EARTH_RADIUS_M * math.cos(math.radians(ORIGIN[0]))
    return east, north


def pick_tiles(count, seed):
    """Steep-view test images that contain people."""

    candidates = []

    for image in sorted((DATASET / "images" / "test").glob("*.jpg")):
        parts = image.stem.split("_")

        if len(parts) < 3 or not parts[2].isdigit() or int(parts[2]) < MIN_PITCH_DEG:
            continue

        label = DATASET / "labels" / "test" / f"{image.stem}.txt"

        if label.exists() and label.read_text().strip():
            candidates.append(image)

    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)

    if len(candidates) < count:
        print(f"Only {len(candidates)} steep-view tiles with people; using all")

    return candidates[:count]


def build(tiles, out, seed):
    """Render the strip and fly over it. Returns the truth."""

    strip_h = HEIGHT * len(tiles)
    strip = np.zeros((strip_h, WIDTH, 3), np.uint8)
    people = []
    m_per_px = gsd()

    # Tile 0 at the bottom (south): the aircraft flies north, image-up.
    for i, image_path in enumerate(tiles):
        image = cv2.imread(str(image_path))
        image = cv2.resize(image, (WIDTH, HEIGHT)) if image.shape[:2] != (HEIGHT, WIDTH) else image
        top = strip_h - HEIGHT * (i + 1)
        strip[top:top + HEIGHT] = image

        label = DATASET / "labels" / "test" / f"{image_path.stem}.txt"

        for line in label.read_text().splitlines():
            parts = line.split()

            if len(parts) < 5:
                continue

            _, xc, yc, w, h = (float(v) for v in parts[:5])
            u, v = xc * WIDTH, top + yc * HEIGHT
            east, north = (u - WIDTH / 2) * m_per_px, (strip_h - v) * m_per_px
            lat, lon = to_latlon(east, north)
            people.append({"id": len(people), "tile": image_path.stem,
                           "east_m": round(east, 2), "north_m": round(north, 2),
                           "lat": lat, "lon": lon,
                           "box_px": [round(w * WIDTH, 1), round(h * HEIGHT, 1)]})

    step_px = SPEED_MS / m_per_px / FPS
    frames = int((strip_h - HEIGHT) / step_px) + 1

    truth = {
        "tiles": [t.stem for t in tiles],
        "gsd_m_per_px": m_per_px,
        "altitude_m": ALTITUDE_M, "speed_ms": SPEED_MS, "fps": FPS,
        "hfov_deg": HFOV_DEG, "frames": frames,
        "length_m": round((strip_h - HEIGHT) * m_per_px, 1),
        "people": people,
    }

    out.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out / "mission.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))

    for k in range(frames):
        top = frame_top(truth, k)
        writer.write(strip[top:top + HEIGHT])

    writer.release()

    with open(out / "telemetry.jsonl", "w", encoding="utf-8") as f:
        for record in telemetry_records(truth, seed):
            f.write(json.dumps(record) + "\n")

    (out / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def telemetry_records(truth, seed):
    """
    The aircraft's true state over each frame, as its instruments report it.

    Each error is a first-order Gauss-Markov process at the sigma
    geolocate.py assumes, drifting with the stated correlation time.
    """

    import geolocate

    rng = np.random.default_rng(seed + 1000)
    strip_h = HEIGHT * len(truth["tiles"])

    def drift(sigma, tau_s):
        a = math.exp(-1.0 / (truth["fps"] * tau_s))
        x = rng.normal(0.0, sigma)
        while True:
            yield x
            x = a * x + math.sqrt(1.0 - a * a) * rng.normal(0.0, sigma)

    gps_sigma = geolocate.SIGMA_GPS_M / math.sqrt(2)
    noise = {
        "east": drift(gps_sigma, GPS_TAU_S), "north": drift(gps_sigma, GPS_TAU_S),
        "alt": drift(geolocate.SIGMA_ALTITUDE_M, GPS_TAU_S),
        "roll": drift(geolocate.SIGMA_ROLL_DEG, ATTITUDE_TAU_S),
        "pitch": drift(geolocate.SIGMA_PITCH_DEG, ATTITUDE_TAU_S),
        "yaw": drift(geolocate.SIGMA_YAW_DEG, ATTITUDE_TAU_S),
    }

    records = []

    for k in range(truth["frames"]):
        # The aircraft is over the frame centre.
        north = (strip_h - (frame_top(truth, k) + HEIGHT / 2)) * truth["gsd_m_per_px"]
        e = {name: next(series) for name, series in noise.items()}
        lat, lon = to_latlon(e["east"], north + e["north"])

        records.append({
            "frame": k, "lat": lat, "lon": lon,
            "altitude_m": truth["altitude_m"] + e["alt"],
            "roll_deg": e["roll"],
            "pitch_deg": e["pitch"],
            "yaw_deg": e["yaw"] % 360.0,
        })

    return records


def regeolocate(truth, records, telemetry):
    """
    The same flight under different instrument errors.

    Which detections the payload made does not depend on telemetry; only
    where it put them does. So one recorded flight can be re-geolocated
    under many independent error draws - the Monte Carlo below - for the
    cost of arithmetic rather than another flight.
    """

    import geolocate

    camera = geolocate.Camera(width=WIDTH, height=HEIGHT,
                              hfov_deg=truth["hfov_deg"], mount_pitch_deg=90.0)
    out = []

    for record in records:
        state = telemetry[record["frame"]]
        detections = []

        for det in record["detections"]:
            det = {k: v for k, v in det.items() if k != "ground"}
            fix = geolocate.locate_detection(camera, det["box"], state)

            if fix is not None:
                det["ground"] = fix

            detections.append(det)

        out.append({**record, "detections": detections,
                    "uav": {"lat": state["lat"], "lon": state["lon"]}})

    return out


def fly(out):
    """Run the payload over the mission, exactly as on the aircraft."""

    import payload

    argv = sys.argv
    sys.argv = ["payload.py", "--source", str(out / "mission.mp4"),
                "--telemetry", str(out / "telemetry.jsonl"),
                "--hfov", str(HFOV_DEG), "--mount-pitch", "90",
                "--jsonl", str(out / "detections.jsonl"),
                "--summary", str(out / "payload_summary.json"),
                "--progress", "100000"]

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            payload.main()
    finally:
        sys.argv = argv

    return json.loads((out / "payload_summary.json").read_text(encoding="utf-8"))


def frame_top(truth, k):
    """Strip row at the top of frame k - as build() rendered it."""

    strip_h = HEIGHT * len(truth["tiles"])
    step_px = truth["speed_ms"] / truth["gsd_m_per_px"] / truth["fps"]
    return int(round(strip_h - HEIGHT - k * step_px))


def load_records(jsonl):
    with open(jsonl, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score(truth, records):
    """
    What the rescue coordinator would have been told, against the truth.

    Identity is settled in pixels, not on the ground. Each frame's truth is
    known exactly - every person's box, moved by the camera - so each
    detection is assigned to the labelled person whose box it sits on, and
    each track to the person most of its sightings sat on. Matching fixes to
    people by ground distance instead mixes up people standing closer
    together than the geolocation error; on this mission it reported real
    finds as duplicates.
    """

    import ground_station

    picture = ground_station.Picture()
    strip_h = HEIGHT * len(truth["tiles"])
    m_per_px = truth["gsd_m_per_px"]

    # Truth boxes in strip pixels.
    people = truth["people"]
    centres = np.array([(p["east_m"] / m_per_px + WIDTH / 2,
                         strip_h - p["north_m"] / m_per_px) for p in people])
    sizes = np.array([p["box_px"] for p in people])

    votes = {}            # track id -> {person: count}
    seen_any = set()      # people under any reported detection

    for record in records:
        picture.add(record)
        top = frame_top(truth, record["frame"])

        for det in record["detections"]:
            x1, y1, x2, y2 = det["box"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + top

            # The person whose box contains this detection's centre,
            # with a little slack for box jitter; nearest if several.
            dx = np.abs(centres[:, 0] - cx) - sizes[:, 0] / 2
            dy = np.abs(centres[:, 1] - cy) - sizes[:, 1] / 2
            inside = np.flatnonzero((dx <= 3) & (dy <= 3))

            if not len(inside):
                person = None
            else:
                d = np.hypot(centres[inside, 0] - cx, centres[inside, 1] - cy)
                person = int(inside[np.argmin(d)])
                seen_any.add(person)

            if det["track_id"] >= 0:
                tally = votes.setdefault(det["track_id"], {})
                tally[person] = tally.get(person, 0) + 1

    per_person = {}
    unlabelled = []
    errors, sigmas, bias = [], [], []

    for track_id, tally in votes.items():
        person = max(tally, key=tally.get)

        if track_id not in picture.targets:
            continue  # never geolocated: not on the coordinator's map

        target = picture.targets[track_id]

        if person is None:
            unlabelled.append(target)
            continue

        per_person.setdefault(person, []).append(track_id)
        fix = picture.best_fix(target)
        east, north = to_local(fix["lat"], fix["lon"])
        de, dn = east - people[person]["east_m"], north - people[person]["north_m"]
        errors.append(math.hypot(de, dn))
        sigmas.append(fix["error_m"])
        bias.append((de, dn))

    total = len(people)
    found = len(per_person)
    bias = np.array(bias) if bias else np.zeros((0, 2))

    return {
        "people": total,
        "found": found,
        "found_share": round(found / total, 3) if total else None,
        "found_including_unconfirmed": len(set(per_person) | seen_any),
        "confirmed_targets": len(picture.targets),
        "duplicate_targets": sum(len(v) - 1 for v in per_person.values()),
        "targets_on_no_labelled_person": len(unlabelled),
        # A target seen once or twice is a flicker; one seen for a whole
        # pass is persistent - a false positive, or a person HIT-UAV did not
        # label. Only looking tells which.
        "unlabelled_target_sightings": sorted(t["sightings"] for t in unlabelled),
        "unconfirmed_detections": len(picture.unconfirmed),
        "error_median_m": round(float(np.median(errors)), 2) if errors else None,
        "error_p90_m": round(float(np.percentile(errors, 90)), 2) if errors else None,
        "reported_sigma_median_m": round(float(np.median(sigmas)), 2) if sigmas else None,
        # Mean offset over this flight. It includes the flight's own GPS and
        # altitude drift, which every target shares, so it cannot show a
        # systematic bias by itself: --monte-carlo averages it over flights.
        "offset_east_m": round(float(bias[:, 0].mean()), 2) if len(bias) else None,
        "offset_north_m": round(float(bias[:, 1].mean()), 2) if len(bias) else None,
        "frames_without_aircraft_state": picture.stale_frames,
        "_errors": errors,
        "_sigmas": sigmas,
    }


def report(truth, result, summary=None):
    print("=" * 72)
    print(f"Simulated mission: {len(truth['tiles'])} tiles, {truth['length_m']:.0f} m "
          f"at {truth['altitude_m']:.0f} m, {truth['speed_ms']:.0f} m/s, "
          f"{truth['fps']:.0f} FPS, {truth['frames']} frames")
    print("=" * 72)

    if summary:
        print(f"payload: {summary.get('pipeline_fps')} FPS on this machine, "
              f"{summary.get('unique_tracks')} tracks, "
              f"ego {summary.get('ego_motion_status')}")

    r = result
    print(f"people on the ground          {r['people']}")
    print(f"found (confirmed target)      {r['found']}  ({r['found_share']:.0%})")
    print(f"found incl. unconfirmed       {r['found_including_unconfirmed']}")
    print(f"confirmed targets             {r['confirmed_targets']}")
    print(f"  second identity for a person {r['duplicate_targets']}")
    print(f"  on no labelled person       {r['targets_on_no_labelled_person']}   "
          f"(sightings {r['unlabelled_target_sightings']})")
    print(f"unconfirmed detections        {r['unconfirmed_detections']}")

    if r["error_median_m"] is not None:
        ok = r["error_median_m"] <= r["reported_sigma_median_m"]
        print(f"position error median / p90   {r['error_median_m']:.2f} / {r['error_p90_m']:.2f} m")
        print(f"reported sigma median         {r['reported_sigma_median_m']:.2f} m   "
              f"ICD median criterion: {'PASS' if ok else 'FAIL'}")
        print(f"mean offset east / north      {r['offset_east_m']:+.2f} / {r['offset_north_m']:+.2f} m   "
              f"(this flight's drift included)")


def monte_carlo(truth, records, draws, seed):
    """
    Is the reported uncertainty honest, and is the ICD's test of it fair?

    The ICD accepts geolocation if, on a flight over surveyed points, the
    median error is no more than the reported position_error_m. Per fix,
    with independent errors, that holds for a correct 1-sigma. But a flight's
    GPS and altitude errors drift slowly, so most of one flight's error is a
    single draw repeated - and a single draw exceeds a correct sigma a good
    fraction of the time. This measures how often an honest payload fails
    the test, and whether the reported sigma covers the error across flights.
    """

    medians, ratios, offsets = [], [], []
    errors, sigmas = [], []

    for draw in range(draws):
        telemetry = telemetry_records(truth, 10_000 + seed * 1000 + draw)
        r = score(truth, regeolocate(truth, records, telemetry))
        medians.append(r["error_median_m"])
        ratios.append(r["error_median_m"] / r["reported_sigma_median_m"])
        errors += r["_errors"]
        sigmas += r["_sigmas"]
        offsets.append((r["offset_east_m"], r["offset_north_m"]))

    errors, sigmas = np.array(errors), np.array(sigmas)
    ratios = np.array(ratios)
    offsets = np.array(offsets)

    # Each flight's mean offset is one sample of the shared drift; averaged
    # over independent flights, what remains is the systematic bias.
    bias = offsets.mean(axis=0)
    bias_se = offsets.std(axis=0) / math.sqrt(len(offsets))

    result = {
        "draws": draws,
        "flight_median_error_m": {
            "median": round(float(np.median(medians)), 2),
            "p90": round(float(np.percentile(medians, 90)), 2),
        },
        "single_flight_pass_rate": round(float(np.mean(ratios <= 1.0)), 3),
        "targets_within_1_sigma": round(float(np.mean(errors <= sigmas)), 3),
        "targets_within_2_sigma": round(float(np.mean(errors <= 2 * sigmas)), 3),
        "rms_error_over_rms_sigma": round(float(
            math.sqrt(np.mean(errors ** 2)) / math.sqrt(np.mean(sigmas ** 2))), 3),
        "bias_east_m": round(float(bias[0]), 2),
        "bias_north_m": round(float(bias[1]), 2),
        "bias_standard_error_m": round(float(bias_se.max()), 2),
    }

    print()
    print(f"Monte Carlo: the same flight under {draws} independent instrument-error draws")
    print(f"  flight median error           median {result['flight_median_error_m']['median']:.2f} m, "
          f"p90 {result['flight_median_error_m']['p90']:.2f} m")
    print(f"  single flights passing ICD    {result['single_flight_pass_rate']:.0%}")
    print(f"  targets within 1 / 2 sigma    {result['targets_within_1_sigma']:.0%} / "
          f"{result['targets_within_2_sigma']:.0%}   (Gaussian, RMS sigma: 63 % / 98 %)")
    print(f"  RMS error / RMS sigma         {result['rms_error_over_rms_sigma']:.2f}   "
          f"(1.0 = sigma exactly right)")
    print(f"  bias east / north             {result['bias_east_m']:+.2f} / "
          f"{result['bias_north_m']:+.2f} m   (standard error {result['bias_standard_error_m']:.2f} m)")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--monte-carlo", type=int, default=0, metavar="N",
                        help="Also re-geolocate the flight under N independent "
                             "instrument-error draws, to test the reported "
                             "uncertainty and the ICD acceptance criterion")
    parser.add_argument("--score", default=None, metavar="JSONL",
                        help="Score an existing stream against the mission "
                             "instead of flying it - for comparing runtimes")
    args = parser.parse_args()

    out = OUTPUT_DIR / f"seed{args.seed}_tiles{args.tiles}"

    if (out / "truth.json").exists():
        truth = json.loads((out / "truth.json").read_text(encoding="utf-8"))
    else:
        truth = build(pick_tiles(args.tiles, args.seed), out, args.seed)

    summary = None

    if args.score:
        jsonl = Path(args.score)
    else:
        summary = fly(out)
        jsonl = out / "detections.jsonl"

    records = load_records(jsonl)
    result = score(truth, records)
    report(truth, result, summary)

    import ground_station
    picture = ground_station.Picture()
    for record in records:
        picture.add(record)
    ground_station.mission_log(picture, out / "mission_log.json")
    cv2.imwrite(str(out / "ground_station.png"), ground_station.render(picture))

    if args.monte_carlo:
        result["monte_carlo"] = monte_carlo(truth, records, args.monte_carlo, args.seed)

    result.pop("_errors")
    result.pop("_sigmas")
    (out / "score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Written: {out / 'score.json'}")


if __name__ == "__main__":
    main()
