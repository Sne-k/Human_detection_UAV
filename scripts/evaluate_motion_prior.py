"""
Camera motion from the aircraft's own state - task A5.

Optical flow measures camera motion from image texture. Over open water,
snow or fog there is none, and section 50 found the limits of every
vision-only remedy: prediction carries the last trusted step across a short
gap but has nothing to carry over ground that is featureless from the start,
and about 3 % of steps there are confidently wrong in ways no vision-only
check caught. The aircraft knows how it is moving. geolocate.predict_step
turns two telemetry states into the image motion they imply, through the
same camera model geolocation uses.

This flies the simulated mission (simulate_mission.py, seed 0: real
imagery, exact camera motion, drifting telemetry errors) with stretches of
the ground replaced by featureless water, and compares camera-motion
policies against the true motion:

  deployed        flow; on failure, repeat the last trusted step
  hint fallback   flow; on failure, the telemetry step
  hint + check    as above, and flow that disagrees with telemetry by more
                  than a tolerance is overruled by it

Scored per step: disagreement with the true step at the frame corners. And
the drift that matters downstream: where the stabilised frame puts the
frame centre at the end, against where it truly is.

A banked turn over water follows: straight and level, repeating the last
trusted step is already right, so telemetry can only add something when
the motion changes while flow is blind.

What this cannot settle: the tolerance. The simulated telemetry's attitude
error moves 0.4 deg between frames, a pessimistic stand-in for a real
autopilot's smooth estimate, so its step error is an upper bound. The
tolerance for flight comes from flight logs.

Usage:

    python scripts/evaluate_motion_prior.py
    python scripts/evaluate_motion_prior.py --end-to-end

--end-to-end flies the real payload over the mission with a stretch of the
ground turned to water - background flattened, people kept, as people in
the sea look to a thermal camera - with and without the telemetry check,
and scores what reaches the ground station.
"""

import argparse
import json
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
MISSION = PROJECT_ROOT / "results" / "mission" / "seed0_tiles24"
OUTPUT_DIR = PROJECT_ROOT / "results" / "motion_prior"

NOISE = 4.0

CONDITIONS = {
    "clear": [],
    "water mid-pass": [(250, 330)],
    "water from start": [(0, 80)],
    "patchy water": [(100 + 40 * i, 112 + 40 * i) for i in range(8)],
}


def water(frame, rng):
    flat = np.full(frame.shape, float(np.median(frame))) + rng.normal(0, NOISE, frame.shape)
    return np.clip(flat, 0, 255).astype(np.uint8)


WATER_FRAMES = (240, 360)


def turn_over_water(residuals, images, payload, realtime_detect):
    """
    A banked turn across featureless water - where repeating the last step
    is wrong, because the heading keeps changing.

    Rendered under a virtual nadir camera (as evaluate_tracker_stress.py)
    turning 3 deg/frame at 20 px/frame, with frames 10-30 of 40 flattened
    to water. The true step is known from the rendering; the telemetry step
    is the true step plus a residual drawn from what predict_step actually
    got wrong on the mission, so its error is the simulator's, not assumed.
    """

    import evaluate_tracker_stress as ets

    rng = np.random.default_rng(1)
    frames_n, water_k = 40, range(10, 30)
    scene = dict(ets.SCENES["turn"], frames=frames_n)
    width, height = ets.VIEW_W, ets.VIEW_H
    disagree = realtime_detect.EgoMotionEstimator.disagreement
    totals = {}

    for path in images:
        image = cv2.imread(str(path))
        people = ets.load_people(path.stem, image.shape[1], image.shape[0])
        world, _, _, _, _, poses = ets.build_scene(image, people, scene, rng)

        mats = []
        frames = []
        for k, (camera, yaw, scale) in enumerate(poses):
            m = np.vstack([ets.view_matrix(camera, yaw, scale), [0, 0, 1]])
            mats.append(m)
            frame = cv2.cvtColor(cv2.warpAffine(world, m[:2], (width, height)), cv2.COLOR_BGR2GRAY)
            frames.append(water(frame, rng) if k in water_k else frame)

        true = [mats[k] @ np.linalg.inv(mats[k - 1]) for k in range(1, frames_n)]
        hints = [s + residuals[rng.integers(len(residuals))] for s in true]

        for name, use_hint, tolerance in (("deployed", False, None),
                                          ("hint + check 20 px", True, 20.0)):
            ego = realtime_detect.EgoMotionEstimator(
                **dict(payload.EGO_SETTINGS, hint_tolerance_px=tolerance))
            ego.update(frames[0])
            errors, truth_cum = [], np.eye(3)

            for k in range(1, frames_n):
                ego.update(frames[k], hints[k - 1] if use_hint else None)
                errors.append(disagree(ego.last_step, true[k - 1], width, height))
                truth_cum = true[k - 1] @ truth_cum

            centre = np.array([width / 2, height / 2, 1.0])
            drift = float(np.linalg.norm((ego.cumulative @ centre - truth_cum @ centre)[:2]))
            water_err = [errors[k - 1] for k in water_k]
            tally = totals.setdefault(name, {"drift": [], "water_step_max": []})
            tally["drift"].append(drift)
            tally["water_step_max"].append(max(water_err))

    print()
    print(f"Banked turn over water: 3 deg/frame, frames 10-30 of 40 featureless, {len(images)} scenes")
    print(f"{'policy':<22s} {'drift at end, median':>21s} {'max':>8s} {'worst water step, median':>25s}")
    out = {}
    for name, tally in totals.items():
        d, s = np.array(tally["drift"]), np.array(tally["water_step_max"])
        out[name] = {"drift_median_px": float(np.median(d)), "drift_max_px": float(d.max()),
                     "water_step_max_median_px": float(np.median(s))}
        print(f"{name:<22s} {np.median(d):19.0f}px {d.max():6.0f}px {np.median(s):23.0f}px")
    return out


def end_to_end(truth, frames_bgr, sm, payload, geolocate):
    """The payload over water, with and without telemetry-predicted motion."""

    import contextlib
    import io

    rng = np.random.default_rng(0)
    out = OUTPUT_DIR / "water_mission"
    out.mkdir(parents=True, exist_ok=True)
    strip_h = sm.HEIGHT * len(truth["tiles"])
    g = truth["gsd_m_per_px"]

    writer = cv2.VideoWriter(str(out / "mission.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             truth["fps"], (sm.WIDTH, sm.HEIGHT))

    for k, frame in enumerate(frames_bgr):
        if WATER_FRAMES[0] <= k < WATER_FRAMES[1]:
            top = sm.frame_top(truth, k)
            flat = water(frame, rng)

            for person in truth["people"]:
                u = person["east_m"] / g + sm.WIDTH / 2
                v = strip_h - person["north_m"] / g - top
                w, h = person["box_px"]
                x1, y1 = max(0, int(u - w / 2) - 3), max(0, int(v - h / 2) - 3)
                x2, y2 = min(sm.WIDTH, int(u + w / 2) + 3), min(sm.HEIGHT, int(v + h / 2) + 3)
                if x2 > x1 and y2 > y1:
                    flat[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

            frame = flat

        writer.write(frame)

    writer.release()

    real_predict = geolocate.predict_step
    rows = {}

    for name, predict in (("without telemetry check", lambda *a: None),
                          ("with telemetry check", real_predict)):
        geolocate.predict_step = predict
        argv = sys.argv
        sys.argv = ["payload.py", "--source", str(out / "mission.mp4"),
                    "--telemetry", str(MISSION / "telemetry.jsonl"),
                    "--hfov", str(truth["hfov_deg"]), "--mount-pitch", "90",
                    "--jsonl", str(out / "detections.jsonl"),
                    "--summary", str(out / "summary.json"), "--progress", "100000"]

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                payload.main()
        finally:
            sys.argv = argv
            geolocate.predict_step = real_predict

        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        records = sm.load_records(out / "detections.jsonl")

        # People whose ground position passes under the camera during the
        # water stretch: the ones this comparison is about.
        top_a, top_b = sm.frame_top(truth, WATER_FRAMES[1]), sm.frame_top(truth, WATER_FRAMES[0])
        in_water = {i for i, p in enumerate(truth["people"])
                    if top_a + sm.HEIGHT * 0.2 <= strip_h - p["north_m"] / g <= top_b + sm.HEIGHT * 0.8}
        water_truth = dict(truth, people=[truth["people"][i] for i in sorted(in_water)])
        water_records = [r for r in records
                         if WATER_FRAMES[0] - 20 <= r["frame"] < WATER_FRAMES[1] + 20]

        result = sm.score(truth, records)
        local = sm.score(water_truth, water_records)
        for r in (result, local):
            r.pop("_errors")
            r.pop("_sigmas")

        rows[name] = {"summary_ego": summary["ego_motion_status"],
                      "overruled": summary.get("ego_motion_overruled"),
                      "mission": result, "water_stretch": local}

    print()
    print(f"End to end: water over frames {WATER_FRAMES[0]}-{WATER_FRAMES[1]}, people kept")
    print(f"{'':<26s} {'ego ok/tel/pred/lost':>22s} {'water: people':>14s} {'found':>6s} "
          f"{'2nd ids':>8s} {'no person':>10s} {'mission found':>14s}")

    for name, r in rows.items():
        e, w, m = r["summary_ego"], r["water_stretch"], r["mission"]
        ego = f"{e['ok']}/{e.get('telemetry', 0)}/{e['predicted']}/{e['lost']}"
        print(f"{name:<26s} {ego:>22s} {w['people']:>14d} {w['found']:>6d} "
              f"{w['duplicate_targets']:>8d} {w['targets_on_no_labelled_person']:>10d} "
              f"{m['found']:>8d}/{m['people']}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerances", type=float, nargs="+", default=[10.0, 20.0, 40.0])
    parser.add_argument("--end-to-end", action="store_true")
    args = parser.parse_args()

    import geolocate
    import payload
    import realtime_detect
    import simulate_mission as sm

    truth = json.loads((MISSION / "truth.json").read_text(encoding="utf-8"))
    telemetry = [json.loads(l) for l in open(MISSION / "telemetry.jsonl", encoding="utf-8")]
    camera = geolocate.Camera(sm.WIDTH, sm.HEIGHT, truth["hfov_deg"], mount_pitch_deg=90.0)

    capture = cv2.VideoCapture(str(MISSION / "mission.mp4"))
    frames, frames_bgr = [], []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames_bgr.append(frame)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    tops = [sm.frame_top(truth, k) for k in range(len(frames))]
    true_steps = []
    for k in range(1, len(frames)):
        step = np.eye(3)
        step[1, 2] = tops[k - 1] - tops[k]   # the ground moves down the frame
        true_steps.append(step)

    hints = [geolocate.predict_step(camera, telemetry[k - 1], telemetry[k])
             for k in range(1, len(frames))]
    width, height = sm.WIDTH, sm.HEIGHT
    disagree = realtime_detect.EgoMotionEstimator.disagreement

    hint_error = np.array([disagree(h, t, width, height) for h, t in zip(hints, true_steps)])
    print("=" * 96)
    print(f"Telemetry-predicted camera motion - {len(frames)} frames of the seed 0 mission")
    print("=" * 96)
    print(f"telemetry step alone, error at corners: median {np.median(hint_error):.1f} px, "
          f"p95 {np.percentile(hint_error, 95):.1f} px, max {hint_error.max():.1f} px")
    print()

    policies = [("deployed", None, None), ("hint fallback", "hint", None)]
    policies += [(f"hint + check {t:g} px", "hint", t) for t in args.tolerances]

    print(f"{'condition':<18s} {'policy':<20s} {'ok':>4s} {'tel':>4s} {'pred':>4s} {'lost':>4s} "
          f"{'overruled':>9s} {'step err p95':>12s} {'max':>7s} {'drift at end':>13s}")

    results = {"telemetry_step_error_px": {
        "median": float(np.median(hint_error)), "p95": float(np.percentile(hint_error, 95)),
        "max": float(hint_error.max())}}

    for condition, stretches in CONDITIONS.items():
        rng = np.random.default_rng(0)
        bad = set()
        for a, b in stretches:
            bad.update(range(a, b))
        seq = [water(f, rng) if k in bad else f for k, f in enumerate(frames)]

        for name, use_hint, tolerance in policies:
            settings = dict(payload.EGO_SETTINGS, hint_tolerance_px=tolerance)
            ego = realtime_detect.EgoMotionEstimator(**settings)
            ego.update(seq[0])
            counts = {"ok": 0, "telemetry": 0, "predicted": 0, "lost": 0}
            errors = []
            true_cumulative = np.eye(3)

            for k in range(1, len(seq)):
                hint = hints[k - 1] if use_hint else None
                ego.update(seq[k], hint)
                counts[ego.status] = counts.get(ego.status, 0) + 1
                errors.append(disagree(ego.last_step, true_steps[k - 1], width, height))
                true_cumulative = true_steps[k - 1] @ true_cumulative

            centre = np.array([width / 2, height / 2, 1.0])
            drift = float(np.linalg.norm((ego.cumulative @ centre - true_cumulative @ centre)[:2]))
            errors = np.array(errors)

            results.setdefault(condition, {})[name] = {
                **counts, "overruled": ego.hint_rejections,
                "step_error_p95": float(np.percentile(errors, 95)),
                "step_error_max": float(errors.max()), "drift_px": drift,
            }

            print(f"{condition:<18s} {name:<20s} {counts['ok']:4d} {counts['telemetry']:4d} "
                  f"{counts['predicted']:4d} {counts['lost']:4d} {ego.hint_rejections:9d} "
                  f"{np.percentile(errors, 95):10.1f}px {errors.max():6.0f}px {drift:11.0f}px")
        print()

    print("Step error: disagreement with the true step at the frame corners. Drift: where the")
    print("stabilised frame puts the frame centre after the whole pass, against the truth.")

    residuals = [h - s for h, s in zip(hints, true_steps) if h is not None]
    import evaluate_tracker_stress as ets
    steep = []
    for path in sorted((ets.DATASET / "images" / "test").glob("*.jpg")):
        parts = path.stem.split("_")
        label = ets.DATASET / "labels" / "test" / f"{path.stem}.txt"
        if (len(parts) > 2 and parts[2].isdigit() and int(parts[2]) >= ets.MIN_PITCH_DEG
                and label.exists() and label.read_text().strip()):
            steep.append(path)
    results["turn_over_water"] = turn_over_water(residuals, steep[::4][:20], payload, realtime_detect)

    if args.end_to_end:
        results["end_to_end"] = end_to_end(truth, frames_bgr, sm, payload, geolocate)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "motion_prior.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Written: {OUTPUT_DIR / 'motion_prior.json'}")


if __name__ == "__main__":
    main()
