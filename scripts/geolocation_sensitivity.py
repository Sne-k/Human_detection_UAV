"""
The software half of the geolocation acceptance test.

payload_icd.md section 8 sets the one quantitative acceptance criterion in the
interface: overfly surveyed markers at 60, 100 and 130 m, and require the
**median error to fall within the reported `position_error_m`, with no
systematic along- or across-track bias**. That needs an aircraft. Everything
except the aircraft can be checked now, against exact geometry:

1. Frame conventions. A sign error in roll, pitch or yaw does not produce
   noise, it produces a confident coordinate in the wrong direction - and the
   ICD notes that a systematic bias is exactly what a convention error looks
   like. Each convention is checked against a case whose answer is known.

2. Sensitivity. How far does the ground fix move per unit error in each
   input - roll, pitch, yaw, altitude, GPS, telemetry timing - across the
   frame and across the operating altitudes?

3. The acceptance criterion, in simulation. Perturb every input by the very
   uncertainties `estimate_error` claims to account for, many times, and
   check whether the median error really falls within the reported value.

Usage:

    python scripts/geolocation_sensitivity.py
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geolocate  # noqa: E402

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "geolocation_sensitivity"

WIDTH, HEIGHT, HFOV = 640, 512, 50.0
ALTITUDES = (60.0, 100.0, 130.0)

# Pixel positions: centre, mid right edge, far corner (bottom of a box there).
POSITIONS = {
    "centre": (320.0, 256.0),
    "edge": (620.0, 256.0),
    "corner": (620.0, 500.0),
}

# The 1-sigma input uncertainties estimate_error() assumes.
SIGMA = {
    "attitude_deg": 1.0,
    "altitude_m": 2.0,
    "gps_m": 2.5,
}


def ground(camera, px, py, altitude, roll=0.0, pitch=0.0, yaw=0.0):
    result = geolocate.project_to_ground(camera, px, py, altitude, roll, pitch, yaw)
    return None if result is None else np.array(result[:2])


def conventions(camera):
    """Known-answer checks. Returns a list of (name, passed, detail)."""

    checks = []

    def check(name, condition, detail):
        checks.append((name, bool(condition), detail))

    nadir = ground(camera, 320, 256, 100.0)
    check("image centre at nadir lands under the aircraft",
          nadir is not None and np.linalg.norm(nadir) < 1e-6, f"{nadir}")

    right = ground(camera, 620, 256, 100.0, yaw=0.0)
    check("heading north: image right is east", right[0] > 0 and abs(right[1]) < 1e-6,
          f"east {right[0]:.2f}, north {right[1]:.2f}")

    top = ground(camera, 320, 10, 100.0, yaw=0.0)
    check("heading north: image top is north (forward)", top[1] > 0 and abs(top[0]) < 1e-6,
          f"east {top[0]:.2f}, north {top[1]:.2f}")

    right_e = ground(camera, 620, 256, 100.0, yaw=90.0)
    check("heading east: image right is south", right_e[1] < 0 and abs(right_e[0]) < 1e-6,
          f"east {right_e[0]:.2f}, north {right_e[1]:.2f}")

    # Rolling right (right wing down) swings the belly out to the LEFT - in a
    # right bank the belly faces out of the turn - so a nadir camera looks
    # west when heading north. The body-to-NED rotation agrees: roll by phi
    # takes the body's down axis to (0, -sin phi, cos phi) in north-east-down.
    # The first version of this check expected east, and was wrong; the
    # geometry was right.
    rolled = ground(camera, 320, 256, 100.0, roll=10.0)
    check("right wing down: belly swings left, centre fix moves west",
          rolled[0] < 0, f"east {rolled[0]:.2f}")

    # Nose up likewise points the belly forward: (sin theta, 0, cos theta).
    pitched = ground(camera, 320, 256, 100.0, pitch=10.0)
    check("nose up: belly points forward, centre fix moves north",
          pitched[1] > 0, f"north {pitched[1]:.2f}")

    expected = 100.0 * math.tan(math.radians(10.0))
    check("10 degrees of roll at 100 m moves the centre fix 100 tan(10) m",
          abs(abs(rolled[0]) - expected) < 0.01, f"{abs(rolled[0]):.3f} vs {expected:.3f}")

    return checks


def sensitivity(camera):
    """Ground displacement per unit error of each input, one at a time."""

    rows = []

    for altitude in ALTITUDES:
        for name, (px, py) in POSITIONS.items():
            base = ground(camera, px, py, altitude)

            def moved(**kwargs):
                return float(np.linalg.norm(
                    ground(camera, px, py, kwargs.pop("alt", altitude), **kwargs) - base))

            rows.append({
                "altitude_m": altitude,
                "position": name,
                "roll_1deg_m": round(moved(roll=1.0), 2),
                "pitch_1deg_m": round(moved(pitch=1.0), 2),
                "yaw_1deg_m": round(moved(yaw=1.0), 2),
                "altitude_2m_m": round(moved(alt=altitude + 2.0), 2),
                # A 50 ms association error during a 25 deg/s roll-in.
                "timing_50ms_m": round(moved(roll=25.0 * 0.05), 2),
            })

    return rows


def monte_carlo(camera, error_model, samples=4000, seed=0):
    """Median radial error against the reported value, per altitude and position."""

    rng = np.random.default_rng(seed)
    rows = []

    for altitude in ALTITUDES:
        for name, (px, py) in POSITIONS.items():
            truth = ground(camera, px, py, altitude)
            reported = error_model(camera, px, py, altitude)

            errors = np.empty((samples, 2))

            for i in range(samples):
                d = rng.normal(0.0, 1.0, 6)
                fix = ground(
                    camera, px, py,
                    altitude + d[0] * SIGMA["altitude_m"],
                    roll=d[1] * SIGMA["attitude_deg"],
                    pitch=d[2] * SIGMA["attitude_deg"],
                    yaw=d[3] * SIGMA["attitude_deg"],
                )
                errors[i] = fix + d[4:6] * SIGMA["gps_m"] - truth

            radial = np.linalg.norm(errors, axis=1)
            median = float(np.median(radial))

            rows.append({
                "altitude_m": altitude,
                "position": name,
                "reported_m": round(reported, 2),
                "median_error_m": round(median, 2),
                "p95_error_m": round(float(np.percentile(radial, 95)), 2),
                "bias_east_m": round(float(errors[:, 0].mean()), 2),
                "bias_north_m": round(float(errors[:, 1].mean()), 2),
                "passes": median <= reported,
            })

    return rows


def shipped_error(camera, px, py, altitude):
    """What the payload reports today: geolocate.estimate_error."""

    result = geolocate.project_to_ground(camera, px, py, altitude, 0.0, 0.0, 0.0)
    return geolocate.estimate_error(altitude, result[2])


def report_mc(title, rows):
    print()
    print(title)
    print(f"{'alt':>5s} {'position':<8s} {'reported':>9s} {'median':>7s} "
          f"{'p95':>6s} {'bias E,N':>12s}  ICD")
    for r in rows:
        print(f"{r['altitude_m']:5.0f} {r['position']:<8s} {r['reported_m']:9.2f} "
              f"{r['median_error_m']:7.2f} {r['p95_error_m']:6.2f} "
              f"{r['bias_east_m']:+5.2f},{r['bias_north_m']:+5.2f}  "
              f"{'pass' if r['passes'] else 'FAIL'}")


def main():
    camera = geolocate.Camera(width=WIDTH, height=HEIGHT, hfov_deg=HFOV,
                              mount_pitch_deg=90.0)

    print("=" * 74)
    print("1. Frame conventions - known answers")
    print("=" * 74)
    checks = conventions(camera)
    for name, passed, detail in checks:
        print(f"  {'ok  ' if passed else 'FAIL'} {name}   [{detail}]")

    print()
    print("=" * 74)
    print("2. Ground displacement per unit input error, metres")
    print("=" * 74)
    rows = sensitivity(camera)
    print(f"{'alt':>5s} {'position':<8s} {'roll 1':>7s} {'pitch 1':>8s} "
          f"{'yaw 1':>6s} {'alt +2m':>8s} {'50 ms':>6s}")
    for r in rows:
        print(f"{r['altitude_m']:5.0f} {r['position']:<8s} {r['roll_1deg_m']:7.2f} "
              f"{r['pitch_1deg_m']:8.2f} {r['yaw_1deg_m']:6.2f} "
              f"{r['altitude_2m_m']:8.2f} {r['timing_50ms_m']:6.2f}")

    print()
    print("=" * 74)
    print("3. ICD acceptance in simulation: median error within reported?")
    print("=" * 74)
    print(f"Inputs perturbed by the uncertainties estimate_error assumes: "
          f"{SIGMA['attitude_deg']} deg on each attitude axis, "
          f"{SIGMA['altitude_m']} m altitude, {SIGMA['gps_m']} m GPS per axis.")

    shipped = monte_carlo(camera, shipped_error)
    report_mc("Reported uncertainty: geolocate.estimate_error (as shipped)", shipped)

    results = {"conventions": [{"check": n, "passed": p, "detail": d}
                               for n, p, d in checks],
               "sensitivity": rows, "monte_carlo_shipped": shipped}

    if hasattr(geolocate, "propagate_error"):
        propagated = monte_carlo(
            camera,
            lambda c, px, py, alt: geolocate.propagate_error(
                c, px, py, {"altitude_m": alt, "roll_deg": 0.0,
                            "pitch_deg": 0.0, "yaw_deg": 0.0}),
        )
        report_mc("Reported uncertainty: geolocate.propagate_error", propagated)
        results["monte_carlo_propagated"] = propagated

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "geolocation_sensitivity.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print()
    print(f"Written: {OUTPUT_DIR / 'geolocation_sensitivity.json'}")


if __name__ == "__main__":
    main()
