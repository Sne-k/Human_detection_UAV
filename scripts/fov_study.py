"""
Sensor resolution and lens field of view, chosen together - task C1.

sensor_resolution_study.py measured how recall falls as a thermal sensor gets
coarser, with the field of view held at 50 degrees. But what the detector
responds to is pixels on the person, and that depends on three things at
once: sensor resolution, lens field of view, and altitude. A cheaper sensor
behind a narrower lens can put as many pixels on a person as an expensive one
behind a wide lens - at the price of a narrower swath.

That trade is the procurement decision, and it needs both sides:

  - **recall**: read from the measured resolution curve, via the linear
    scale a (sensor, lens) pair implies relative to the reference geometry
  - **swath**: how much ground each pass covers, and so how much area is
    searched per hour

For a search aircraft the quantity that matters is roughly people found per
hour, proportional to area swept per hour times recall. Area alone favours
the widest lens; recall alone the narrowest.

Assumptions, stated rather than buried:

  - The reference is the project's working geometry: HIT-UAV's 640 x 512
    frames treated as a 50 degree horizontal field of view. Every GSD figure
    in this project uses it; it has not been confirmed against the dataset's
    capture metadata.
  - Recall at a given scale comes from MT-005's measured curve. It is not
    extrapolated: above the native scale it is reported as "at least" the
    native recall, and below the smallest measured scale it is not reported.
  - Coverage assumes 20 m/s and three frames per target for confirmation, as
    coverage_requirements.py does.

Usage:

    python scripts/fov_study.py
    python scripts/fov_study.py --altitude 60
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "fov_study"

REFERENCE_WIDTH_PX = 640
REFERENCE_HFOV_DEG = 50.0
SPEED_MS = 20.0
CONFIRM_FRAMES = 3

# Measured: MT-005, sensor_resolution_study.py. (linear scale, recall)
RECALL_CURVE = [(0.25, 0.6254), (0.4, 0.8407), (0.5, 0.8774),
                (0.6, 0.8905), (1.0, 0.9288)]

SENSORS = [(160, 120), (256, 192), (320, 256), (384, 288), (640, 512)]
HFOVS = [24.0, 32.0, 42.0, 50.0, 57.0, 69.0]


def linear_scale(width_px, hfov_deg):
    """Pixels on a target relative to the reference, at the same altitude."""

    return (width_px / REFERENCE_WIDTH_PX) * (
        math.tan(math.radians(REFERENCE_HFOV_DEG) / 2.0)
        / math.tan(math.radians(hfov_deg) / 2.0)
    )


def recall_at(scale):
    """(recall, qualifier). Interpolated in log scale; never extrapolated."""

    lo_scale, hi_scale = RECALL_CURVE[0][0], RECALL_CURVE[-1][0]

    if scale < lo_scale:
        return None, "below measured range"

    if scale >= hi_scale:
        return RECALL_CURVE[-1][1], "at least"

    for (s0, r0), (s1, r1) in zip(RECALL_CURVE, RECALL_CURVE[1:]):
        if s0 <= scale <= s1:
            w = (math.log(scale) - math.log(s0)) / (math.log(s1) - math.log(s0))
            return r0 + w * (r1 - r0), ""

    return None, "below measured range"


def geometry(width_px, height_px, hfov_deg, altitude_m):
    hfov = math.radians(hfov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * height_px / width_px)

    swath = 2.0 * altitude_m * math.tan(hfov / 2.0)
    along = 2.0 * altitude_m * math.tan(vfov / 2.0)

    return {
        "swath_m": swath,
        "gsd_m_per_px": swath / width_px,
        "area_km2_per_h": swath * SPEED_MS * 3600.0 / 1e6,
        "confirm_fps": CONFIRM_FRAMES * SPEED_MS / along,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--altitude", type=float, default=100.0)
    args = parser.parse_args()

    rows = []
    reference = None

    for width, height in SENSORS:
        for hfov in HFOVS:
            scale = linear_scale(width, hfov)
            recall, qualifier = recall_at(scale)
            g = geometry(width, height, hfov, args.altitude)

            row = {
                "sensor": f"{width}x{height}",
                "hfov_deg": hfov,
                "linear_scale": round(scale, 3),
                "person_px": f"{12 * scale:.1f}x{19 * scale:.1f}",
                "recall": None if recall is None else round(recall, 4),
                "recall_qualifier": qualifier,
                **{k: round(v, 3) for k, v in g.items()},
            }

            if recall is not None:
                row["found_per_h"] = g["area_km2_per_h"] * recall

            if width == 640 and hfov == REFERENCE_HFOV_DEG:
                reference = row

            rows.append(row)

    for row in rows:
        if "found_per_h" in row:
            row["found_vs_reference"] = round(row["found_per_h"] / reference["found_per_h"], 3)

    print("=" * 96)
    print(f"Sensor and lens at {args.altitude:.0f} m, {SPEED_MS:.0f} m/s - "
          f"relative to 640 x 512 at {REFERENCE_HFOV_DEG:.0f} deg")
    print("=" * 96)
    print(f"{'sensor':<9s} {'HFOV':>5s} {'person px':>10s} {'recall':>14s} "
          f"{'swath':>7s} {'km2/h':>6s} {'found/h':>8s} {'confirm FPS':>12s}")

    for row in rows:
        if row["recall"] is None:
            recall = "  (not measured)"
            found = "-"
        else:
            recall = f"{'>=' if row['recall_qualifier'] == 'at least' else '  '}{row['recall']:.3f}"
            found = f"{row['found_vs_reference']:.2f}x"

        print(f"{row['sensor']:<9s} {row['hfov_deg']:5.0f} {row['person_px']:>10s} "
              f"{recall:>14s} {row['swath_m']:6.0f}m {row['area_km2_per_h']:6.2f} "
              f"{found:>8s} {row['confirm_fps']:12.2f}")

    print()
    print("found/h is area swept per hour x recall, relative to the reference.")
    print("Above the native scale recall is reported as a floor, not a gain.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"fov_study_{int(args.altitude)}m.json").write_text(
        json.dumps({"altitude_m": args.altitude, "rows": rows}, indent=2),
        encoding="utf-8")
    print(f"Written: {OUTPUT_DIR / f'fov_study_{int(args.altitude)}m.json'}")


if __name__ == "__main__":
    main()
