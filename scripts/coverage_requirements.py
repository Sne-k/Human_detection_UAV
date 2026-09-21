"""
Derive the detection frame-rate requirement from the mission, not from habit.

"How fast must the detector run?" is usually answered with a reflex - 30 FPS,
because that is what video looks like. That is the wrong question for a search
payload. A UAV flying a search pattern needs two things:

  1. Every point of ground must pass through at least one processed frame,
     otherwise the search has holes in it.
  2. A person must stay in view for enough consecutive frames to be tracked
     and classified as moving or stationary.

Both are set by the aircraft's ground speed and the camera's along-track
footprint, not by perceptual smoothness. The result is a far lower requirement
than video frame rates, which matters because it decides what companion
computer the aircraft actually needs.

Mission parameters come from the airframe designed in the previous phase:
approximately 5 kg MTOW, 20 m/s cruise, 20 km range, ~1 hour endurance. The
HIT-UAV thermal imagery was captured between 60 and 130 m altitude, which is
the band the detector has been trained and evaluated on.

Usage:

    python scripts/coverage_requirements.py
    python scripts/coverage_requirements.py --speed 15 --altitude 80 --vfov 40
"""

import argparse
import math

# Airframe, from the previous-phase design.
CRUISE_SPEED_MS = 20.0

# Operating band of the thermal training data.
ALTITUDES_M = (60, 80, 100, 130)

# Default camera geometry: a 640 x 512 thermal sensor with a 50 degree
# horizontal field of view, giving ~40.9 degrees vertically.
HFOV_DEG = 50.0
SENSOR_W = 640
SENSOR_H = 512

# Frames a track needs before the movement classifier will commit to a state.
# Set by HISTORY_FRAMES in realtime_detect.py.
TRACK_FRAMES_REQUIRED = 15

# Minimum frames for a detection to be trusted at all, rather than treated as
# a single-frame artefact.
CONFIRM_FRAMES = 3

# Along-track overlap between consecutive frames. Search patterns are normally
# flown with overlap so that a target near a frame edge is not lost.
OVERLAP = 0.30


def vfov_from_hfov(hfov_deg, width, height):
    hfov = math.radians(hfov_deg)

    return math.degrees(
        2.0 * math.atan(math.tan(hfov / 2.0) * (height / width))
    )


def footprint(altitude_m, fov_deg):
    """Ground width covered by a field of view at nadir."""

    return 2.0 * altitude_m * math.tan(math.radians(fov_deg) / 2.0)


def analyse(speed_ms, altitude_m, hfov_deg, vfov_deg):
    across = footprint(altitude_m, hfov_deg)
    along = footprint(altitude_m, vfov_deg)

    # Time for the aircraft to traverse its own along-track footprint. A point
    # on the ground is in view for roughly this long.
    dwell_s = along / speed_ms

    # Coverage: one frame per footprint, reduced by the overlap requirement.
    coverage_fps = 1.0 / (dwell_s * (1.0 - OVERLAP))

    # Confirmation: several frames on the same target.
    confirm_fps = CONFIRM_FRAMES / dwell_s

    # Tracking and movement classification need a full history window.
    tracking_fps = TRACK_FRAMES_REQUIRED / dwell_s

    return {
        "altitude_m": altitude_m,
        "across_track_m": across,
        "along_track_m": along,
        "dwell_s": dwell_s,
        "coverage_fps": coverage_fps,
        "confirm_fps": confirm_fps,
        "tracking_fps": tracking_fps,
        "swath_rate_km2_per_h": across * speed_ms * 3.6 / 1000.0,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--speed", type=float, default=CRUISE_SPEED_MS)
    parser.add_argument("--hfov", type=float, default=HFOV_DEG)
    parser.add_argument("--vfov", type=float, default=None)
    parser.add_argument("--width", type=int, default=SENSOR_W)
    parser.add_argument("--height", type=int, default=SENSOR_H)

    args = parser.parse_args()

    vfov = args.vfov or vfov_from_hfov(args.hfov, args.width, args.height)

    print("=" * 70)
    print("Detection frame-rate requirement, derived from the mission")
    print("=" * 70)
    print(f"Cruise speed   : {args.speed:.0f} m/s")
    print(f"Camera         : {args.width} x {args.height}, "
          f"{args.hfov:.0f} deg H / {vfov:.1f} deg V")
    print(f"Along-track overlap required: {OVERLAP:.0%}")
    print(f"Movement classifier needs {TRACK_FRAMES_REQUIRED} frames per track")
    print()

    header = (
        f"{'Alt':>5s} {'swath':>8s} {'along':>8s} {'dwell':>8s}  "
        f"{'coverage':>9s} {'confirm':>9s} {'tracking':>9s}"
    )
    print(header)
    print(f"{'(m)':>5s} {'(m)':>8s} {'(m)':>8s} {'(s)':>8s}  "
          f"{'(FPS)':>9s} {'(FPS)':>9s} {'(FPS)':>9s}")
    print("-" * len(header))

    rows = []

    for altitude in ALTITUDES_M:
        r = analyse(args.speed, altitude, args.hfov, vfov)
        rows.append(r)

        print(
            f"{r['altitude_m']:5.0f} {r['across_track_m']:8.1f} "
            f"{r['along_track_m']:8.1f} {r['dwell_s']:8.2f}  "
            f"{r['coverage_fps']:9.2f} {r['confirm_fps']:9.2f} "
            f"{r['tracking_fps']:9.2f}"
        )

    worst = max(rows, key=lambda r: r["tracking_fps"])

    print()
    print("=" * 70)
    print("Requirement")
    print("=" * 70)
    print(
        f"Worst case is the lowest altitude ({worst['altitude_m']:.0f} m), "
        f"where the footprint is smallest\nand the ground passes through the "
        f"frame fastest ({worst['dwell_s']:.2f} s dwell)."
    )
    print()
    print(f"  Coverage only, no gaps in the search   "
          f">= {worst['coverage_fps']:5.2f} FPS")
    print(f"  Plus {CONFIRM_FRAMES}-frame detection confirmation      "
          f">= {worst['confirm_fps']:5.2f} FPS")
    print(f"  Plus tracking and movement detection   "
          f">= {worst['tracking_fps']:5.2f} FPS")
    print()
    print("The binding requirement is movement detection, not coverage.")
    print("Coverage alone is satisfied by well under 1 FPS, because the")
    print("aircraft takes seconds to traverse its own footprint.")
    print()
    print(f"Area search rate at {worst['altitude_m']:.0f} m: "
          f"{worst['swath_rate_km2_per_h']:.1f} km2/h")

    print()
    print("=" * 70)
    print("What this means for hardware selection")
    print("=" * 70)
    print(
        "A detector that manages a few frames per second satisfies the whole\n"
        "mission. The reflex target of 30 FPS would oversize the companion\n"
        "computer by roughly an order of magnitude, and on a weight- and\n"
        "power-constrained airframe that is a real cost.\n"
    )
    print(
        "If the movement-detection requirement is relaxed - reporting where\n"
        "people are, without classifying whether they are moving - the\n"
        "requirement drops to the confirmation row and a far smaller board\n"
        "becomes sufficient."
    )


if __name__ == "__main__":
    main()
