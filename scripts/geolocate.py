"""
Convert pixel detections into ground coordinates.

The payload architecture requires the companion computer to report *where a
person is*, not where they appear in an image:

    Camera -> Companion Computer -> Detection -> GPS / UAV position
           -> Detection coordinates -> Telemetry -> Ground Station

A bounding box in pixels is not actionable for a rescue team. This module
converts a detection into a latitude/longitude on the ground, given the UAV's
position and attitude and the camera's geometry.

Method
------

The ground is modelled as a horizontal plane at a known elevation. For each
detection:

1. The box's ground-contact point (bottom-centre of the box, where a standing
   person meets the ground) is converted to a normalised image coordinate.
2. That is turned into a ray in camera coordinates using the camera's field of
   view.
3. The ray is rotated into the world frame by the camera mount angle and the
   UAV's roll, pitch and yaw.
4. The ray is intersected with the ground plane.
5. The resulting local East/North offset is converted to latitude/longitude.

This is a flat-earth approximation, which is appropriate here: at the
project's 60-130 m operating altitudes the ground footprint is on the order of
100 m, where earth curvature is far below the error contributed by attitude
and altitude uncertainty.

Accuracy
--------

The dominant error source is attitude, not arithmetic. At altitude h, an
attitude error of d radians moves the ground point by roughly h * d /
cos^2(theta) for a look angle theta. At 100 m, a 1-degree attitude error is
about 1.7 m on the ground at nadir, and considerably worse at oblique angles.
`estimate_error` reports this so the ground station can be told how much to
trust each fix rather than being handed a bare coordinate.

The bottom-centre convention also assumes the person is standing on the
modelled ground plane. A person on a roof or a slope will be projected to the
wrong place, and terrain elevation is not modelled.

Usage:

    python scripts/geolocate.py --demo
    python scripts/geolocate.py --jsonl detections.jsonl --telemetry flight.jsonl \
        --out located.jsonl
"""

import argparse
import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

EARTH_RADIUS_M = 6378137.0


# ---------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------

class Camera:
    """
    Pinhole camera defined by its field of view and mounting angle.

    `pitch_deg` is the mount angle below horizontal: 90 looks straight down
    (nadir), 0 looks at the horizon. The HIT-UAV imagery was captured between
    30 and 90 degrees, which is the range this is expected to operate over.
    """

    def __init__(
        self,
        width,
        height,
        hfov_deg,
        vfov_deg=None,
        mount_pitch_deg=90.0,
        mount_yaw_deg=0.0,
    ):
        self.width = width
        self.height = height

        self.hfov = math.radians(hfov_deg)

        if vfov_deg is not None:
            self.vfov = math.radians(vfov_deg)
        else:
            # Derive from the sensor aspect ratio, assuming square pixels.
            self.vfov = 2.0 * math.atan(
                math.tan(self.hfov / 2.0) * (height / width)
            )

        self.mount_pitch = math.radians(mount_pitch_deg)
        self.mount_yaw = math.radians(mount_yaw_deg)

    def ray(self, px, py):
        """
        Direction vector for a pixel, in camera frame.

        Camera frame: x right, y down, z forward along the optical axis.
        """

        # Normalised coordinates in [-1, 1], origin at image centre.
        nx = (2.0 * px / self.width) - 1.0
        ny = (2.0 * py / self.height) - 1.0

        x = nx * math.tan(self.hfov / 2.0)
        y = ny * math.tan(self.vfov / 2.0)

        return (x, y, 1.0)


# ---------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------
#
# Camera frame : x right, y down, z forward along the optical axis.
# Body frame   : x forward (nose), y right (starboard), z down.
# World frame  : North, East, Down (NED).
#
# These are the standard aerospace conventions. Using them rather than an
# ad-hoc sequence of axis rotations means the attitude handling is the usual
# direction cosine matrix and can be checked against known cases.


def camera_to_body(ray, elevation_rad):
    """
    Rotate a camera-frame ray into the body frame.

    `elevation_rad` is the camera's downward mount angle from the nose:
    0 looks forward, pi/2 looks straight down.

    At pi/2 the optical axis maps to body down (0, 0, 1), and image-down maps
    to body aft (-1, 0, 0), which is what a nadir camera with image-up toward
    the nose actually does.
    """

    rx, ry, rz = ray

    se = math.sin(elevation_rad)
    ce = math.cos(elevation_rad)

    forward = -ry * se + rz * ce
    right = rx
    down = ry * ce + rz * se

    return (forward, right, down)


def body_to_ned(vector, roll, pitch, yaw):
    """Standard body-to-NED direction cosine matrix (Z-Y-X order)."""

    x, y, z = vector

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    north = (
        x * (cy * cp)
        + y * (cy * sp * sr - sy * cr)
        + z * (cy * sp * cr + sy * sr)
    )

    east = (
        x * (sy * cp)
        + y * (sy * sp * sr + cy * cr)
        + z * (sy * sp * cr - cy * sr)
    )

    down = (
        x * (-sp)
        + y * (cp * sr)
        + z * (cp * cr)
    )

    return (north, east, down)


# ---------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------

def project_to_ground(camera, px, py, altitude_m, roll_deg, pitch_deg, yaw_deg):
    """
    Intersect the ray through a pixel with a horizontal ground plane.

    Returns (east_m, north_m, slant_range_m), or None when the ray points at
    or above the horizon and never meets the ground.
    """

    ray = camera.ray(px, py)

    body = camera_to_body(ray, camera.mount_pitch)

    north, east, down = body_to_ned(
        body,
        math.radians(roll_deg),
        math.radians(pitch_deg),
        math.radians(yaw_deg) + camera.mount_yaw,
    )

    if down <= 1e-6:
        return None

    scale = altitude_m / down

    ground_east = east * scale
    ground_north = north * scale

    slant = math.sqrt(ground_east ** 2 + ground_north ** 2 + altitude_m ** 2)

    return ground_east, ground_north, slant


def offset_to_latlon(lat_deg, lon_deg, east_m, north_m):
    """Flat-earth offset from a reference latitude/longitude."""

    lat = lat_deg + math.degrees(north_m / EARTH_RADIUS_M)

    lon = lon_deg + math.degrees(
        east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat_deg)))
    )

    return lat, lon


def estimate_error(altitude_m, slant_range_m, attitude_error_deg=1.0,
                   altitude_error_m=2.0, gps_error_m=2.5):
    """
    Ground-position error budget, in metres.

    Attitude error dominates and grows with look angle, which is why it is
    scaled by slant range rather than altitude.
    """

    attitude_term = slant_range_m * math.radians(attitude_error_deg)

    # An altitude error scales the whole projection proportionally.
    altitude_term = (
        slant_range_m * (altitude_error_m / altitude_m)
        if altitude_m > 0 else 0.0
    )

    return math.sqrt(
        attitude_term ** 2 + altitude_term ** 2 + gps_error_m ** 2
    )


# ---------------------------------------------------------------------
# Detection handling
# ---------------------------------------------------------------------

def locate_detection(camera, box, telemetry):
    """
    Geolocate one detection.

    The ground-contact point is the bottom-centre of the box, where a standing
    person meets the ground. The box centre would sit roughly at waist height
    and project further from the aircraft.
    """

    x1, y1, x2, y2 = box

    px = (x1 + x2) / 2.0
    py = y2

    result = project_to_ground(
        camera,
        px, py,
        telemetry["altitude_m"],
        telemetry.get("roll_deg", 0.0),
        telemetry.get("pitch_deg", 0.0),
        telemetry.get("yaw_deg", 0.0),
    )

    if result is None:
        return None

    east, north, slant = result

    lat, lon = offset_to_latlon(
        telemetry["lat"], telemetry["lon"], east, north
    )

    return {
        "lat": round(lat, 7),
        "lon": round(lon, 7),
        "east_m": round(east, 2),
        "north_m": round(north, 2),
        "slant_range_m": round(slant, 2),
        "position_error_m": round(
            estimate_error(telemetry["altitude_m"], slant), 2
        ),
    }


def process(detections_path, telemetry_path, camera, output_path):
    """Join a detection stream with a telemetry stream, frame by frame."""

    telemetry = {}

    with open(telemetry_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            telemetry[record["frame"]] = record

    located = 0
    skipped = 0

    with open(detections_path, encoding="utf-8") as src, \
            open(output_path, "w", encoding="utf-8") as dst:

        for line in src:
            if not line.strip():
                continue

            frame = json.loads(line)

            state = telemetry.get(frame["frame"])

            if state is None:
                skipped += 1
                continue

            for detection in frame.get("detections", []):
                fix = locate_detection(camera, detection["box"], state)

                if fix is None:
                    detection["ground"] = None
                    skipped += 1
                else:
                    detection["ground"] = fix
                    located += 1

            frame["uav"] = {
                "lat": state["lat"],
                "lon": state["lon"],
                "altitude_m": state["altitude_m"],
            }

            dst.write(json.dumps(frame) + "\n")

    print(f"Geolocated {located} detections, skipped {skipped}.")
    print(f"Written: {output_path}")


# ---------------------------------------------------------------------
# Demo / self-check
# ---------------------------------------------------------------------

def demo():
    """
    Sanity checks with known geometry, at the project's operating altitudes.

    A nadir camera should place an image-centre detection directly beneath the
    aircraft, and the footprint should scale linearly with altitude.
    """

    camera = Camera(
        width=640, height=512,
        hfov_deg=50.0,
        mount_pitch_deg=90.0,
    )

    print("Camera: 640x512, 50 deg HFOV, nadir mount")
    print(f"Derived VFOV: {math.degrees(camera.vfov):.1f} deg")
    print()

    base = {"lat": 12.9716, "lon": 77.5946, "altitude_m": 100.0}

    print("1. Nadir check - a ray through the exact image centre")

    centre = project_to_ground(
        camera, camera.width / 2, camera.height / 2, 100.0, 0, 0, 0
    )

    print(f"   east {centre[0]:+.3f} m  north {centre[1]:+.3f} m  "
          f"(expect 0, 0 - directly below the aircraft)")

    print()
    print("   A detection box projects from its BOTTOM edge, not its centre,")
    print("   so a box straddling the image centre lands slightly aft:")

    fix = locate_detection(
        camera, [310, 246, 330, 266], {**base, "roll_deg": 0, "pitch_deg": 0}
    )

    expected = 100.0 * math.tan(
        math.atan(((2 * 266 / camera.height) - 1) * math.tan(camera.vfov / 2))
    )

    print(f"   box bottom 10 px below centre -> north {fix['north_m']:+.2f} m "
          f"(predicted {-expected:+.2f} m)")
    print(f"   error estimate +/- {fix['position_error_m']:.1f} m")
    print()

    print("2. Ground footprint vs altitude - expect linear scaling")
    print("   (HIT-UAV was captured across exactly this 60-130 m band)")

    for altitude in (60, 100, 130):
        corner = project_to_ground(
            camera, camera.width, camera.height, altitude, 0, 0, 0
        )

        width = 2 * altitude * math.tan(camera.hfov / 2)

        print(f"   {altitude:3d} m: footprint {width:6.1f} m wide, "
              f"corner at {corner[0]:+7.1f} E {corner[1]:+7.1f} N")

    print()
    print("3. Attitude sensitivity at 100 m - the dominant error source")

    for roll in (0, 1, 5, 10):
        fix = locate_detection(
            camera, [310, 246, 330, 266],
            {**base, "roll_deg": roll, "pitch_deg": 0},
        )

        print(f"   roll {roll:2d} deg -> east {fix['east_m']:+7.2f} m "
              f"(a {roll} deg error moves the fix by that much)")

    print()
    print("4. Oblique view, 30 deg off nadir - error grows with slant range")

    oblique = Camera(
        width=640, height=512, hfov_deg=50.0, mount_pitch_deg=60.0
    )

    fix = locate_detection(
        oblique, [310, 246, 330, 266], {**base, "roll_deg": 0, "pitch_deg": 0}
    )

    print(f"   ground point {fix['north_m']:+.1f} m north, "
          f"slant range {fix['slant_range_m']:.1f} m")
    print(f"   error estimate +/- {fix['position_error_m']:.1f} m "
          f"(vs {estimate_error(100, 100):.1f} m at nadir)")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--jsonl", help="Detection stream from realtime_detect.py")
    parser.add_argument("--telemetry", help="Per-frame UAV state, JSONL")
    parser.add_argument("--out", default="located.jsonl")

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--hfov", type=float, default=50.0)
    parser.add_argument("--mount-pitch", type=float, default=90.0)

    args = parser.parse_args()

    if args.demo:
        demo()
        return

    if not args.jsonl or not args.telemetry:
        raise SystemExit(
            "Provide --jsonl and --telemetry, or run with --demo"
        )

    camera = Camera(
        width=args.width,
        height=args.height,
        hfov_deg=args.hfov,
        mount_pitch_deg=args.mount_pitch,
    )

    process(args.jsonl, args.telemetry, camera, args.out)


if __name__ == "__main__":
    main()
