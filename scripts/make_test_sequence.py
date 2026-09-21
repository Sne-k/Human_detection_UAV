"""
Synthesise a UAV-like test sequence from a single HIT-UAV thermal image.

The HIT-UAV dataset is a collection of independent still frames, so it cannot
be used to test tracking or movement detection directly. This script builds a
short video with known ground truth instead:

  - A viewport smaller than the source image pans (and optionally rotates)
    across the image, which reproduces the dominant effect of UAV motion: every
    stationary person slides across the frame.

  - One person patch is copied from the image and composited along a
    trajectory of its own, so the sequence contains exactly one genuinely
    moving person among stationary ones.

That gives a sequence where the correct answer is known: every real person in
the source image is stationary in the world, and only the composited person is
moving. A movement classifier that does not compensate for camera motion will
report everyone as moving, which is what makes this a useful test.

No upscaling is applied, so the thermal texture the detector sees is the
original sensor data.

Usage:

    python scripts/make_test_sequence.py
    python scripts/make_test_sequence.py --frames 180 --no-moving-person
    python scripts/make_test_sequence.py --image 1_60_30_0_00007 --rotate 2.0
"""

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

DATASET_ROOT = PROJECT_ROOT / "dataset" / "hit_uav_person"
OUTPUT_DIR = PROJECT_ROOT / "results" / "test_sequence"

# A test image with many annotated persons.
DEFAULT_IMAGE = "1_60_80_0_00707"

VIEWPORT = (512, 384)  # width, height
FRAMES = 120
FPS = 15.0


def load_labels(stem, width, height):
    """Ground-truth person boxes in source-image pixel coordinates."""

    label_path = DATASET_ROOT / "labels" / "test" / f"{stem}.txt"

    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text().splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        _, xc, yc, w, h = (float(v) for v in parts[:5])

        xc *= width
        yc *= height
        w *= width
        h *= height

        boxes.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))

    return boxes


def pick_patch(image, boxes, margin=3):
    """Take the largest annotated person as the patch to animate."""

    if not boxes:
        return None

    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    height, width = image.shape[:2]

    x1 = int(max(0, x1 - margin))
    y1 = int(max(0, y1 - margin))
    x2 = int(min(width, x2 + margin))
    y2 = int(min(height, y2 + margin))

    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    return image[y1:y2, x1:x2].copy()


def composite(frame, patch, centre):
    """
    Paste a person patch with a soft elliptical edge.

    A hard rectangular paste leaves a visible seam that the detector can latch
    onto; feathering keeps the composited person looking like sensor data.
    """

    patch_h, patch_w = patch.shape[:2]

    x0 = int(round(centre[0] - patch_w / 2))
    y0 = int(round(centre[1] - patch_h / 2))

    height, width = frame.shape[:2]

    if x0 < 0 or y0 < 0 or x0 + patch_w > width or y0 + patch_h > height:
        return False

    mask = np.zeros((patch_h, patch_w), dtype=np.float32)

    cv2.ellipse(
        mask,
        (patch_w // 2, patch_h // 2),
        (max(1, patch_w // 2 - 1), max(1, patch_h // 2 - 1)),
        0, 0, 360,
        1.0,
        -1,
    )

    mask = cv2.GaussianBlur(mask, (5, 5), 0)[..., None]

    region = frame[y0:y0 + patch_h, x0:x0 + patch_w].astype(np.float32)

    blended = patch.astype(np.float32) * mask + region * (1.0 - mask)

    frame[y0:y0 + patch_h, x0:x0 + patch_w] = blended.astype(np.uint8)

    return True


def build(args):
    image_path = DATASET_ROOT / "images" / "test" / f"{args.image}.jpg"

    if not image_path.exists():
        raise SystemExit(f"Source image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise SystemExit(f"Could not read {image_path}")

    height, width = image.shape[:2]

    view_w, view_h = args.viewport

    if view_w >= width or view_h >= height:
        raise SystemExit(
            f"Viewport {view_w}x{view_h} must be smaller than the "
            f"source image {width}x{height}"
        )

    boxes = load_labels(args.image, width, height)

    patch = None if args.no_moving_person else pick_patch(image, boxes)

    if patch is None and not args.no_moving_person:
        print("WARNING: no usable person patch found; sequence will be "
              "stationary only")

    # Pan range available inside the source image.
    span_x = width - view_w
    span_y = height - view_h

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_path = OUTPUT_DIR / f"{args.image}_pan.mp4"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (view_w, view_h),
    )

    truth = {
        "source_image": args.image,
        "source_size": [width, height],
        "viewport": [view_w, view_h],
        "frames": args.frames,
        "fps": args.fps,
        "rotate_deg_amplitude": args.rotate,
        "stationary_persons_in_source": len(boxes),
        "moving_person_composited": patch is not None,
        "moving_person_speed_px_per_frame": (
            args.moving_speed if patch is not None else 0.0
        ),
        "note": (
            "Every annotated person in the source image is stationary in the "
            "world frame. Only the composited person moves."
        ),
    }

    for index in range(args.frames):
        phase = index / max(1, args.frames - 1)

        # Smooth diagonal sweep, decelerating at both ends.
        eased = 0.5 - 0.5 * math.cos(math.pi * phase)

        offset_x = span_x * eased
        offset_y = span_y * eased * 0.6

        angle = args.rotate * math.sin(2 * math.pi * phase)

        # Rotate about the viewport centre, then translate to the pan offset.
        centre = (offset_x + view_w / 2.0, offset_y + view_h / 2.0)

        rotation = cv2.getRotationMatrix2D(centre, angle, 1.0)

        rotation[0, 2] -= offset_x
        rotation[1, 2] -= offset_y

        frame = cv2.warpAffine(
            image,
            rotation,
            (view_w, view_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        if patch is not None:
            # Trajectory in viewport coordinates: the composited person walks
            # across the frame independently of the camera pan.
            travel = args.moving_speed * index

            start_x = view_w * 0.15
            start_y = view_h * 0.55

            composite(
                frame,
                patch,
                (
                    start_x + travel,
                    start_y + travel * 0.25,
                ),
            )

        writer.write(frame)

    writer.release()

    truth_path = OUTPUT_DIR / f"{args.image}_pan.json"
    truth_path.write_text(json.dumps(truth, indent=2))

    print(f"Source image:        {image_path.name} ({width}x{height})")
    print(f"Annotated persons:   {len(boxes)}")
    print(f"Composited person:   {'yes' if patch is not None else 'no'}")
    print(f"Frames:              {args.frames} @ {args.fps} fps")
    print(f"Viewport:            {view_w}x{view_h}")
    print(f"Pan span:            {span_x} x {span_y} px")
    print()
    print(f"Written: {video_path}")
    print(f"Written: {truth_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument(
        "--viewport",
        type=int,
        nargs=2,
        default=list(VIEWPORT),
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument(
        "--rotate",
        type=float,
        default=1.5,
        help="Peak camera roll in degrees; 0 disables rotation",
    )
    parser.add_argument(
        "--moving-speed",
        type=float,
        default=1.4,
        help="Composited person speed in pixels per frame",
    )
    parser.add_argument("--no-moving-person", action="store_true")

    build(parser.parse_args())


if __name__ == "__main__":
    main()
