"""
Tracker stress suite - task B4.

evaluate_tracking_gate.py asks whether tracking lets people through. This
asks whether it keeps them apart and keeps them whole, in the situations
that break trackers: several people moving at once, two walking straight
through each other, the camera turning, the aircraft descending, and the
low frame rates of search mode.

Each scene is a real steep-view (80-90 deg) HIT-UAV test image as the
ground - its labelled people standing still - plus walkers: copies of a person from the same image,
composited onto known straight paths. A virtual camera flies over it with
a known pose per frame (translation, rotation, scale), and the payload's
own detector, ego-motion estimator, tracker and movement classifier run on
the rendered frames. Every person's box is known in every frame.

Scored per person, over the frames they are in view:

  coverage       share of frames a confirmed track reported them
  confirmed      people reported as a confirmed track at least once
  identities     distinct track IDs a person was reported under; one is right
  switches       times a person's reported ID changed between sightings
  merged IDs     track IDs that were reported on two different people - the
                 identity swap that sends a team after the wrong person
  moving         walkers: share of their definite labels that say moving
  false moving   standing people labelled moving

Scenes, at 100 m (0.146 m/px) and 7 FPS unless stated:

  cruise      ground drifts 20 px/frame; three walkers at 1.4 m/s
  runners     as cruise, walkers at 4 m/s
  crossing    two walkers on one line, head-on, passing 4 px apart
  turn        as cruise, camera yawing 3 deg/frame (21 deg/s, a banked turn)
  descent     as cruise, 100 m to 71 m over the sequence (scale 1.0 - 1.4)
  search      search mode, 1.4 FPS: ground 100 px/frame, walkers 7 px/frame

Every association setting runs on identical detector and ego-motion output,
so differences are the tracker's alone.

Usage:

    python scripts/evaluate_tracker_stress.py
    python scripts/evaluate_tracker_stress.py --images 20 --assoc 1.0 1.5 2.0
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)
DATASET = PROJECT_ROOT / "dataset" / "hit_uav_person"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tracker_stress"

VIEW_W, VIEW_H = 640, 512
MATCH_IOU = 0.2
MIN_PITCH_DEG = 80
GSD_M = 0.146
FPS = 7.0

WALK_PX = 1.4 / GSD_M / FPS      # 1.4 m/s
RUN_PX = 4.0 / GSD_M / FPS       # 4 m/s

# name: frames, fps, ground drift px/frame, yaw deg/frame, final scale,
#       walker speed px/frame, walker layout
SCENES = {
    "cruise": dict(frames=24, fps=FPS, drift=20.0, yaw=0.0, scale=1.0,
                   speed=WALK_PX, layout="scatter"),
    "runners": dict(frames=24, fps=FPS, drift=20.0, yaw=0.0, scale=1.0,
                    speed=RUN_PX, layout="scatter"),
    "crossing": dict(frames=24, fps=FPS, drift=20.0, yaw=0.0, scale=1.0,
                     speed=WALK_PX, layout="crossing"),
    "turn": dict(frames=24, fps=FPS, drift=20.0, yaw=3.0, scale=1.0,
                 speed=WALK_PX, layout="scatter"),
    "descent": dict(frames=24, fps=FPS, drift=20.0, yaw=0.0, scale=1.4,
                    speed=WALK_PX, layout="scatter"),
    "search": dict(frames=14, fps=1.4, drift=100.0, yaw=0.0, scale=1.0,
                   speed=WALK_PX * FPS / 1.4, layout="scatter"),
}


def load_people(stem, width, height):
    path = DATASET / "labels" / "test" / f"{stem}.txt"
    boxes = []

    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            _, xc, yc, w, h = (float(v) for v in parts[:5])
            boxes.append((xc * width, yc * height, w * width, h * height))

    return boxes


def cut_patch(image, person, margin=3):
    xc, yc, w, h = person
    x1, y1 = int(max(0, xc - w / 2 - margin)), int(max(0, yc - h / 2 - margin))
    x2 = int(min(image.shape[1], xc + w / 2 + margin))
    y2 = int(min(image.shape[0], yc + h / 2 + margin))
    return image[y1:y2, x1:x2].copy(), (w, h)


def paste(canvas, patch, centre):
    """Feathered paste, as make_test_sequence.composite."""

    ph, pw = patch.shape[:2]
    x0, y0 = int(round(centre[0] - pw / 2)), int(round(centre[1] - ph / 2))

    if x0 < 0 or y0 < 0 or x0 + pw > canvas.shape[1] or y0 + ph > canvas.shape[0]:
        return

    mask = np.zeros((ph, pw), np.float32)
    cv2.ellipse(mask, (pw // 2, ph // 2), (max(1, pw // 2 - 1), max(1, ph // 2 - 1)),
                0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)[..., None]
    region = canvas[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
    canvas[y0:y0 + ph, x0:x0 + pw] = (patch * mask + region * (1 - mask)).astype(np.uint8)


def build_scene(image, people, scene, rng):
    """World canvas, walkers' paths, and the camera's pose per frame."""

    frames, drift = scene["frames"], scene["drift"]
    travel = drift * (frames - 1)
    pad = int(travel / 2 + 400)
    world = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    h, w = image.shape[:2]

    standing = [(xc + pad, yc + pad, bw, bh) for xc, yc, bw, bh in people]

    # Walkers: copies of the image's largest person on straight paths through
    # the middle of what the camera will see.
    source = max(people, key=lambda p: p[2] * p[3])
    patch, (pw, ph) = cut_patch(image, source)
    centre = np.array([pad + w / 2, pad + h / 2])
    walkers = []

    if scene["layout"] == "crossing":
        # Head-on along one line, 4 px apart laterally, meeting mid-sequence.
        direction = np.array([1.0, 0.0])
        meet = centre
        half = scene["speed"] * (frames - 1) / 2
        walkers.append((meet - direction * half + [0, -2], direction * scene["speed"]))
        walkers.append((meet + direction * half + [0, 2], -direction * scene["speed"]))
    else:
        for _ in range(3):
            angle = rng.uniform(0, 2 * math.pi)
            velocity = np.array([math.cos(angle), math.sin(angle)]) * scene["speed"]
            start = centre + rng.uniform(-150, 150, 2) - velocity * (frames - 1) / 2
            walkers.append((start, velocity))

    poses = []

    for k in range(frames):
        phase = k - (frames - 1) / 2
        # The ground moves down the frame: the camera moves up the world.
        camera = centre + np.array([0.0, -drift * phase])
        yaw = math.radians(scene["yaw"] * phase)
        scale = 1.0 + (scene["scale"] - 1.0) * k / max(1, frames - 1)
        poses.append((camera, yaw, scale))

    return world, standing, walkers, patch, (pw, ph), poses


def view_matrix(camera, yaw, scale):
    """World pixels to frame pixels."""

    c, s = math.cos(yaw) * scale, math.sin(yaw) * scale
    return np.array([
        [c, -s, VIEW_W / 2 - (c * camera[0] - s * camera[1])],
        [s, c, VIEW_H / 2 - (s * camera[0] + c * camera[1])],
    ])


def truth_boxes(objects, matrix, scale):
    """Frame boxes of objects given as (cx, cy, w, h) in world pixels, and
    which are at least half in view."""

    boxes, visible = [], []

    for cx, cy, bw, bh in objects:
        x, y = matrix @ np.array([cx, cy, 1.0])
        w, h = bw * scale, bh * scale
        boxes.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        visible.append(0 <= x < VIEW_W and 0 <= y < VIEW_H)

    return np.array(boxes).reshape(-1, 4), np.array(visible)


def iou_matrix(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match(truth, predicted):
    if not len(truth) or not len(predicted):
        return []

    m = iou_matrix(truth, predicted)
    pairs, used_t, used_p = [], set(), set()

    for ti, pi in sorted(np.argwhere(m >= MATCH_IOU), key=lambda tp: -m[tp[0], tp[1]]):
        if ti not in used_t and pi not in used_p:
            used_t.add(ti)
            used_p.add(pi)
            pairs.append((int(ti), int(pi)))

    return pairs


def render(scene_parts, detector, ego, scene):
    """Frames through detector and ego-motion, once, shared by all settings."""

    world, standing, walkers, patch, (pw, ph), poses = scene_parts
    prepared = []

    for k, (camera, yaw, scale) in enumerate(poses):
        canvas = world.copy()
        positions = [start + velocity * k for start, velocity in walkers]

        for p in positions:
            paste(canvas, patch, p)

        matrix = view_matrix(camera, yaw, scale)
        frame = cv2.warpAffine(canvas, matrix, (VIEW_W, VIEW_H), flags=cv2.INTER_LINEAR)

        objects = standing + [(p[0], p[1], pw, ph) for p in positions]
        boxes, visible = truth_boxes(objects, matrix, scale)

        cumulative = ego.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).copy()
        det_boxes, scores, _ = detector(frame)

        prepared.append((k / scene["fps"], frame, det_boxes, scores, cumulative,
                         boxes, visible))

    return prepared, len(standing)


def track(prepared, n_standing, assoc, payload, realtime_detect):
    payload.ASSOC_SCALE = assoc
    tracker = payload.build_tracker(FPS, payload.CONF)
    movement = realtime_detect.MovementTracker()
    n = len(prepared[0][5])

    timeline = [[] for _ in range(n)]          # per person: ID or None per visible frame
    detected = [0] * n                         # frames the detector alone found them
    labels = [[] for _ in range(n)]
    owners = {}                                # track id -> people it was reported on

    for t, frame, boxes, scores, cumulative, truth, visible in prepared:
        rows, world = payload.track_frame(tracker, boxes, scores, frame, cumulative)
        ids = [int(r[4]) for r in rows]
        states = []

        for row, wb in zip(rows, world):
            tid = int(row[4])
            movement.update(tid, ((wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2), 0, False, t)
            states.append(movement.state(tid)[0])

        pairs = dict(match(truth, rows[:, :4] if len(rows) else np.zeros((0, 4))))
        found = dict(match(truth, np.asarray(boxes).reshape(-1, 4)))

        for person in np.flatnonzero(visible):
            detected[person] += int(person) in found
            pi = pairs.get(int(person))
            timeline[person].append(None if pi is None else ids[pi])

            if pi is not None:
                owners.setdefault(ids[pi], set()).add(int(person))
                labels[person].append(states[pi])

    result = {}

    for group, people in (("standing", range(n_standing)), ("walkers", range(n_standing, n))):
        people = [p for p in people if timeline[p]]
        frames = sum(len(timeline[p]) for p in people)
        covered = sum(sum(i is not None for i in timeline[p]) for p in people)
        confirmed = [p for p in people if any(i is not None for i in timeline[p])]
        identities = sum(len({i for i in timeline[p] if i is not None}) for p in confirmed)
        switches = 0

        for p in confirmed:
            seen = [i for i in timeline[p] if i is not None]
            switches += sum(a != b for a, b in zip(seen, seen[1:]))

        definite = [s for p in people for s in labels[p] if s in ("moving", "stationary")]
        moving = sum(s == "moving" for s in definite)

        result[group] = {
            "people": len(people), "frames": frames, "covered": covered,
            "detected": sum(detected[p] for p in people),
            "confirmed": len(confirmed), "identities": identities,
            "switches": switches, "definite_labels": len(definite), "moving_labels": moving,
        }

    result["merged_ids"] = sum(len(v) > 1 for v in owners.values())
    return result


def add(total, part):
    for key, value in part.items():
        if isinstance(value, dict):
            add(total.setdefault(key, {}), value)
        else:
            total[key] = total.get(key, 0) + value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=40)
    parser.add_argument("--assoc", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import payload
    import realtime_detect

    # Steep views only. The virtual camera is nadir: a turn rotates the
    # image in-plane and a descent zooms it, which is what a downward camera
    # sees. Rotating an oblique image instead shows standing people tilted
    # over, which no camera would - and which cut the detector's recall in a
    # first trial on two images.
    images = []
    for path in sorted((DATASET / "images" / "test").glob("*.jpg")):
        parts = path.stem.split("_")
        label = DATASET / "labels" / "test" / f"{path.stem}.txt"
        if (len(parts) > 2 and parts[2].isdigit() and int(parts[2]) >= MIN_PITCH_DEG
                and label.exists() and label.read_text().strip()):
            images.append(path)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(images)
    images = images[:args.images]

    detector = payload.Detector([payload.DEFAULT_THERMAL], device=args.device, conf=payload.CONF)
    totals = {s: {a: {} for a in args.assoc} for s in args.scenes}
    started = time.perf_counter()

    for n, path in enumerate(images, 1):
        image = cv2.imread(str(path))
        people = load_people(path.stem, image.shape[1], image.shape[0])

        for name in args.scenes:
            scene = SCENES[name]
            parts = build_scene(image, people, scene, rng)
            prepared, n_standing = render(parts, detector, payload.make_ego_estimator(), scene)

            for assoc in args.assoc:
                add(totals[name][assoc], track(prepared, n_standing, assoc, payload, realtime_detect))

        if n % 10 == 0 or n == len(images):
            print(f"  {n}/{len(images)} images, {time.perf_counter() - started:.0f}s")

    payload.ASSOC_SCALE = 1.0

    print()
    print("=" * 112)
    print(f"Tracker stress - {len(images)} test images per scene; association box scale "
          f"{', '.join(f'{a:g}x' for a in args.assoc)}")
    print("=" * 112)
    print(f"{'scene':<9s} {'assoc':>5s} | {'walkers: det':>12s} {'cover':>6s} {'confirmed':>10s} "
          f"{'ids/person':>10s} {'switch':>6s} {'moving':>7s} | "
          f"{'standing: det':>13s} {'cover':>6s} {'ids/person':>10s} {'false mv':>8s} | {'merged':>6s}")

    for name in args.scenes:
        for assoc in args.assoc:
            t = totals[name][assoc]
            w, s = t["walkers"], t["standing"]

            def share(a, b):
                return f"{a / b:.0%}" if b else "-"

            print(f"{name:<9s} {assoc:5.2g} | {share(w['detected'], w['frames']):>12s} "
                  f"{share(w['covered'], w['frames']):>6s} "
                  f"{w['confirmed']:4d}/{w['people']:<5d} "
                  f"{(w['identities'] / w['confirmed']) if w['confirmed'] else 0:10.2f} "
                  f"{w['switches']:6d} {share(w['moving_labels'], w['definite_labels']):>7s} | "
                  f"{share(s['detected'], s['frames']):>13s} {share(s['covered'], s['frames']):>6s} "
                  f"{(s['identities'] / s['confirmed']) if s['confirmed'] else 0:10.2f} "
                  f"{s['moving_labels']:8d} | {t['merged_ids']:6d}")

    print()
    print("det: share of in-view frames the detector alone found - the ceiling for cover.")
    print("cover: share of in-view frames reported by a confirmed track. ids/person: "
          "distinct IDs per confirmed person (1 is right).")
    print("switch: ID changes along a person's sightings. merged: track IDs reported on "
          "two different people.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "tracker_stress.json"
    out.write_text(json.dumps({s: {str(a): v for a, v in d.items()} for s, d in totals.items()},
                              indent=2), encoding="utf-8")
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
