"""
Movement classification when frames go missing - task B3.

The movement decision is displacement over the last 15 *observations* of a
track, against a 15 px threshold. Its history is counted in frames, not
seconds, so anything that changes how much time 15 frames span changes what
the threshold means: a companion computer slower than planned, a search-mode
frame rate, a camera that drops frames, or a detector that misses a person
for a few frames. None of this had been tested.

This replays the synthetic pan sequence through the real payload with frames
removed, and scores every movement label against the known answer. The
sequence (make_test_sequence.py) has one composited person on a known
trajectory, who moves, among 31 annotated people who do not.

Patterns:

  every k   keep one frame in k: the sequence at 15/k FPS, as a slower
            computer or search mode would see it
  random    each frame dropped independently with probability p
  burst     runs of consecutive frames lost, as from a stalled camera or bus

Scored per labelled detection - a detection with state moving or
stationary; edge and unknown are withheld labels, counted but not scored:

  mover     share of the moving person's labels that say moving, the
            seconds from its first sighting to its first "moving", and how
            many track identities seen more than once it was split across
            (ids; one is right)
  others    labels on stationary people that say moving - the error that
            sends a rescue team after a person who is not going anywhere
  margin    the smallest displacement among the mover's labels against the
            largest among stationary people's: the room the threshold has

Usage:

    python scripts/evaluate_frame_drops.py
    python scripts/evaluate_frame_drops.py --long
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)
SEQUENCE = PROJECT_ROOT / "results" / "test_sequence" / "1_60_80_0_00707_pan"
OUTPUT_DIR = PROJECT_ROOT / "results" / "frame_drops"

# The composited person's trajectory, from make_test_sequence.py: in
# viewport pixels, starting at (0.15 w, 0.55 h), moving 1.4 px/frame in x
# and a quarter of that in y.
MOVER_SPEED = 1.4
MOVER_RADIUS_PX = 12.0

PATTERNS = [
    ("every 1 (15 FPS)", "every", 1),
    ("every 2 (7.5 FPS)", "every", 2),
    ("every 3 (5 FPS)", "every", 3),
    ("every 5 (3 FPS)", "every", 5),
    ("every 8 (1.9 FPS)", "every", 8),
    ("every 11 (1.4 FPS)", "every", 11),
    ("random 20%", "random", 0.2),
    ("random 40%", "random", 0.4),
    ("burst 4 of 12", "burst", (4, 12)),
    ("burst 8 of 24", "burst", (8, 24)),
]


def kept_indices(kind, param, frames, seed=0):
    if kind == "every":
        return list(range(0, frames, param))

    if kind == "random":
        rng = np.random.default_rng(seed)
        keep = [i for i in range(frames) if i == 0 or rng.random() >= param]
        return keep

    lost, period = param
    return [i for i in range(frames) if i % period < period - lost]


def mover_centre(index, width, height):
    travel = MOVER_SPEED * index
    return (0.15 * width + travel, 0.55 * height + travel * 0.25)


def run(kind, param, truth, video, out_dir, seed=0):
    import payload
    import realtime_detect

    frames_total = truth["frames"]
    keep = kept_indices(kind, param, frames_total, seed)
    keep_set = set(keep)
    fps = truth["fps"] * len(keep) / frames_total

    real_open = realtime_detect.open_source

    def open_source(source):
        frames, _, size, _ = real_open(source)

        def generator():
            # Each frame carries its capture time, as a live camera's would,
            # so the payload can see the gaps.
            for i, frame in enumerate(frames):
                if i in keep_set:
                    yield frame, i / truth["fps"]

        return generator(), fps, size, len(keep)

    jsonl = out_dir / "run.jsonl"
    argv = sys.argv
    realtime_detect.open_source = open_source
    sys.argv = ["payload.py", "--source", str(video), "--device", "cpu",
                "--jsonl", str(jsonl), "--summary", str(out_dir / "summary.json"),
                "--progress", "100000"]

    try:
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            payload.main()
    finally:
        realtime_detect.open_source = real_open
        sys.argv = argv

    width, height = truth["viewport"]
    mover = {"moving": 0, "stationary": 0, "withheld": 0, "disp": [],
             "first_seen": None, "first_moving": None, "tracks": {},
             "sightings": 0, "unconfirmed": 0}
    others = {"moving": 0, "stationary": 0, "withheld": 0, "disp": [],
              "false_tracks": set()}

    for line in jsonl.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        source_index = keep[record["frame"]]
        mx, my = mover_centre(source_index, width, height)

        for det in record["detections"]:
            x1, y1, x2, y2 = det["box"]
            is_mover = np.hypot((x1 + x2) / 2 - mx, (y1 + y2) / 2 - my) <= MOVER_RADIUS_PX
            tally = mover if is_mover else others
            state = det["state"]

            if is_mover and mover["first_seen"] is None:
                mover["first_seen"] = source_index

            if is_mover:
                mover["sightings"] += 1
                mover["unconfirmed"] += det["track_id"] < 0

            if is_mover and det["track_id"] >= 0:
                mover["tracks"][det["track_id"]] = mover["tracks"].get(det["track_id"], 0) + 1

            if state in ("moving", "stationary"):
                tally[state] += 1
                tally["disp"].append(det.get("displacement_px", 0.0))
            else:
                tally["withheld"] += 1

            if state == "moving":
                if is_mover and mover["first_moving"] is None:
                    mover["first_moving"] = source_index
                if not is_mover and det["track_id"] >= 0:
                    others["false_tracks"].add(det["track_id"])

    labelled = mover["moving"] + mover["stationary"]
    latency = (None if mover["first_moving"] is None
               else (mover["first_moving"] - mover["first_seen"]) / truth["fps"])

    return {
        "frames_kept": len(keep),
        "fps": round(fps, 2),
        "mover_labels": labelled,
        "mover_sightings": mover["sightings"],
        "mover_unconfirmed": mover["unconfirmed"],
        "mover_tracks": sum(1 for n in mover["tracks"].values() if n > 1),
        "mover_stationary_labels": mover["stationary"],
        "mover_moving_share": round(mover["moving"] / labelled, 3) if labelled else None,
        "mover_latency_s": None if latency is None else round(latency, 2),
        "mover_min_disp_px": round(min(mover["disp"]), 1) if mover["disp"] else None,
        "others_labels": others["moving"] + others["stationary"],
        "others_false_moving": others["moving"],
        "others_false_tracks": len(others["false_tracks"]),
        "others_max_disp_px": round(max(others["disp"]), 1) if others["disp"] else None,
        "others_withheld": others["withheld"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3,
                        help="Repeats for the random patterns")
    parser.add_argument("--long", action="store_true",
                        help="Use a 600-frame version of the sequence, long "
                             "enough to reach labels at search-mode rates")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sequence = SEQUENCE

    if args.long:
        # The standard sequence is 120 frames. At one frame in 11 that is 11
        # processed frames - fewer than the 15 clean motion steps movement
        # labels wait for - so search-mode rates need a longer pass. Built
        # here rather than in results/test_sequence, which tests depend on.
        from types import SimpleNamespace

        import make_test_sequence as mts

        mts.OUTPUT_DIR = OUTPUT_DIR / "long"
        sequence = mts.OUTPUT_DIR / f"{mts.DEFAULT_IMAGE}_pan"

        if not sequence.with_suffix(".mp4").exists():
            mts.build(SimpleNamespace(
                image=mts.DEFAULT_IMAGE, frames=600, fps=mts.FPS,
                viewport=list(mts.VIEWPORT), rotate=1.5, moving_speed=1.4,
                no_moving_person=False))

    truth = json.loads(sequence.with_suffix(".json").read_text(encoding="utf-8"))
    video = sequence.with_suffix(".mp4")

    print("=" * 112)
    print(f"Movement labels under frame loss - {video.name}, "
          f"{truth['frames']} frames at {truth['fps']:.0f} FPS")
    print("=" * 112)
    print(f"{'pattern':<20s} {'kept':>5s} {'mover moving':>13s} {'latency':>8s} "
          f"{'seen':>5s} {'-1':>4s} {'ids':>4s} {'mover min':>10s} {'others max':>11s} {'false moving':>13s} {'tracks':>7s}")

    results = {}

    for name, kind, param in PATTERNS:
        seeds = range(args.seeds) if kind == "random" else [0]

        for seed in seeds:
            label = name if kind != "random" else f"{name} s{seed}"
            r = run(kind, param, truth, video, OUTPUT_DIR, seed)
            results[label] = r

            share = "-" if r["mover_moving_share"] is None else f"{r['mover_moving_share']:.0%} of {r['mover_labels']}"
            latency = "-" if r["mover_latency_s"] is None else f"{r['mover_latency_s']:.1f} s"
            mover_min = "-" if r["mover_min_disp_px"] is None else f"{r['mover_min_disp_px']:.1f}"
            others_max = "-" if r["others_max_disp_px"] is None else f"{r['others_max_disp_px']:.1f}"

            print(f"{label:<20s} {r['frames_kept']:5d} {share:>13s} {latency:>8s} "
                  f"{r['mover_sightings']:5d} {r['mover_unconfirmed']:4d} "
                  f"{r['mover_tracks']:4d} {mover_min:>10s} {others_max:>11s} "
                  f"{r['others_false_moving']:>5d} of {r['others_labels']:<5d} "
                  f"{r['others_false_tracks']:>5d}")

    print()
    print("seen: frames the mover was reported in; -1: of those, unconfirmed.")
    print("mover min / others max: displacement (px) over the history window; "
          "the threshold is 15 px.")

    name = "frame_drops_long.json" if args.long else "frame_drops.json"
    (OUTPUT_DIR / name).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Written: {OUTPUT_DIR / name}")


if __name__ == "__main__":
    main()
