"""
Real-time trial on a laptop camera, and what it can honestly tell you.

The hardware has not arrived, so the only live sensor available is the
laptop's webcam. That is worth running - but it is worth being precise about
what it tests, because the obvious reading of the result is the wrong one.

**The webcam is a visible-light camera. The deployed detector is thermal.**

MT-011b was trained entirely on 640 x 512 LWIR frames captured from 60-130 m.
It has never seen visible light. Pointed at a webcam it will detect almost
nothing, and that is not a defect - a thermal detector receiving RGB input is
being asked a question it was never built to answer.

The RGB model is no better matched. MT-004 was trained on VisDrone, where a
person is around 12 x 19 px seen from altitude. A face filling half the frame
at 60 cm is as far outside its domain as thermal is outside a webcam's.

So there are three things you can run, and they answer different questions:

    --mode pipeline   Stock COCO weights. Detects you reliably, so real
                      detections flow through tracking, movement
                      classification and geolocation. **Tests the system,
                      not our detector.** This is the useful demo.

    --mode rgb        MT-004, this project's RGB model. Right modality,
                      wrong viewpoint and scale. Expect little.

    --mode thermal    MT-011b, the deployed detector. Wrong modality
                      entirely. Expect nothing, and understand why.

What a webcam trial genuinely validates: that capture, inference, weighted box
fusion, ByteTrack association, ego-motion compensation, movement
classification and the JSONL emitter all run together on live input at a
measured frame rate, without stalling or leaking. Every one of those has been
tested on recorded files and none on a live camera.

What it cannot validate: detection accuracy, geolocation accuracy, or
behaviour at altitude. Those need the thermal camera and an airframe.

Usage:

    python scripts/webcam_trial.py
    python scripts/webcam_trial.py --mode thermal
    python scripts/webcam_trial.py --mode pipeline --jsonl out.jsonl
    python scripts/webcam_trial.py --camera 1 --seconds 30
"""

import argparse
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

MODES = {
    "pipeline": {
        "weights": "yolo26n.pt",
        "imgsz": [480, 640],
        "conf": 0.35,
        "label": "stock COCO weights",
        "expect": (
            "Should detect you reliably. This exercises the whole payload "
            "with real detections - it does NOT test this project's detector."
        ),
    },
    "rgb": {
        "model": "MT-004",
        "conf": 0.25,
        "label": "MT-004, this project's RGB model",
        "expect": (
            "Trained on aerial VisDrone imagery where a person is about "
            "12 x 19 px. A close-up at desk distance is far outside that "
            "domain, so expect few or no detections."
        ),
    },
    "thermal": {
        "model": "MT-011b",
        "conf": 0.10,
        "label": "MT-011b, the deployed thermal detector",
        "expect": (
            "Trained only on LWIR thermal. A webcam produces visible light, "
            "so expect essentially nothing. Running it is still informative: "
            "it shows concretely why the thermal camera is the binding "
            "purchase decision."
        ),
    },
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=list(MODES), default="pipeline")
    parser.add_argument("--camera", default="0", help="Camera index")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--seconds", type=float, default=None,
                        help="Stop after this many seconds")
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--save-video", default=None)
    parser.add_argument("--device", default="cpu",
                        help="cpu matches the target board; 0 uses the GPU")
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    mode = MODES[args.mode]

    print("=" * 74)
    print(f"Webcam trial - {args.mode}")
    print("=" * 74)
    print(f"detector   {mode['label']}")
    print(f"camera     index {args.camera}  (visible light)")
    print(f"device     {args.device}")
    print()
    print("What to expect:")

    for line in _wrap(mode["expect"], 68):
        print(f"  {line}")

    print()
    print("What this validates:")
    print("  capture -> detect -> fuse -> track -> movement -> JSONL,")
    print("  running together on live input at a measured frame rate.")
    print()
    print("What it does not:")
    print("  detection accuracy, geolocation accuracy, or anything about")
    print("  altitude. Those need the thermal camera and an airframe.")
    print()
    print("Geolocation is disabled - it needs MAVLink attitude and altitude,")
    print("and a laptop on a desk has neither.")
    print()
    print("Press q in the video window to stop.")
    print("=" * 74)
    print()

    argv = ["payload.py", "--source", str(args.camera), "--device", args.device]

    if "weights" in mode:
        weights = PROJECT_ROOT / mode["weights"]

        if not weights.exists():
            raise SystemExit(
                f"{weights} not found.\n"
                f"Stock weights download on first use: "
                f"python -c \"from ultralytics import YOLO; YOLO('yolo26n.pt')\""
            )

        argv += ["--weights", str(weights)]
        argv += ["--imgsz", str(mode["imgsz"][0]), str(mode["imgsz"][1])]

        # Stock COCO weights detect 80 classes; only person is relevant.
        argv += ["--classes", "0"]
    else:
        argv += ["--model", mode["model"]]

    argv += ["--conf", str(args.conf if args.conf is not None else mode["conf"])]

    if not args.no_show:
        argv += ["--show"]

    if args.jsonl:
        argv += ["--jsonl", args.jsonl]

    if args.save_video:
        argv += ["--save-video", args.save_video]

    if args.seconds:
        # payload.py counts frames, not seconds. 30 fps is the usual webcam
        # rate; the cap is a convenience, not a precise timer.
        argv += ["--max-frames", str(int(args.seconds * 30))]

    sys.argv = argv

    import payload

    payload.main()


def _wrap(text, width):
    words = text.split()
    lines, current = [], ""

    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()

    if current:
        lines.append(current)

    return lines


if __name__ == "__main__":
    main()
