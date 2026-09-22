"""
Run a queue of experiments end to end, unattended.

Each experiment in this project needs the same four things after it trains:
test-split metrics, predictions at the operating threshold, the custom failure
-mechanism breakdown, and a comparison against the frozen MT-005 baseline.
Doing that by hand between runs wastes the hours the GPU is otherwise idle.

This driver chains them. It waits for any run already in progress, then for
each queued experiment trains it if it is not already on disk and evaluates it
either way. A failure in one step is logged and the queue continues, because
an unattended run that stops at the first problem wastes the whole night.

The queue is defined in QUEUE below. Each entry is a training configuration
expressed as a difference from MT-005; everything not named is inherited, so
the result stays attributable to the named change.

Usage:

    python scripts/run_experiment_queue.py
    python scripts/run_experiment_queue.py --only MT-011
    python scripts/run_experiment_queue.py --evaluate-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

SCRIPTS = Path(__file__).resolve().parent
PYTHON = sys.executable

TRAINING_ROOT = PROJECT_ROOT / "runs" / "detect" / "results" / "training"
ANALYSIS_ROOT = PROJECT_ROOT / "runs" / "detect" / "results" / "error_analysis"
DATASET = PROJECT_ROOT / "dataset" / "hit_uav_person"

LOG_DIR = PROJECT_ROOT / "results" / "experiment_queue"

DEPLOY_SHAPE = (512, 640)
CONF = 0.25
EPOCHS = 50

# Each entry is expressed as a difference from MT-005. Anything not named here
# is inherited from MT-005's configuration by train_experiment.py.
QUEUE = [
    {
        "name": "MT-010",
        "hypothesis": "Box-loss weight 15.0 corrects undersized boxes",
        "train": None,                     # already running or complete
    },
    {
        "name": "MT-011",
        "hypothesis": (
            "Two-stage aerial pretraining: VisDrone RGB weights transfer "
            "aerial human shape and scale to thermal"
        ),
        "train": [
            "--mosaic", "1.0",
            "--model", str(TRAINING_ROOT / "MT-004" / "weights" / "best.pt"),
        ],
    },
    {
        "name": "MT-013",
        "hypothesis": (
            "NWD localisation loss: an IoU-based loss carries no gradient "
            "once a near-miss box stops overlapping, which is 56% of the "
            "measured failures. NWD degrades smoothly instead"
        ),
        "train": ["--mosaic", "1.0", "--nwd", "0.5"],
    },
]

# Not queued automatically. architecture_budget.py measured the P2 head at
# 4.0-6.6 FPS against a binding 6.7 FPS requirement, so it cannot deploy on a
# bare Pi 5 at 60 m. The only question it can still answer is whether its
# accuracy would justify buying an accelerator - and that question is not live
# until the free techniques have been tried and failed.
#
# The criteria below are fixed before MT-011's result is known, so the
# decision is auditable rather than rationalised afterwards.
MT012 = {
    "name": "MT-012",
    "hypothesis": (
        "P2 stride-4 detection head gives small targets more resolution. "
        "Vetoed for deployment at 60 m by architecture_budget.py; would only "
        "run to establish whether the accuracy justifies an accelerator"
    ),
    "train": [
        "--mosaic", "1.0",
        "--model", "yolo26n-p2.yaml",
        "--transfer", "yolo26n.pt",
    ],
}

MT005_BASELINE = {"map50": 0.9330, "recall": 0.9288}

# A seed repeat of the one experiment that beat the baseline. MT-011 is now a
# deployment recommendation resting on a single training run, and its margin
# is 2,492 people against 2,470. Everything is identical except the seed, so
# the difference between MT-011 and MT-011b is run-to-run variation - the
# noise floor against which every other result in this project should be read.
MT011B = {
    "name": "MT-011b",
    "hypothesis": (
        "Seed repeat of MT-011. Measures how much of the accepted result is "
        "the change under test and how much is run-to-run variation"
    ),
    "train": [
        "--mosaic", "1.0",
        "--model", str(TRAINING_ROOT / "MT-004" / "weights" / "best.pt"),
        "--seed", "1",
    ],
}


def mt012_gate(summaries, stream):
    """
    Should MT-012 run at all?

    Three conditions, all of which must hold. Any one of them failing means
    the run cannot change a decision, and a run that cannot change a decision
    is not worth the GPU time.
    """

    log("", stream)
    log("=" * 70, stream)
    log("MT-012 decision gate", stream)
    log("=" * 70, stream)

    mt011 = next((s for s in summaries if s["experiment"] == "MT-011"), None)

    if not mt011 or "test" not in mt011:
        log("MT-011 produced no metrics. Cannot evaluate the gate, so MT-012",
            stream)
        log("does not run - an unexplained failure is not evidence for", stream)
        log("spending compute on a model that cannot deploy anyway.", stream)
        return False

    gain = mt011["test"]["map50"] - MT005_BASELINE["map50"]

    log(f"MT-011 mAP@50 {mt011['test']['map50']:.4f} vs baseline "
        f"{MT005_BASELINE['map50']:.4f}  ({gain:+.4f})", stream)
    log("", stream)

    # 1. A free technique that already works makes a costly one redundant.
    condition_1 = gain < 0.005
    log(f"  1. MT-011 did NOT already solve it (gain < +0.5 pts): "
        f"{'PASS' if condition_1 else 'FAIL'}", stream)

    if not condition_1:
        log("     MT-011 gained more than half a point at zero inference cost.",
            stream)
        log("     A 1.3-1.5x technique has nothing left to justify.", stream)

    # 2. NWD targets the same failure mode for free, so it goes first.
    #    P2 is only worth measuring once the free option has been exhausted.
    nwd_done = (TRAINING_ROOT / "MT-013" / "weights" / "best.pt").exists()
    log(f"  2. NWD (MT-013) already tried and insufficient: "
        f"{'PASS' if nwd_done else 'FAIL'}", stream)

    if not nwd_done:
        log("     NWD attacks the same near-miss localisation failure at zero", stream)
        log("     inference cost, so it strictly dominates a P2 head. Spending", stream)
        log("     compute on the dominated option first is the wrong order.", stream)

    # 3. It must be deployable on something. On a bare Pi 5 it is not.
    log("  3. An accelerator is under consideration: UNKNOWN - project "
        "decision, not a measurement", stream)

    verdict = condition_1 and nwd_done

    log("", stream)
    log(f"Verdict: MT-012 {'RUNS' if verdict else 'does NOT run'}", stream)

    if not verdict:
        log("", stream)
        log("This is not a refusal to test the idea. P2 remains the most", stream)
        log("-supported technique in the surveyed literature and stays on the", stream)
        log("list as a contingency if an accelerator is ever bought. It is", stream)
        log("simply not the next thing worth an hour of GPU time.", stream)

    return verdict


def log(message, stream=None):
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"

    print(line, flush=True)

    if stream is not None:
        stream.write(line + "\n")
        stream.flush()


def epochs_done(name):
    csv = TRAINING_ROOT / name / "results.csv"

    if not csv.exists():
        return 0

    lines = [l for l in csv.read_text(encoding="utf-8").splitlines() if l.strip()]

    return max(0, len(lines) - 1)


def training_complete(name):
    return (
        (TRAINING_ROOT / name / "weights" / "best.pt").exists()
        and epochs_done(name) >= EPOCHS
    )


def wait_for(name, stream, poll=60):
    """Block until an in-progress run finishes writing its weights."""

    if not (TRAINING_ROOT / name).exists():
        return

    while not training_complete(name):
        done = epochs_done(name)
        log(f"{name}: waiting, {done}/{EPOCHS} epochs", stream)
        time.sleep(poll)

    log(f"{name}: training complete", stream)


def run(command, stream, label):
    log(f"$ {label}", stream)

    env = dict(os.environ)
    env["HDU_ROOT"] = str(PROJECT_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        command, env=env, cwd=str(SCRIPTS.parent),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )

    output = (result.stdout or "") + (result.stderr or "")

    stream.write(output + "\n")
    stream.flush()

    if result.returncode != 0:
        log(f"  FAILED (exit {result.returncode}) - see log", stream)
        for line in output.strip().splitlines()[-12:]:
            log(f"  | {line}", stream)
        return False

    return True


def train(entry, stream):
    name = entry["name"]

    if entry["train"] is None:
        return True

    if training_complete(name):
        log(f"{name}: already trained, skipping", stream)
        return True

    if (TRAINING_ROOT / name).exists():
        log(f"{name}: partial run present, waiting for it", stream)
        wait_for(name, stream)
        return training_complete(name)

    log(f"{name}: training - {entry['hypothesis']}", stream)

    return run(
        [PYTHON, str(SCRIPTS / "train_experiment.py"), "--name", name] + entry["train"],
        stream, f"train {name}",
    )


def evaluate(entry, stream):
    """Test metrics, predictions at the operating point, mechanism breakdown."""

    name = entry["name"]
    weights = TRAINING_ROOT / name / "weights" / "best.pt"

    if not weights.exists():
        log(f"{name}: no weights, skipping evaluation", stream)
        return None

    from ultralytics import YOLO

    summary = {"experiment": name, "hypothesis": entry["hypothesis"]}

    # --- Test-split metrics -------------------------------------------
    log(f"{name}: test-split metrics", stream)

    try:
        metrics = YOLO(str(weights)).val(
            data=str(DATASET / "data.yaml"),
            split="test", imgsz=DEPLOY_SHAPE, batch=8,
            verbose=False, plots=False,
            project=str(PROJECT_ROOT / "results" / "experiment_queue"),
            name=f"{name}-val", exist_ok=True,
        )

        summary["test"] = {
            "map50": round(float(metrics.box.map50), 4),
            "map50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        }

        log(f"  mAP@50 {summary['test']['map50']:.4f}  "
            f"P {summary['test']['precision']:.4f}  "
            f"R {summary['test']['recall']:.4f}", stream)
    except Exception as exc:
        log(f"  metrics FAILED: {exc}", stream)

    # --- Predictions at the operating threshold -----------------------
    analysis = f"{name}-test-analysis"

    if not (ANALYSIS_ROOT / analysis / "labels").exists():
        log(f"{name}: predictions at conf {CONF}", stream)

        try:
            YOLO(str(weights)).predict(
                source=str(DATASET / "images" / "test"),
                imgsz=DEPLOY_SHAPE, conf=CONF,
                save=False, save_txt=True, save_conf=True,
                project=str(ANALYSIS_ROOT), name=analysis,
                exist_ok=True, verbose=False, stream=False,
            )
        except Exception as exc:
            log(f"  predictions FAILED: {exc}", stream)
    else:
        log(f"{name}: predictions already present", stream)

    # --- Failure-mechanism breakdown ----------------------------------
    if (ANALYSIS_ROOT / analysis / "labels").exists():
        run([PYTHON, str(SCRIPTS / "evaluate_experiment.py"),
             "--labels", analysis, "--label", name],
            stream, f"evaluate_experiment {name}")

        # --- The mission-derived criterion, alongside the conventional one
        run([PYTHON, str(SCRIPTS / "operational_criterion.py"),
             "--labels", analysis],
            stream, f"operational_criterion {name}")

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--evaluate-only", action="store_true")

    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"queue-{stamp}.log"

    queue = QUEUE

    if args.only:
        # MT-012 is held outside QUEUE behind its gate, but naming it
        # explicitly is a deliberate override of that gate.
        queue = [e for e in (QUEUE + [MT012, MT011B]) if e["name"] in args.only]

    summaries = []

    with log_path.open("w", encoding="utf-8") as stream:
        log("=" * 70, stream)
        log(f"Experiment queue: {', '.join(e['name'] for e in queue)}", stream)
        log(f"Log: {log_path}", stream)
        log("=" * 70, stream)

        for entry in queue:
            name = entry["name"]

            log("", stream)
            log("-" * 70, stream)
            log(f"{name}", stream)
            log("-" * 70, stream)

            if not args.evaluate_only:
                if entry["train"] is None:
                    wait_for(name, stream)
                elif not train(entry, stream):
                    log(f"{name}: training failed, continuing to next", stream)
                    continue

            summary = evaluate(entry, stream)

            if summary:
                summaries.append(summary)

                (LOG_DIR / "queue_results.json").write_text(
                    json.dumps(summaries, indent=2), encoding="utf-8"
                )

        # --- Comparison table -----------------------------------------
        log("", stream)
        log("=" * 70, stream)
        log("Queue complete", stream)
        log("=" * 70, stream)

        header = (f"{'Experiment':<12s} {'mAP@50':>9s} {'mAP@50-95':>11s} "
                  f"{'precision':>11s} {'recall':>9s}")
        log(header, stream)
        log("-" * len(header), stream)

        for summary in summaries:
            test = summary.get("test")

            if not test:
                log(f"{summary['experiment']:<12s} {'(no metrics)':>9s}", stream)
                continue

            log(f"{summary['experiment']:<12s} {test['map50']:9.4f} "
                f"{test['map50_95']:11.4f} {test['precision']:11.4f} "
                f"{test['recall']:9.4f}", stream)

        log("", stream)
        log("MT-005 frozen baseline: mAP@50 0.9330, recall 0.9288 at IoU 0.50",
            stream)
        log(f"Results: {LOG_DIR / 'queue_results.json'}", stream)

        # --- Should MT-012 run? ---------------------------------------
        if not args.only and not args.evaluate_only:
            if mt012_gate(summaries, stream):
                log("", stream)
                log("-" * 70, stream)
                log("MT-012", stream)
                log("-" * 70, stream)

                if train(MT012, stream):
                    summary = evaluate(MT012, stream)

                    if summary:
                        summaries.append(summary)

                        (LOG_DIR / "queue_results.json").write_text(
                            json.dumps(summaries, indent=2), encoding="utf-8"
                        )


if __name__ == "__main__":
    main()
