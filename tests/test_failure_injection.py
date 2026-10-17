"""
Failure injection - task B2.

The ICD's requirements for a failing payload: it must never affect flight
control, and losing storage must not stop detection or telemetry. The first
holds by construction - the MAVLink link is read-only. The rest are tested
here by breaking things on purpose and checking the run survives.

Every fault here used to end the run. A search payload that loses the rest of
a mission to one corrupt frame, one exception or a full disk has chosen the
wrong failure mode.

The runtime tests drive the real payload in-process and need the deployed
weights and the synthetic test video. Where those are absent - a fresh clone,
a CI runner - they skip, and the fast suite still runs.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import payload
import realtime_detect

ROOT = Path(os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1]))
SEQUENCE = ROOT / "results" / "test_sequence"
WEIGHTS = ROOT / "runs" / "detect" / "results" / "training" / payload.DEFAULT_THERMAL / "weights" / "best.pt"

needs_artifacts = pytest.mark.skipif(
    not (WEIGHTS.exists() and (SEQUENCE / "ego_on.mp4").exists()),
    reason="needs the deployed weights and results/test_sequence (set HDU_ROOT)",
)


def run_payload(tmp_path, monkeypatch, *extra, frames=40):
    """Run payload.main on the test sequence; return the summary."""

    import sys

    monkeypatch.setenv("HDU_ROOT", str(ROOT))
    summary = tmp_path / "summary.json"
    argv = ["payload.py", "--source", str(SEQUENCE / "ego_on.mp4"),
            "--device", "cpu", "--conf", "0.10", "--max-frames", str(frames),
            "--summary", str(summary), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    payload.main()
    return json.loads(summary.read_text(encoding="utf-8"))


def faulty_source(every, fault):
    """Wrap open_source so every Nth frame is replaced by `fault(frame)`."""

    real = realtime_detect.open_source

    def open_source(source):
        frames, fps, size, total = real(source)

        def generator():
            for i, frame in enumerate(frames):
                yield fault(frame) if i % every == every - 1 else frame

        return generator(), fps, size, total

    return open_source


@needs_artifacts
@pytest.mark.parametrize("name,fault", [
    ("none", lambda f: None),
    ("empty", lambda f: np.zeros((0, 0, 3), np.uint8)),
    ("grayscale", lambda f: f[:, :, 0]),
    ("wrong size", lambda f: f[:100, :100]),
])
def test_corrupt_frames_are_skipped(tmp_path, monkeypatch, name, fault):
    monkeypatch.setattr(realtime_detect, "open_source", faulty_source(5, fault))
    summary = run_payload(tmp_path, monkeypatch)
    assert summary["bad_frames_skipped"] == 8
    assert summary["frames"] == 40


@needs_artifacts
def test_inference_exception_costs_one_frame(tmp_path, monkeypatch):
    real_call = payload.Detector.__call__
    calls = {"n": 0}

    def flaky(self, frame):
        calls["n"] += 1
        if calls["n"] % 7 == 0:
            raise RuntimeError("injected inference failure")
        return real_call(self, frame)

    monkeypatch.setattr(payload.Detector, "__call__", flaky)
    summary = run_payload(tmp_path, monkeypatch)
    assert summary["frame_errors"] == 40 // 7
    assert summary["frames"] == 40


@needs_artifacts
def test_full_disk_stops_the_stream_not_detection(tmp_path, monkeypatch):
    class FullDisk:
        def __init__(self):
            self.writes = 0

        def write(self, text):
            self.writes += 1
            if self.writes > 10:
                raise OSError(28, "No space left on device")

        def flush(self):
            pass

        def close(self):
            pass

    class DiskPath(type(Path())):
        def open(self, *args, **kwargs):
            if self.suffix == ".jsonl":
                return FullDisk()
            return super().open(*args, **kwargs)

    monkeypatch.setattr(payload, "Path", DiskPath)
    summary = run_payload(tmp_path, monkeypatch, "--jsonl", str(tmp_path / "out.jsonl"))
    assert summary["stream_failed"] and "No space left" in summary["stream_failed"]
    assert summary["frames"] == 40


@needs_artifacts
def test_telemetry_stopping_gives_no_coordinates_not_wrong_ones(tmp_path, monkeypatch):
    """Telemetry for the first half only: later frames get uav null, no fix."""

    lines = (SEQUENCE / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    truncated = tmp_path / "half.jsonl"
    truncated.write_text("\n".join(lines[:20]), encoding="utf-8")
    out = tmp_path / "out.jsonl"

    summary = run_payload(tmp_path, monkeypatch, "--telemetry", str(truncated),
                          "--jsonl", str(out))

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    late = [r for r in records if r["frame"] >= 25]

    assert summary["frames_without_valid_state"] > 0
    assert late and all(r["uav"] is None for r in late)
    assert all(d.get("ground") is None for r in late for d in r["detections"])


def test_camera_dropout_reconnects(monkeypatch):
    """No weights needed: a fake camera that drops out, then comes back."""

    class FakeCapture:
        opened = 0

        def __init__(self, index):
            FakeCapture.opened += 1
            self.reads = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return 0

        def read(self):
            self.reads += 1
            # The first device dies after 3 frames; the second delivers 5 and
            # dies; every later one opens but never delivers - the wedged
            # camera that must not be retried forever.
            if FakeCapture.opened == 1 and self.reads > 3:
                return False, None
            if FakeCapture.opened == 2 and self.reads > 5:
                return False, None
            if FakeCapture.opened > 2:
                return False, None
            return True, np.zeros((4, 4, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(realtime_detect.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(realtime_detect.time, "sleep", lambda s: None)
    monkeypatch.setattr(realtime_detect, "RECONNECT_S", 0.5)

    frames, *_ = realtime_detect.open_source("0")
    received = sum(1 for _ in frames)

    # 3 before the dropout, 5 after reconnecting; then devices that open but
    # never deliver, which must end the run once RECONNECT_S passes without a
    # frame rather than loop forever.
    assert received == 8
    assert FakeCapture.opened > 2


def test_file_end_is_not_a_fault(tmp_path):
    """Reaching the end of a recording ends the run; it must not reconnect."""

    import cv2

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    for _ in range(4):
        writer.write(np.zeros((32, 32, 3), np.uint8))
    writer.release()

    frames, *_ = realtime_detect.open_source(str(path))
    assert sum(1 for _ in frames) == 4
