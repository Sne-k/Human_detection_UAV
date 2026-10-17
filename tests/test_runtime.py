"""
The runtime's geometry and post-processing: each test pins a property that a
past bug violated, so the bug cannot come back unnoticed.
"""

import numpy as np
import pytest

import payload


# ------------------------------------------------------------- letterbox

@pytest.mark.parametrize("height,width", [
    (512, 640), (480, 640), (384, 512),      # every frame size used so far
    (481, 641), (250, 330), (333, 517),      # odd sizes, where it used to drift
])
def test_letterbox_round_trip_is_exact(height, width):
    """Forward and inverse share one geometry (training_log.md section 49)."""

    new_w, new_h, left, top = payload.letterbox_geometry(height, width, (512, 640))

    for x in (0.0, width * 0.37, width - 1.0):
        for y in (0.0, height * 0.61, height - 1.0):
            cx = x * (new_w / width) + left
            cy = y * (new_h / height) + top

            assert (cx - left) * (width / new_w) == pytest.approx(x, abs=1e-9)
            assert (cy - top) * (height / new_h) == pytest.approx(y, abs=1e-9)


def test_letterbox_output_shape():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert payload.letterbox(image, (512, 640)).shape == (512, 640, 3)


# ---------------------------------------------------------------- fusion

def _cluster():
    boxes = np.array([[10.0, 10, 22, 29], [11, 10, 23, 29], [12, 11, 24, 30]])
    scores = np.array([0.9, 0.5, 0.3])
    return boxes, scores, ["m", "m", "m"]


def test_single_model_fusion_keeps_the_seed_score():
    """The measured configuration kept the seed's score; the runtime averaged."""

    _, fused_scores, _ = payload.weighted_box_fusion(*_cluster(), 0.6, score="seed")
    assert fused_scores.tolist() == pytest.approx([0.9])


def test_ensemble_fusion_averages():
    _, fused_scores, _ = payload.weighted_box_fusion(*_cluster(), 0.6, score="mean")
    assert fused_scores.tolist() == pytest.approx([(0.9 + 0.5 + 0.3) / 3])


def test_fusion_rejects_unknown_score_rule():
    with pytest.raises(ValueError):
        payload.weighted_box_fusion(*_cluster(), 0.6, score="max")


def test_fused_box_is_confidence_weighted():
    boxes, scores, sources = _cluster()
    fused, _, _ = payload.weighted_box_fusion(boxes, scores, sources, 0.6)
    expected = (boxes * scores[:, None]).sum(axis=0) / scores.sum()
    assert fused[0] == pytest.approx(expected)


# --------------------------------------------------------------- tracker

@pytest.mark.parametrize("conf", [0.05, 0.10, 0.25, 0.35])
def test_tracker_thresholds_follow_the_operating_point(conf):
    """They were fixed at 0.25 whatever --conf said."""

    tracker = payload.build_tracker(7.0, conf)
    assert tracker.args.track_high_thresh == conf
    assert tracker.args.new_track_thresh == conf
    assert tracker.args.track_low_thresh <= conf


def test_default_tracker_thresholds_unchanged():
    tracker = payload.build_tracker(7.0, 0.25)
    assert (tracker.args.track_high_thresh, tracker.args.track_low_thresh,
            tracker.args.new_track_thresh) == (0.25, 0.1, 0.25)


# ------------------------------------------------- stabilised tracking

def test_transform_boxes_identity_and_round_trip():
    boxes = np.array([[100.0, 50, 112, 69], [300, 200, 330, 260]])
    assert payload.transform_boxes(boxes, np.eye(3)) == pytest.approx(boxes)

    theta = np.radians(7.0)
    m = np.array([[1.1 * np.cos(theta), -1.1 * np.sin(theta), 23.0],
                  [1.1 * np.sin(theta), 1.1 * np.cos(theta), -9.0],
                  [0.0, 0.0, 1.0]])
    back = payload.transform_boxes(payload.transform_boxes(boxes, m), np.linalg.inv(m))
    assert back == pytest.approx(boxes)


def _drift_track(drift_px, frames=12, stabilise=True):
    """A person drifting at drift_px per frame; returns emitted track count."""

    tracker = payload.build_tracker(7.0, 0.25)
    emitted = 0

    for t in range(frames):
        x = 100.0 + drift_px * t
        boxes = np.array([[x, 200.0, x + 12.0, 219.0]])
        scores = np.array([0.8])

        # Camera motion that exactly explains the drift: first frame -> now.
        cumulative = np.eye(3)
        cumulative[0, 2] = drift_px * t

        rows, _ = payload.track_frame(tracker, boxes, scores, None,
                                      cumulative if stabilise else None)
        emitted += len(rows)

    return emitted


def test_raw_pixel_tracking_collapses_at_flight_speed():
    """The failure found in section 49, pinned so the evidence stays."""

    assert _drift_track(20.0, stabilise=False) <= 2


def test_stabilised_tracking_holds_at_flight_speed():
    assert _drift_track(20.0, stabilise=True) >= 11


def test_singular_transform_falls_back_to_raw_pixels():
    tracker = payload.build_tracker(7.0, 0.25)
    boxes = np.array([[10.0, 10, 22, 29]])
    rows, world = payload.track_frame(tracker, boxes, np.array([0.9]), None,
                                      np.zeros((3, 3)))
    assert len(rows) == len(world)


# ----------------------------------------------------------------- misc

def test_default_model_is_the_deployed_one():
    """DEFAULT_THERMAL used to be defined and never used."""

    import inspect
    source = inspect.getsource(payload.main)
    assert "args.model or DEFAULT_THERMAL" in source
    assert payload.DEFAULT_THERMAL in payload.MODEL_SHAPES
