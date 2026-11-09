"""
The ONNX Runtime detector must return exactly what the PyTorch one does.

Every companion-computer frame rate in this project was measured with ONNX
Runtime, and on the Raspberry Pi it is the engine the payload runs - so the
boxes it produces must be the boxes every accuracy figure was measured on.
"""

import cv2
import numpy as np
import pytest

import payload
from test_failure_injection import ROOT, needs_artifacts

ONNX = (ROOT / "runs" / "detect" / "results" / "training" / payload.DEFAULT_THERMAL
        / "weights" / "best.onnx")


def test_nms_keeps_the_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    scores = np.array([0.5, 0.9, 0.7])
    assert list(payload.nms(boxes, scores, 0.5)) == [1, 2]
    assert list(payload.nms(boxes, scores, 0.99)) == [1, 2, 0]


@needs_artifacts
@pytest.mark.skipif(not ONNX.exists(), reason="needs the exported best.onnx")
@pytest.mark.parametrize("conf", [0.25, 0.05])
def test_onnx_and_torch_detectors_agree(conf):
    images = sorted((ROOT / "dataset" / "hit_uav_person" / "images" / "test").glob("*.jpg"))[::40]
    torch_det = payload.Detector([payload.DEFAULT_THERMAL], device="cpu", conf=conf, backend="torch")
    onnx_det = payload.Detector([payload.DEFAULT_THERMAL], device="cpu", conf=conf, backend="onnx")

    for path in images:
        frame = cv2.imread(str(path))
        a_boxes, a_scores, _ = torch_det(frame)
        b_boxes, b_scores, _ = onnx_det(frame)

        assert len(a_boxes) == len(b_boxes)
        order_a, order_b = np.argsort(-a_scores), np.argsort(-b_scores)
        assert b_boxes[order_b] == pytest.approx(a_boxes[order_a], abs=0.05)
        assert b_scores[order_b] == pytest.approx(a_scores[order_a], abs=1e-3)
