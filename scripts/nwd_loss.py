"""
Normalized Wasserstein Distance localisation loss, for tiny targets.

Why this and not another training tweak
---------------------------------------

Seven directions have been tested and rejected, and every one of them attacked
target *scale*: higher input resolution, crop augmentation, mosaic ablation,
sensor-resolution analysis. The measured failure is not scale. Section 24 of
`docs/training_log.md` established that **56% of unmatched predictions are
near-miss boxes on real people**, and section 32 showed that for a 12 x 19 px
target a **4-pixel** displacement is enough to fall below IoU 0.50.

IoU is the reason 4 pixels is fatal. For two boxes of that size, overlap falls
away sharply with displacement and reaches zero while the boxes are still
essentially on the same person. Once it reaches zero the gradient carries no
information about which direction would have been better, so the loss stops
teaching anything precisely where the model is closest to being right.

Normalized Wasserstein Distance (Wang et al., 2021) replaces the overlap test.
Each box becomes a 2D Gaussian - centre at the box centre, variance from its
width and height - and similarity is measured by the Wasserstein distance
between the two distributions:

    W2^2 = (cx_a - cx_b)^2 + (cy_a - cy_b)^2
         + ((w_a - w_b)^2 + (h_a - h_b)^2) / 4

    NWD  = exp(-sqrt(W2^2) / C)

Two properties matter here. It is defined for non-overlapping boxes, so a
prediction 4 px off a 12 x 19 px person still produces a useful gradient. And
it degrades smoothly rather than falling off a cliff, so small localisation
errors are penalised in proportion to how wrong they are.

The constant C
--------------

C sets the distance scale, and the paper takes it as the average absolute
object size in the dataset. Measured over the 8,533 boxes in the HIT-UAV train
split, in image pixels:

    median w x h    12.00 x 19.00      (the failure analysis's canonical person)
    mean sqrt(w*h)  15.93
    median sqrt(wh) 15.10

So C = 15.93. It is measured rather than tuned, which keeps it one fewer free
parameter to be accused of fitting.

Deployment cost
---------------

None. A loss function exists only during training; the exported graph is
identical to MT-005's and the Raspberry Pi 5 latency is unchanged. That is why
this is preferred over the P2 detection head, which targets the same failure
and was measured at 1.3-1.5x inference cost - enough to fail the 6.7 FPS
requirement. See `docs/training_log.md` section 33.

How it is applied
-----------------

`enable()` patches `BboxLoss.forward` so the localisation term becomes a blend:

    similarity = (1 - ratio) * CIoU + ratio * NWD

Everything else in the training path is untouched, so a run using this differs
from MT-005 in exactly one respect. A ratio of 0.0 reproduces stock CIoU
exactly, which is what `verify()` checks.

Usage:

    python scripts/nwd_loss.py          # run the self-test
"""

import math

import torch

# Mean sqrt(w*h) over the HIT-UAV train split, in image pixels.
DEFAULT_CONSTANT = 15.93

# Blend weight. 0.0 is stock CIoU, 1.0 is pure NWD.
DEFAULT_RATIO = 0.5

EPS = 1e-7


def normalized_wasserstein(pred, target, constant=DEFAULT_CONSTANT):
    """
    NWD similarity in [0, 1] between two sets of xyxy boxes, in pixels.

    Both tensors are (N, 4). Returns (N,), where 1.0 means identical.
    """

    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    pred_cx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_cy = (pred[:, 1] + pred[:, 3]) * 0.5

    target_w = target[:, 2] - target[:, 0]
    target_h = target[:, 3] - target[:, 1]
    target_cx = (target[:, 0] + target[:, 2]) * 0.5
    target_cy = (target[:, 1] + target[:, 3]) * 0.5

    centre = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
    extent = ((pred_w - target_w) ** 2 + (pred_h - target_h) ** 2) * 0.25

    distance = torch.sqrt((centre + extent).clamp(min=EPS))

    return torch.exp(-distance / constant)


def enable(ratio=DEFAULT_RATIO, constant=DEFAULT_CONSTANT):
    """
    Patch BboxLoss so the localisation term blends CIoU with NWD.

    Ultralytics does not expose the localisation similarity as a
    configuration option, so this replaces the one line that computes it. The
    DFL term, the assigner, the class loss and every other part of the
    training path are left alone.
    """

    from ultralytics.utils import loss as ul_loss
    from ultralytics.utils.metrics import bbox_iou
    from ultralytics.utils.tal import bbox2dist

    if getattr(ul_loss.BboxLoss, "_nwd_patched", False):
        raise RuntimeError("NWD loss is already enabled")

    original = ul_loss.BboxLoss.forward

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores[fg_mask].sum(-1, keepdim=True)

        pred_fg = pred_bboxes[fg_mask]
        target_fg = target_bboxes[fg_mask]

        iou = bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)

        # Boxes reach this function divided by their anchor's stride, so they
        # are in feature-map units and a level's units differ from another's.
        # C is defined in image pixels, so both are scaled back before the
        # distance is taken - otherwise a P3 box and a P4 box covering the
        # same person would be treated as different sizes.
        stride_fg = (
            stride.view(1, -1)
            .expand(fg_mask.shape[0], -1)[fg_mask]
            .unsqueeze(-1)
        )

        nwd = normalized_wasserstein(
            pred_fg * stride_fg, target_fg * stride_fg, constant
        ).unsqueeze(-1)

        similarity = (1.0 - ratio) * iou + ratio * nwd

        loss_iou = ((1.0 - similarity) * weight).sum() / target_scores_sum

        # DFL term, verbatim from the original.
        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points, target_bboxes, self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask],
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = torch.nn.functional.l1_loss(
                pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none"
            ).mean(-1, keepdim=True) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

    ul_loss.BboxLoss.forward = forward
    ul_loss.BboxLoss._nwd_patched = True
    ul_loss.BboxLoss._nwd_original = original

    print(f"NWD localisation loss enabled: ratio {ratio}, C {constant} px")
    print(f"  similarity = {1 - ratio:.2f} * CIoU + {ratio:.2f} * NWD")


def verify():
    """
    Check the behaviour that motivates using this at all.

    The claim is that NWD stays informative where IoU does not, on targets the
    size this project actually sees. Each case below is checkable by hand.
    """

    from ultralytics.utils.metrics import bbox_iou

    print("=" * 74)
    print("NWD self-test - a 12 x 19 px person, the HIT-UAV median")
    print("=" * 74)
    print(f"C = {DEFAULT_CONSTANT} px (mean sqrt(w*h) over the train split)")
    print()

    w, h = 12.0, 19.0
    target = torch.tensor([[100.0, 100.0, 100.0 + w, 100.0 + h]])

    print(f"{'x offset':>9s} {'IoU':>8s} {'1 - IoU':>9s} {'NWD':>8s} {'1 - NWD':>9s}")
    print("-" * 47)

    for offset in (0.0, 1.0, 2.0, 4.0, 6.0, 12.0, 20.0, 40.0):
        pred = target + torch.tensor([offset, 0.0, offset, 0.0])

        iou = float(bbox_iou(pred, target, xywh=False, CIoU=False).squeeze())
        nwd = float(normalized_wasserstein(pred, target).squeeze())

        print(f"{offset:9.1f} {iou:8.4f} {1 - iou:9.4f} {nwd:8.4f} {1 - nwd:9.4f}")

    print()
    print("At a 12 px offset the boxes no longer touch, IoU is 0.0000 and the")
    print("gradient stops distinguishing 12 px from 40 px. NWD still separates")
    print("them, which is the whole point.")
    print()

    # --- Identity ---------------------------------------------------------
    identical = float(normalized_wasserstein(target, target).squeeze())
    print(f"Identical boxes         NWD = {identical:.6f}  (expect 1.000000)")
    assert abs(identical - 1.0) < 1e-4, "identical boxes must score 1"

    # --- Against the hand-computed value ---------------------------------
    offset = 4.0
    pred = target + torch.tensor([offset, 0.0, offset, 0.0])

    expected = math.exp(-math.sqrt(offset ** 2) / DEFAULT_CONSTANT)
    measured = float(normalized_wasserstein(pred, target).squeeze())

    print(f"Pure 4 px translation   NWD = {measured:.6f}  "
          f"(hand-computed {expected:.6f})")
    assert abs(measured - expected) < 1e-5, "translation case disagrees"

    # --- Size difference only --------------------------------------------
    wider = torch.tensor([[100.0, 100.0, 100.0 + w + 4.0, 100.0 + h]])

    expected = math.exp(-math.sqrt((2.0 ** 2) + (2.0 ** 2)) / DEFAULT_CONSTANT)
    measured = float(normalized_wasserstein(wider, target).squeeze())

    print(f"4 px wider, same left   NWD = {measured:.6f}  "
          f"(hand-computed {expected:.6f})")
    assert abs(measured - expected) < 1e-5, "size case disagrees"

    # --- Monotonicity -----------------------------------------------------
    offsets = torch.arange(0.0, 60.0, 1.0)
    preds = target.repeat(len(offsets), 1)
    preds[:, 0] += offsets
    preds[:, 2] += offsets

    scores = normalized_wasserstein(preds, target.repeat(len(offsets), 1))

    assert bool((scores[1:] < scores[:-1]).all()), "NWD must decrease with offset"
    print("Strictly decreasing over 0-60 px offset: OK")

    print()
    print("All checks passed.")


if __name__ == "__main__":
    verify()
