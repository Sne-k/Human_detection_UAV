"""
Telemetry matching, geolocation and the output contract.
"""

import json
import math
import threading
import time

import numpy as np
import pytest

import geolocate
import telemetry
import validate_jsonl


# ------------------------------------------------------------- telemetry

def _buffer():
    b = telemetry.StateBuffer()
    b.add_position(0.0, 10.0, 20.0, 100.0)
    b.add_position(0.2, 10.0002, 20.0, 102.0)
    b.add_attitude(0.0, 0.0, 0.0, 350.0)
    b.add_attitude(0.1, 10.0, 0.0, 10.0)
    return b


def test_interpolates_and_unwraps_yaw_through_north():
    s = _buffer().at(0.05)
    assert s["roll_deg"] == pytest.approx(5.0)
    assert s["altitude_m"] == pytest.approx(100.5)
    assert min(s["yaw_deg"], 360.0 - s["yaw_deg"]) == pytest.approx(0.0, abs=1e-9)


def test_refuses_before_first_sample():
    assert _buffer().at(-0.01) is None


def test_refuses_once_the_link_has_stalled():
    assert _buffer().at(0.1 + telemetry.HOLD_ATTITUDE_S + 0.01) is None


def test_refuses_across_a_dropout():
    b = telemetry.StateBuffer()
    for t in (0.0, 1.0):
        b.add_position(t, 1, 1, 1)
        b.add_attitude(t, 0, 0, 0)
    assert b.at(0.5) is None


def test_ready_only_once_newer_telemetry_exists():
    b = _buffer()
    assert b.ready(0.09)
    assert not b.ready(0.15)  # attitude ends at 0.1


def test_clock_aligner_rejects_queuing_delay():
    clock = telemetry.ClockAligner()
    for k in range(50):
        boot = k * 0.1
        clock.observe(boot, 1000.0 + boot + 0.004 + (0.03 if k % 3 else 0.0))
    assert clock.offset == pytest.approx(1000.004)


def test_load_jsonl_reads_frame_indexed_records(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps({
        "frame": f, "lat": 1.0, "lon": 2.0, "altitude_m": 100.0,
        "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0}) for f in range(3)))
    b = telemetry.load_jsonl(path, fps=10.0)
    assert b.at(0.1)["state_age_s"] == 0.0


def test_mavlink_on_the_wire():
    """Real packets over loopback UDP, with link delay and a foreign clock."""

    import mavlink_sim

    port = 14571
    trajectory = mavlink_sim.Trajectory(duration_s=10.0)
    buffer = telemetry.StateBuffer()
    receiver = telemetry.MavlinkReceiver(f"udpin:127.0.0.1:{port}", buffer).start()
    time.sleep(0.3)

    result = {}
    sender = threading.Thread(target=lambda: result.update(zero=mavlink_sim.send(
        f"udpout:127.0.0.1:{port}", trajectory, 3.0, log=lambda *_: None)))
    sender.start()
    sender.join()
    time.sleep(0.2)
    receiver.stop()

    assert receiver.counts["ATTITUDE"] >= 25
    assert receiver.counts["GLOBAL_POSITION_INT"] >= 12

    # Bounds allow ~20 ms of clock-alignment error, which scheduler jitter on
    # a loaded machine can cause (0.11 m and 0.1 deg are only ~5 ms at 20 m/s
    # and 25 deg/s, and flaked). Using the latest message instead of matching
    # by time is off by up to 6 m and 3.4 deg, so this still catches it.
    for t_flight in (1.0, 1.55, 2.2):
        state = buffer.at(result["zero"] + t_flight)
        truth = trajectory.at(t_flight)
        assert state is not None
        assert state["lat"] == pytest.approx(truth["lat"], abs=4.5e-6)  # 0.5 m
        assert state["roll_deg"] == pytest.approx(truth["roll_deg"], abs=0.5)


# ------------------------------------------------------------- geolocation

@pytest.fixture
def camera():
    return geolocate.Camera(width=640, height=512, hfov_deg=50.0)


def _ground(camera, px, py, alt=100.0, roll=0.0, pitch=0.0, yaw=0.0):
    return geolocate.project_to_ground(camera, px, py, alt, roll, pitch, yaw)[:2]


def test_conventions_heading(camera):
    east, north = _ground(camera, 620, 256)
    assert east > 0 and abs(north) < 1e-6          # image right is east
    east, north = _ground(camera, 620, 256, yaw=90.0)
    assert north < 0 and abs(east) < 1e-6          # heading east: right is south


def test_conventions_roll_and_pitch(camera):
    """Right wing down swings the belly left; nose up points it forward."""

    east, _ = _ground(camera, 320, 256, roll=10.0)
    assert east == pytest.approx(-100.0 * math.tan(math.radians(10.0)))
    _, north = _ground(camera, 320, 256, pitch=10.0)
    assert north == pytest.approx(100.0 * math.tan(math.radians(10.0)))


@pytest.mark.parametrize("altitude,px,py", [
    (60.0, 320, 256), (100.0, 620, 500), (130.0, 620, 500)])
def test_reported_error_bounds_the_median(camera, altitude, px, py):
    """The ICD acceptance criterion, simulated: median error within reported."""

    rng = np.random.default_rng(1)
    state = {"altitude_m": altitude, "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0}
    reported = geolocate.propagate_error(camera, px, py, state)
    truth = np.array(_ground(camera, px, py, altitude))

    errors = []
    for _ in range(800):
        d = rng.normal(0, 1, 6)
        fix = np.array(_ground(camera, px, py, altitude + 2.0 * d[0],
                               roll=d[1], pitch=d[2], yaw=d[3]))
        errors.append(np.linalg.norm(fix + 2.5 * d[4:6] - truth))

    assert np.median(errors) <= reported


def test_no_uncertainty_means_no_coordinate():
    """
    A ray just below the horizon hits the ground, but its one-degree pitch
    perturbation does not - so no uncertainty exists, and the ICD forbids a
    coordinate without one.
    """

    horizon = geolocate.Camera(width=640, height=512, hfov_deg=50.0,
                               mount_pitch_deg=0.0)
    state = {"lat": 1.0, "lon": 2.0, "altitude_m": 100.0,
             "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0}

    # About 0.5 degrees below the horizon, at the bottom edge of a box.
    assert geolocate.project_to_ground(horizon, 320, 262, 100.0, 0, 0, 0) is not None
    assert geolocate.propagate_error(horizon, 320, 262, state) is None
    assert geolocate.locate_detection(horizon, [314, 240, 326, 262], state) is None


# ---------------------------------------------------------------- contract

GOOD = {"frame": 3, "timestamp_s": 0.2, "persons": 1,
        "uav": {"lat": 1.0, "lon": 2.0, "altitude_m": 100.0, "state_age_s": 0.05},
        "detections": [{"track_id": 4, "box": [1.0, 2.0, 13.0, 21.0],
                        "confidence": 0.7, "state": "unknown",
                        "ground": {"lat": 1.0, "lon": 2.0, "east_m": 0.0,
                                   "north_m": 0.0, "slant_range_m": 100.0,
                                   "position_error_m": 4.3}}]}


def test_valid_record_passes():
    errors, warnings = validate_jsonl.validate_record(GOOD)
    assert errors == [] and warnings == []


@pytest.mark.parametrize("mutate,fragment", [
    (lambda r: r.update(persons=2), "persons is 2"),
    (lambda r: r["detections"][0].update(track_id=0), "track_id"),
    (lambda r: r["detections"][0].update(box=[5, 5, 3, 9]), "x1 < x2"),
    (lambda r: r["detections"][0].update(confidence=1.4), "confidence"),
    (lambda r: r["detections"][0].update(state="walking"), "state"),
    (lambda r: r["detections"][0]["ground"].pop("position_error_m"), "position_error_m"),
    (lambda r: r.update(uav=None), "no aircraft state"),
])
def test_contract_violations_are_caught(mutate, fragment):
    record = json.loads(json.dumps(GOOD))
    mutate(record)
    errors, _ = validate_jsonl.validate_record(record)
    assert any(fragment in e for e in errors), errors


def test_frames_must_increase():
    later = json.loads(json.dumps(GOOD))
    errors, _ = validate_jsonl.validate_record(GOOD, previous=later)
    assert any("does not increase" in e for e in errors)


def test_a_long_recording_keeps_its_start(tmp_path):
    """A live buffer keeps 30 s; a recording loaded whole must keep it all.
    It did not, and every frame of a replay before its last 30 s lost its
    ground fix (training_log section 53)."""

    path = tmp_path / "long.jsonl"
    path.write_text("\n".join(json.dumps({
        "frame": f, "lat": 1.0, "lon": 2.0, "altitude_m": 100.0,
        "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0}) for f in range(700)))
    buffer = telemetry.load_jsonl(path, fps=7.0)

    assert buffer.at(1.0) is not None
    assert buffer.at(99.0) is not None


def test_ground_point_follows_the_view_angle():
    """From overhead the person is the centre of the box; from the side, its
    bottom. Using the bottom at nadir put every fix half a box south
    (training_log section 53)."""

    box = [300.0, 200.0, 312.0, 220.0]
    nadir = geolocate.Camera(640, 512, 50.0, mount_pitch_deg=90.0)
    side = geolocate.Camera(640, 512, 50.0, mount_pitch_deg=0.0)
    oblique = geolocate.Camera(640, 512, 50.0, mount_pitch_deg=45.0)

    assert geolocate.ground_point(nadir, box) == pytest.approx((306.0, 210.0))
    assert geolocate.ground_point(side, box) == pytest.approx((306.0, 220.0))
    assert 210.0 < geolocate.ground_point(oblique, box)[1] < 220.0


def test_predicted_step_follows_the_conventions():
    """Flying north moves the ground down the frame; descending zooms in;
    the image-to-ground model and its inverse agree exactly."""

    camera = geolocate.Camera(640, 512, 50.0)
    state = {"lat": 12.97, "lon": 77.59, "altitude_m": 100.0,
             "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0}

    north = dict(state, lat=12.97 + math.degrees(2.914 / geolocate.EARTH_RADIUS_M))
    step = geolocate.predict_step(camera, state, north)
    assert step[1, 2] == pytest.approx(20.0, abs=0.1)
    assert step[0, 2] == pytest.approx(0.0, abs=0.1)

    lower = dict(state, altitude_m=90.0)
    assert geolocate.predict_step(camera, state, lower)[0, 0] == pytest.approx(100 / 90, abs=1e-3)

    e, n, _ = geolocate.project_to_ground(camera, 100, 400, 100.0, 5, -2, 40)
    assert geolocate.ground_to_pixel(camera, e, n, 100.0, 5, -2, 40) == pytest.approx((100, 400))
