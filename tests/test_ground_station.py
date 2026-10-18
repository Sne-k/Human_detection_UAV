"""
Ground station and the UDP stream (task D1, interface IF-6).

What must hold: confirmed targets and unconfirmed detections are kept apart;
a moving person is reported where they are now, not where they were best
measured; a corrupt datagram is dropped without ending the session; and on
the payload side, a link that refuses every datagram costs nothing but a
count.
"""

import json
import socket

import pytest

import ground_station
from test_failure_injection import needs_artifacts, run_payload


def record(frame, detections, uav=True):
    rec = {"frame": frame, "timestamp_s": frame / 7, "persons": len(detections),
           "detections": detections, "ego_motion": "ok"}
    rec["uav"] = {"lat": 12.97, "lon": 77.59} if uav else None
    return rec


def detection(track_id, lat, lon, error_m, confidence=0.5, state="stationary"):
    return {"track_id": track_id, "confidence": confidence, "state": state,
            "ground": {"lat": lat, "lon": lon, "position_error_m": error_m}}


def test_targets_and_unconfirmed_are_kept_apart(tmp_path):
    picture = ground_station.Picture()
    picture.add(record(0, [detection(4, 12.9701, 77.59, 5.0),
                           detection(-1, 12.9702, 77.59, 5.0)]))
    picture.add(record(1, [detection(4, 12.9701, 77.59, 4.0, confidence=0.7)]))
    picture.add(record(2, [], uav=False))

    assert list(picture.targets) == [4]
    assert picture.targets[4]["sightings"] == 2
    assert picture.targets[4]["best_confidence"] == 0.7
    assert len(picture.unconfirmed) == 1
    assert picture.stale_frames == 1

    log = tmp_path / "log.json"
    ground_station.mission_log(picture, log)
    out = json.loads(log.read_text(encoding="utf-8"))
    assert [t["track_id"] for t in out["targets"]] == [4]
    assert out["targets"][0]["position_error_m"] == 4.0
    assert out["unconfirmed_detections"] == 1

    assert ground_station.render(picture).shape == (700, 1100, 3)


def test_moving_target_is_reported_where_it_is_now():
    picture = ground_station.Picture()
    picture.add(record(0, [detection(7, 12.9701, 77.59, 3.0)]))
    picture.add(record(1, [detection(7, 12.9705, 77.59, 6.0, state="moving")]))

    best = picture.best_fix(picture.targets[7])
    assert best["lat"] == 12.9705


def test_udp_drops_corrupt_datagrams():
    sock = ground_station.udp_socket("127.0.0.1:0")
    port = sock.getsockname()[1]
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    sender.sendto(b"\xff not json", ("127.0.0.1", port))
    sender.sendto(json.dumps(record(3, [])).encode(), ("127.0.0.1", port))

    received = []
    for item in ground_station.records_from_udp(sock):
        if item is not None:
            received.append(item)
            break

    assert [r["frame"] for r in received] == [3]
    sock.close()
    sender.close()


@needs_artifacts
def test_payload_streams_every_record_over_udp(tmp_path, monkeypatch):
    sock = ground_station.udp_socket("127.0.0.1:0")
    port = sock.getsockname()[1]
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)

    summary = run_payload(tmp_path, monkeypatch, "--udp", f"127.0.0.1:{port}",
                          "--jsonl", str(tmp_path / "out.jsonl"), frames=20)

    received = []
    for item in ground_station.records_from_udp(sock):
        if item is None:
            break
        received.append(item)
    sock.close()

    assert summary["udp_sent"] == summary["frames"] == 20
    assert summary["udp_errors"] == 0
    lines = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    assert received == [json.loads(line) for line in lines]


@needs_artifacts
def test_a_refusing_link_never_stops_detection(tmp_path, monkeypatch):
    # Broadcast without SO_BROADCAST: every sendto raises.
    summary = run_payload(tmp_path, monkeypatch, "--udp", "255.255.255.255:14660",
                          frames=10)

    assert summary["frames"] == 10
    assert summary["udp_sent"] == 0
    assert summary["udp_errors"] == 10


def test_a_still_target_is_placed_by_all_its_fixes():
    """Picking the single tightest fix was biased (training_log section 53);
    fixes are now averaged by inverse variance, and the uncertainty claimed
    is the tightest single fix's - not divided by sqrt(N), because flight
    errors drift and one pass largely repeats one error."""

    picture = ground_station.Picture()
    picture.add(record(0, [detection(5, 12.9700, 77.59, 4.0)]))
    picture.add(record(1, [detection(5, 12.9702, 77.59, 4.0)]))
    picture.add(record(2, [detection(5, 12.9710, 77.59, 40.0)]))

    best = picture.best_fix(picture.targets[5])
    assert best["lat"] == pytest.approx(12.9701, abs=2e-5)
    assert best["error_m"] == 4.0
    assert best["fixes_combined"] == 3
