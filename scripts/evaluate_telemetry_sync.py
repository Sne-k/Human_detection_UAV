"""
How much geolocation error does telemetry timing cost?

Receives a simulated flight over real MAVLink (mavlink_sim.py), then asks, at
many instants: what attitude would the payload have used for a frame captured
now, and how far is it from the truth?

Two answers are compared:

  latest      the most recently received message - what a straightforward
              implementation does, and roughly what frame-index lookup
              degrades to once telemetry no longer arrives one record per frame
  matched     telemetry.StateBuffer: clock-aligned, interpolated to the
              frame's capture time, refused across dropouts

Attitude error is converted to ground error with the ICD's figure of about
1.75 m per degree at 100 m.

Usage:

    python scripts/evaluate_telemetry_sync.py
    python scripts/evaluate_telemetry_sync.py --seconds 30
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "telemetry_sync"

M_PER_DEG_AT_100M = 1.75
PORT = 14561


def angle_error(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=26.0)
    args = parser.parse_args()

    import mavlink_sim
    import telemetry

    trajectory = mavlink_sim.Trajectory(duration_s=args.seconds + 5.0)
    buffer = telemetry.StateBuffer()

    receiver = telemetry.MavlinkReceiver(
        f"udpin:127.0.0.1:{PORT}", buffer, record=True
    ).start()
    time.sleep(0.5)

    result = {}

    def transmit():
        result["zero"] = mavlink_sim.send(
            f"udpout:127.0.0.1:{PORT}", trajectory, args.seconds,
            log=lambda *_: None,
        )

    print(f"Flying {args.seconds:.0f} s over MAVLink/UDP, turn from 10 s ...")

    sender = threading.Thread(target=transmit)
    sender.start()
    sender.join()
    time.sleep(0.3)
    receiver.stop()

    zero = result["zero"]
    arrivals = receiver.arrivals

    attitude_arrivals = [(a, v) for a, k, v in arrivals if k == "ATTITUDE"]

    print(f"received {len(arrivals)} packets "
          f"({receiver.counts['ATTITUDE']} ATTITUDE, "
          f"{receiver.counts['GLOBAL_POSITION_INT']} GLOBAL_POSITION_INT)")
    print(f"estimated clock offset error: "
          f"{(receiver.aligner.offset - (zero - 37.0)) * 1000:+.1f} ms "
          f"(link delay 2-40 ms)")
    print()

    rows = {"latest": {"straight": [], "turn": []},
            "matched": {"straight": [], "turn": []}}
    refused = 0

    for t_flight in np.arange(1.0, args.seconds - 0.5, 0.0137):
        local = zero + t_flight
        truth = trajectory.at(t_flight)
        phase = "turn" if 10.0 <= t_flight <= 22.0 else "straight"

        # latest received at the frame's capture time
        prior = [v for a, v in attitude_arrivals if a <= local]

        if prior:
            roll, pitch, yaw = prior[-1]
            rows["latest"][phase].append(max(
                angle_error(roll, truth["roll_deg"]),
                angle_error(pitch, truth["pitch_deg"]),
            ))

        state = buffer.at(local)

        if state is None:
            refused += 1
            continue

        rows["matched"][phase].append(max(
            angle_error(state["roll_deg"], truth["roll_deg"]),
            angle_error(state["pitch_deg"], truth["pitch_deg"]),
        ))

    header = (f"{'method':<9s} {'phase':<9s} {'median deg':>11s} "
              f"{'p95 deg':>8s} {'max deg':>8s} {'max ground m':>13s}")
    print(header)
    print("-" * len(header))

    summary = {}

    for method in ("latest", "matched"):
        for phase in ("straight", "turn"):
            e = np.asarray(rows[method][phase])
            entry = {
                "samples": int(len(e)),
                "median_deg": round(float(np.median(e)), 3),
                "p95_deg": round(float(np.percentile(e, 95)), 3),
                "max_deg": round(float(e.max()), 3),
                "max_ground_m_at_100m": round(float(e.max()) * M_PER_DEG_AT_100M, 2),
            }
            summary[f"{method}_{phase}"] = entry

            print(f"{method:<9s} {phase:<9s} {entry['median_deg']:11.3f} "
                  f"{entry['p95_deg']:8.3f} {entry['max_deg']:8.3f} "
                  f"{entry['max_ground_m_at_100m']:13.2f}")

    print()
    print(f"queries refused for stale or missing state: {refused}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "telemetry_sync.json").write_text(
        json.dumps({"seconds": args.seconds, "refused": refused,
                    "results": summary}, indent=2), encoding="utf-8")
    print(f"Written: {OUTPUT_DIR / 'telemetry_sync.json'}")


if __name__ == "__main__":
    main()
