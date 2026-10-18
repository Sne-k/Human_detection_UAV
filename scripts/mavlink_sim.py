"""
A flight controller on a laptop: real MAVLink packets from a known trajectory.

Testing MAVLink ingestion needs MAVLink. ArduPilot SITL provides it, but it
is a substantial build on Windows and simulates far more than the payload
reads. This sends exactly the two messages the payload consumes - ATTITUDE at
10 Hz and GLOBAL_POSITION_INT at 5 Hz, the ICD minimums - as real packets over
UDP, with the properties that make ingestion hard:

  - its own boot clock, offset arbitrarily from the receiver's
  - random link delay on every packet, so arrival time is not sample time
  - a flight that turns, so attitude changes quickly enough for timing errors
    to matter

Because the trajectory is known exactly, whatever the receiver reconstructs
can be checked against truth at any instant.

The flight: straight and level at 20 m/s, then a 25 degree banked coordinated
turn entered and left at 25 deg/s of roll, then straight again. In the turn
the yaw rate is g tan(bank) / v, about 13 deg/s.

Usage:

    python scripts/mavlink_sim.py --target udpout:127.0.0.1:14550
    python scripts/mavlink_sim.py --write-telemetry flight.jsonl --rate 15
"""

import argparse
import json
import math
import random
import sys
import time

import numpy as np

G = 9.81
SPEED_MS = 20.0
ALTITUDE_M = 100.0
PITCH_DEG = 2.0
HOME = (12.9716, 77.5946)
HEADING_DEG = 35.0

ATTITUDE_HZ = 10.0
POSITION_HZ = 5.0

EARTH_RADIUS_M = 6_378_137.0


class Trajectory:
    """A known flight, sampled densely and interpolated for exact truth."""

    def __init__(self, duration_s=60.0, dt=0.001, turn_start_s=10.0,
                 turn_length_s=12.0, bank_deg=25.0, roll_rate_dps=25.0):
        n = int(duration_s / dt) + 1
        t = np.arange(n) * dt

        ramp = bank_deg / roll_rate_dps
        t1 = turn_start_s
        t2 = t1 + ramp
        t3 = t1 + turn_length_s - ramp
        t4 = t1 + turn_length_s

        roll = np.interp(t, [0, t1, t2, t3, t4, duration_s],
                         [0, 0, bank_deg, bank_deg, 0, 0])

        yaw_rate = np.degrees(G * np.tan(np.radians(roll)) / SPEED_MS)
        yaw = HEADING_DEG + np.concatenate([[0.0], np.cumsum(yaw_rate[1:] * dt)])

        heading = np.radians(yaw)
        north = np.concatenate([[0.0], np.cumsum(SPEED_MS * np.cos(heading[1:]) * dt)])
        east = np.concatenate([[0.0], np.cumsum(SPEED_MS * np.sin(heading[1:]) * dt)])

        lat0 = math.radians(HOME[0])

        self.t = t
        self.roll = roll
        self.yaw = yaw  # unwrapped, degrees
        self.lat = HOME[0] + np.degrees(north / EARTH_RADIUS_M)
        self.lon = HOME[1] + np.degrees(east / (EARTH_RADIUS_M * math.cos(lat0)))
        self.duration_s = duration_s

    def at(self, t):
        """Exact state at time t (seconds since the flight controller booted)."""

        return {
            "lat": float(np.interp(t, self.t, self.lat)),
            "lon": float(np.interp(t, self.t, self.lon)),
            "altitude_m": ALTITUDE_M,
            "roll_deg": float(np.interp(t, self.t, self.roll)),
            "pitch_deg": PITCH_DEG,
            "yaw_deg": float(np.interp(t, self.t, self.yaw)) % 360.0,
        }


def schedule(duration_s, delay_ms=(2.0, 40.0), seed=0):
    """(send_time, kind, sample_time) for every packet, ordered by send time."""

    rng = random.Random(seed)
    events = []

    for hz, kind in ((ATTITUDE_HZ, "ATTITUDE"), (POSITION_HZ, "GLOBAL_POSITION_INT")):
        k = 0

        while k / hz <= duration_s:
            sample = k / hz
            events.append((sample + rng.uniform(*delay_ms) / 1000.0, kind, sample))
            k += 1

    return sorted(events)


def send(target, trajectory, duration_s, boot_offset_s=37.0, seed=0, log=print,
         clock=time.perf_counter):
    """
    Transmit the flight in real time.

    `boot_offset_s` sets how long the simulated flight controller had been up
    when the flight started, so its clock disagrees with the receiver's.
    Returns the local clock time corresponding to boot time zero, which a
    test needs to evaluate truth at a local timestamp.
    """

    from pymavlink import mavutil

    link = mavutil.mavlink_connection(target, source_system=1, source_component=1)

    start = clock()
    boot_epoch = start - boot_offset_s  # local time at which the FC "booted"

    for send_at, kind, sample in schedule(duration_s, seed=seed):
        wait = start + send_at - clock()

        if wait > 0:
            time.sleep(wait)

        state = trajectory.at(sample)
        boot_ms = int(round((sample + boot_offset_s) * 1000))

        if kind == "ATTITUDE":
            yaw = math.radians(state["yaw_deg"])
            yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            link.mav.attitude_send(
                boot_ms,
                math.radians(state["roll_deg"]),
                math.radians(state["pitch_deg"]),
                yaw, 0.0, 0.0, 0.0,
            )
        else:
            link.mav.global_position_int_send(
                boot_ms,
                int(round(state["lat"] * 1e7)),
                int(round(state["lon"] * 1e7)),
                int(ALTITUDE_M * 1000),
                int(ALTITUDE_M * 1000),
                0, 0, 0,
                int(state["yaw_deg"] * 100) % 36000,
            )

    log(f"sent {duration_s:.0f} s of telemetry to {target}")

    # Local time = boot time + boot_epoch, and the trajectory's time zero is
    # boot time boot_offset_s.
    return boot_epoch + boot_offset_s


def write_telemetry(path, trajectory, rate_hz, duration_s):
    """Recorded telemetry with timestamps, for replay through payload.py."""

    with open(path, "w", encoding="utf-8") as f:
        k = 0

        while k / rate_hz <= duration_s:
            t = k / rate_hz
            f.write(json.dumps({"t": round(t, 4), **trajectory.at(t)}) + "\n")
            k += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="udpout:127.0.0.1:14550")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--write-telemetry", default=None)
    parser.add_argument("--rate", type=float, default=10.0)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    trajectory = Trajectory(duration_s=max(args.seconds, 30.0) + 1.0)

    if args.write_telemetry:
        write_telemetry(args.write_telemetry, trajectory, args.rate, args.seconds)
        print(f"written {args.write_telemetry}")
        return

    send(args.target, trajectory, args.seconds)


if __name__ == "__main__":
    main()
