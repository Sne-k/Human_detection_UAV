"""
Aircraft state for geolocation: received over MAVLink, matched by time.

Interface IF-4 in docs/payload_icd.md. Geolocation needs the aircraft's
position and attitude at the instant a frame was captured. Two things make
that harder than a lookup.

**The state does not arrive per frame.** A flight controller sends ATTITUDE
(roll, pitch, yaw) and GLOBAL_POSITION_INT (latitude, longitude, altitude) as
separate messages at their own rates - the ICD asks for at least 10 Hz and
5 Hz - while the camera delivers frames at whatever rate the companion
computer sustains. None of these line up. The payload previously looked
telemetry up by frame index, which worked only because the synthetic test
sequence was generated with one record per frame. Real telemetry has no frame
index at all.

**The flight controller has its own clock.** MAVLink messages carry
`time_boot_ms`, milliseconds since the flight controller booted. The frame is
timestamped by the companion computer. Matching them needs the offset between
the two clocks.

Why it matters: the ICD's own error budget puts one degree of attitude error
at about 1.75 m on the ground from 100 m. In a turn rolling at 30 deg/s, a
50 ms association error is 1.5 degrees - 2.6 m of error that the reported
uncertainty would not include.

What this module does
---------------------

`StateBuffer` keeps position and attitude as two separate time series and
answers "what was the state at time t" by linear interpolation between the
samples either side, with yaw unwrapped so a heading crossing north does not
interpolate the long way round.

It **refuses to answer** rather than guess. If the samples either side of t
are further apart than `max_gap_s` - a dropout - or t is more than `hold_s`
past the newest sample - the link has stalled - it returns None, and the
payload reports the detection without a ground coordinate. A stale state
produces no coordinate; it never produces a confident wrong one.

`MavlinkReceiver` reads a MAVLink connection on a background thread and feeds
the buffer, converting flight-controller time to the local clock with a
windowed minimum of (arrival - boot time). Queuing delay on the link only ever
adds to that difference, so its minimum over a window is the best estimate of
the true offset; the window lets it follow the slow drift between two
crystals over a long flight.

`load_jsonl` reads recorded telemetry, by timestamp when records carry one and
by frame index otherwise, so existing recordings keep working.

Usage:

    python scripts/telemetry.py --listen udpin:0.0.0.0:14550
    python scripts/telemetry.py --selftest
"""

import argparse
import collections
import json
import math
import sys
import threading
import time
from bisect import bisect_left

POSITION_FIELDS = ("lat", "lon", "altitude_m")
ATTITUDE_FIELDS = ("roll_deg", "pitch_deg", "yaw_deg")

# Samples further apart than this cannot be interpolated between: something
# was lost in the gap. Generous against the ICD's 10 Hz / 5 Hz minimum rates.
MAX_GAP_S = 0.5

# How long past the newest sample a state may be held. Attitude changes fast
# and is held briefly; position changes slowly and tolerates more.
HOLD_ATTITUDE_S = 0.15
HOLD_POSITION_S = 0.30

# Window over which the flight-controller clock offset is estimated.
CLOCK_WINDOW_S = 10.0

# The one clock that frame timestamps and telemetry arrivals must share.
# perf_counter rather than monotonic: both are monotonic, but on Windows
# monotonic ticks at about 15.6 ms, coarser than the timing being measured
# here. Measured: it put the clock-offset estimate 6 ms early, which a
# minimum over delays should never do.
CLOCK = time.perf_counter


class _Series:
    """One group of fields sampled together, interpolated in time."""

    def __init__(self, fields, angular=(), history_s=30.0):
        self.fields = fields
        self.angular = set(angular)
        self.history_s = history_s
        self.times = []
        self.values = []
        self.lock = threading.Lock()

    def add(self, t, values):
        with self.lock:
            # Messages can arrive out of order on a lossy link.
            index = bisect_left(self.times, t)

            if index < len(self.times) and self.times[index] == t:
                self.values[index] = values
            else:
                self.times.insert(index, t)
                self.values.insert(index, values)

            cutoff = self.times[-1] - self.history_s

            while self.times and self.times[0] < cutoff:
                self.times.pop(0)
                self.values.pop(0)

    def at(self, t, max_gap_s, hold_s):
        """Interpolated values at t and the age of the data used, or None."""

        with self.lock:
            if not self.times:
                return None

            index = bisect_left(self.times, t)

            if index < len(self.times) and self.times[index] == t:
                return dict(self.values[index]), 0.0

            if index == 0:
                return None  # before the first sample: nothing to go on

            if index == len(self.times):
                age = t - self.times[-1]

                if age > hold_s:
                    return None  # the link has stalled

                return dict(self.values[-1]), age

            t0, t1 = self.times[index - 1], self.times[index]

            if t1 - t0 > max_gap_s:
                return None  # a dropout spans t

            v0, v1 = self.values[index - 1], self.values[index]
            w = (t - t0) / (t1 - t0)

            out = {}

            for field in self.fields:
                a, b = v0[field], v1[field]

                if field in self.angular:
                    # Unwrap so 359 -> 1 interpolates through north.
                    delta = (b - a + 180.0) % 360.0 - 180.0
                    out[field] = (a + w * delta) % 360.0
                else:
                    out[field] = a + w * (b - a)

            return out, min(t - t0, t1 - t)


class StateBuffer:
    """Position and attitude, interpolated to any requested time."""

    def __init__(self, max_gap_s=MAX_GAP_S, hold_attitude_s=HOLD_ATTITUDE_S,
                 hold_position_s=HOLD_POSITION_S):
        self.position = _Series(POSITION_FIELDS)
        self.attitude = _Series(ATTITUDE_FIELDS, angular=("yaw_deg",))
        self.max_gap_s = max_gap_s
        self.hold_attitude_s = hold_attitude_s
        self.hold_position_s = hold_position_s

    def add_position(self, t, lat, lon, altitude_m):
        self.position.add(t, {"lat": lat, "lon": lon, "altitude_m": altitude_m})

    def add_attitude(self, t, roll_deg, pitch_deg, yaw_deg):
        self.attitude.add(t, {
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg % 360.0,
        })

    def at(self, t):
        """
        Aircraft state at time t, or None if it cannot be known honestly.

        The result carries `state_age_s`: the larger distance, across the two
        series, between t and the nearest sample actually used. It is the
        honest measure of how current the state behind a coordinate was.
        """

        position = self.position.at(t, self.max_gap_s, self.hold_position_s)
        attitude = self.attitude.at(t, self.max_gap_s, self.hold_attitude_s)

        if position is None or attitude is None:
            return None

        state = {**position[0], **attitude[0]}
        state["state_age_s"] = round(max(position[1], attitude[1]), 4)

        return state

    def ready(self, t):
        """
        True once both series hold a sample at or after t.

        At the moment a frame is captured, the next telemetry sample has not
        arrived yet, so the state can only be held, not interpolated - and a
        held attitude is up to a sample period plus link delay old. Waiting
        until this returns True is what makes interpolation possible live.
        """

        with self.position.lock, self.attitude.lock:
            return (
                bool(self.position.times) and self.position.times[-1] >= t
                and bool(self.attitude.times) and self.attitude.times[-1] >= t
            )

    def __len__(self):
        return len(self.position.times) + len(self.attitude.times)


def load_jsonl(path, fps):
    """
    Recorded telemetry into a buffer, in stream time (seconds from start).

    A record's time is its `t` or `timestamp_s` field when present, and
    `frame / fps` otherwise - the format of the synthetic test sequence. A
    record may carry position fields, attitude fields or both.
    """

    buffer = StateBuffer()

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if "t" in record:
                t = float(record["t"])
            elif "timestamp_s" in record:
                t = float(record["timestamp_s"])
            else:
                t = record["frame"] / fps

            if all(k in record for k in POSITION_FIELDS):
                buffer.add_position(
                    t, record["lat"], record["lon"], record["altitude_m"]
                )

            if all(k in record for k in ATTITUDE_FIELDS):
                buffer.add_attitude(
                    t, record["roll_deg"], record["pitch_deg"], record["yaw_deg"]
                )

    return buffer


class ClockAligner:
    """
    Flight-controller boot time to the local clock.

    offset = arrival_time - boot_time is inflated by every millisecond a
    message spends queued or in transit, never deflated. So the minimum over a
    recent window is the best estimate of the true offset, and the window lets
    it follow the slow drift between the two clocks.
    """

    def __init__(self, window_s=CLOCK_WINDOW_S):
        self.window_s = window_s
        self.samples = collections.deque()

    def observe(self, boot_s, arrival_s):
        self.samples.append((arrival_s, arrival_s - boot_s))

        while self.samples and arrival_s - self.samples[0][0] > self.window_s:
            self.samples.popleft()

    @property
    def offset(self):
        if not self.samples:
            return None

        return min(offset for _, offset in self.samples)

    def to_local(self, boot_s):
        return None if self.offset is None else boot_s + self.offset


class MavlinkReceiver:
    """
    Read ATTITUDE and GLOBAL_POSITION_INT from a MAVLink connection.

    `connection` is a pymavlink connection string - `udpin:0.0.0.0:14550`
    for a UDP stream, `COM3` or `/dev/ttyAMA0` with a baud rate for the
    flight controller's UART. Read-only: the payload never writes to the
    flight controller (ICD section 5).

    Altitude is `relative_alt`, height above the home position. The ICD
    requires height above ground; over terrain that differs from home, a
    rangefinder (DISTANCE_SENSOR) is needed, which this does not yet read.
    """

    def __init__(self, connection, buffer, baud=57600, clock=None,
                 record=False):
        from pymavlink import mavutil

        self.link = mavutil.mavlink_connection(connection, baud=baud)
        self.buffer = buffer
        self.clock = clock or CLOCK
        self.aligner = ClockAligner()
        self.counts = collections.Counter()
        self.last_message = None
        # Raw (arrival, kind, values) for evaluation; off in flight.
        self.arrivals = [] if record else None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                message = self.link.recv_match(
                    type=["ATTITUDE", "GLOBAL_POSITION_INT"],
                    blocking=True,
                    timeout=0.5,
                )
            except Exception:
                # A malformed packet or a transient serial error must not kill
                # the reader: the payload outlives a bad message.
                self.counts["errors"] += 1
                continue

            if message is None:
                continue

            arrival = self.clock()
            boot_s = message.time_boot_ms / 1000.0

            self.aligner.observe(boot_s, arrival)
            t = self.aligner.to_local(boot_s)

            kind = message.get_type()
            self.counts[kind] += 1
            self.last_message = arrival

            if kind == "ATTITUDE":
                values = (
                    math.degrees(message.roll),
                    math.degrees(message.pitch),
                    math.degrees(message.yaw),
                )
                self.buffer.add_attitude(t, *values)
            else:
                values = (
                    message.lat / 1e7,
                    message.lon / 1e7,
                    message.relative_alt / 1000.0,
                )
                self.buffer.add_position(t, *values)

            if self.arrivals is not None:
                self.arrivals.append((arrival, kind, values))


def _selftest():
    """Interpolation, unwrap and refusal cases, each checkable by hand."""

    b = StateBuffer()

    b.add_position(0.0, 10.0, 20.0, 100.0)
    b.add_position(0.2, 10.0002, 20.0, 102.0)
    b.add_attitude(0.0, 0.0, 0.0, 350.0)
    b.add_attitude(0.1, 10.0, 0.0, 10.0)

    s = b.at(0.05)
    assert abs(s["roll_deg"] - 5.0) < 1e-9, s
    assert abs(s["yaw_deg"] - 0.0) < 1e-9 or abs(s["yaw_deg"] - 360.0) < 1e-9, s
    assert abs(s["altitude_m"] - 100.5) < 1e-9, s
    print("interpolation and yaw unwrap through north   ok")

    assert b.at(-0.01) is None
    print("before the first sample -> refused           ok")

    assert b.at(0.1 + HOLD_ATTITUDE_S + 0.01) is None
    print("attitude stalled past the hold -> refused    ok")

    gap = StateBuffer()
    gap.add_position(0.0, 1, 1, 1)
    gap.add_position(1.0, 1, 1, 1)
    gap.add_attitude(0.0, 0, 0, 0)
    gap.add_attitude(1.0, 0, 0, 0)
    assert gap.at(0.5) is None
    print("dropout wider than max_gap -> refused        ok")

    clock = ClockAligner()
    for k in range(50):
        boot = k * 0.1
        clock.observe(boot, 1000.0 + boot + 0.004 + (0.03 if k % 3 else 0.0))
    assert abs(clock.offset - 1000.004) < 1e-9
    print("clock offset rejects queuing delay           ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default=None,
                        help="pymavlink connection string, e.g. udpin:0.0.0.0:14550")
    parser.add_argument("--baud", type=int, default=57600)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.selftest or not args.listen:
        _selftest()
        return

    buffer = StateBuffer()
    receiver = MavlinkReceiver(args.listen, buffer, baud=args.baud).start()

    end = CLOCK() + args.seconds

    while CLOCK() < end:
        time.sleep(1.0)
        state = buffer.at(CLOCK() - 0.05)
        print(dict(receiver.counts), state)

    receiver.stop()


if __name__ == "__main__":
    main()
