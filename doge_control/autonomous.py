import math

from .manual import index_by_tick
from .model import ROOT
from .preview import Decision


STOP_DISTANCE_M = 0.35
RANGE_TIMEOUT_TICKS = 5
DEFAULT_RANGE = ROOT / "examples/front_range.jsonl"
ROUTE = ((25, 0.10, "route_leg_1"), (35, 0.0, "route_pause"), (65, 0.08, "route_leg_2"))


def decode_range(row):
    distance = row.get("distance_m")
    if set(row) != {"tick", "distance_m"} or type(distance) not in (int, float):
        raise ValueError("range event requires distance_m")
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("distance_m must be finite and non-negative")
    return distance


class AutonomousSource:
    def __init__(self, rows):
        self.events = index_by_tick(rows, decode_range)
        self.last_distance = None
        self.last_tick = None
        self.stop = None

    def __call__(self, tick):
        if self.stop is not None:
            return self.stop
        if tick in self.events:
            self.last_distance, self.last_tick = self.events[tick], tick
        for end_tick, speed, reason in ROUTE:
            if tick < end_tick:
                if self.last_tick is None or tick - self.last_tick >= RANGE_TIMEOUT_TICKS:
                    self.stop = Decision(0.0, "range_missing_or_stale")
                elif self.last_distance <= STOP_DISTANCE_M:
                    self.stop = Decision(0.0, "obstacle")
                return self.stop or Decision(speed, reason)
        self.stop = Decision(0.0, "route_complete")
        return self.stop


def configure(parser):
    parser.add_argument("--range", dest="range_file", default=DEFAULT_RANGE, help="front range JSONL file")
