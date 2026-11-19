from contextlib import nullcontext
import json
from pathlib import Path
import sys

from .model import ROOT
from .preview import Decision


COMMAND_TIMEOUT_TICKS = 10
DEFAULT_COMMANDS = ROOT / "examples/operator.jsonl"


def read_rows(path):
    stream = nullcontext(sys.stdin) if path == "-" else Path(path).open(encoding="utf-8")
    rows = []
    with stream as lines:
        for line_number, line in enumerate(lines, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def index_by_tick(rows, decode):
    events = {}
    last_tick = -1
    for row in rows:
        tick = row.get("tick") if isinstance(row, dict) else None
        if type(tick) is not int or tick <= last_tick:
            raise ValueError("event ticks must be non-negative integers in strictly increasing order")
        events[tick] = decode(row)
        last_tick = tick
    return events


def decode_command(row):
    if set(row) == {"tick", "stop"} and row["stop"] is True:
        return Decision(0.0, "operator_stop")
    if set(row) == {"tick", "forward_mps"} and type(row["forward_mps"]) in (int, float):
        return Decision(row["forward_mps"], "operator")
    raise ValueError("operator event requires either forward_mps or stop=true")


class ManualSource:
    def __init__(self, rows):
        self.events = index_by_tick(rows, decode_command)
        self.last = None
        self.last_tick = None

    def __call__(self, tick):
        if self.last is not None and self.last.reason == "operator_stop":
            return self.last
        event = self.events.get(tick)
        if event is not None:
            self.last, self.last_tick = event, tick
        if self.last is None:
            return Decision(0.0, "operator_wait")
        if tick - self.last_tick >= COMMAND_TIMEOUT_TICKS:
            return Decision(0.0, "operator_timeout")
        return self.last


def configure(parser):
    parser.add_argument("--commands", default=DEFAULT_COMMANDS, help="operator JSONL file; '-' reads stdin")
