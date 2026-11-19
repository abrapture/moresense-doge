"""Bounded route with the archived RL V1 actor and front-range interlock."""

from ..autonomous import AutonomousSource, configure
from ..manual import read_rows
from ..preview import Preview
from ..rl import RlV1


def build(args, model):
    return Preview(model, AutonomousSource(read_rows(args.range_file)), RlV1.load(model))
