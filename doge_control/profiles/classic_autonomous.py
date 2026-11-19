"""Bounded route with a deterministic trot and a front-range interlock."""

from ..autonomous import AutonomousSource, configure
from ..classic import ClassicTrot
from ..manual import read_rows
from ..preview import Preview


def build(args, model):
    return Preview(model, AutonomousSource(read_rows(args.range_file)), ClassicTrot(model))
