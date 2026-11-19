"""Operator replay with the archived RL V1 actor."""

from ..manual import ManualSource, configure, read_rows
from ..preview import Preview
from ..rl import RlV1


def build(args, model):
    return Preview(model, ManualSource(read_rows(args.commands)), RlV1.load(model))
