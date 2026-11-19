"""Operator replay with a deterministic trot."""

from ..classic import ClassicTrot
from ..manual import ManualSource, configure, read_rows
from ..preview import Preview


def build(args, model):
    return Preview(model, ManualSource(read_rows(args.commands)), ClassicTrot(model))
