import argparse
from importlib import import_module, util
import json

from .model import RobotModel
from .preview import Decision, Preview


MODES = ("classic-manual", "classic-autonomous", "rl-manual", "rl-autonomous")


def tick_count(value):
    ticks = int(value)
    if not 1 <= ticks <= 5000:
        raise argparse.ArgumentTypeError("--ticks must be in 1..5000")
    return ticks


def stand(args, model):
    return Preview(model, lambda tick: Decision(0.0, "stand"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline joint-target preview; no hardware output")
    commands = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ticks", type=tick_count, default=64, help="number of 20 ms samples")
    commands.add_parser("stand", parents=[common], help="print the archived stand target").set_defaults(build=stand)
    for mode in MODES:
        module_name = f"doge_control.profiles.{mode.replace('-', '_')}"
        if util.find_spec(module_name) is not None:
            profile = import_module(module_name)
            mode_parser = commands.add_parser(mode, parents=[common], help=profile.__doc__)
            profile.configure(mode_parser)
            mode_parser.set_defaults(build=profile.build)

    args = parser.parse_args(argv)
    try:
        preview = args.build(args, RobotModel.load())
        for _ in range(args.ticks):
            print(json.dumps({"mode": args.mode, **preview.step()}, allow_nan=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
