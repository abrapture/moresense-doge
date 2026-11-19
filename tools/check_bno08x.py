"""Run BNO08x unit and Isaac Lab integration checks on the Linux Docker host."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="isaac-lab-kitless")
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--num-envs", type=int, default=8)
    args = parser.parse_args()
    if os.name == "nt":
        raise SystemExit("Run on the Linux server that has your Isaac Lab Docker image.")
    package = Path(__file__).resolve().parent.parent
    (package / "runtime/cache/torch/kernels").mkdir(parents=True, exist_ok=True)
    (package / "runtime/warp-cache").mkdir(parents=True, exist_ok=True)
    base = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{package}:/robot", "-w", "/robot", "-e", "PYTHONPATH=/robot",
            "-e", "PYTHONUNBUFFERED=1", "-e", "WARP_CACHE_PATH=/robot/runtime/warp-cache",
            "-e", "XDG_CACHE_HOME=/robot/runtime/cache"]
    subprocess.run(base + ["--entrypoint", "python", args.image, "-m", "unittest", "discover",
                          "-s", "training/tests", "-v"], check=True)
    if not args.unit_only:
        subprocess.run(base + ["--gpus", "all", "--entrypoint", "python", args.image,
                              "-m", "training.check_isaac_imu", "--num-envs", str(args.num_envs)], check=True)


if __name__ == "__main__":
    main()
