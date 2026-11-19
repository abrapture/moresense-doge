"""Run the provisional Newton standing test on the Linux Docker host."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--view', action='store_true')
    p.add_argument('--port', type=int, default=8084)
    p.add_argument('--seconds', type=float, default=10)
    p.add_argument('--image', default='isaac-lab-kitless')
    args = p.parse_args()
    if os.name == 'nt':
        p.error('Run this on the Linux server.')
    if not 1024 <= args.port <= 65535:
        p.error('port must be between 1024 and 65535')
    package = Path(__file__).resolve().parent.parent
    workspace = package.parent
    relative = package.relative_to(workspace).as_posix()
    base = ['docker', 'run', '--rm', '--gpus', 'all', '--user', f'{os.getuid()}:{os.getgid()}',
            '-v', f'{workspace}:/robot', '-w', '/robot', '-e', 'PYTHONPATH=/robot',
            '-e', 'PYTHONUNBUFFERED=1', '-e', f'WARP_CACHE_PATH=/robot/{relative}/runtime/warp-cache',
            '-e', f'XDG_CACHE_HOME=/robot/{relative}/runtime/cache']
    options = ['--dt', '0.001', '--kp', '6', '--kd', '0.2', '--pose-candidate', f'/robot/{relative}/verification/stand_pose_candidate.json']
    calibration = f'/robot/{relative}/calibration'
    if args.view:
        base += ['-p', f'127.0.0.1:{args.port}:8080']
        command = ['-m', 'moresense_doge.training.stand_view', '--calibration', calibration, '--wall-seconds', '600'] + options
    else:
        command = ['-m', 'moresense_doge.training.stand_check', '--calibration', calibration,
                   '--seconds', str(args.seconds), '--report', f'/robot/{relative}/verification/stand_candidate_check.json'] + options
    subprocess.run(base + ['--entrypoint', 'python', args.image] + command, check=True)


if __name__ == '__main__':
    main()
