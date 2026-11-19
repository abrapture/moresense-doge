"""Run the kinematic pose editor next to an existing Isaac Lab Docker image.

Host-side launcher: python3 tools/start_pose_editor.py
Uses an isolated venv on the bind mount; never installs into Isaac Lab's venv.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='isaac-lab-kitless')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--name', default='moresense-poser')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if os.name == 'nt':
        raise SystemExit('Run this launcher on the Linux server, inside the extracted model folder.')
    if not 1024 <= args.port <= 65535:
        raise SystemExit('Port must be between 1024 and 65535.')
    model_dir = Path(__file__).resolve().parent.parent
    workspace = model_dir.parent
    relative_model = model_dir.relative_to(workspace).as_posix()
    mount = f'{workspace}:/robot'
    uid_gid = f'{os.getuid()}:{os.getgid()}'
    run(['docker','image','inspect',args.image], stdout=subprocess.DEVNULL)
    existing = subprocess.run(['docker','inspect','--format','{{.State.Running}}',args.name],
                              text=True,capture_output=True)
    if existing.returncode == 0:
        state=existing.stdout.strip()
        raise SystemExit(f'Container {args.name} already exists (running={state}). '
                         'Use its current viewer or stop that named container before relaunching.')
    base = ['docker','run','--rm','--user',uid_gid,'-v',mount,
            '-e','UV_CACHE_DIR=/robot/.poser-uv-cache','-e','PYTHONUNBUFFERED=1']
    venv = workspace/'.poser-venv'
    # The kitless image may omit ensurepip. uv installs into this isolated venv.
    run(base+['--entrypoint','python',args.image,'-m','venv','--without-pip',
              '--system-site-packages','/robot/.poser-venv'])
    python = '/robot/.poser-venv/bin/python'
    probe = "import importlib.metadata as m; assert m.version('viser')=='1.1.1'; assert m.version('yourdfpy')=='0.0.60'"
    checked = subprocess.run(base+['--entrypoint',python,args.image,'-c',probe],
                             stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if checked.returncode != 0:
        run(base+['--entrypoint','uv',args.image,'pip','install','--python',python,
                  '-r',f'/robot/{relative_model}/tools/poser_requirements.txt'])
    editor=f'/robot/{relative_model}/tools/pose_editor.py'
    model_name = next(name for name in ('moresense_doge_simple.urdf', 'moresense_doge_coacd.urdf', 'moresense_doge.urdf') if (model_dir / 'urdf' / name).is_file())
    urdf=f'/robot/{relative_model}/urdf/{model_name}'
    saved=f'/robot/{relative_model}/calibration'
    command=[editor,'--urdf',urdf,'--save-dir',saved]
    run(base+['--entrypoint',python,args.image]+command+['--check'])
    if args.check_only:
        return
    # Port is bound to server loopback, matching the user's existing SSH-tunnel setup.
    start=['docker','run','--rm','-d','--name',args.name,'--user',uid_gid,
           '-p',f'127.0.0.1:{args.port}:8080','-v',mount,'-e','PYTHONUNBUFFERED=1',
           '--workdir',f'/robot/{relative_model}','--entrypoint',python,args.image]
    result=run(start+command+['--host','0.0.0.0','--port','8080'],text=True,capture_output=True)
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        logs=subprocess.run(['docker','logs','--tail','20',args.name],text=True,capture_output=True)
        if 'POSE_EDITOR_READY' in logs.stdout+logs.stderr:
            break
        state=subprocess.run(['docker','inspect','--format','{{.State.Running}}',args.name],text=True,capture_output=True)
        if state.returncode != 0 or state.stdout.strip() != 'true':
            raise SystemExit('Editor exited during startup.\n'+logs.stdout+logs.stderr)
        time.sleep(1)
    else:
        raise SystemExit(f'Editor did not become ready within 60 seconds. Check: docker logs {args.name}')
    print(json.dumps({'container':args.name,'container_id':result.stdout.strip(),
        'server_url':f'http://127.0.0.1:{args.port}', 'mode':'KINEMATIC_POSE_EDITOR_NO_PHYSICS',
        'saved_files':str(model_dir/'calibration'), 'gpu_used':False},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
