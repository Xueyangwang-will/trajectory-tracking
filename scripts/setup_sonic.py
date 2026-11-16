"""Install the pinned external controller and only the resources this project uses."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

UPSTREAM = 'https://github.com/NVlabs/GR00T-WholeBodyControl.git'
REVISION = '1e1ec4f5f27e6c3e03048f693bd7c60cce04afda'
ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    'g1/scene_29dof.xml',
    'policy/release/model_encoder.onnx',
    'policy/release/model_decoder.onnx',
    'planner/target_vel/V2/planner_sonic.onnx',
)


def run(*command, cwd=None):
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
    subprocess.run(command, cwd=cwd, env=env, check=True)


def check(checkout):
    deploy = checkout / 'gear_sonic_deploy'
    missing = [str(deploy / name) for name in REQUIRED if not (deploy / name).is_file()]
    if missing:
        raise SystemExit('Missing external resources:\n' + '\n'.join(missing))
    # A Git LFS pointer is not a usable robot mesh or ONNX model.
    for name in REQUIRED:
        with (deploy / name).open('rb') as stream:
            if stream.read(64).startswith(b'version https://git-lfs.github.com/spec/'):
                raise SystemExit(f'Unresolved Git LFS pointer: {deploy / name}')
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(deploy / 'g1/scene_29dof.xml'))
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise SystemExit('Expected the 29-DOF free-base G1 simulation model')
    print(f'External controller ready: {checkout}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path,
                        default=Path(os.environ.get('SONIC_ROOT', ROOT / 'external/sonic')))
    parser.add_argument('--check', action='store_true', help='Validate an existing installation only')
    args = parser.parse_args()
    checkout = args.checkout.expanduser().resolve()
    if args.check:
        check(checkout)
        return
    if not checkout.exists():
        checkout.mkdir(parents=True)
        run('git', 'init', str(checkout))
        run('git', 'remote', 'add', 'origin', UPSTREAM, cwd=checkout)
        run('git', 'config', 'remote.origin.promisor', 'true', cwd=checkout)
        run('git', 'config', 'remote.origin.partialclonefilter', 'blob:none', cwd=checkout)
        run('git', 'fetch', '--filter=blob:none', '--depth=1', 'origin', REVISION, cwd=checkout)
        run('git', 'sparse-checkout', 'init', '--cone', cwd=checkout)
        run('git', 'sparse-checkout', 'set', 'gear_sonic_deploy/g1', cwd=checkout)
        run('git', 'checkout', '--detach', 'FETCH_HEAD', cwd=checkout)
    elif not (checkout / '.git').exists():
        raise SystemExit(f'Existing checkout is not a Git repository: {checkout}')
    # Reused checkouts are never reset or cleaned. Only G1 LFS resources and
    # deployment models are fetched; unrelated training checkpoints stay out.
    run('git', 'lfs', 'pull', '--include=gear_sonic_deploy/g1/**', '--exclude=', cwd=checkout)
    run(sys.executable, str(checkout / 'download_from_hf.py'),
        '--output-dir', str(checkout / 'gear_sonic_deploy'))
    check(checkout)


if __name__ == '__main__':
    main()
