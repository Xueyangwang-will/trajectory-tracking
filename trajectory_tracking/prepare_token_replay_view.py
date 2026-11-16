"""Prepare visualization of NEW decoder-only simulation states, with a path overlay.

This runs after run_token_decoder.py simulate; the optional reference path is
visual-only and is never fed to the decoder or to physics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args()
    with np.load(args.replay/'episode.npz') as source:
        arrays={key:np.array(source[key]) for key in source.files}
    with np.load(args.reference) as source:
        path=np.array(source['path_xy'])
    summary=json.loads((args.replay/'summary.json').read_text())
    assert summary['completed'] and len(summary['loaded_models'])==1
    n=len(arrays['t'])
    goal_heading=np.arctan2(*(path[-1]-path[-30])[::-1])
    yaw=Rotation.from_quat(arrays['qpos'][:,[4,5,6,3]]).as_euler('xyz')[:,2]
    error=np.arctan2(np.sin(goal_heading-yaw),np.cos(goal_heading-yaw))
    distance=np.linalg.norm(arrays['qpos'][:,:2]-path[-1],axis=1)
    arrays.update(path_xy=path,phase=np.zeros(n,dtype=np.int64),goal_distance=distance,
                  goal_heading_error=error)
    viz=args.replay/'visualization';viz.mkdir(exist_ok=True)
    np.savez_compressed(viz/'episode.npz',**arrays)
    meta={**summary,'trajectory_name':'U-turn / decoder-only replay',
          'phase_names':['token_decoder'],
          'render_caption':'SONIC decoder only | external tokens + live state | 25 FPS',
          'duration_s':float(arrays['t'][-1]),'control_hz':50,
          'endpoint_error_m':float(distance[-1]),'final_heading_error_deg':float(np.rad2deg(abs(error[-1]))),
          'visualization_state_source':str((args.replay/'episode.npz').resolve()),
          'reference_path_usage':'visual overlay only, after closed-loop simulation'}
    (viz/'summary.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({key:meta[key] for key in ['frames','duration_s','endpoint_error_m','final_heading_error_deg','reference_max_abs_error']},indent=2))


if __name__=='__main__':main()
