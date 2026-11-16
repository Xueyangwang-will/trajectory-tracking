"""Validate synchronized training data and export link-level motion at 50 Hz."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.ndimage import uniform_filter1d
from trajectory_tracking.collect import ROOT, DEPLOY, DT, ORDER, session
from trajectory_tracking.arm_export import export


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--allow-incomplete',action='store_true',help='Export diagnostic data without marking it successful')
    args=parser.parse_args()
    z=np.load(args.directory/'episode.npz')
    t=z['t'];n=len(t)
    assert np.allclose(np.diff(t),DT,atol=1e-10)
    assert z['qpos'].shape==(n,36) and z['qvel'].shape==(n,35)
    assert z['token'].shape==(n,64) and z['action'].shape==(n,29)
    for key in z.files:
        assert np.isfinite(z[key]).all(),key
    assert np.allclose(np.linalg.norm(z['qpos'][:,3:7],axis=1),1,atol=1e-5)
    assert np.allclose(z['token'],z['decoder_obs'][:,:64])
    config=json.loads((args.directory/'summary.json').read_text())
    if config.get('planner_mode')=='waypoints':
        active=z['planner_has_target'].astype(bool)
        assert active.any(), 'Waypoint mode must actually activate the ONNX waypoint flag'
        assert np.all(z['planner_prediction_frames'][active]==44)
        targets=z['planner_target_positions'][active]
        np.testing.assert_allclose(targets[:,:,2],.788740,atol=1e-6)
        assert np.max(abs(targets[:,:,:2]))>1
        config['waypoint_active_replans']=int(active.sum())
        config['waypoint_total_replans']=len(active)
    # Re-run actual model inference to catch offset mistakes and stale-token logging.
    enc=session(DEPLOY/'policy/release/model_encoder.onnx')
    dec=session(DEPLOY/'policy/release/model_decoder.onnx')
    for i in np.linspace(0,n-1,21).astype(int):
        np.testing.assert_allclose(enc.run(None,{'obs_dict':z['encoder_obs'][i:i+1]})[0][0],z['token'][i],atol=1e-5)
        np.testing.assert_allclose(dec.run(None,{'obs_dict':z['decoder_obs'][i:i+1]})[0][0],z['action'][i],atol=1e-5)
    arm_config=export(args.directory,z)
    np.savez_compressed(args.directory/'tokens_50hz.npz',fps=np.array(50),t=t,
                        token=z['token'],encoder_mode=np.zeros(n,dtype=np.int64))
    summary=config
    mask=(t>12)&(z['phase']==1)&(z['progress']<z['path_s'][-1]-2)
    speed=np.linalg.norm(z['qvel'][:,:2],axis=1)
    smooth=uniform_filter1d(speed,50)
    along_command=np.sum(z['qvel'][:,:2]*np.c_[np.cos(z['command'][:,1]),np.sin(z['command'][:,1])],axis=1)
    summary.update(validation='passed: 50 Hz, finite values, normalized quaternions, synchronized tokens, 21 encoder/decoder replay checks',
        settled_cross_track_rmse_m=float(np.sqrt(np.mean(z['cross_track_error'][mask]**2))) if mask.any() else None,
        settled_cruise_mean_speed_mps=float(speed[mask].mean()) if mask.any() else None,
        settled_along_path_speed_mps=float(np.gradient(z['progress'],DT)[mask].mean()) if mask.any() else None,
        settled_along_command_speed_mps=float(along_command[mask].mean()) if mask.any() else None,
        arm_export=arm_config,
        settled_cruise_1s_speed_rmse_mps=float(np.sqrt(np.mean((smooth[mask]-.6)**2))) if mask.any() else None)
    summary['minimum_goal_distance_m']=float(z['goal_distance'].min())
    summary['minimum_goal_distance_time_s']=float(t[np.argmin(z['goal_distance'])])
    assert summary['completed'] or args.allow_incomplete, 'Episode incomplete, do not use as successful training rollout'
    entered=np.flatnonzero(z['goal_distance']<=summary['arrival_radius_m'])
    if len(entered):
        first=entered[0]
        summary['arrival_trigger_distance_m']=float(z['goal_distance'][first])
        assert not z['position_arrived'][:first].any() and z['position_arrived'][first:].all()
        assert np.all(z['phase'][first:]>=3), 'Position lock cannot return to path tracking'
    else:
        assert not z['position_arrived'].any()
        summary['arrival_trigger_distance_m']=None
        summary['failure_reason']=('Fall/nonfinite state before arrival' if summary.get('termination_reason')=='fall_or_nonfinite'
                                   else 'Timeout: never entered arrival radius; low-speed approach stalled outside threshold')
    if summary['completed']:
        assert summary['idle_simulated_duration_s']>=1.2-1e-8
        assert summary['final_heading_error_deg']<=15
        assert summary['final_0_3s_max_speed_mps']<.12
        idle=np.flatnonzero(z['phase']==3)
        assert np.all(z['command'][idle,2]==0)
        assert np.max(np.ptp(z['qpos'][idle],axis=0))>1e-5, 'Standing must be simulated, not frozen'
    summary['training_success']=bool(summary['completed'])
    braking=z['phase']==2
    u=np.clip((1.2-z['goal_distance'][braking])/.9,0,1)
    expected=.15+.45*.5*(1+np.cos(np.pi*u))
    if summary.get('terminal_controller')=='predictive':
        expected=np.where(z['goal_distance'][braking]<.55,np.maximum(expected,.28),expected)
        refinement=z['phase']==5
        if refinement.any():
            assert z['position_arrived'][refinement].all()
            assert np.ptp(z['t'][refinement])<=1.+1e-8
            assert np.all(z['command'][refinement,0]==.28)
    np.testing.assert_allclose(z['command'][braking,0],expected,atol=1e-10)
    np.testing.assert_array_equal(z['planner_speed'],z['command'][:,0])
    (args.directory/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
