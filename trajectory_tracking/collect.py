"""Closed-loop SONIC trajectory collection in free-base MuJoCo, no DDS/hardware.

Run from repo root: python -m trajectory_tracking.collect
All per-tick arrays share t[k]: state before action[k], token[k] drives that action.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

from trajectory_tracking.paths import ROOT, DEPLOY
from trajectory_tracking.sonic_decoder import (
    SonicDecoder, RobotState, POLICY_TO_MUJOCO as ORDER,
    DEFAULT_JOINT_POS as DEFAULT, ARMATURE as ARM,
    EFFORT_FOR_ACTION_SCALE as EFFORT, ACTION_SCALE as SCALE, KP, KD,
)
DT = .02


def rot(q):
    return Rotation.from_quat(np.asarray(q)[..., [1,2,3,0]])


def yaw(q):
    w,x,y,z = q
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))


def sample(frames, indices):
    indices = np.clip(indices, 0, len(frames)-1)
    a = np.floor(indices).astype(int)
    b = np.minimum(a+1,len(frames)-1)
    w = (indices-a)[:,None]
    result = frames[a]*(1-w)+frames[b]*w
    # Pairwise quaternion slerp with sign correction.
    q0,q1 = frames[a,3:7], frames[b,3:7].copy()
    dots = np.sum(q0*q1,axis=1)
    q1[dots<0] *= -1
    theta = np.arccos(np.clip(abs(dots),0,1))[:,None]
    sin = np.sin(theta)
    near = sin < 1e-6
    safe = np.where(near,1,sin)
    qr = np.sin((1-w)*theta)/safe*q0+np.sin(w*theta)/safe*q1
    qr = np.where(near,(1-w)*q0+w*q1,qr)
    result[:,3:7] = qr/np.linalg.norm(qr,axis=1,keepdims=True)
    return result


def session(path, threads=2):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])


class Sonic:
    def __init__(self,seed=42):
        self.seed=seed
        self.encoder = session(DEPLOY/'policy/release/model_encoder.onnx')
        self.decoder_runner = SonicDecoder()
        self.decoder = self.decoder_runner.session
        self.planner = session(DEPLOY/'planner/target_vel/V2/planner_sonic.onnx',4)
        self.plan = np.tile(np.r_[0,0,.788740,1,0,0,0,DEFAULT],(100,1))
        self.frame = 0
        self.calls = 0
        self.timings = []
        self.planner_records=[]
        assert self.encoder.get_inputs()[0].shape == [1,1762]
        assert self.decoder.get_inputs()[0].shape == [1,994]

    def replan(self, speed, heading, mode, turn_in_place=False,waypoints=None,gt_xy=None):
        if waypoints is not None:
            # Global targets require the reference context and GT odometry to
            # share a translation origin. This shifts only planner reference XY;
            # it never writes the physical robot root or joint states.
            self.plan[:,:2]+=np.asarray(gt_xy)-sample(self.plan,np.array([self.frame]))[0,:2]
        # Continue the motion context, as in the upstream deployment runtime.
        context = sample(self.plan,self.frame+2+np.arange(4)*50/30)
        direction = np.array([[np.cos(heading),np.sin(heading),0]],np.float32)
        inputs = dict(context_mujoco_qpos=context[None].astype(np.float32),
                      target_vel=np.array([max(speed,.01)],np.float32),
                      mode=np.array([mode],np.int64),movement_direction=np.zeros_like(direction) if turn_in_place else direction,
                      facing_direction=direction,random_seed=np.array([self.seed+self.calls],np.int64),
                      height=np.array([-1],np.float32),has_specific_target=np.zeros((1,1),np.int64),
                      specific_target_positions=np.zeros((1,4,3),np.float32),
                      specific_target_headings=np.zeros((1,4),np.float32),
                      allowed_pred_num_tokens=np.array([[1]*6+[0]*5],np.int64))
        if waypoints is not None:
            positions,headings=waypoints
            inputs['has_specific_target'][:]=1
            inputs['specific_target_positions'][0]=positions
            inputs['specific_target_headings'][0]=headings
            inputs['allowed_pred_num_tokens'][:]=0
            inputs['allowed_pred_num_tokens'][0,5]=1 # 11 tokens = 44 frames at 30 Hz
        inputs = {i.name: inputs[i.name] for i in self.planner.get_inputs()}
        start=time.perf_counter()
        q,n=self.planner.run(None,inputs)
        self.timings.append(time.perf_counter()-start)
        n=int(np.asarray(n).item())
        assert 4 <= n <= 64
        if waypoints is not None:assert n==44
        self.planner_records.append(dict(context=context.copy(),has_target=int(inputs['has_specific_target'].item()),
            target_positions=inputs['specific_target_positions'][0].copy(),
            target_headings=inputs['specific_target_headings'][0].copy(),prediction_frames=n))
        new=sample(q[0,:n],np.arange(int(n*50/30))*30/50)
        # Insert two frames ahead, eight-frame crossfade (upstream deployment convention).
        old=sample(self.plan,self.frame+np.arange(len(new)+2))
        out=old.copy()
        out[2:]=new
        for j in range(min(8,len(new))):
            w=j/8
            out[2+j]=(1-w)*old[2+j]+w*new[j]
            pair=np.stack([old[2+j],new[j]])
            out[2+j,3:7]=sample(pair,np.array([w]))[0,3:7]
        self.plan=out
        self.frame=0
        self.calls+=1

    def act(self, data):
        idx=np.minimum(self.frame+np.arange(10)*5,len(self.plan)-1)
        refs=self.plan[idx]
        velocities=np.diff(self.plan[:,7:],axis=0)/DT
        velocities=np.vstack([velocities,velocities[-1]])
        q=rot(data.qpos[3:7])
        ori=(q.inv()*rot(refs[:,3:7])).as_matrix()[:,:,:2].reshape(-1)
        # Encoder mode 0: only joint q/dq and relative anchor orientation are active.
        # Layout taken verbatim from release/observation_config.yaml.
        enc=np.zeros(1762,np.float32)
        enc[4:294]=refs[:,7:][:,ORDER].ravel()
        enc[294:584]=velocities[idx][:,ORDER].ravel()
        enc[601:661]=ori
        token=self.encoder.run(None,{'obs_dict':enc[None]})[0][0]
        decoded=self.decoder_runner.step(token,RobotState.from_mujoco(data.qpos,data.qvel))
        ref=self.plan[min(self.frame,len(self.plan)-1)].copy()
        self.frame+=1
        return decoded.joint_target,token,decoded.action,ref,enc,decoded.observation


class PathTracker:
    def __init__(self,length=24.,amplitude=2.,lookahead=.8,speed=.6,arrival_radius=.18,path_xy=None):
        x=np.linspace(0,length,6001)
        # One long smooth S: zero lateral slope at both ends, plus straight run-in/out.
        u=np.clip((x-2)/(length-4),0,1)
        y=amplitude*np.sin(2*np.pi*u)**3
        self.path=np.c_[x,y] if path_xy is None else np.asarray(path_xy,dtype=float).copy()
        if self.path.ndim!=2 or self.path.shape[1]!=2 or len(self.path)<30 or not np.isfinite(self.path).all():
            raise ValueError('Path must have at least 30 finite XY points')
        if np.any(np.linalg.norm(np.diff(self.path,axis=0),axis=1)<1e-8):
            raise ValueError('Consecutive path points must be distinct')
        self.s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.path,axis=0),axis=1))]
        self.lookahead=lookahead
        self.speed=speed
        self.arrival_radius=arrival_radius
        self.last_s=0.
        self.v=0.
        self.heading=0.
        self.idle_since=None
        self.arrived_since=None
        self.goal_heading=float(np.arctan2(*(self.path[-1]-self.path[-30])[::-1]))
        self.braking_since=None
        self.heading_error=0.
        self.distance=float('inf')

    def cosine_speed(self,distance):
        # Decelerate over [1.20, 0.30] m, with zero slope at both boundaries.
        u=np.clip((1.2-distance)/(.9),0,1)
        return .15+(.6-.15)*.5*(1+np.cos(np.pi*u))

    def waypoints(self,progress,speed):
        # Last token of an 11-token/44-frame horizon: frames 40..43, 30 Hz.
        # At cruise, the first target is 0.6 * 40/30 = 0.8 m along the path.
        target_s=np.minimum(progress+speed*np.arange(40,44)/30,self.s[-1])
        xy=np.column_stack([np.interp(target_s,self.s,self.path[:,d]) for d in range(2)])
        tangent=np.gradient(self.path,axis=0)
        tangent_heading=np.unwrap(np.arctan2(tangent[:,1],tangent[:,0]))
        headings=np.interp(target_s,self.s,tangent_heading)
        return np.c_[xy,np.full(4,.788740)],headings

    def command(self,t,pos,actual_yaw=0.,actual_vxy=None):
        # Local segment projection; no time-indexed target that can run away from robot.
        lo=max(0,np.searchsorted(self.s,self.last_s-.4)-1)
        hi=min(len(self.s)-1,np.searchsorted(self.s,self.last_s+2.))
        a=self.path[lo:hi]; ab=self.path[lo+1:hi+1]-a
        f=np.clip(np.sum((pos-a)*ab,axis=1)/np.sum(ab*ab,axis=1),0,1)
        proj=a+f[:,None]*ab
        j=np.argmin(np.sum((pos-proj)**2,axis=1))
        i=lo+j
        progress=self.s[i]+f[j]*(self.s[i+1]-self.s[i])
        self.last_s=max(self.last_s,progress)
        tangent=ab[j]/np.linalg.norm(ab[j])
        error=np.cross(np.r_[tangent,0],np.r_[pos-proj[j],0])[2]
        target_s=min(progress+self.lookahead,self.s[-1])
        target=np.array([np.interp(target_s,self.s,self.path[:,d]) for d in range(2)])
        heading=np.arctan2(*(target-pos)[::-1])
        distance=np.linalg.norm(self.path[-1]-pos)
        self.distance=float(distance)
        self.heading_error=float(np.arctan2(np.sin(self.goal_heading-actual_yaw),np.cos(self.goal_heading-actual_yaw)))
        desired=self.speed
        phase=1
        if t<2:
            desired=0;phase=0;heading=0
        elif distance<=1.2:
            if self.braking_since is None:self.braking_since=t
            desired=self.cosine_speed(distance)
            phase=2
        if self.arrived_since is None and t>=2 and distance<=self.arrival_radius:
            self.arrived_since=t
        if self.arrived_since is not None:
            # Position arrival is permanent. Never point back towards the goal.
            heading=self.goal_heading
            target=self.path[-1].copy()
            desired=.15;phase=4
            if self.idle_since is not None or abs(self.heading_error)<=np.deg2rad(15):
                if self.idle_since is None:self.idle_since=t
                desired=0;phase=3
        # Startup ramp only. Braking uses the specified cosine command exactly.
        self.v=min(desired,self.v+.20*DT) if phase in [0,1] else desired
        delta=np.arctan2(np.sin(heading-self.heading),np.cos(heading-self.heading))
        self.heading+=np.clip(delta,-.6*DT,.6*DT)
        mode=0 if phase in [0,3] else 1
        return self.v,self.heading,mode,error,progress,target,phase


class PrecisionTracker(PathTracker):
    """Velocity-mode terminal controller; no root edits or waypoint injection."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.filtered_closing_speed=0.
        self.terminal_brake_done=False
        self.predicted_remaining=float('inf')

    def command(self,t,pos,actual_yaw=0.,actual_vxy=None):
        old_idle=self.idle_since
        result=list(super().command(t,pos,actual_yaw,actual_vxy))
        direction=np.array([np.cos(self.goal_heading),np.sin(self.goal_heading)])
        v=float(np.asarray(actual_vxy)@direction) if actual_vxy is not None else 0.
        self.filtered_closing_speed+=DT/(.2+DT)*(v-self.filtered_closing_speed)
        if self.arrived_since is None:
            if result[-1]==2 and self.distance<.55:
                # The original 0.15 m/s command often yields a stationary pose.
                self.v=max(result[0],.28)
                result[0]=self.v
            return tuple(result)
        if not self.terminal_brake_done:
            self.idle_since=old_idle
            remaining=float((self.path[-1]-pos)@direction)
            self.predicted_remaining=remaining-.25*max(0,self.filtered_closing_speed)
            # Position stays latched. No direction back towards the goal is used.
            # Finish once predicted rest position is near goal, or after 1 s.
            self.terminal_brake_done=(self.predicted_remaining<=.045 or remaining<=0
                                      or t-self.arrived_since>=1.)
            if not self.terminal_brake_done:
                self.v=.28
                return (.28,self.heading,1,*result[3:6],5)
            if abs(self.heading_error)<=np.deg2rad(15):
                self.idle_since=t
                self.v=0.
                return (0.,self.heading,0,*result[3:6],3)
            self.v=.15
            return (.15,self.heading,1,*result[3:6],4)
        return tuple(result)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'outputs/s_curve_v3')
    p.add_argument('--arrival-radius',type=float,default=.18)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--planner-mode',choices=['velocity','waypoints'],default='velocity')
    p.add_argument('--terminal-controller',choices=['legacy','predictive'],default='legacy')
    p.add_argument('--path-file',type=Path,help='JSON with points [N,2], name and optional id')
    p.add_argument('--duration',type=float,default=100)
    p.add_argument('--length',type=float,default=24)
    p.add_argument('--amplitude',type=float,default=2)
    p.add_argument('--offset',type=float,default=.35,help='Initial lateral offset, metres')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    route=json.loads(args.path_file.read_text()) if args.path_file else None
    sonic=Sonic(seed=args.seed)
    tracker_class=PrecisionTracker if args.terminal_controller=='predictive' else PathTracker
    tracker=tracker_class(args.length,args.amplitude,arrival_radius=args.arrival_radius,
                          path_xy=route['points'] if route else None)
    model=mujoco.MjModel.from_xml_path(str(DEPLOY/'g1/scene_29dof.xml'))
    model.opt.timestep=.002
    data=mujoco.MjData(model)
    data.qpos[:]=np.r_[0,args.offset,.788740,1,0,0,0,DEFAULT]
    mujoco.mj_forward(model,data)
    logs={k:[] for k in ['t','qpos','qvel','token','action','reference_qpos','target_joint_pos',
                           'command','cross_track_error','progress','lookahead_xy','phase','encoder_obs','decoder_obs','planner_speed',
                           'goal_distance','goal_heading_error','position_arrived']}
    completed=False
    termination_reason='time_limit'
    for k in range(round(args.duration/DT)):
        t=k*DT
        speed,heading,mode,error,progress,target,phase=tracker.command(t,data.qpos[:2],yaw(data.qpos[3:7]),data.qvel[:2])
        # Pure SONIC: pass the requested speed directly, without a speed PI servo.
        planner_speed=speed
        if k%5==0 or (logs['phase'] and phase!=logs['phase'][-1]):
            points=tracker.waypoints(progress,speed) if args.planner_mode=='waypoints' and phase in [1,2] else None
            sonic.replan(planner_speed,heading,mode,turn_in_place=(phase==4),waypoints=points,gt_xy=data.qpos[:2])
        target_q,token,action,ref,enc,obs=sonic.act(data)
        values=[t,data.qpos.copy(),data.qvel.copy(),token,action,ref,target_q,
                [speed,heading,mode],error,progress,target,phase,enc,obs,planner_speed,
                tracker.distance,tracker.heading_error,tracker.arrived_since is not None]
        for key,value in zip(logs,values):logs[key].append(value)
        # Check recorded, physically simulated states. Never duplicate a hold frame.
        if tracker.idle_since is not None and t-tracker.idle_since>=1.2-1e-8:
            recent=np.array(logs['qvel'][-16:])
            stable=np.max(np.linalg.norm(recent[:,:2],axis=1))<.12
            if stable and abs(tracker.heading_error)<=np.deg2rad(15):
                completed=True
                termination_reason='completed'
                break
        for _ in range(10):
            data.ctrl[:]=np.clip(KP*(target_q-data.qpos[7:])-KD*data.qvel[6:],
                                 model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
            mujoco.mj_step(model,data)
        if k%250==0:
            print(f't={t:.1f} s={progress:.2f}/{tracker.s[-1]:.2f} xy={data.qpos[:2]} z={data.qpos[2]:.3f} e={error:.3f} v={np.linalg.norm(data.qvel[:2]):.3f}',flush=True)
        if not np.isfinite(data.qpos).all() or data.qpos[2]<.4:
            print('FAILED: fall/nonfinite state; retaining diagnostic data',flush=True)
            termination_reason='fall_or_nonfinite'
            break
    arrays={key:np.array(v) for key,v in logs.items()}
    arrays.update(path_xy=tracker.path,path_s=tracker.s,frequency_hz=np.array(50))
    for key in ['context','has_target','target_positions','target_headings','prediction_frames']:
        arrays['planner_'+key]=np.array([r[key] for r in sonic.planner_records])
    np.savez_compressed(args.output/'episode.npz',**arrays)
    tail=arrays['t']>arrays['t'][-1]-.3
    cruise=(arrays['phase']==1)&(arrays['t']>10)&(arrays['progress']<tracker.s[-1]-2)
    summary=dict(completed=completed,termination_reason=termination_reason,frames=len(arrays['t']),duration_s=float(arrays['t'][-1]),
                 path_length_m=float(tracker.s[-1]),progress_m=float(arrays['progress'][-1]),
                 cross_track_rmse_m=float(np.sqrt(np.mean(arrays['cross_track_error']**2))),
                 cross_track_max_m=float(np.max(abs(arrays['cross_track_error']))),
                 endpoint_error_m=float(np.linalg.norm(data.qpos[:2]-tracker.path[-1])),
                 final_0_3s_max_speed_mps=float(np.max(np.linalg.norm(arrays['qvel'][tail,:2],axis=1))),
                 final_heading_error_deg=float(np.rad2deg(abs(arrays['goal_heading_error'][-1]))),
                 position_arrived_time_s=tracker.arrived_since,idle_start_time_s=tracker.idle_since,
                 idle_simulated_duration_s=float(arrays['t'][-1]-tracker.idle_since) if tracker.idle_since is not None else 0.,
                 braking_start_time_s=tracker.braking_since,goal_heading_rad=tracker.goal_heading,
                 braking_start_distance_m=1.2,braking_end_distance_m=.30,arrival_radius_m=tracker.arrival_radius,
                 heading_tolerance_deg=15,minimum_idle_hold_s=1.2,
                 pure_sonic=True,speed_pi_enabled=False,
                 planner_random_seed=args.seed,
                 planner_mode=args.planner_mode,
                 trajectory_name=route['name'] if route else 'S-curve',
                 trajectory_id=route.get('id') if route else None,
                 trajectory_file=str(args.path_file.resolve()) if args.path_file else None,
                 terminal_controller=args.terminal_controller,
                 terminal_min_command_mps=.28 if args.terminal_controller=='predictive' else .15,
                 terminal_speed_floor_distance_m=.55 if args.terminal_controller=='predictive' else None,
                 terminal_prediction_seconds=.25 if args.terminal_controller=='predictive' else None,
                 terminal_rest_margin_m=.045 if args.terminal_controller=='predictive' else None,
                 waypoint_horizon_frames=44 if args.planner_mode=='waypoints' else None,
                 waypoint_gt_xy_context_alignment=args.planner_mode=='waypoints',
                 cruise_mean_speed_mps=float(np.mean(np.linalg.norm(arrays['qvel'][cruise,:2],axis=1))) if cruise.any() else None,
                 minimum_root_height_m=float(arrays['qpos'][:,2].min()),
                 planner_mean_wall_seconds=float(np.mean(sonic.timings)),
                 nominal_speed_mps=.6,lookahead_m=.8,frequency_hz=50,initial_lateral_offset_m=args.offset,
                 backend='MuJoCo free-base dynamics + official SONIC ONNX encoder/decoder/planner, CPU',
                 quaternion_order='wxyz',joint_order='MuJoCo; action and proprioception use IsaacLab ORDER',
                 phase_names=['warmup','tracking','braking','idle','heading_alignment','terminal_braking'],
                 model_sha256={str(f.relative_to(DEPLOY)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [
                     DEPLOY/'policy/release/model_encoder.onnx',DEPLOY/'policy/release/model_decoder.onnx',
                     DEPLOY/'planner/target_vel/V2/planner_sonic.onnx']})
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
