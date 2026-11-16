"""Offline arm-only pose processing. Never called by the SONIC control loop."""
import json
from pathlib import Path
import mujoco
import numpy as np
from trajectory_tracking.collect import DEFAULT, DEPLOY, DT, ORDER

# MuJoCo joint order: each arm is shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw.
ARM_JOINTS=np.arange(15,29)
LIMITS=np.array([.45,.30,.35,.50,.25,.20,.20]*2)
RETAIN=.20


def soften(qpos):
    out=np.asarray(qpos).copy()
    delta=out[:,7+ARM_JOINTS]-DEFAULT[ARM_JOINTS]
    out[:,7+ARM_JOINTS]=DEFAULT[ARM_JOINTS]+RETAIN*np.clip(delta,-LIMITS,LIMITS)
    return out


def motion_fields(qpos,qvel):
    model=mujoco.MjModel.from_xml_path(str(DEPLOY/'g1/scene_29dof.xml'))
    data=mujoco.MjData(model)
    pos=[];quat=[];vel=[]
    for q,v in zip(qpos,qvel):
        data.qpos[:]=q;data.qvel[:]=v;mujoco.mj_forward(model,data)
        pos.append(data.xpos[1:].copy());quat.append(data.xquat[1:].copy())
        bv=np.empty((model.nbody-1,6))
        for j in range(1,model.nbody):
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,j,bv[j-1],0)
        vel.append(bv)
    vel=np.array(vel)
    return dict(qpos=qpos,qvel=qvel,joint_pos=qpos[:,7:],joint_vel=qvel[:,6:],
                body_pos_w=np.array(pos),body_quat_w=np.array(quat),
                body_lin_vel_w=vel[:,:,3:],body_ang_vel_w=vel[:,:,:3],
                joint_names=np.array([model.joint(j).name for j in range(1,model.njnt)]),
                body_names=np.array([model.body(j).name for j in range(1,model.nbody)]),
                isaaclab_joint_order=ORDER)


def export(folder,z):
    folder=Path(folder)
    raw=motion_fields(z['qpos'],z['qvel'])
    q=soften(z['qpos'])
    v=z['qvel'].copy()
    v[:,6+ARM_JOINTS]=np.gradient(q[:,7+ARM_JOINTS],DT,axis=0,edge_order=2)
    processed=motion_fields(q,v)
    np.testing.assert_array_equal(q[:,:22],z['qpos'][:,:22])
    np.testing.assert_array_equal(v[:,:21],z['qvel'][:,:21])
    unchanged=[i for i,name in enumerate(raw['body_names']) if not any(s in name for s in ['shoulder','elbow','wrist'])]
    for key in ['body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']:
        np.testing.assert_allclose(processed[key][:,unchanged],raw[key][:,unchanged],rtol=0,atol=1e-12)
    for filename,fields in [('motion_50hz.npz',raw),('motion_arms_50hz.npz',processed)]:
        np.savez_compressed(folder/filename,fps=np.array(50),t=z['t'],**fields)
    deviation_raw=z['qpos'][:,7+ARM_JOINTS]-DEFAULT[ARM_JOINTS]
    deviation_export=q[:,7+ARM_JOINTS]-DEFAULT[ARM_JOINTS]
    config=dict(retain_ratio=RETAIN,pre_scale_limits_rad=dict(shoulder_pitch=.45,shoulder_roll=.30,
                shoulder_yaw=.35,elbow=.50,wrist_roll=.25,wrist_pitch=.20,wrist_yaw=.20),
                baseline_joint_angles_rad=DEFAULT[ARM_JOINTS].tolist(),
                joint_names=raw['joint_names'][ARM_JOINTS].tolist(),
                online_arm_constraint=False,token_modified=False,
                raw_arm_max_deviation_deg=float(np.rad2deg(abs(deviation_raw).max())),
                exported_arm_max_deviation_deg=float(np.rad2deg(abs(deviation_export).max())),
                root_waist_legs_unchanged=True)
    (folder/'arm_export.json').write_text(json.dumps(config,indent=2))
    return config
