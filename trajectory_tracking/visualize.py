"""Export a static diagnostic figure and a MuJoCo replay with a trajectory panel."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','1')
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter1d
from trajectory_tracking.arm_export import DEFAULT,ARM_JOINTS

from trajectory_tracking.paths import DEPLOY


def main():
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    p.add_argument('--video',action='store_true')
    p.add_argument('--video-fps',type=int,choices=[10,25,50],default=25)
    args=p.parse_args()
    z=np.load(args.directory/'episode.npz')
    meta=json.loads((args.directory/'summary.json').read_text())
    t=z['t'];q=z['qpos'];path=z['path_xy'];phase=z['phase']
    processed=np.load(args.directory/'motion_arms_50hz.npz')['qpos']
    speed=np.linalg.norm(z['qvel'][:,:2],axis=1)
    smooth=uniform_filter1d(speed,50,mode='nearest')
    forward=np.sum(z['qvel'][:,:2]*np.c_[np.cos(z['command'][:,1]),np.sin(z['command'][:,1])],axis=1)
    forward_smooth=uniform_filter1d(forward,50,mode='nearest')
    fig,axs=plt.subplots(3,2,figsize=(15,12),layout='constrained')
    ax=axs[0,0]
    ax.plot(*path.T,'--',color='#7a7a7a',label='Reference path')
    sc=ax.scatter(*q[:,:2].T,c=t,s=3,cmap='viridis',label='SONIC + GT odometry')
    ax.plot(*q[0,:2],'o',color='#dd6b20',label='Start (0.35 m offset)')
    ax.plot(*path[-1],'*',ms=12,color='#e53e3e',label='Goal')
    ax.axis('equal');ax.set(xlabel='World X [m]',ylabel='World Y [m]',title='Free-base MuJoCo rollout')
    ax.legend(fontsize=8);fig.colorbar(sc,ax=ax,label='Simulation time [s]')
    ax=axs[0,1]
    ax.plot(t,z['cross_track_error'],color='#257ab4')
    ax.axhline(0,color='k',lw=.6)
    ax.set(xlabel='Time [s]',ylabel='Signed cross-track error [m]',title='GT odometry feedback: gradual path recovery')
    ax=axs[1,0]
    ax.plot(t,speed,color='#b9d3e8',lw=.5,label='GT instantaneous speed')
    ax.plot(t,smooth,label='GT speed, 1 s average')
    ax.plot(t,forward_smooth,label='GT along-command, 1 s average',color='#4a9b58')
    ax.plot(t,z['command'][:,0],'--',label='Desired speed')
    ax.axhline(.6,color='#d87827',ls=':',label='Cruise 0.6 m/s')
    ax.set(xlabel='Time [s]',ylabel='Speed [m/s]',title='Acceleration, cruise and terminal stop');ax.legend(fontsize=8)
    ax=axs[1,1]
    stop=t>max(0,t[-1]-12)
    ax.plot(t[stop],z['goal_distance'][stop],label='GT distance to goal')
    ax.axhline(meta['arrival_radius_m'],color='k',ls=':',label=f"Position latch: {meta['arrival_radius_m']:.2f} m")
    ax.axhline(1.2,color='#caa25b',ls=':',label='Cosine brake: 1.20 m')
    heading_ax=ax.twinx()
    heading_ax.plot(t[stop],np.rad2deg(abs(z['goal_heading_error'][stop])),color='#a45491',label='GT heading error')
    heading_ax.axhline(15,color='#a45491',ls=':',alpha=.5)
    heading_ax.set_ylabel('Absolute heading error [deg]')
    ax.set(xlabel='Time [s]',ylabel='Distance [m]',title=f"Terminal position / heading | final heading {meta['final_heading_error_deg']:.2f} deg")
    ax.legend(fontsize=8)
    ax=axs[2,0]
    raw_delta=np.rad2deg(abs(q[:,7+ARM_JOINTS]-DEFAULT[ARM_JOINTS]))
    out_delta=np.rad2deg(abs(processed[:,7+ARM_JOINTS]-DEFAULT[ARM_JOINTS]))
    ax.plot(t,raw_delta.max(axis=1),color='#818d96',lw=1,label='Raw arm max deviation')
    ax.plot(t,out_delta.max(axis=1),color='#1689d2',label='Exported arm max deviation')
    ax.set(xlabel='Time [s]',ylabel='Deviation from natural stand [deg]',title='Offline arms: clip shoulder/elbow/wrist, retain 20%')
    ax.legend(fontsize=8)
    ax=axs[2,1]
    ax.plot(t[stop],speed[stop],label='GT horizontal speed')
    ax.plot(t[stop],z['command'][stop,0],'--',label='SONIC speed command')
    ax.axhline(.12,color='k',ls=':',label='Final stability check')
    if meta['idle_start_time_s'] is not None:
        ax.axvline(meta['idle_start_time_s'],color='#28835c',ls='--',label='Standing mode start')
    ax.set(xlabel='Time [s]',ylabel='Speed [m/s]',title=f"Real simulated standing hold: {meta['idle_simulated_duration_s']:.2f} s")
    ax.legend(fontsize=8)
    for ax in axs.ravel():
        ax.grid(alpha=.2)
        for i,name,color in [(0,'warmup','#d8d8d8'),(2,'braking','#fff0bb'),(3,'idle','#d4edda')]:
            inds=np.flatnonzero(phase==i)
            if len(inds) and ax is not axs[0,0]:ax.axvspan(t[inds[0]],t[inds[-1]],color=color,alpha=.35,zorder=-2)
    axs[1,1].set_xlim(t[stop][0],t[-1])
    axs[2,1].set_xlim(t[stop][0],t[-1])
    fig.suptitle(f"SONIC / {meta.get('trajectory_name','S-curve')} | 50 Hz motion + 64-D tokens | lookahead 0.8 m | complete={meta['completed']}")
    fig.savefig(args.directory/'tracking.png',dpi=160)
    plt.close(fig)
    if not args.video:return
    model=mujoco.MjModel.from_xml_path(str(DEPLOY/'g1/scene_29dof.xml'))
    model.vis.global_.offwidth=640;model.vis.global_.offheight=640
    data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=640,width=640)
    cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
    cam.distance=3.3;cam.azimuth=125;cam.elevation=-20
    panel_fig,panel_ax=plt.subplots(figsize=(4,6.4),dpi=100)
    panel_fig.subplots_adjust(left=.19,right=.97,bottom=.10,top=.87)
    panel_ax.plot(path[:,1],path[:,0],'--',color='#939393',lw=2,label='Reference')
    trail,=panel_ax.plot([],[],color='#1689d2',lw=2,label='GT rollout')
    dot,=panel_ax.plot([],[],'o',color='#e35b35',ms=8)
    look,=panel_ax.plot([],[],'x',color='#bf8f00',ms=8,label='0.8 m lookahead')
    panel_ax.set(xlim=(path[:,1].min()-.7,path[:,1].max()+.7),
                 ylim=(path[:,0].min()-.7,path[:,0].max()+.7),xlabel='Y [m]',ylabel='X [m]')
    panel_ax.set_aspect('equal',adjustable='box')
    panel_ax.plot(path[-1,1],path[-1,0],'*',ms=10,color='#bd4c31',label='Goal')
    panel_ax.legend(loc='upper left');panel_ax.grid(alpha=.2)
    with imageio.get_writer(args.directory/'rollout.mp4',fps=args.video_fps,codec='libx264',quality=7,macro_block_size=1) as writer:
        for k in range(0,len(t),50//args.video_fps):
            data.qpos[:]=q[k];mujoco.mj_forward(model,data)
            cam.lookat[:]=q[k,:3]+[0,0,.15]
            renderer.update_scene(data,camera=cam)
            # Draw nearby original path as collision-free visual capsules.
            for j in range(0,len(path)-5,15):
                a,b=path[j],path[min(j+15,len(path)-1)]
                if np.linalg.norm(a-q[k,:2])>5:continue
                scene=renderer.scene
                geom=scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.95,.65,.12,1]))
                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.012,np.r_[a,.015],np.r_[b,.015])
                scene.ngeom+=1
            frame=renderer.render().copy()
            data.qpos[:]=processed[k];mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=cam)
            frame_export=renderer.render().copy()
            trail.set_data(q[:k+1,1],q[:k+1,0]);dot.set_data([q[k,1]],[q[k,0]])
            look.set_data([z['lookahead_xy'][k,1]],[z['lookahead_xy'][k,0]])
            panel_ax.set_title(f"{t[k]:.2f} s | {meta['phase_names'][int(phase[k])]}\nGT speed {smooth[k]:.2f} m/s | error {z['cross_track_error'][k]:+.2f} m",fontsize=10)
            panel_fig.canvas.draw()
            panel=np.asarray(panel_fig.canvas.buffer_rgba())[:,:,:3]
            canvas=Image.fromarray(np.concatenate([frame,frame_export,panel],axis=1))
            draw=ImageDraw.Draw(canvas)
            draw.rectangle((8,8,630,45),fill=(245,245,245))
            draw.text((18,18),'RAW SONIC simulation | original arms + original tokens',fill=(20,30,40))
            draw.rectangle((648,8,1270,45),fill=(245,245,245))
            draw.text((658,18),'OFFLINE arm export | clip + retain 20% | same root / legs',fill=(20,30,40))
            writer.append_data(np.asarray(canvas))
            if k%500==0:print(f'render {k}/{len(t)}',flush=True)
    renderer.close();plt.close(panel_fig)
    meta['video_fps']=args.video_fps
    (args.directory/'summary.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':main()
