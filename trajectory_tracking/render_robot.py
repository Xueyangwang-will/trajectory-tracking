"""Robot-only 25 FPS view of original, physically simulated qpos (no arm edits)."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','1')
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from trajectory_tracking.collect import DEPLOY, yaw


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    z=np.load(args.directory/'episode.npz');meta=json.loads((args.directory/'summary.json').read_text())
    q=z['qpos'];path=z['path_xy'];t=z['t'];phase=z['phase']
    model=mujoco.MjModel.from_xml_path(str(DEPLOY/'g1/scene_29dof.xml'))
    model.vis.global_.offwidth=1280;model.vis.global_.offheight=720
    model.vis.quality.shadowsize=2048
    data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=720,width=1280)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.distance=2.9;camera.elevation=-18
    heading=0.
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    path_segments=np.arange(0,len(path)-1,18)
    def capsule(start,end,radius,color):
        scene=renderer.scene
        if scene.ngeom>=scene.maxgeom:return
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.asarray(color,np.float32))
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,radius,start,end)
        scene.ngeom+=1
    with imageio.get_writer(args.directory/'robot.mp4',fps=25,codec='libx264',quality=8,macro_block_size=1,
                           ffmpeg_params=['-movflags','+faststart']) as writer:
        for k in range(0,len(q),2):
            data.qpos[:]=q[k];mujoco.mj_forward(model,data)
            target_yaw=yaw(q[k,3:7]);delta=np.arctan2(np.sin(target_yaw-heading),np.cos(target_yaw-heading))
            heading+=.07*delta
            camera.lookat[:]=q[k,:3]+[0,0,-.08];camera.azimuth=135+np.rad2deg(heading)
            renderer.update_scene(data,camera=camera)
            for j in path_segments:
                if np.linalg.norm(path[j]-q[k,:2])>5:continue
                end=min(j+18,len(path)-1)
                capsule(np.r_[path[j],.012],np.r_[path[end],.012],.014,[.98,.69,.15,1])
            # A ground-level circle marks the actual 18 cm arrival radius.
            if np.linalg.norm(path[-1]-q[k,:2])<4:
                angle=np.linspace(0,2*np.pi,49)
                circle=path[-1]+.18*np.c_[np.cos(angle),np.sin(angle)]
                for a,b in zip(circle[:-1],circle[1:]):
                    capsule(np.r_[a,.018],np.r_[b,.018],.009,[.2,.8,.5,1])
            frame=Image.fromarray(renderer.render())
            draw=ImageDraw.Draw(frame)
            draw.rectangle((14,14,815,83),fill=(241,246,249))
            caption=meta.get('render_caption','SONIC / G1 | Original MuJoCo physics rollout | 25 FPS')
            draw.text((27,23),caption,font=font,fill=(25,45,60))
            text=f"{meta['trajectory_name']}   {t[k]:05.2f}s   {meta['phase_names'][int(phase[k])]}   goal distance {z['goal_distance'][k]:.2f} m"
            draw.text((27,51),text,font=font,fill=(25,45,60))
            if k==min(300,(len(q)-1)//2*2):frame.save(args.directory/'robot-preview.jpg',quality=90)
            writer.append_data(np.asarray(frame))
    renderer.close()
    print(f"Robot view ready: {args.directory} ({t[-1]:.2f} s)",flush=True)


if __name__=='__main__':main()
