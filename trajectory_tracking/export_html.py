"""Bundle rollout, synchronized charts and NPZ downloads into one offline HTML."""
import argparse
import base64
import json
import re
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
from trajectory_tracking.arm_export import DEFAULT,ARM_JOINTS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--web-directory',type=Path)
    parser.add_argument('--compare',type=Path,help='Previous episode directory for threshold comparison')
    args=parser.parse_args()
    folder=args.directory
    z=np.load(folder/'episode.npz')
    v=z['qvel'][:,:2]
    heading=z['command'][:,1]
    processed=np.load(folder/'motion_arms_50hz.npz')['qpos']
    forward=(v*np.c_[np.cos(heading),np.sin(heading)]).sum(axis=1)
    def packed(x):return np.round(x,5).tolist()
    data=dict(t=packed(z['t']),xy=packed(z['qpos'][:,:2]),path=packed(z['path_xy'][::10]),
              lookahead=packed(z['lookahead_xy']),error=packed(z['cross_track_error']),
              speed=packed(np.linalg.norm(v,axis=1)),desired=packed(z['command'][:,0]),
              forward_smooth=packed(uniform_filter1d(forward,50,mode='nearest')),
              phase=z['phase'].tolist(),progress=packed(z['progress']),tokens=packed(z['token']),
              goal_distance=packed(z['goal_distance']),heading_error=packed(np.rad2deg(abs(z['goal_heading_error']))),
              position_arrived=z['position_arrived'].tolist(),
              arm_raw=packed(np.rad2deg(abs(z['qpos'][:,7+ARM_JOINTS]-DEFAULT[ARM_JOINTS])).max(axis=1)),
              arm_processed=packed(np.rad2deg(abs(processed[:,7+ARM_JOINTS]-DEFAULT[ARM_JOINTS])).max(axis=1)),
              token_min=float(z['token'].min()),token_max=float(z['token'].max()),
              summary=json.loads((folder/'summary.json').read_text()))
    if args.compare:
        data['comparison']=json.loads((args.compare/'summary.json').read_text())
    template=Path(__file__).with_name('report.template.html').read_text()
    # Arrival radius is a report parameter, distinct from the cosine brake endpoint.
    radius=data['summary']['arrival_radius_m']
    template=template.replace('0.30 m',f'{radius:.2f} m').replace('[[1.2,\'1.20 m 开始减速\'],[.3,',f"[[1.2,'1.20 m 开始减速'],[{radius},")
    version='3' if radius==.18 else '2'
    template=template.replace('· V2','· V'+version)
    if data['summary'].get('terminal_controller')=='predictive':
        template=template.replace('1.2 m 内余弦减速至 0.15 m/s。','1.2 m 内开始余弦减速；最后 0.55 m 内速度指令不低于 0.28 m/s。')
    template=template.replace('视频 25 FPS',f"视频 {data['summary'].get('video_fps',25)} FPS")
    assets=[('POSTER','preview.png'),('VIDEO','rollout.mp4'),('MOTION','motion_50hz.npz'),
            ('ARMS','motion_arms_50hz.npz'),('TOKENS','tokens_50hz.npz'),('FIGURE','tracking.png')]
    template=template.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')))
    if args.web_directory:
        import shutil
        args.web_directory.mkdir(parents=True,exist_ok=True)
        web=template
        for name,file in assets:
            shutil.copyfile(folder/file,args.web_directory/file)
            web=re.sub(r'data:[^;"\s]+;base64,__'+name+'__',file+'?v='+version,web)
        (args.web_directory/'index.html').write_text(web)
        for file in ['summary.json','arm_export.json']:
            shutil.copyfile(folder/file,args.web_directory/file)
    for name,file in assets:
        template=template.replace('__'+name+'__',base64.b64encode((folder/file).read_bytes()).decode())
    output=folder/'sonic-s-curve.html'
    output.write_text(template)
    print(f'{output.resolve()} ({output.stat().st_size/1024**2:.1f} MiB)')


if __name__=='__main__':main()
