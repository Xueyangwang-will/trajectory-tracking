"""Validate ten different path geometries and export their comparison report."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    root=args.directory;web=root/'web';web.mkdir(exist_ok=True)
    definitions=json.loads((root/'routes/routes.json').read_text())
    trials=[];summaries=[]
    for route in definitions:
        run=root/f"trial-{route['id']:02d}"
        with (run/'validate.log').open('w') as log:
            subprocess.run([sys.executable,'-m','trajectory_tracking.validate_export',str(run),'--allow-incomplete'],stdout=log,stderr=subprocess.STDOUT,check=True)
        z=np.load(run/'episode.npz');s=json.loads((run/'summary.json').read_text())
        length=float(z['path_s'][-1])
        middle=(z['progress']>=min(5,length*.25))&(z['progress']<length-2)&(z['phase']==1)
        s.update(trajectory_label=route['label'],entered_radius=bool(z['position_arrived'].any()),
            final_recorded_distance_m=float(z['goal_distance'][-1]),
            final_within_radius=bool(z['goal_distance'][-1]<=s['arrival_radius_m']),
            middle_rmse_m=float(np.sqrt(np.mean(z['cross_track_error'][middle]**2))) if middle.any() else None,
            middle_max_error_m=float(abs(z['cross_track_error'][middle]).max()) if middle.any() else None)
        (run/'summary.json').write_text(json.dumps(s,indent=2,ensure_ascii=False));summaries.append(s)
        ids=np.unique(np.r_[np.arange(0,len(z['t']),5),len(z['t'])-1]).astype(int)
        def pack(x):return np.round(x,5).tolist()
        trials.append(dict(id=route['id'],name=route['label'],summary=s,t=pack(z['t'][ids]),
            xy=pack(z['qpos'][ids,:2]),path=pack(z['path_xy'][::5]),goal=pack(z['path_xy'][-1]),
            distance=pack(z['goal_distance'][ids]),error=pack(z['cross_track_error'][ids]),
            heading_error=pack(np.rad2deg(abs(z['goal_heading_error'][ids]))),
            speed=pack(np.linalg.norm(z['qvel'][ids,:2],axis=1)),desired=pack(z['command'][ids,0]),
            phase=z['phase'][ids].tolist()))
        dst=web/run.name;dst.mkdir(exist_ok=True)
        for name in ['summary.json','arm_export.json','episode.npz','motion_50hz.npz','motion_arms_50hz.npz','tokens_50hz.npz']:
            shutil.copyfile(run/name,dst/name)
        print(f"{route['id']:02d} {route['name']}: completed={s['completed']} final={s['final_recorded_distance_m']*100:.2f} cm",flush=True)
    distances=np.array([s['final_recorded_distance_m'] for s in summaries])
    stats=dict(paths=len(trials),entered=sum(s['entered_radius'] for s in summaries),
        completed=sum(s['completed'] for s in summaries),final_within=sum(s['final_within_radius'] for s in summaries),
        median_cm=float(np.median(distances)*100),mean_cm=float(distances.mean()*100),
        min_cm=float(distances.min()*100),max_cm=float(distances.max()*100))
    (root/'routes_results.json').write_text(json.dumps(dict(stats=stats,summaries=summaries),indent=2,ensure_ascii=False))
    shutil.copyfile(root/'routes_results.json',web/'batch.json')
    columns=['trajectory_id','trajectory_name','trajectory_label','path_length_m','planner_random_seed','entered_radius',
        'completed','final_within_radius','final_recorded_distance_m','middle_rmse_m','middle_max_error_m',
        'final_heading_error_deg','idle_simulated_duration_s','duration_s']
    with (web/'batch.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)
    fig,axes=plt.subplots(2,5,figsize=(20,8),layout='constrained')
    for ax,tr in zip(axes.ravel(),trials):
        p=np.array(tr['path']);q=np.array(tr['xy']);s=tr['summary']
        ax.plot(*p.T,'--',color='#8b969e',lw=1,label='Reference');ax.plot(*q.T,color='#167abd',lw=1,label='GT rollout')
        ax.plot(*q[0],'o',color='#d47b2e',ms=4);ax.plot(*p[-1],'*',color='#ba493d',ms=8)
        ax.set_title(f"{tr['id']:02d} {s['trajectory_name']}\nend {s['final_recorded_distance_m']*100:.2f} cm / complete={s['completed']}",fontsize=10)
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.set_xlabel('X [m]');ax.set_ylabel('Y [m]')
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f"10 distinct geometries / same seed 43 / {stats['completed']}/10 completed / speed command 0.6 m/s, lookahead 0.8 m")
    fig.savefig(web/'batch.png',dpi=140);plt.close(fig)
    html=Path(__file__).with_name('routes.template.html').read_text()
    (web/'index.html').write_text(html.replace('__DATA__',json.dumps(dict(stats=stats,trials=trials),separators=(',',':'),ensure_ascii=False)))
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
