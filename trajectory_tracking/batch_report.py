"""Validate repeated 18 cm trials and build a local, offline batch comparison."""
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
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--baseline',type=Path,help='Velocity-mode trial directory for paired comparisons')
    parser.add_argument('--previous-cohort',type=Path,help='Previous independent cohort for aggregate comparisons')
    args=parser.parse_args()
    folder=args.directory
    web=folder/'web';web.mkdir(parents=True,exist_ok=True)
    summaries=[];trials=[];paired=[]
    for run in sorted(folder.glob('trial-*')):
        with (run/'validate.log').open('w') as log:
            subprocess.run([sys.executable,'-m','trajectory_tracking.validate_export',
                            str(run),'--allow-incomplete'],stdout=log,stderr=subprocess.STDOUT,check=True)
        s=json.loads((run/'summary.json').read_text())
        z=np.load(run/'episode.npz')
        s['final_recorded_distance_m']=float(z['goal_distance'][-1])
        s['entered_radius']=bool(z['position_arrived'].any())
        s['final_within_radius']=bool(s['final_recorded_distance_m']<=s['arrival_radius_m'])
        mask=(z['t']>12)&(z['phase']==1)&(z['progress']<z['path_s'][-1]-2)
        s['middle_max_abs_cross_track_m']=float(np.max(abs(z['cross_track_error'][mask])))
        if args.baseline and (args.baseline/run.name/'episode.npz').exists():
            base=np.load(args.baseline/run.name/'episode.npz')
            bs=json.loads((args.baseline/run.name/'summary.json').read_text())
            pair=dict(seed=s['planner_random_seed'])
            candidate='precision' if s.get('terminal_controller')=='predictive' else 'waypoints'
            for tag,episode,summary in [('velocity',base,bs),(candidate,z,s)]:
                # Match physical path section, rather than time, across unequal speeds.
                common=(episode['progress']>=5)&(episode['progress']<episode['path_s'][-1]-2)&(episode['phase']==1)
                err=episode['cross_track_error'][common]
                pair[tag]=dict(rmse_cm=float(np.sqrt(np.mean(err**2))*100),
                    max_cm=float(np.max(abs(err))*100),
                    along_path_speed_mps=float(np.gradient(episode['progress'],.02)[common].mean()),
                    final_cm=float(episode['goal_distance'][-1]*100),completed=summary['completed'])
            paired.append(pair)
        (run/'summary.json').write_text(json.dumps(s,indent=2))
        summaries.append(s)
        # Display only is downsampled; all exports and exact metrics are 50 Hz.
        ids=np.unique(np.r_[np.arange(0,len(z['t']),5),len(z['t'])-1]).astype(int)
        def pack(x):return np.round(x,5).tolist()
        trials.append(dict(seed=s['planner_random_seed'],summary=s,t=pack(z['t'][ids]),
                      xy=pack(z['qpos'][ids,:2]),distance=pack(z['goal_distance'][ids]),
                      error=pack(z['cross_track_error'][ids]),heading_error=pack(np.rad2deg(abs(z['goal_heading_error'][ids]))),
                      speed=pack(np.linalg.norm(z['qvel'][ids,:2],axis=1)),phase=z['phase'][ids].tolist()))
        target=web/run.name;target.mkdir(exist_ok=True)
        for name in ['summary.json','arm_export.json','episode.npz','motion_50hz.npz','motion_arms_50hz.npz','tokens_50hz.npz']:
            shutil.copyfile(run/name,target/name)
        print(f"seed {s['planner_random_seed']}: entered={s['entered_radius']} completed={s['completed']} final={s['final_recorded_distance_m']*100:.2f} cm",flush=True)
    final=np.array([s['final_recorded_distance_m'] for s in summaries])
    stats=dict(trials=len(summaries),entered_count=sum(s['entered_radius'] for s in summaries),
               completed_count=sum(s['completed'] for s in summaries),
               final_within_count=sum(s['final_within_radius'] for s in summaries),
               final_mean_cm=float(final.mean()*100),final_median_cm=float(np.median(final)*100),
               final_min_cm=float(final.min()*100),final_max_cm=float(final.max()*100))
    cohorts=None
    if args.previous_cohort:
        previous=json.loads((args.previous_cohort/'batch.json').read_text())
        all_summaries=previous['summaries']+summaries
        distances=np.array([s['final_recorded_distance_m'] for s in all_summaries])
        combined=dict(trials=len(all_summaries),entered_count=sum(s['entered_radius'] for s in all_summaries),
            completed_count=sum(s['completed'] for s in all_summaries),
            final_within_count=sum(s['final_within_radius'] for s in all_summaries),
            final_mean_cm=float(distances.mean()*100),final_median_cm=float(np.median(distances)*100),
            final_min_cm=float(distances.min()*100),final_max_cm=float(distances.max()*100))
        cohorts=dict(previous=previous['stats'],current=stats,combined=combined)
    reference=np.load(next(folder.glob('trial-*'))/'episode.npz')['path_xy']
    kind='precision' if summaries[0].get('terminal_controller')=='predictive' else 'waypoints'
    payload=dict(stats=stats,trials=trials,path=np.round(reference[::10],5).tolist(),paired=paired,comparison_kind=kind,cohorts=cohorts)
    (folder/'batch.json').write_text(json.dumps(dict(stats=stats,summaries=summaries,paired=paired,cohorts=cohorts),indent=2))
    shutil.copyfile(folder/'batch.json',web/'batch.json')
    columns=['planner_random_seed','entered_radius','completed','final_within_radius','final_recorded_distance_m',
             'minimum_goal_distance_m','final_heading_error_deg','idle_simulated_duration_s','duration_s','settled_cross_track_rmse_m']
    with (web/'batch.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)
    fig,axs=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    axs[0].plot(*reference.T,'--',color='gray',label='Path')
    angle=np.linspace(0,2*np.pi,200)
    axs[1].plot(24+.18*np.cos(angle),.18*np.sin(angle),'--',color='gray',label='18 cm arrival radius')
    for tr in trials:
        a=np.array(tr['xy']);s=tr['summary'];time=np.array(tr['t']);dist=np.array(tr['distance'])
        axs[0].plot(*a.T,lw=1,label=f"Seed {tr['seed']}")
        axs[1].plot(*a.T,lw=1)
        axs[1].plot(*a[-1],'o',ms=5)
        if s['braking_start_time_s'] is not None:
            mask=time>=s['braking_start_time_s']
            axs[2].plot(time[mask]-s['braking_start_time_s'],dist[mask]*100,label=f"Seed {tr['seed']}")
    axs[0].set(xlabel='X [m]',ylabel='Y [m]',title='GT path tracking');axs[0].axis('equal');axs[0].legend(fontsize=8)
    axs[1].plot(24,0,'*',color='black',ms=10,label='Goal');axs[1].set(xlim=(23.3,24.5),ylim=(-.5,.5),xlabel='X [m]',ylabel='Y [m]',title='Goal approach and final position');axs[1].set_aspect('equal');axs[1].legend(fontsize=8)
    axs[2].axhline(18,color='black',ls='--',label='18 cm threshold');axs[2].set(xlabel='Seconds since braking started',ylabel='Goal distance [cm]',title='Terminal distance');axs[2].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.2)
    fig.suptitle(f"18 cm arrival / {stats['entered_count']}/{stats['trials']} entered / {stats['completed_count']} completed / command 0.6 m/s, lookahead 0.8 m")
    fig.savefig(web/'batch.png',dpi=150);plt.close(fig)
    template=Path(__file__).with_name('batch.template.html').read_text()
    (web/'index.html').write_text(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':'))))
    previous=folder.parent/'s_curve_v3/web'
    if previous.is_dir():
        target=web/'previous';target.mkdir(exist_ok=True)
        for file in previous.iterdir():
            if file.is_file():shutil.copyfile(file,target/file.name)
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
