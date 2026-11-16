"""Ten geometrically distinct, arc-length-resampled paths, starting along +X."""
import argparse
import json
from pathlib import Path
import numpy as np


def resample(points,spacing=.005):
    p=np.asarray(points,dtype=float)
    keep=np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-9]
    p=p[keep];s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    ss=np.linspace(0,s[-1],int(np.ceil(s[-1]/spacing))+1)
    return np.c_[np.interp(ss,s,p[:,0]),np.interp(ss,s,p[:,1])]


def s_path(length,amplitude,cycles=1):
    x=np.linspace(0,length,12001)
    u=np.clip((x-2)/(length-4),0,1)
    return np.c_[x,amplitude*np.sin(2*np.pi*cycles*u)**3]


def lane_change(sign):
    x=np.linspace(0,22,12001);u=np.clip((x-3)/12,0,1)
    y=sign*2.5*(10*u**3-15*u**4+6*u**5)
    return np.c_[x,y]


def corner(sign,radius=4):
    start=np.c_[np.linspace(0,3,601),np.zeros(601)]
    theta=np.linspace(0,np.pi/2,1601)
    arc=np.c_[3+radius*np.sin(theta),sign*radius*(1-np.cos(theta))]
    end=np.c_[np.full(601,3+radius),sign*(radius+np.linspace(0,3,601))]
    return np.concatenate([start,arc,end])


def hairpin():
    r=3.5;theta=np.linspace(-np.pi/2,np.pi/2,3001)
    start=np.c_[np.linspace(0,4,801),np.zeros(801)]
    arc=np.c_[4+r*np.cos(theta),r+r*np.sin(theta)]
    end=np.c_[4-np.linspace(0,3,601),np.full(601,2*r)]
    return np.concatenate([start,arc,end])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    args.directory.mkdir(parents=True,exist_ok=True)
    routes=[('Narrow S','窄幅 S',s_path(22,1.8)),('Mirrored S','反向 S',s_path(24,-2.2)),
            ('Wide S','宽幅 S',s_path(28,3)),('Gentle S','缓弯 S',s_path(18,1)),
            ('Double S','双 S',s_path(32,1.6,2)),('Left lane change','左变道',lane_change(1)),
            ('Right lane change','右变道',lane_change(-1)),('Left 90 degree turn','左转 90°',corner(1)),
            ('Right 90 degree turn','右转 90°',corner(-1)),('U-turn','U 形回转',hairpin())]
    entries=[]
    for i,(name,label,p) in enumerate(routes,1):
        points=resample(p)
        obj=dict(id=i,name=name,label=label,points=points.tolist())
        filename=args.directory/f'route-{i:02d}.json';filename.write_text(json.dumps(obj,separators=(',',':')))
        entries.append(dict(id=i,name=name,label=label,file=filename.name,
            length_m=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
            goal_xy=points[-1].tolist()))
    (args.directory/'routes.json').write_text(json.dumps(entries,indent=2,ensure_ascii=False))
    print(json.dumps(entries,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
