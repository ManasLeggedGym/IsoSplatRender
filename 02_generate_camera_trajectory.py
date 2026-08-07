import json, math, argparse
import numpy as np


def load_cameras_json(path):
    with open(path) as f:
        cams = json.load(f)

    c2w_list = []
    for cam in cams:
        R   = np.array(cam["rotation"], dtype=np.float64)
        pos = np.array(cam["position"], dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        c2w[:3,  3] = pos
        c2w_list.append(c2w)

    intr = {
        "fx":     float(cams[0]["fx"]),
        "fy":     float(cams[0]["fy"]),
        "width":  int(cams[0]["width"]),
        "height": int(cams[0]["height"]),
    }

    return c2w_list, intr


def mat_to_quat(R):
    trace = R[0,0]+R[1,1]+R[2,2]
    if trace > 0:
        s=0.5/math.sqrt(trace+1.0)
        w=0.25/s; x=(R[2,1]-R[1,2])*s; y=(R[0,2]-R[2,0])*s; z=(R[1,0]-R[0,1])*s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=2.0*math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])
        w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s=2.0*math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])
        w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s=2.0*math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])
        w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    q=np.array([w,x,y,z],dtype=np.float64)
    return q/np.linalg.norm(q)

def quat_to_mat(q):
    w,x,y,z=q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ], dtype=np.float64)

def slerp(R1, R2, t):
    q1=mat_to_quat(R1); q2=mat_to_quat(R2)
    if np.dot(q1,q2)<0: q2=-q2
    dot=float(np.clip(np.dot(q1,q2),-1.0,1.0))
    if dot>0.9995:
        q=q1+t*(q2-q1)
    else:
        theta0=math.acos(dot); theta=theta0*t
        q=q1*math.cos(theta)+((q2-q1*dot)/math.sin(theta0))*math.sin(theta)
    return quat_to_mat(q/np.linalg.norm(q))


def generate_novel_cameras(c2w_list, n_views):
    N=len(c2w_list); novel=[]
    for i in range(n_views):
        frac=i*(N-1)/max(n_views-1,1)
        i0=int(math.floor(frac)); i1=min(i0+1,N-1); t=frac-i0
        pos=(1.0-t)*c2w_list[i0][:3,3]+t*c2w_list[i1][:3,3]
        R=slerp(c2w_list[i0][:3,:3],c2w_list[i1][:3,:3],t)
        c2w=np.eye(4,dtype=np.float64)
        c2w[:3,:3]=R; c2w[:3,3]=pos
        novel.append(c2w)
    return novel


def write_novel_cameras_json(novel_c2w, intr, outpath):
    entries = []
    for i, c2w in enumerate(novel_c2w):
        entry = {
            "fx":       intr["fx"],
            "fy":       intr["fy"],
            "height":   intr["height"],
            "id":       i,
            "img_name": f"novel_view_{i:04d}",
            "position": c2w[:3, 3].tolist(),
            "rotation": c2w[:3, :3].tolist(),
            "width":    intr["width"],
        }
        entries.append(entry)

    with open(outpath, "w") as f:
        json.dump(entries, f, separators=(",", ":"))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cameras", required=True)
    p.add_argument("--outfile", default="novel_cameras.json")
    p.add_argument("--n_views", type=int, default=36)
    args=p.parse_args()

    c2w_list, intr = load_cameras_json(args.cameras)
    novel_c2w = generate_novel_cameras(c2w_list, args.n_views)
    write_novel_cameras_json(novel_c2w, intr, args.outfile)

if __name__=="__main__":
    main()
