import os, json, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def load_ply(path: str) -> dict:
    plydata = PlyData.read(path)
    v = plydata["vertex"]

    xyz       = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    opacities = v["opacity"].astype(np.float32)
    scales    = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    quats     = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)

    SH_C0 = 0.28209479177387814
    f_dc  = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    colors = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)

    return {
        "xyz":     torch.tensor(xyz,       device=DEVICE),
        "opacity": torch.tensor(opacities, device=DEVICE),
        "scale":   torch.tensor(scales,    device=DEVICE),
        "quat":    torch.tensor(quats,     device=DEVICE),
        "color":   torch.tensor(colors,    device=DEVICE),
    }


LABEL_BALL      = 1
LABEL_CUPBOARD1 = 2
LABEL_CUPBOARD2 = 3

LABEL_COLORS = {
    1: (255, 0,   0),
    2: (0,   255, 0),
    3: (0,   0,   255),
}

def merge_gaussians(plys: dict) -> dict:
    keys = ["xyz", "opacity", "scale", "quat", "color"]
    merged = {k: torch.cat([plys[lbl][k] for lbl in sorted(plys)]) for k in keys}
    merged["label"] = torch.cat([
        torch.full((plys[lbl]["xyz"].shape[0],), lbl, dtype=torch.long, device=DEVICE)
        for lbl in sorted(plys)
    ])
    return merged


def load_cameras_json(path: str):
    with open(path) as f:
        cams = json.load(f)

    c2w_list = []
    for cam in cams:
        R   = np.array(cam["rotation"], dtype=np.float32)
        pos = np.array(cam["position"], dtype=np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3,  3] = pos
        c2w_list.append(c2w)

    fx = float(cams[0]["fx"])
    fy = float(cams[0]["fy"])
    W  = int(cams[0]["width"])
    H  = int(cams[0]["height"])
    cx = float(cams[0].get("cx", W / 2.0))
    cy = float(cams[0].get("cy", H / 2.0))

    return c2w_list, fx, fy, cx, cy, W, H


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    R = c2w[:3, :3]; t = c2w[:3, 3]
    R_inv = R.T
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R_inv
    w2c[:3,  3] = -R_inv @ t
    return w2c


def slerp_rotation(R1: np.ndarray, R2: np.ndarray, t: float) -> np.ndarray:
    def mat_to_quat(R):
        trace = R[0,0]+R[1,1]+R[2,2]
        if trace > 0:
            s = 0.5/math.sqrt(trace+1.0)
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
        q=np.array([w,x,y,z],dtype=np.float32)
        return q/np.linalg.norm(q)

    def quat_to_mat(q):
        w,x,y,z=q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
        ],dtype=np.float32)

    q1=mat_to_quat(R1); q2=mat_to_quat(R2)
    if np.dot(q1,q2)<0: q2=-q2
    dot=np.clip(np.dot(q1,q2),-1,1)
    if dot>0.9995:
        q=q1+(t*(q2-q1))
    else:
        theta0=math.acos(dot); theta=theta0*t
        q=q1*math.cos(theta)+((q2-q1*dot)/math.sin(theta0))*math.sin(theta)
    q/=np.linalg.norm(q)
    return quat_to_mat(q)


def generate_novel_cameras(c2w_list: list, n_views: int) -> list:
    N = len(c2w_list)
    novel = []

    for i in range(n_views):
        frac  = i * (N - 1) / max(n_views - 1, 1)
        idx0  = int(math.floor(frac))
        idx1  = min(idx0 + 1, N - 1)
        t     = frac - idx0

        c2w0 = c2w_list[idx0]
        c2w1 = c2w_list[idx1]

        pos = (1 - t) * c2w0[:3, 3] + t * c2w1[:3, 3]
        R = slerp_rotation(c2w0[:3, :3], c2w1[:3, :3], t)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3,  3] = pos
        novel.append(c2w)

    return novel


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    N = q.shape[0]
    R = torch.zeros(N, 3, 3, device=q.device)
    R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y-w*z); R[:,0,2]=2*(x*z+w*y)
    R[:,1,0]=2*(x*y+w*z);   R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z-w*x)
    R[:,2,0]=2*(x*z-w*y);   R[:,2,1]=2*(y*z+w*x); R[:,2,2]=1-2*(x*x+y*y)
    return R


def build_2d_cov(scale, quat, w2c, fx, fy, z_cam):
    S     = torch.exp(scale)
    R     = quat_to_rot(quat)
    RS    = R * S.unsqueeze(1)
    Sig3D = RS @ RS.transpose(1,2)

    W    = w2c[:3,:3]
    Wn   = W.unsqueeze(0).expand(Sig3D.shape[0],-1,-1)
    Sigc = Wn @ Sig3D @ Wn.transpose(1,2)

    z = z_cam.clamp(min=1e-4)
    J = torch.zeros(z.shape[0], 2, 3, device=scale.device)
    J[:,0,0] = fx / z
    J[:,1,1] = fy / z

    cov2D = J @ Sigc @ J.transpose(1,2)
    cov2D[:,0,0] += 0.3
    cov2D[:,1,1] += 0.3
    return cov2D


def rasterize(gaussians, c2w, W, H, fx, fy, cx, cy, bg=(0.,0.,0.)):
    w2c = torch.tensor(c2w_to_w2c(c2w), device=DEVICE)

    xyz     = gaussians["xyz"]
    opacity = torch.sigmoid(gaussians["opacity"])
    scale   = gaussians["scale"]
    quat    = F.normalize(gaussians["quat"], dim=1)
    color   = gaussians["color"]
    label   = gaussians["label"]
    N       = xyz.shape[0]

    xyz_h   = torch.cat([xyz, torch.ones(N,1,device=DEVICE)], dim=1)
    xyz_cam = (w2c @ xyz_h.T).T
    x_c, y_c, z_c = xyz_cam[:,0], xyz_cam[:,1], xyz_cam[:,2]

    valid = z_c > 0.01
    if valid.sum() == 0:
        return (torch.zeros(H,W,3,device=DEVICE),
                torch.zeros(H,W,dtype=torch.long,device=DEVICE))

    x_c=x_c[valid]; y_c=y_c[valid]; z_c=z_c[valid]
    opacity=opacity[valid]; scale=scale[valid]; quat=quat[valid]
    color=color[valid]; label=label[valid]

    u = (x_c/z_c)*fx + cx
    v = (y_c/z_c)*fy + cy

    order   = torch.argsort(z_c, descending=True)
    u=u[order]; v=v[order]; z_c=z_c[order]
    opacity=opacity[order]; scale=scale[order]; quat=quat[order]
    color=color[order]; label=label[order]

    cov2D = build_2d_cov(scale, quat, w2c, fx, fy, z_c)
    det   = (cov2D[:,0,0]*cov2D[:,1,1] - cov2D[:,0,1]**2).clamp(min=1e-6)
    inv_c = torch.zeros_like(cov2D)
    inv_c[:,0,0] =  cov2D[:,1,1]/det
    inv_c[:,1,1] =  cov2D[:,0,0]/det
    inv_c[:,0,1] = inv_c[:,1,0] = -cov2D[:,0,1]/det

    r_u = (3.0*torch.sqrt(cov2D[:,0,0].clamp(min=0))).ceil().long().clamp(max=64)
    r_v = (3.0*torch.sqrt(cov2D[:,1,1].clamp(min=0))).ceil().long().clamp(max=64)

    acc_rgb   = torch.zeros(H, W, 3, device=DEVICE)
    acc_alpha = torch.zeros(H, W,    device=DEVICE)
    NUM_LABELS = 4
    acc_lbl   = torch.zeros(H, W, NUM_LABELS, device=DEVICE)

    u_i = u.round().long()
    v_i = v.round().long()

    for i in range(u_i.shape[0]):
        ui=int(u_i[i]); vi=int(v_i[i])
        ru=int(r_u[i]); rv=int(r_v[i])

        x0=max(0,ui-ru); x1=min(W,ui+ru+1)
        y0=max(0,vi-rv); y1=min(H,vi+rv+1)
        if x0>=x1 or y0>=y1:
            continue

        xs = torch.arange(x0,x1,device=DEVICE,dtype=torch.float32)
        ys = torch.arange(y0,y1,device=DEVICE,dtype=torch.float32)
        gx,gy = torch.meshgrid(xs,ys,indexing="xy")

        dx = gx - float(u[i])
        dy = gy - float(v[i])
        ic = inv_c[i]
        maha = (ic[0,0]*dx*dx + (ic[0,1]+ic[1,0])*dx*dy + ic[1,1]*dy*dy).clamp(min=0)
        gw   = torch.exp(-0.5*maha)

        alpha   = float(opacity[i]) * gw
        T       = 1.0 - acc_alpha[y0:y1,x0:x1]
        contrib = alpha * T

        acc_rgb  [y0:y1,x0:x1]               += contrib.unsqueeze(-1) * color[i]
        acc_alpha[y0:y1,x0:x1]               += contrib
        acc_lbl  [y0:y1,x0:x1,int(label[i])] += contrib

    T_fin     = (1.0 - acc_alpha).unsqueeze(-1)
    bg_t      = torch.tensor(bg, device=DEVICE, dtype=torch.float32)
    rgb_image = (acc_rgb + T_fin*bg_t).clamp(0,1)
    lbl_image = acc_lbl.argmax(dim=-1)

    return rgb_image, lbl_image


def save_rgb(t, path):
    arr = (t.cpu().numpy()*255).clip(0,255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def save_masks(lbl_t, out_dir, prefix, label_map):
    lbl = lbl_t.cpu().numpy()
    for lid, name in label_map.items():
        mask = ((lbl==lid).astype(np.uint8)*255)
        Image.fromarray(mask,mode="L").save(
            os.path.join(out_dir, f"{prefix}_mask_{name}.png"))
    H,W = lbl.shape
    col = np.zeros((H,W,3),dtype=np.uint8)
    for lid,rgb in LABEL_COLORS.items():
        col[lbl==lid]=rgb
    Image.fromarray(col).save(os.path.join(out_dir,f"{prefix}_mask_overlay.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ball",    required=True)
    p.add_argument("--cup1",    required=True)
    p.add_argument("--cup2",    required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--outdir",  default="output")
    p.add_argument("--n_views", type=int, default=36)
    p.add_argument("--bg", type=float, nargs=3, default=[0.,0.,0.], metavar=("R","G","B"))
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rgb_dir  = os.path.join(args.outdir,"rgb");   os.makedirs(rgb_dir, exist_ok=True)
    mask_dir = os.path.join(args.outdir,"masks"); os.makedirs(mask_dir,exist_ok=True)

    plys = {
        LABEL_BALL:      load_ply(args.ball),
        LABEL_CUPBOARD1: load_ply(args.cup1),
        LABEL_CUPBOARD2: load_ply(args.cup2),
    }
    gaussians = merge_gaussians(plys)

    c2w_list, fx, fy, cx, cy, W, H = load_cameras_json(args.cameras)
    novel_c2w = generate_novel_cameras(c2w_list, args.n_views)

    label_map = {
        LABEL_BALL:      "ball",
        LABEL_CUPBOARD1: "cupboard1",
        LABEL_CUPBOARD2: "cupboard2",
    }

    for i, c2w in enumerate(novel_c2w):
        prefix = f"view_{i:04d}"
        rgb, lbl = rasterize(gaussians, c2w, W, H, fx, fy, cx, cy, bg=tuple(args.bg))
        save_rgb(rgb,  os.path.join(rgb_dir,  f"{prefix}_rgb.png"))
        save_masks(lbl, mask_dir, prefix, label_map)


if __name__ == "__main__":
    main()
