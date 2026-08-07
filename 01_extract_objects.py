import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

def load_ply(ply_path: str) -> dict:
    plydata = PlyData.read(ply_path)
    v = plydata["vertex"]
    return {p.name: np.array(v[p.name]) for p in v.properties}

def save_ply(gaussians: dict, mask: np.ndarray, out_path: str):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return
    vertex_data = np.zeros(len(idx), dtype=[(name, gaussians[name].dtype) for name in gaussians.keys()])
    for name in gaussians.keys():
        vertex_data[name] = gaussians[name][idx]
    PlyData([PlyElement.describe(vertex_data, "vertex")]).write(out_path)

def load_cameras_json(json_path: str) -> list:
    with open(json_path) as f:
        raw = json.load(f)
    cameras = []
    for cam in raw:
        R_cw = np.array(cam["rotation"], dtype=np.float64)
        t_w  = np.array(cam["position"], dtype=np.float64)
        R_wc = R_cw.T
        t_wc = -R_wc @ t_w
        cameras.append({
            "img_name": cam["img_name"],
            "w":        int(cam["width"]),
            "h":        int(cam["height"]),
            "fx":       float(cam["fx"]),
            "fy":       float(cam["fy"]),
            "cx":       float(cam.get("cx", cam["width"]  / 2.0)),
            "cy":       float(cam.get("cy", cam["height"] / 2.0)),
            "R_wc":     R_wc,
            "t_wc":     t_wc,
        })
    return cameras

def project_gaussians(xyz_world: np.ndarray, cam: dict):
    p_cam = (cam["R_wc"] @ xyz_world.T).T + cam["t_wc"]
    valid = p_cam[:, 2] > 0.01
    z  = np.where(valid, p_cam[:, 2], 1.0)
    px = cam["fx"] * p_cam[:, 0] / z + cam["cx"]
    py = cam["fy"] * p_cam[:, 1] / z + cam["cy"]
    return px, py, valid

def find_mask(mask_dir: str, img_name: str):
    stem = Path(img_name).stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = Path(mask_dir) / (stem + ext)
        if candidate.exists():
            return np.array(Image.open(candidate).convert("L")) > 127
    return None

def accumulate_votes(xyz: np.ndarray, cameras: list, mask_dir: str, view_num: int = -1):
    votes = np.zeros(len(xyz), dtype=np.float32)
    total_views = 0
    cams_used = cameras if (view_num <= 0 or view_num >= len(cameras)) else [cameras[i] for i in np.linspace(0, len(cameras) - 1, view_num, dtype=int)]

    for cam in cams_used:
        mask = find_mask(mask_dir, cam["img_name"])
        if mask is None:
            continue
        h, w = mask.shape
        px, py, valid = project_gaussians(xyz, cam)
        ix = np.round(px).astype(np.int32)
        iy = np.round(py).astype(np.int32)
        in_bounds = valid & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        safe_ix = np.where(in_bounds, ix, 0)
        safe_iy = np.where(in_bounds, iy, 0)
        votes[in_bounds &  mask[safe_iy, safe_ix]] += 1.0
        votes[in_bounds & ~mask[safe_iy, safe_ix]] -= 1.0
        total_views += 1
    return votes, total_views

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ply",     default="scene_output.ply")
    p.add_argument("--cameras", default="cameras.json")
    p.add_argument("--masks",   nargs="+", default=["masks-ball", "masks-cupboard", "masks-cupboard-2"])
    p.add_argument("--out",     default="extracted_objects")
    p.add_argument("--gamma",   type=float, default=0.1)
    p.add_argument("--views",   type=int, default=-1)
    args = p.parse_args()

    base = Path(__file__).parent
    ply_path, cameras_path, out_dir = base / args.ply, base / args.cameras, base / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ply_path.exists() or not cameras_path.exists():
        sys.exit(1)

    gaussians = load_ply(str(ply_path))
    cameras   = load_cameras_json(str(cameras_path))
    xyz       = np.stack([gaussians["x"], gaussians["y"], gaussians["z"]], axis=1)

    for mask_dir in args.masks:
        mask_path = base / mask_dir
        if not mask_path.is_dir():
            continue
        votes, total_views = accumulate_votes(xyz, cameras, str(mask_path), view_num=args.views)
        if total_views > 0:
            fg_mask = (votes / float(total_views)) > args.gamma
            save_ply(gaussians, fg_mask, str(out_dir / f"{mask_path.name}.ply"))

if __name__ == "__main__":
    main()