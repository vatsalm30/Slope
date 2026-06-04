"""
Depth Anything 3 incremental GLB generation.

DA3 predicts depth + camera poses directly (no VGGT needed).
Uses DA3NESTED-GIANT-LARGE-1.1 (1.4B params, metric depth + any-view).

Setup on the A100 machine:
    git clone https://github.com/ByteDance-Seed/depth-anything-3
    cd depth-anything-3 && pip install -e .
    cd ..
"""
import os
import sys
import glob
import re

import numpy as np
import trimesh
import matplotlib
from scipy.spatial.transform import Rotation
from PIL import Image

SLOPE_DIR = os.path.dirname(os.path.abspath(__file__))
DA3_DIR = os.path.join(SLOPE_DIR, "depth-anything-3")
sys.path.insert(0, DA3_DIR)

import torch
from depth_anything_3.api import DepthAnything3

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16 if device == "cuda" else None

CONF_THRES = 85.0
DA3_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


# ── GLB helpers ────────────────────────────────────────────────────────────────

def get_opengl_conversion_matrix():
    matrix = np.identity(4)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def transform_points(transformation, points, dim=None):
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]
    transformation = transformation.swapaxes(-1, -2)
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]
    return points[..., :dim].reshape(*initial_shape, dim)


def compute_camera_faces(cone_shape):
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)
    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone
        faces_list.extend([
            (v1, v2, v2_offset), (v1, v1_offset, v3), (v3_offset, v2, v3),
            (v1, v2, v2_offset_2), (v1, v1_offset_2, v3), (v3_offset_2, v2, v3),
        ])
    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def integrate_camera_into_scene(scene, transform, face_colors, scene_scale):
    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height
    complete_transform = transform @ get_opengl_conversion_matrix() @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()
    vertices_combined = np.concatenate([
        camera_cone_shape.vertices,
        0.95 * camera_cone_shape.vertices,
        transform_points(slight_rotation, camera_cone_shape.vertices),
    ])
    vertices_transformed = transform_points(complete_transform, vertices_combined)
    mesh_faces = compute_camera_faces(camera_cone_shape)
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def apply_scene_alignment(scene_3d, extrinsics_4x4):
    opengl = get_opengl_conversion_matrix()
    align = np.eye(4)
    align[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    scene_3d.apply_transform(np.linalg.inv(extrinsics_4x4[0]) @ opengl @ align)
    return scene_3d


# ── Unproject depth to world points using DA3 intrinsics/extrinsics ───────────

def unproject_da3(depth, extrinsics, intrinsics):
    """
    depth:      (S, H, W) metres
    extrinsics: (S, 3, 4) world-to-camera
    intrinsics: (S, 3, 3) pixel-unit K matrices
    Returns:    (S, H, W, 3) world points
    """
    S, H, W = depth.shape
    world_pts = np.zeros((S, H, W, 3), dtype=np.float32)
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)   # (H, W)

    for i in range(S):
        fx, fy = intrinsics[i, 0, 0], intrinsics[i, 1, 1]
        cx, cy = intrinsics[i, 0, 2], intrinsics[i, 1, 2]
        d = depth[i]
        cam_pts = np.stack([
            (uu - cx) / fx * d,
            (vv - cy) / fy * d,
            d,
            np.ones((H, W), dtype=np.float32),
        ], axis=-1)   # (H, W, 4)
        E = np.eye(4)
        E[:3] = extrinsics[i]
        c2w = np.linalg.inv(E)
        world_pts[i] = (cam_pts @ c2w.T)[..., :3]

    return world_pts


# ── GLB assembly ──────────────────────────────────────────────────────────────

def build_glb(depth_np, extrinsics_np, intrinsics_np, images_np):
    """
    depth_np:      (S, H, W)
    extrinsics_np: (S, 3, 4)
    intrinsics_np: (S, 3, 3)
    images_np:     (S, H, W, 3) float32 0-1
    """
    world_pts = unproject_da3(depth_np, extrinsics_np, intrinsics_np)  # (S, H, W, 3)
    vertices_3d = world_pts.reshape(-1, 3)
    colors_rgb  = (images_np.reshape(-1, 3) * 255).astype(np.uint8)

    # Confidence proxy: use inverse depth variance (flatter = more confident)
    conf = 1.0 / (depth_np.reshape(-1) + 1e-6)
    conf_threshold = np.percentile(conf, CONF_THRES)
    conf_mask = conf >= conf_threshold

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb  = colors_rgb[conf_mask]

    # Statistical outlier removal
    if len(vertices_3d) > 50:
        from scipy.spatial import cKDTree
        _k = min(20, len(vertices_3d) - 1)
        tree = cKDTree(vertices_3d)
        dists, _ = tree.query(vertices_3d, k=_k + 1)
        mean_d = dists[:, 1:].mean(axis=1)
        sor_mask = mean_d < mean_d.mean() + 2.0 * mean_d.std()
        vertices_3d = vertices_3d[sor_mask]
        colors_rgb  = colors_rgb[sor_mask]

    if vertices_3d.size == 0:
        vertices_3d = np.array([[1, 0, 0]], dtype=np.float32)
        colors_rgb  = np.array([[255, 255, 255]], dtype=np.uint8)
        scene_scale = 1.0
    else:
        lo = np.percentile(vertices_3d, 5, axis=0)
        hi = np.percentile(vertices_3d, 95, axis=0)
        scene_scale = float(np.linalg.norm(hi - lo))

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")
    scene_3d = trimesh.Scene()
    scene_3d.add_geometry(trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb))

    S = len(extrinsics_np)
    extrinsics_4x4 = np.zeros((S, 4, 4))
    extrinsics_4x4[:, :3, :4] = extrinsics_np
    extrinsics_4x4[:, 3, 3] = 1

    for i in range(S):
        c2w = np.linalg.inv(extrinsics_4x4[i])
        rgba = colormap(i / S)
        color = tuple(int(255 * x) for x in rgba[:3])
        integrate_camera_into_scene(scene_3d, c2w, color, scene_scale)

    scene_3d = apply_scene_alignment(scene_3d, extrinsics_4x4)
    return scene_3d


# ── Image selection & ordering (mirrors run_vggt_all.py) ──────────────────────

def image_sharpness(img_path):
    img = np.array(Image.open(img_path).convert("L").resize((256, 192)), dtype=np.float32)
    lap = img[:-2, 1:-1] + img[2:, 1:-1] + img[1:-1, :-2] + img[1:-1, 2:] - 4 * img[1:-1, 1:-1]
    return lap.var()


def pick_best_image(marker_dir):
    imgs = sorted(glob.glob(os.path.join(marker_dir, "*.png")))
    if not imgs:
        return None
    if len(imgs) == 1:
        return imgs[0]
    return imgs[int(np.argmax([image_sharpness(p) for p in imgs]))]


def get_markers(scenario_path, scenario_name):
    markers = sorted(
        (d for d in os.listdir(scenario_path)
         if not d.startswith(".") and os.path.isdir(os.path.join(scenario_path, d))),
        key=lambda x: int(re.search(r"\d+", x).group())
    )
    if "downhill" in scenario_name:
        markers = list(reversed(markers))
    return markers


# ── Inference ─────────────────────────────────────────────────────────────────

def run_glb(model, image_names, out_path):
    with torch.no_grad():
        if dtype is not None:
            with torch.amp.autocast(device, dtype=dtype):
                pred = model.inference(image_names)
        else:
            pred = model.inference(image_names)

    def to_np(x):
        if hasattr(x, "cpu"):
            return x.cpu().float().numpy()
        return np.array(x, dtype=np.float32)

    depth_np     = to_np(pred.depth)       # (S, H, W) metres
    extrinsic_np = to_np(pred.extrinsics)  # (S, 3, 4)
    intrinsic_np = to_np(pred.intrinsics)  # (S, 3, 3)

    S, H, W = depth_np.shape
    images_np = np.stack([
        np.array(Image.open(p).convert("RGB").resize((W, H)), dtype=np.float32) / 255.0
        for p in image_names
    ])  # (S, H, W, 3)

    glb = build_glb(depth_np, extrinsic_np, intrinsic_np, images_np)
    glb.export(out_path)
    print(f"    Saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Loading DA3 model ({DA3_MODEL_ID})...")
    model = DepthAnything3.from_pretrained(DA3_MODEL_ID).to(device)
    model.eval()
    print("Model ready.\n")

    media_dir  = os.path.join(SLOPE_DIR, "media")
    output_dir = os.path.join(SLOPE_DIR, "output_glbs", "dav3")
    os.makedirs(output_dir, exist_ok=True)

    scenarios = sorted(
        d for d in os.listdir(media_dir)
        if not d.startswith(".") and os.path.isdir(os.path.join(media_dir, d))
    )

    for scenario in scenarios:
        scenario_path = os.path.join(media_dir, scenario)
        print(f"\n[{scenario}]")

        markers = get_markers(scenario_path, scenario)
        print(f"  Marker order: {markers}")

        all_images = []
        for marker in markers:
            img = pick_best_image(os.path.join(scenario_path, marker))
            if img:
                all_images.append(img)
                print(f"  {marker}: {os.path.basename(img)}")

        for k in range(1, len(all_images) + 1):
            out_path = os.path.join(output_dir, f"{scenario}_k{k:02d}.glb")
            if os.path.exists(out_path):
                print(f"  k={k:02d}: exists, skipping")
                continue
            print(f"  k={k:02d}: running DA3 on {k} image(s)...")
            run_glb(model, all_images[:k], out_path)

    print("\nDone. GLBs in:", output_dir)


if __name__ == "__main__":
    main()
