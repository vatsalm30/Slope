"""
DAv3 (Depth Anything V2 Metric Outdoor) + VGGT poses → incremental GLBs.

Uses VGGT for camera pose estimation and DAv3 for depth prediction.
This gives a direct comparison: same poses, different depth model.

Install deps on the A100 machine before running:
    pip install transformers accelerate
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
VGGT_DIR = os.path.join(SLOPE_DIR, "vggt")
sys.path.insert(0, VGGT_DIR)

import torch
import torch.nn.functional as F
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16 if device == "cuda" else None

CONF_THRES = 85.0
DAV3_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large"


# ── GLB helpers (same as run_vggt_all.py) ─────────────────────────────────────

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
    opengl_transform = get_opengl_conversion_matrix()
    complete_transform = transform @ opengl_transform @ rot_45_degree
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


def apply_scene_alignment(scene_3d, extrinsics_matrices):
    opengl_conversion_matrix = get_opengl_conversion_matrix()
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    initial_transformation = np.linalg.inv(extrinsics_matrices[0]) @ opengl_conversion_matrix @ align_rotation
    scene_3d.apply_transform(initial_transformation)
    return scene_3d


def predictions_to_glb(predictions, conf_thres=50.0, show_cam=True):
    pred_world_points = predictions["world_points_from_depth"]
    pred_world_points_conf = predictions.get(
        "depth_conf", np.ones_like(pred_world_points[..., 0])
    )

    images = predictions["images"]
    camera_matrices = predictions["extrinsic"]

    vertices_3d = pred_world_points.reshape(-1, 3)
    if images.ndim == 4 and images.shape[1] == 3:
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    conf_threshold = np.percentile(conf, conf_thres) if conf_thres > 0.0 else 0.0
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)
    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if len(vertices_3d) > 50:
        from scipy.spatial import cKDTree as _KDTree
        _k = min(20, len(vertices_3d) - 1)
        _tree = _KDTree(vertices_3d)
        _dists, _ = _tree.query(vertices_3d, k=_k + 1)
        _mean_dists = _dists[:, 1:].mean(axis=1)
        _thresh = _mean_dists.mean() + 2.0 * _mean_dists.std()
        vertices_3d = vertices_3d[_mean_dists < _thresh]
        colors_rgb = colors_rgb[_mean_dists < _thresh]

    if vertices_3d.size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")
    scene_3d = trimesh.Scene()
    scene_3d.add_geometry(trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb))

    num_cameras = len(camera_matrices)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = camera_matrices
    extrinsics_matrices[:, 3, 3] = 1

    if show_cam:
        for i in range(num_cameras):
            camera_to_world = np.linalg.inv(extrinsics_matrices[i])
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)

    scene_3d = apply_scene_alignment(scene_3d, extrinsics_matrices)
    return scene_3d


# ── Image selection & ordering (shared with run_vggt_all.py) ──────────────────

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
    scores = [image_sharpness(img) for img in imgs]
    return imgs[int(np.argmax(scores))]


def get_markers(scenario_path, scenario_name):
    markers = sorted(
        (d for d in os.listdir(scenario_path)
         if not d.startswith(".") and os.path.isdir(os.path.join(scenario_path, d))),
        key=lambda x: int(re.search(r"\d+", x).group())
    )
    if "downhill" in scenario_name:
        markers = list(reversed(markers))
    return markers


# ── DAv3 depth prediction ──────────────────────────────────────────────────────

def predict_depth_dav3(dav3_model, dav3_processor, img_path, target_hw):
    """
    Run DAv3 on a single image and return a depth map resized to target_hw (H, W).
    Returns a numpy array in metres.
    """
    raw = Image.open(img_path).convert("RGB")
    inputs = dav3_processor(images=raw, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dav3_model(**inputs)
    # post_process_depth_estimation returns list of {"predicted_depth": tensor (H, W)}
    result = dav3_processor.post_process_depth_estimation(outputs, target_sizes=[target_hw])
    depth = result[0]["predicted_depth"].cpu().numpy().astype(np.float32)   # (H, W) metres
    # Resize to VGGT image dimensions
    depth_t = torch.from_numpy(depth)[None, None]   # (1,1,H,W)
    depth_t = F.interpolate(depth_t, size=target_hw, mode="bilinear", align_corners=False)
    return depth_t[0, 0].numpy()   # (H, W)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_glb(vggt_model, dav3_model, dav3_processor, image_names, out_path):
    images = load_and_preprocess_images(image_names).to(device)
    H, W = images.shape[-2], images.shape[-1]

    # VGGT forward for camera poses
    with torch.no_grad():
        if dtype is not None:
            with torch.amp.autocast(device, dtype=dtype):
                vggt_pred = vggt_model(images)
        else:
            vggt_pred = vggt_model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(vggt_pred["pose_enc"], (H, W))

    # Convert VGGT tensors to numpy
    images_np = vggt_pred["images"].cpu().numpy().squeeze(0)   # (S, 3, H, W)
    extrinsic_np = extrinsic.cpu().numpy().squeeze(0) if isinstance(extrinsic, torch.Tensor) else extrinsic
    intrinsic_np = intrinsic.cpu().numpy().squeeze(0) if isinstance(intrinsic, torch.Tensor) else intrinsic
    vggt_depth_np = vggt_pred["depth"].cpu().numpy().squeeze(0)   # (S, H, W, 1)

    # DAv3 depth for each image
    dav3_depths = []
    for img_path in image_names:
        d = predict_depth_dav3(dav3_model, dav3_processor, img_path, (H, W))
        dav3_depths.append(d)
    dav3_depth_np = np.stack(dav3_depths, axis=0)[..., np.newaxis]   # (S, H, W, 1)

    # Scale DAv3 depth to match VGGT coordinate system
    scale = np.median(vggt_depth_np) / (np.median(dav3_depth_np) + 1e-8)
    dav3_depth_scaled = dav3_depth_np * scale

    predictions = {
        "images":    images_np,
        "extrinsic": extrinsic_np,
        "intrinsic": intrinsic_np,
    }
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        dav3_depth_scaled, extrinsic_np, intrinsic_np
    )

    glb = predictions_to_glb(predictions, conf_thres=CONF_THRES, show_cam=True)
    glb.export(out_path)
    print(f"    Saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {device}, dtype: {dtype}")

    print("Loading VGGT model (for poses)...")
    vggt_model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    vggt_model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, map_location=device))
    vggt_model.eval().to(device)

    print(f"Loading DAv3 model ({DAV3_MODEL_ID})...")
    dav3_processor = AutoImageProcessor.from_pretrained(DAV3_MODEL_ID)
    dav3_model = AutoModelForDepthEstimation.from_pretrained(DAV3_MODEL_ID).to(device)
    dav3_model.eval()
    print("Models ready.\n")

    media_dir = os.path.join(SLOPE_DIR, "media")
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
            print(f"  k={k:02d}: running DAv3+VGGT on {k} image(s)...")
            run_glb(vggt_model, dav3_model, dav3_processor, all_images[:k], out_path)

    print("\nDone. GLBs in:", output_dir)


if __name__ == "__main__":
    main()
