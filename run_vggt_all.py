import os
import sys
import glob

import numpy as np
import trimesh
import matplotlib
from scipy.spatial.transform import Rotation
import copy

SLOPE_DIR = os.path.dirname(os.path.abspath(__file__))
VGGT_DIR = os.path.join(SLOPE_DIR, "vggt")
sys.path.insert(0, VGGT_DIR)

import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ── GLB helpers inlined from vggt/visual_util.py (gradio-free) ────────────────

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


def predictions_to_glb(predictions, conf_thres=50.0, show_cam=True,
                        prediction_mode="Predicted Pointmap"):
    print("Building GLB scene")

    if "Pointmap" in prediction_mode and "world_points" in predictions:
        pred_world_points = predictions["world_points"]
        pred_world_points_conf = predictions.get(
            "world_points_conf", np.ones_like(pred_world_points[..., 0])
        )
    else:
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
            world_to_camera = extrinsics_matrices[i]
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)

    scene_3d = apply_scene_alignment(scene_3d, extrinsics_matrices)
    print("GLB scene built")
    return scene_3d


# ── Main ───────────────────────────────────────────────────────────────────────

CONF_THRES = 85.0  # keep top 15% most confident points


def pick_center_image(marker_dir):
    imgs = sorted(glob.glob(os.path.join(marker_dir, "*.png")))
    if not imgs:
        return None
    return imgs[len(imgs) // 2]


def run_scenario(model, device, scenario_path, out_path):
    markers = sorted(
        d for d in os.listdir(scenario_path)
        if not d.startswith(".") and os.path.isdir(os.path.join(scenario_path, d))
    )
    image_names = []
    for marker in markers:
        img = pick_center_image(os.path.join(scenario_path, marker))
        if img:
            image_names.append(img)

    print(f"  {len(image_names)} views: {[os.path.basename(p) for p in image_names]}")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"  Image tensor: {images.shape}")

    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16 if device == "cuda" else None

    with torch.no_grad():
        if dtype is not None:
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
    )

    glb = predictions_to_glb(predictions, conf_thres=CONF_THRES, show_cam=True,
                              prediction_mode="Predicted Pointmap")
    glb.export(out_path)
    print(f"  Saved → {out_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print("Model ready.\n")

    media_dir = os.path.join(SLOPE_DIR, "media")
    output_dir = os.path.join(SLOPE_DIR, "output_glbs")
    os.makedirs(output_dir, exist_ok=True)

    scenarios = sorted(
        d for d in os.listdir(media_dir)
        if not d.startswith(".") and os.path.isdir(os.path.join(media_dir, d))
    )

    for scenario in scenarios:
        scenario_path = os.path.join(media_dir, scenario)
        out_path = os.path.join(output_dir, f"{scenario}.glb")
        print(f"\n[{scenario}]")
        run_scenario(model, device, scenario_path, out_path)

    print("\nDone. GLBs in:", output_dir)


if __name__ == "__main__":
    main()
