"""
BEV (bird's-eye-view) video slope estimation.

The drone flies along the road with a nadir (straight-down) camera, so the
gravity direction coincides with the camera's optical axis — NOT +Y as in the
marker GLBs. Slope is invisible in a single frame's appearance; it is recovered
from geometry instead, via two independent paths:

  Path A — VGGT multi-view:
      Reconstruct the whole corridor from N frames. Gravity = mean camera
      optical axis. Sample the ground below each camera (90th-percentile depth
      in a local cylinder, skipping trees) and regress absolute ground height
      against the along-flight coordinate. Works for both fixed-altitude and
      terrain-following flights because heights are absolute, not
      camera-relative. Also exports a GLB for inspection.

  Path B — DA3 per-frame depth plane:
      For each frame, predict a metric depth map. With a nadir camera the
      depth map directly encodes terrain elevation, so the tilt of a robust
      plane fit = surface slope. Vegetation pixels are excluded by color and
      outliers (trees, cars) by iterative rejection. Slope angle is invariant
      to the monocular scale ambiguity. Travel direction is recovered with
      phase correlation between frame pairs to label uphill/downhill.

Usage (A100 machine):
    python analyze_bev_video.py media/DJI_0290.mp4 --method both --n-frames 30

Extract frames only (e.g. locally, then rsync):
    python analyze_bev_video.py media/DJI_0290.mp4 --method extract
"""

import os
import sys
import glob
import argparse

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

SLOPE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SLOPE_DIR, "vggt"))
sys.path.insert(0, os.path.join(SLOPE_DIR, "depth-anything-3", "src"))

PAIR_DT      = 0.7    # seconds between a frame and its motion-pair companion
CONF_KEEP    = 40.0   # VGGT: keep points above this confidence percentile for slope math
GROUND_PCT   = 90     # ground = this percentile of depth in cylinder (skips trees)
MAD_SCALE    = 5.0
DA3_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
VGGT_URL     = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


# ── Step 1: frame extraction ───────────────────────────────────────────────────

def extract_frames(video_path, n_frames, frames_dir):
    """Sample 1 fps, score sharpness, keep the sharpest frame per time bin.
    Saves half-res JPEGs plus a +PAIR_DT companion for motion estimation."""
    pairs_dir = os.path.join(frames_dir, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)

    existing = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if len(existing) == n_frames:
        print(f"  Reusing {n_frames} cached frames in {frames_dir}")
        return existing

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    # score one candidate per second
    candidates = []   # (t_sec, sharpness)
    t = 0.0
    while t < duration - PAIR_DT:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (480, 270))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        candidates.append((t, cv2.Laplacian(gray, cv2.CV_32F).var()))
        t += 1.0
    print(f"  Scored {len(candidates)} candidate frames over {duration:.0f}s")

    # sharpest frame per time bin → even coverage of the flight
    bins = np.array_split(np.arange(len(candidates)), n_frames)
    chosen = []
    for b in bins:
        if len(b) == 0:
            continue
        best = max(b, key=lambda i: candidates[i][1])
        chosen.append(candidates[best][0])

    frame_paths = []
    for i, t in enumerate(chosen):
        out = os.path.join(frames_dir, f"frame_{i:03d}.jpg")
        pair = os.path.join(pairs_dir, f"frame_{i:03d}_p.jpg")
        for path, ts in [(out, t), (pair, t + PAIR_DT)]:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (w // 2, h // 2))
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_paths.append(out)
        print(f"  frame_{i:03d}.jpg  ←  t={t:.0f}s")

    cap.release()
    return frame_paths


# ── Path A: VGGT multi-view ────────────────────────────────────────────────────

def vggt_analysis(frame_paths, out_glb):
    import torch
    from run_vggt_all import predictions_to_glb
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from scipy.spatial import cKDTree

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = (torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8)
             else torch.float16 if device == "cuda" else None)
    if device != "cuda":
        print("  WARNING: no CUDA — this is heavy; intended for the A100 machine.")

    print("  Loading VGGT...")
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_URL, map_location=device))
    model.eval().to(device)

    images = load_and_preprocess_images(frame_paths).to(device)
    print(f"  Inference on {images.shape[0]} frames...")
    with torch.no_grad():
        if dtype is not None:
            with torch.amp.autocast(device, dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
    )

    glb = predictions_to_glb(predictions, conf_thres=85.0, show_cam=True,
                              prediction_mode="Predicted Pointmap")
    glb.export(out_glb)
    print(f"  GLB → {out_glb}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── slope math in raw VGGT world coords, gravity = mean camera optical axis
    E = predictions["extrinsic"]                       # (S,3,4) world-to-camera
    R, tv = E[:, :3, :3], E[:, :3, 3]
    C = -np.einsum("sji,sj->si", R, tv)                # camera centers, R^T @ t
    g = R[:, 2, :].mean(axis=0)                        # mean optical axis = gravity (down)
    g /= np.linalg.norm(g)

    if "world_points" in predictions:
        pts  = predictions["world_points"].reshape(-1, 3)
        conf = predictions.get("world_points_conf", np.ones(len(pts))).reshape(-1)
    else:
        pts  = predictions["world_points_from_depth"].reshape(-1, 3)
        conf = predictions.get("depth_conf", np.ones(len(pts))).reshape(-1)

    keep = np.isfinite(pts).all(axis=1) & (conf >= np.percentile(conf, CONF_KEEP))
    pts = pts[keep]
    if len(pts) > 500_000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 500_000, replace=False)]

    h = pts @ g                                        # absolute height, down-positive
    cam_h = C @ g

    # flight direction in the plane ⊥ g
    C_horiz = C - np.outer(cam_h, g)
    _, _, Vt = np.linalg.svd(C_horiz - C_horiz.mean(0), full_matrices=False)
    e1 = Vt[0]
    e2 = np.cross(g, e1)

    pts_2d = np.c_[pts @ e1, pts @ e2]
    cam_2d = np.c_[C @ e1, C @ e2]
    s_cam = cam_2d[:, 0]

    spacing = np.median(np.abs(np.diff(np.sort(s_cam))))
    radius = max(0.5 * spacing, 1e-6)

    tree = cKDTree(pts_2d)
    ground_h, valid = [], []
    for i in range(len(C)):
        idx = tree.query_ball_point(cam_2d[i], radius)
        if len(idx) < 50:
            ground_h.append(np.nan)
            valid.append(False)
        else:
            ground_h.append(np.percentile(h[idx], GROUND_PCT))
            valid.append(True)
    ground_h = np.array(ground_h)
    valid = np.array(valid)

    if valid.sum() >= 3:                               # MAD filter
        med = np.median(ground_h[valid])
        mad = np.median(np.abs(ground_h[valid] - med))
        valid &= np.abs(np.nan_to_num(ground_h, nan=np.inf) - med) < MAD_SCALE * max(mad, 1e-9)

    if valid.sum() < 2:
        print("  Not enough valid ground samples.")
        return None

    grad = np.polyfit(s_cam[valid], ground_h[valid], 1)[0]
    slope_deg = float(np.degrees(np.arctan(abs(grad))))
    # s ordered along flight (frame index increases along flight): orient e1 accordingly
    flight_sign = np.sign(s_cam[-1] - s_cam[0]) or 1.0
    direction = "downhill" if grad * flight_sign > 0 else "uphill"

    cam_grad = np.polyfit(s_cam, cam_h, 1)[0]
    cam_slope = float(np.degrees(np.arctan(abs(cam_grad))))
    print(f"  Ground slope : {slope_deg:.2f}° {direction}  (cameras used: {valid.sum()}/{len(C)})")
    print(f"  Camera-path slope: {cam_slope:.2f}° (≈0° ⇒ fixed-altitude flight)")

    return {
        "slope": slope_deg, "direction": direction, "cam_slope": cam_slope,
        "s": s_cam * flight_sign, "ground_h": ground_h, "cam_h": cam_h, "valid": valid,
    }


# ── Path B: DA3 per-frame depth plane ──────────────────────────────────────────

def _vegetation_mask(img_path, shape_hw):
    """True = usable pixel (not vegetation)."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (shape_hw[1], shape_hw[0]))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 90) & (hsv[..., 1] > 60)
    return ~green


def _robust_plane(X, Y, Z, iters=3, sigma=2.5):
    A = np.c_[X, Y, np.ones_like(X)]
    mask = np.ones(len(Z), bool)
    coef = np.zeros(3)
    resid = np.zeros(len(Z))
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[mask], Z[mask], rcond=None)
        resid = A @ coef - Z
        std = resid[mask].std()
        mask = np.abs(resid) < sigma * max(std, 1e-9)
    return coef, mask, resid  # Z = aX + bY + c; final inlier mask; residuals (all pts)


def _travel_direction(frame_path, pair_path):
    """Drone motion direction in image coords via phase correlation. None if unreliable."""
    g1 = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.imread(pair_path), cv2.COLOR_BGR2GRAY)
    g1 = cv2.resize(g1, (960, 540)).astype(np.float32)
    g2 = cv2.resize(g2, (960, 540)).astype(np.float32)
    win = cv2.createHanningWindow((960, 540), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(g1, g2, win)
    if response < 0.05 or (abs(dx) < 0.5 and abs(dy) < 0.5):
        return None
    v = np.array([-dx, -dy])                # scene shift is opposite to drone motion
    return v / np.linalg.norm(v)


def dav3_analysis(frame_paths, frames_dir):
    import torch
    from depth_anything_3.api import DepthAnything3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading DA3 ({DA3_MODEL_ID})...")
    model = DepthAnything3.from_pretrained(DA3_MODEL_ID).to(device)
    model.eval()

    slopes, signed_grads, roughs, infracs = [], [], [], []
    for fp in frame_paths:
        with torch.no_grad():
            pred = model.inference([fp])

        def to_np(x):
            return x.cpu().float().numpy() if hasattr(x, "cpu") else np.asarray(x, dtype=np.float32)

        depth = to_np(pred.depth)[0]        # (H,W)
        K     = to_np(pred.intrinsics)[0]   # (3,3)
        H, W  = depth.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        usable = _vegetation_mask(fp, (H, W)) & np.isfinite(depth) & (depth > 0)
        vs, us = np.where(usable)
        if len(vs) < 1000:
            print(f"    {os.path.basename(fp)}: skipped (mask too small)")
            continue
        if len(vs) > 60_000:
            sel = np.random.default_rng(0).choice(len(vs), 60_000, replace=False)
            vs, us = vs[sel], us[sel]

        D = depth[vs, us]
        X = (us - cx) / fx * D          # +X right
        Y = (vs - cy) / fy * D          # +Y down (image), horizontal in world
        (a, b, _), inliers, resid = _robust_plane(X, Y, D)

        slope = float(np.degrees(np.arctan(np.hypot(a, b))))
        rough = float(np.median(np.abs(resid)) / max(np.median(D), 1e-9))
        infrac = float(inliers.mean())
        slopes.append(slope)
        roughs.append(rough)
        infracs.append(infrac)

        pair = os.path.join(frames_dir, "pairs",
                            os.path.basename(fp).replace(".jpg", "_p.jpg"))
        u = _travel_direction(fp, pair) if os.path.exists(pair) else None
        # +ve: deeper ahead = downhill; nan when motion estimation failed
        signed_grads.append(a * u[0] + b * u[1] if u is not None else np.nan)

        print(f"    {os.path.basename(fp)}: {slope:.2f}°  "
              f"(rough {rough:.3f}, inliers {infrac:.0%})")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    if not slopes:
        return None
    slopes = np.array(slopes)
    roughs = np.array(roughs)
    infracs = np.array(infracs)
    signed_grads = np.array(signed_grads)

    # ── quality gate: drop frames with rough / fragmented plane fits (trees) ──
    rough_thresh = max(2.5 * np.median(roughs), 0.005)
    good = (roughs < rough_thresh) & (infracs > 0.6)
    if good.sum() < 3:
        print("  Quality gate left <3 frames — keeping all frames instead.")
        good = np.ones(len(slopes), bool)
    rejected = np.where(~good)[0]
    if len(rejected):
        print(f"  Quality gate: kept {good.sum()}/{len(slopes)} frames "
              f"(rejected: {', '.join(map(str, rejected))})")

    med = float(np.median(slopes[good]))
    mad = float(np.median(np.abs(slopes[good] - med)))

    votes = signed_grads[good]
    votes = votes[np.isfinite(votes)]
    if len(votes):
        down_votes = int(np.sum(votes > 0))
        direction = "downhill" if down_votes > len(votes) / 2 else "uphill"
        dir_note = f"{direction} ({down_votes}/{len(votes)} clean frames agree)"
    else:
        direction, dir_note = None, "n/a (motion estimation failed)"

    print(f"  Per-frame slope (clean frames): median {med:.2f}° ± {mad:.2f}° (MAD), direction: {dir_note}")
    return {"slopes": slopes, "good": good, "median": med, "mad": mad,
            "direction": direction, "dir_note": dir_note}


# ── Visualization ──────────────────────────────────────────────────────────────

def visualise(res_a, res_b, video_name, out_png):
    n = (res_a is not None) + (res_b is not None)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    axes = np.atleast_1d(axes)
    col = 0

    if res_a is not None:
        ax = axes[col]; col += 1
        s, gh, ch, ok = res_a["s"], res_a["ground_h"], res_a["cam_h"], res_a["valid"]
        order = np.argsort(s)
        ax.plot(s[order], -ch[order], "r*-", ms=8, lw=1, label="camera path")
        ax.plot(s[order][ok[order]], -gh[order][ok[order]], "o-", c="dodgerblue",
                ms=6, lw=1.5, label="ground below camera")
        bad = order[~ok[order]]
        if len(bad):
            ax.plot(s[bad], -np.nan_to_num(gh[bad], nan=np.nanmean(gh)), "x",
                    c="orange", ms=8, label="excluded")
        ax.set_xlabel("Along-flight distance (VGGT units)")
        ax.set_ylabel("Height (up)")
        ax.set_title(f"Path A — VGGT multi-view\n"
                     f"{res_a['slope']:.2f}° {res_a['direction']} "
                     f"(camera path: {res_a['cam_slope']:.2f}°)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    if res_b is not None:
        ax = axes[col]
        sl = res_b["slopes"]
        good = res_b.get("good", np.ones(len(sl), bool))
        idx = np.arange(len(sl))
        ax.plot(idx[good], sl[good], "o-", c="steelblue", ms=5, label="clean frames")
        if (~good).any():
            ax.plot(idx[~good], sl[~good], "x", c="tomato", ms=8,
                    label="rejected (rough fit — trees)")
        ax.axhline(res_b["median"], color="red", ls="--",
                   label=f"median {res_b['median']:.2f}° ± {res_b['mad']:.2f}°")
        if res_a is not None:
            ax.axhline(res_a["slope"], color="green", ls=":",
                       label=f"VGGT {res_a['slope']:.2f}°")
        ax.set_xlabel("Frame index (along flight)")
        ax.set_ylabel("Slope (°)")
        ax.set_title(f"Path B — DA3 per-frame depth plane\ndirection: {res_b['dir_note']}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(f"{video_name} — BEV slope estimation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {out_png}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="BEV video slope estimation")
    ap.add_argument("video")
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--method", choices=["both", "vggt", "dav3", "extract"], default="both")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.video))[0]
    frames_dir = os.path.join(SLOPE_DIR, "bev_frames", stem)
    out_dir = os.path.join(SLOPE_DIR, "output_glbs", "bev")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[1/3] Extracting frames from {args.video}")
    frame_paths = extract_frames(args.video, args.n_frames, frames_dir)
    if args.method == "extract":
        print("Done (extract-only).")
        return

    res_a = res_b = None
    if args.method in ("both", "vggt"):
        print(f"\n[2/3] Path A — VGGT multi-view")
        res_a = vggt_analysis(frame_paths, os.path.join(out_dir, f"{stem}_vggt.glb"))

    if args.method in ("both", "dav3"):
        print(f"\n[3/3] Path B — DA3 per-frame depth")
        res_b = dav3_analysis(frame_paths, frames_dir)

    print(f"\n{'═'*50}")
    print(f" BEV slope summary — {stem}")
    print(f"{'═'*50}")
    if res_a:
        print(f" Path A (VGGT) : {res_a['slope']:.2f}° {res_a['direction']}")
    if res_b:
        print(f" Path B (DA3)  : {res_b['median']:.2f}° ± {res_b['mad']:.2f}°, {res_b['dir_note']}")
    print(f"{'═'*50}")

    visualise(res_a, res_b, stem, os.path.join(out_dir, f"{stem}_slope.png"))


if __name__ == "__main__":
    main()
