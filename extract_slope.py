import glob
import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D

matplotlib.use("Agg")


def _flight_dir(pts_xz):
    _, _, Vt = np.linalg.svd(pts_xz - pts_xz.mean(0), full_matrices=False)
    return Vt[0]


def _ground_point(terrain, cam, search_radius):
    """
    Find the road surface below a camera. Takes the 90th-percentile Y point
    within a horizontal cylinder — deep enough to be road (not wall), but
    robust against stray noise artifacts at extreme depths.
    """
    xz_dists = np.linalg.norm(terrain[:, [0, 2]] - cam[[0, 2]], axis=1)
    candidates = terrain[xz_dists < search_radius]
    if len(candidates) == 0:
        return None, np.inf
    below = candidates[candidates[:, 1] > cam[1]]
    if len(below) == 0:
        below = candidates
    y90 = np.percentile(below[:, 1], 90)
    # take the point closest to the 90th percentile
    idx = np.argmin(np.abs(below[:, 1] - y90))
    pt = below[idx]
    dist_xz = np.linalg.norm(pt[[0, 2]] - cam[[0, 2]])
    return pt, dist_xz


def process_fixed_altitude(scene, terrain, cameras, name):
    """
    Drone at constant absolute altitude. Slope comes from how far the ground
    is below the drone at each position. Wall is excluded by taking the
    furthest-down point (ground), not the nearest in XZ (which may be wall).
    """
    scene_width = np.ptp(terrain[:, [0, 2]])
    search_r = scene_width * 0.08   # search cylinder radius

    ground_pts, ok = [], []
    for cam in cameras:
        pt, _ = _ground_point(terrain, cam, search_r)
        if pt is not None:
            ground_pts.append(pt)
            ok.append(True)
        else:
            ground_pts.append(cam)
            ok.append(False)
    ground_pts = np.array(ground_pts)
    ok = np.array(ok)

    # vertical MAD filter to remove remaining outliers
    med_y = np.median(ground_pts[ok, 1])
    mad_y = np.median(np.abs(ground_pts[ok, 1] - med_y))
    vert_ok = np.abs(ground_pts[:, 1] - med_y) < 5.0 * max(mad_y, 1e-6)
    ok = ok & vert_ok

    bp_good = ground_pts[ok]
    cam_good = cameras[ok]
    print(f"  Cameras used: {ok.sum()}/{len(cameras)}")

    fd = _flight_dir(cam_good[:, [0, 2]])
    s = (bp_good[:, [0, 2]] - bp_good[:, [0, 2]].mean(0)) @ fd
    slope_overall = np.degrees(np.arctan(abs(np.polyfit(s, bp_good[:, 1], 1)[0])))
    print(f"Overall slope : {slope_overall:.2f}°  [fixed-altitude / ground-distance method]")

    return ground_pts, ok, slope_overall, fd


def process_fixed_distance(cameras):
    """
    Drone follows terrain at constant AGL height. Camera altitude change IS
    the terrain slope. Wall is irrelevant — camera height was set physically
    by the drone, not by the point cloud.
    """
    fd = _flight_dir(cameras[:, [0, 2]])
    s = (cameras[:, [0, 2]] - cameras[:, [0, 2]].mean(0)) @ fd
    order = np.argsort(s)
    cams_ord = cameras[order]
    slope_overall = np.degrees(np.arctan(abs(np.polyfit(s[order], cams_ord[:, 1], 1)[0])))
    ok = np.ones(len(cameras), dtype=bool)
    print(f"  Cameras used: {ok.sum()}/{len(cameras)}")
    print(f"Overall slope : {slope_overall:.2f}°  [fixed-distance / camera-altitude method]")
    return cameras, ok, slope_overall, fd


def _pairwise(pts, ok, fd):
    s_all = (pts[:, [0, 2]] - pts[:, [0, 2]].mean(0)) @ fd
    order = np.argsort(s_all)
    pts_ord = pts[order]
    ok_ord  = ok[order]

    print(f"\nPairwise slopes (consecutive pairs):")
    print(f"  {'Pair':>10}   {'horiz dist':>12}   {'ΔY':>8}   {'slope':>8}   {'note':>10}")
    pair_slopes, pair_labels, pair_outlier = [], [], []
    for i in range(len(pts_ord) - 1):
        a, b   = pts_ord[i], pts_ord[i + 1]
        horiz  = np.linalg.norm(b[[0, 2]] - a[[0, 2]])
        dY     = abs(b[1] - a[1])
        slope  = np.degrees(np.arctan(dY / horiz)) if horiz > 1e-6 else float("nan")
        ia, ib = order[i] + 1, order[i + 1] + 1
        out    = not (ok_ord[i] and ok_ord[i + 1])
        note   = "⚠ outlier" if out else ""
        print(f"  cam{ia:2d}→cam{ib:2d}   {horiz:10.3f} m   {dY:6.3f} m   {slope:6.2f}°   {note}")
        pair_slopes.append(slope)
        pair_labels.append(f"c{ia}→c{ib}")
        pair_outlier.append(out)
    return pair_slopes, pair_labels, pair_outlier, order, ok_ord


def _plot(terrain, cameras, blue_pts, ok, slope_overall, normal, centroid,
          pair_slopes, pair_labels, pair_outlier, glb_path, out_png, subtitle):
    fig = plt.figure(figsize=(16, 9))
    ax3 = fig.add_subplot(121, projection="3d")

    rng = np.random.default_rng(42)
    sub = rng.choice(len(terrain), size=min(40_000, len(terrain)), replace=False)
    pts_sub = terrain[sub]
    hv = -pts_sub[:, 1]
    cols = plt.cm.terrain((hv - hv.min()) / max(hv.max() - hv.min(), 1e-9))[:, :3]
    ax3.scatter(pts_sub[:, 0], pts_sub[:, 2], pts_sub[:, 1],
                c=cols, s=0.3, alpha=0.35, rasterized=True)

    for cam, bp, good in zip(cameras, blue_pts, ok):
        if good:
            ax3.plot([cam[0], bp[0]], [cam[2], bp[2]], [cam[1], bp[1]], color="gold", lw=1.8)
            ax3.scatter(*[[v] for v in [bp[0], bp[2], bp[1]]], c="dodgerblue", s=60, marker="o", zorder=6)
            ax3.scatter(*[[v] for v in [cam[0], cam[2], cam[1]]], c="red", s=80, marker="*", zorder=7)
        else:
            ax3.scatter(*[[v] for v in [cam[0], cam[2], cam[1]]], c="orange", s=80, marker="X", zorder=7)

    bp_good = blue_pts[ok]
    sx = (bp_good[:, 0].max() - bp_good[:, 0].min()) + 0.2
    sz = (bp_good[:, 2].max() - bp_good[:, 2].min()) + 0.2
    cx, cz = centroid[0], centroid[2]
    corners = [[cx + dx, cz + dz,
                centroid[1] - (normal[0]*dx + normal[2]*dz) / (normal[1] + 1e-12)]
               for dx, dz in [(-sx/2,-sz/2),(sx/2,-sz/2),(sx/2,sz/2),(-sx/2,sz/2)]]
    ax3.add_collection3d(Poly3DCollection([[[c[0], c[1], c[2]] for c in corners]],
                         alpha=0.2, facecolor="cyan", edgecolor="cyan", lw=0.5))
    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Z (m)"); ax3.set_zlabel("Y +down (m)")
    ax3.set_title(f"3-D scene\nOverall slope = {slope_overall:.2f}°", fontsize=10)
    ax3.view_init(elev=20, azim=-60)
    ax3.legend(handles=[
        Line2D([0],[0], color="gold", lw=2, label="gravity lines"),
        Line2D([0],[0], marker="*", color="red", lw=0, ms=9, label="cameras (used)"),
        Line2D([0],[0], marker="X", color="orange", lw=0, ms=9, label="camera (outlier)"),
        Line2D([0],[0], marker="o", color="dodgerblue", lw=0, ms=7, label="ground points"),
    ], fontsize=7, loc="upper left")

    ax2 = fig.add_subplot(122)
    bar_colors = ["tomato" if out else "steelblue" for out in pair_outlier]
    bars = ax2.bar(pair_labels, pair_slopes, color=bar_colors, edgecolor="white", width=0.6)
    ax2.axhline(slope_overall, color="red", lw=1.5, linestyle="--",
                label=f"overall {slope_overall:.2f}°")
    ax2.set_ylabel("Slope (°)")
    ax2.set_xlabel("Consecutive pair")
    ax2.set_title("Pairwise slopes", fontsize=10)
    ax2.tick_params(axis="x", rotation=45)

    good_slopes = [v for v, out in zip(pair_slopes, pair_outlier) if np.isfinite(v) and not out]
    y_max = (max(good_slopes) if good_slopes else slope_overall) * 1.5 + 2
    ax2.set_ylim(0, y_max)
    for bar, val, out in zip(bars, pair_slopes, pair_outlier):
        if not np.isfinite(val):
            continue
        display = min(val, y_max * 0.97)
        bar.set_height(display)
        label = f"{val:.1f}°" + (" ↑" if val > y_max else "")
        ax2.text(bar.get_x() + bar.get_width()/2, display + 0.1,
                 label, ha="center", va="bottom", fontsize=7,
                 color="tomato" if out else "black")
    ax2.legend(handles=[
        plt.Rectangle((0,0),1,1, color="steelblue", label="good pair"),
        plt.Rectangle((0,0),1,1, color="tomato",    label="outlier pair"),
        Line2D([0],[0], color="red", lw=1.5, linestyle="--", label=f"overall {slope_overall:.2f}°"),
    ], fontsize=7)

    name = os.path.splitext(os.path.basename(glb_path))[0]
    plt.suptitle(f"{name} — {subtitle}", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_png}\n")


def process_glb(glb_path, out_png):
    scene = trimesh.load(glb_path)
    terrain = np.asarray(scene.geometry["geometry_0"].vertices, dtype=np.float64)
    terrain = terrain[np.isfinite(terrain).all(axis=1)]

    n_cams = sum(1 for k in scene.geometry if k.startswith("geometry_") and k != "geometry_0")
    cameras = np.array([scene.geometry[f"geometry_{i}"].vertices[1]
                        for i in range(1, n_cams + 1)], dtype=np.float64)

    name = os.path.splitext(os.path.basename(glb_path))[0]
    is_fixed_dist = "fixed_distance" in name

    if is_fixed_dist:
        blue_pts, ok, slope_overall, fd = process_fixed_distance(cameras)
        subtitle = "fixed-distance / camera-altitude method"
    else:
        blue_pts, ok, slope_overall, fd = process_fixed_altitude(scene, terrain, cameras, name)
        subtitle = "fixed-altitude / ground-distance method"

    bp_good = blue_pts[ok]
    centroid = bp_good.mean(0)
    _, _, Vt3 = np.linalg.svd(bp_good - centroid, full_matrices=False)
    normal = Vt3[-1] * np.sign(Vt3[-1, 1])

    pair_slopes, pair_labels, pair_outlier, order, ok_ord = _pairwise(blue_pts, ok, fd)

    _plot(terrain, cameras, blue_pts, ok, slope_overall, normal, centroid,
          pair_slopes, pair_labels, pair_outlier, glb_path, out_png, subtitle)


def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_glbs")
    # Support both flat layout (legacy) and model subfolders (vggt/ dav3/)
    glbs = sorted(glob.glob(os.path.join(base, "*.glb")))
    for model_dir in ["vggt", "dav3"]:
        glbs += sorted(glob.glob(os.path.join(base, model_dir, "*.glb")))
    if not glbs:
        print("No GLB files found in output_glbs/")
        return
    for glb_path in glbs:
        name = os.path.splitext(os.path.basename(glb_path))[0]
        out_png = os.path.join(os.path.dirname(glb_path), f"{name}_slope.png")
        print(f"\n[{name}]")
        process_glb(glb_path, out_png)


if __name__ == "__main__":
    main()
