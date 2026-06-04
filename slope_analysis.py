"""
Terrain Slope Analysis from a VGGT / DA3 GLB file.

Usage:
    python slope_analysis.py <path_to_glb>

The GLB must contain:
    geometry_0          — terrain point cloud
    geometry_1..N       — camera frustum meshes (apex = vertices[1])

Two methods are applied based on the filename:
    fixed_altitude      — Drone flew at constant absolute height.
                          Slope is inferred from how far the ground is below
                          each camera position (ground-distance method).
                          A wall-avoidance filter selects the deepest ground
                          point in a vertical cylinder below each camera.

    fixed_distance2ground — Drone followed terrain at constant AGL height.
                          The camera trajectory itself traces the terrain.
                          Slope is read directly from camera altitude change.
"""

import sys
import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D

matplotlib.use("Agg")

# ── Parameters ─────────────────────────────────────────────────────────────────
SEARCH_RADIUS_FRAC = 0.08   # cylinder radius = this fraction of scene width
GROUND_PERCENTILE  = 90     # use 90th-percentile depth (avoids wall, avoids noise)
MAD_SCALE          = 5.0    # vertical outlier rejection: keep within N×MAD of median


# ── Core geometry helpers ──────────────────────────────────────────────────────

def flight_direction(pts_xz):
    """Principal horizontal direction of a set of XZ positions."""
    _, _, Vt = np.linalg.svd(pts_xz - pts_xz.mean(0), full_matrices=False)
    return Vt[0]


def ground_point_below(terrain, camera, radius):
    """
    Find the road surface below a camera.

    Searches a vertical cylinder of the given radius and returns the point at
    the 90th-percentile depth — deep enough to be the road surface (not the top
    of a wall, which sits closer to the drone) but robust against stray noise.
    """
    xz_dist = np.linalg.norm(terrain[:, [0, 2]] - camera[[0, 2]], axis=1)
    nearby = terrain[xz_dist < radius]
    if len(nearby) == 0:
        return None

    below = nearby[nearby[:, 1] > camera[1]]   # only points below camera (+Y=down)
    if len(below) == 0:
        below = nearby

    y90 = np.percentile(below[:, 1], GROUND_PERCENTILE)
    return below[np.argmin(np.abs(below[:, 1] - y90))]


# ── Slope estimation methods ───────────────────────────────────────────────────

def slope_fixed_altitude(terrain, cameras):
    """
    For fixed-altitude flights: slope comes from how the ground depth below the
    drone varies along the flight path.

    Returns: slope_deg, direction, ground_points, valid_mask
    """
    scene_width = np.ptp(terrain[:, [0, 2]])
    radius = scene_width * SEARCH_RADIUS_FRAC

    ground_pts, valid = [], []
    for cam in cameras:
        pt = ground_point_below(terrain, cam, radius)
        if pt is not None:
            ground_pts.append(pt)
            valid.append(True)
        else:
            ground_pts.append(cam.copy())
            valid.append(False)

    ground_pts = np.array(ground_pts)
    valid = np.array(valid)

    # Vertical MAD filter: drop points far from the median ground height
    if valid.sum() > 0:
        med = np.median(ground_pts[valid, 1])
        mad = np.median(np.abs(ground_pts[valid, 1] - med))
        valid &= np.abs(ground_pts[:, 1] - med) < MAD_SCALE * max(mad, 1e-6)

    if valid.sum() < 2:
        return None, None, ground_pts, valid

    gp = ground_pts[valid]
    fd = flight_direction(cameras[valid, :][:, [0, 2]])
    s  = (gp[:, [0, 2]] - gp[:, [0, 2]].mean(0)) @ fd

    gradient   = np.polyfit(s, gp[:, 1], 1)[0]
    slope_deg  = np.degrees(np.arctan(abs(gradient)))

    # +Y=down: positive gradient → terrain descends → downhill
    order     = np.argsort(s)
    direction = "downhill" if gp[order[-1], 1] > gp[order[0], 1] else "uphill"

    return slope_deg, direction, ground_pts, valid


def slope_fixed_distance(cameras):
    """
    For fixed-distance-to-ground flights: the drone physically follows the
    terrain at constant AGL height, so the camera altitude change IS the slope.

    Returns: slope_deg, direction
    """
    if len(cameras) < 2:
        return None, None

    fd    = flight_direction(cameras[:, [0, 2]])
    s     = (cameras[:, [0, 2]] - cameras[:, [0, 2]].mean(0)) @ fd
    order = np.argsort(s)
    cams  = cameras[order]

    gradient  = np.polyfit(s[order], cams[:, 1], 1)[0]
    slope_deg = np.degrees(np.arctan(abs(gradient)))

    direction = "downhill" if cams[-1, 1] > cams[0, 1] else "uphill"

    return slope_deg, direction


# ── Pairwise slopes ────────────────────────────────────────────────────────────

def pairwise_slopes(pts, valid, fd):
    """Compute slope between each consecutive pair of points along the flight."""
    s     = (pts[:, [0, 2]] - pts[:, [0, 2]].mean(0)) @ fd
    order = np.argsort(s)
    pts_o = pts[order]
    ok_o  = valid[order]

    pairs = []
    for i in range(len(pts_o) - 1):
        a, b   = pts_o[i], pts_o[i + 1]
        horiz  = np.linalg.norm(b[[0, 2]] - a[[0, 2]])
        dY     = abs(b[1] - a[1])
        angle  = np.degrees(np.arctan(dY / horiz)) if horiz > 1e-6 else float("nan")
        is_out = not (ok_o[i] and ok_o[i + 1])
        pairs.append({
            "label":   f"c{order[i]+1}→c{order[i+1]+1}",
            "horiz":   horiz,
            "dY":      dY,
            "slope":   angle,
            "outlier": is_out,
        })
    return pairs


# ── Visualisation ──────────────────────────────────────────────────────────────

def visualise(terrain, cameras, ground_pts, valid, slope_deg, direction,
              pairs, method_label, glb_path, out_png):

    fig = plt.figure(figsize=(16, 9))

    # ── left: 3-D point cloud + cameras + fitted plane ─────────────────────
    ax3 = fig.add_subplot(121, projection="3d")

    rng = np.random.default_rng(42)
    sub = rng.choice(len(terrain), min(40_000, len(terrain)), replace=False)
    pts = terrain[sub]
    hv  = -pts[:, 1]
    cols = plt.cm.terrain((hv - hv.min()) / max(hv.max() - hv.min(), 1e-9))[:, :3]
    ax3.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=cols, s=0.3, alpha=0.35, rasterized=True)

    for cam, gp, ok in zip(cameras, ground_pts, valid):
        if ok:
            ax3.plot([cam[0], gp[0]], [cam[2], gp[2]], [cam[1], gp[1]], color="gold", lw=1.5)
            ax3.scatter(*[[v] for v in [gp[0],  gp[2],  gp[1]]],  c="dodgerblue", s=50,  marker="o", zorder=6)
            ax3.scatter(*[[v] for v in [cam[0], cam[2], cam[1]]], c="red",        s=70,  marker="*", zorder=7)
        else:
            ax3.scatter(*[[v] for v in [cam[0], cam[2], cam[1]]], c="orange",     s=70,  marker="X", zorder=7)

    gp_good = ground_pts[valid]
    if len(gp_good) >= 3:
        centroid = gp_good.mean(0)
        _, _, Vt = np.linalg.svd(gp_good - centroid, full_matrices=False)
        nrm = Vt[-1] * np.sign(Vt[-1, 1])
        sx  = gp_good[:, 0].max() - gp_good[:, 0].min() + 0.2
        sz  = gp_good[:, 2].max() - gp_good[:, 2].min() + 0.2
        cx, cz = centroid[0], centroid[2]
        corners = [[cx+dx, cz+dz, centroid[1] - (nrm[0]*dx + nrm[2]*dz) / (nrm[1]+1e-12)]
                   for dx, dz in [(-sx/2,-sz/2),(sx/2,-sz/2),(sx/2,sz/2),(-sx/2,sz/2)]]
        ax3.add_collection3d(Poly3DCollection([[[c[0],c[1],c[2]] for c in corners]],
                             alpha=0.2, facecolor="cyan", edgecolor="cyan", lw=0.5))

    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Z (m)"); ax3.set_zlabel("Y +down (m)")
    ax3.set_title(f"Slope = {slope_deg:.2f}° {direction}", fontsize=11)
    ax3.view_init(elev=20, azim=-60)
    ax3.legend(handles=[
        Line2D([0],[0], color="gold",      lw=2,   label="gravity lines"),
        Line2D([0],[0], marker="*", color="red",        lw=0, ms=9, label="cameras"),
        Line2D([0],[0], marker="X", color="orange",     lw=0, ms=9, label="camera (excluded)"),
        Line2D([0],[0], marker="o", color="dodgerblue", lw=0, ms=7, label="ground points"),
    ], fontsize=7, loc="upper left")

    # ── right: pairwise bar chart ──────────────────────────────────────────
    ax2 = fig.add_subplot(122)
    labels  = [p["label"]   for p in pairs]
    slopes  = [p["slope"]   for p in pairs]
    outlier = [p["outlier"] for p in pairs]
    colors  = ["tomato" if o else "steelblue" for o in outlier]

    bars = ax2.bar(labels, slopes, color=colors, edgecolor="white", width=0.6)
    ax2.axhline(slope_deg, color="red", lw=1.5, linestyle="--",
                label=f"overall {slope_deg:.2f}° {direction}")

    good_s = [v for v, o in zip(slopes, outlier) if np.isfinite(v) and not o]
    y_max  = (max(good_s) if good_s else slope_deg) * 1.5 + 2
    ax2.set_ylim(0, y_max)

    for bar, val, out in zip(bars, slopes, outlier):
        if not np.isfinite(val):
            continue
        disp = min(val, y_max * 0.97)
        bar.set_height(disp)
        lbl  = f"{val:.1f}°" + (" ↑" if val > y_max else "")
        ax2.text(bar.get_x() + bar.get_width()/2, disp + 0.1, lbl,
                 ha="center", va="bottom", fontsize=7,
                 color="tomato" if out else "black")

    ax2.set_ylabel("Slope (°)")
    ax2.set_xlabel("Consecutive camera pair")
    ax2.set_title("Pairwise slopes along flight path", fontsize=10)
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(handles=[
        plt.Rectangle((0,0),1,1, color="steelblue", label="good pair"),
        plt.Rectangle((0,0),1,1, color="tomato",    label="excluded pair"),
        Line2D([0],[0], color="red", lw=1.5, linestyle="--", label=f"overall {slope_deg:.2f}°"),
    ], fontsize=7)

    name = os.path.splitext(os.path.basename(glb_path))[0]
    plt.suptitle(f"{name}  |  {slope_deg:.2f}° {direction}  |  {method_label}",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_png}")


# ── Main ───────────────────────────────────────────────────────────────────────

def analyse(glb_path):
    print(f"\nAnalysing: {glb_path}")

    scene   = trimesh.load(glb_path)
    terrain = np.asarray(scene.geometry["geometry_0"].vertices, dtype=np.float64)
    terrain = terrain[np.isfinite(terrain).all(axis=1)]

    n_cams  = sum(1 for k in scene.geometry if k.startswith("geometry_") and k != "geometry_0")
    cameras = np.array([scene.geometry[f"geometry_{i}"].vertices[1]
                        for i in range(1, n_cams + 1)], dtype=np.float64)

    name = os.path.splitext(os.path.basename(glb_path))[0]

    if "fixed_distance" in name:
        method_label = "fixed-distance-to-ground: camera-altitude method"
        slope_deg, direction = slope_fixed_distance(cameras)
        if slope_deg is None:
            print("  Not enough cameras.")
            return
        # For visualisation: use cameras as both "camera" and "ground reference"
        ground_pts = cameras.copy()
        valid      = np.ones(len(cameras), dtype=bool)
        fd         = flight_direction(cameras[:, [0, 2]])

    else:
        method_label = "fixed-altitude: ground-distance method"
        slope_deg, direction, ground_pts, valid = slope_fixed_altitude(terrain, cameras)
        if slope_deg is None:
            print("  Not enough valid ground points.")
            return
        fd = flight_direction(cameras[valid, :][:, [0, 2]])

    print(f"  Method  : {method_label}")
    print(f"  Slope   : {slope_deg:.2f}°")
    print(f"  Direction: {direction}")
    print(f"  Cameras used: {valid.sum()}/{len(cameras)}")

    pairs = pairwise_slopes(ground_pts, valid, fd)
    print(f"\n  {'Pair':>10}   {'horiz':>8}   {'ΔY':>7}   {'slope':>7}   {'note':>9}")
    for p in pairs:
        note = "⚠ excluded" if p["outlier"] else ""
        angle = f"{p['slope']:.2f}°" if np.isfinite(p["slope"]) else "nan"
        print(f"  {p['label']:>10}   {p['horiz']:>7.3f}m   {p['dY']:>6.3f}m   {angle:>7}   {note}")

    out_png = os.path.splitext(glb_path)[0] + "_slope.png"
    visualise(terrain, cameras, ground_pts, valid, slope_deg, direction,
              pairs, method_label, glb_path, out_png)

    return slope_deg, direction


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python slope_analysis.py <path_to_glb> [<path_to_glb> ...]")
        sys.exit(1)

    results = []
    for glb_path in sys.argv[1:]:
        if not os.path.exists(glb_path):
            print(f"File not found: {glb_path}")
            continue
        result = analyse(glb_path)
        if result:
            results.append((os.path.basename(glb_path), *result))

    if results:
        print(f"\n{'─'*55}")
        print(f"  {'File':<35}  {'Slope':>7}  Direction")
        print(f"{'─'*55}")
        for name, slope, direction in results:
            print(f"  {name:<35}  {slope:>6.2f}°  {direction}")
        print(f"{'─'*55}")
