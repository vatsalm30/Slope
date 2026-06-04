import sys
import glob
import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

matplotlib.use("Agg")

OUTLIER_THRESH_M = 0.05   # drop cameras with no terrain within 5 cm horizontally


def process_glb(glb_path, out_png):
    scene = trimesh.load(glb_path)

    terrain = np.asarray(scene.geometry["geometry_0"].vertices, dtype=np.float64)

    n_cams = sum(1 for k in scene.geometry if k.startswith("geometry_") and k != "geometry_0")
    cameras = np.array([scene.geometry[f"geometry_{i}"].vertices[1]
                        for i in range(1, n_cams + 1)], dtype=np.float64)

    tree = cKDTree(terrain[:, [0, 2]])
    dists, idxs = tree.query(cameras[:, [0, 2]], k=1)
    blue_pts = terrain[idxs]
    ok = dists < OUTLIER_THRESH_M

    bp_good  = blue_pts[ok]
    cam_good = cameras[ok]

    cam_xz = cam_good[:, [0, 2]] - cam_good[:, [0, 2]].mean(0)
    _, _, Vt = np.linalg.svd(cam_xz, full_matrices=False)
    flight_dir = Vt[0]

    s = (bp_good[:, [0, 2]] - bp_good[:, [0, 2]].mean(0)) @ flight_dir
    slope_overall = np.degrees(np.arctan(abs(np.polyfit(s, bp_good[:, 1], 1)[0])))
    print(f"Overall slope : {slope_overall:.2f}°")

    s_all = (blue_pts[:, [0, 2]] - blue_pts[:, [0, 2]].mean(0)) @ flight_dir
    order = np.argsort(s_all)
    blue_ordered = blue_pts[order]
    ok_ordered   = ok[order]

    print(f"\nPairwise slopes (consecutive camera pairs):")
    print(f"  {'Pair':>10}   {'horiz dist':>12}   {'ΔY':>8}   {'slope':>8}   {'note':>10}")
    pair_slopes = []
    pair_labels = []
    for i in range(len(blue_ordered) - 1):
        a, b   = blue_ordered[i], blue_ordered[i + 1]
        horiz  = np.linalg.norm(b[[0, 2]] - a[[0, 2]])
        dY     = abs(b[1] - a[1])
        slope  = np.degrees(np.arctan(dY / horiz)) if horiz > 1e-6 else float("nan")
        ia, ib = order[i] + 1, order[i + 1] + 1
        note   = "" if (ok_ordered[i] and ok_ordered[i + 1]) else "⚠ outlier"
        print(f"  cam{ia:2d}→cam{ib:2d}   {horiz:10.3f} m   {dY:6.3f} m   {slope:6.2f}°   {note}")
        pair_slopes.append(slope)
        pair_labels.append(f"c{ia}→c{ib}")

    fig = plt.figure(figsize=(16, 9))

    ax3 = fig.add_subplot(121, projection="3d")
    rng = np.random.default_rng(42)
    sub = rng.choice(len(terrain), size=min(40_000, len(terrain)), replace=False)
    pts_sub = terrain[sub]
    hv = -pts_sub[:, 1]
    cols = plt.cm.terrain((hv - hv.min()) / max(hv.max() - hv.min(), 1e-9))[:, :3]
    ax3.scatter(pts_sub[:, 0], pts_sub[:, 2], pts_sub[:, 1],
                c=cols, s=0.3, alpha=0.35, rasterized=True)

    for i, (cam, bp, good) in enumerate(zip(cameras, blue_pts, ok)):
        if good:
            ax3.plot([cam[0], bp[0]], [cam[2], bp[2]], [cam[1], bp[1]],
                     color="gold", lw=1.8)
            ax3.scatter(*[[v] for v in [bp[0], bp[2], bp[1]]],
                        c="dodgerblue", s=60, marker="o", zorder=6)
            ax3.scatter(*[[v] for v in [cam[0], cam[2], cam[1]]],
                        c="red", s=80, marker="*", zorder=7)
        else:
            ax3.scatter(*[[v] for v in [cam[0], cam[2], cam[1]]],
                        c="orange", s=80, marker="X", zorder=7)

    centroid = bp_good.mean(0)
    _, _, Vt3 = np.linalg.svd(bp_good - centroid)
    nrm = Vt3[-1] * np.sign(Vt3[-1] @ np.array([0., 1., 0.]))
    sx = (bp_good[:, 0].max() - bp_good[:, 0].min()) + 0.2
    sz = (bp_good[:, 2].max() - bp_good[:, 2].min()) + 0.2
    cx, cz = centroid[0], centroid[2]
    corners = [[cx + dx, cz + dz,
                centroid[1] - (nrm[0]*dx + nrm[2]*dz) / (nrm[1] + 1e-12)]
               for dx, dz in [(-sx/2,-sz/2),(sx/2,-sz/2),(sx/2,sz/2),(-sx/2,sz/2)]]
    plane_verts = [[c[0], c[1], c[2]] for c in corners]
    ax3.add_collection3d(Poly3DCollection([plane_verts], alpha=0.2,
                         facecolor="cyan", edgecolor="cyan", lw=0.5))

    ax3.set_xlabel("X (m)");  ax3.set_ylabel("Z (m)");  ax3.set_zlabel("Y +down (m)")
    ax3.set_title(f"3-D scene\nOverall slope = {slope_overall:.2f}°", fontsize=10)
    ax3.view_init(elev=20, azim=-60)

    from matplotlib.lines import Line2D
    ax3.legend(handles=[
        Line2D([0],[0], color="gold", lw=2, label="gravity lines"),
        Line2D([0],[0], marker="*", color="red", lw=0, ms=9, label="cameras (used)"),
        Line2D([0],[0], marker="X", color="orange", lw=0, ms=9, label="camera (outlier)"),
        Line2D([0],[0], marker="o", color="dodgerblue", lw=0, ms=7, label="blue points"),
    ], fontsize=7, loc="upper left")

    ax2 = fig.add_subplot(122)
    colors_bar = ["steelblue"] * len(pair_slopes)
    bars = ax2.bar(pair_labels, pair_slopes, color=colors_bar, edgecolor="white", width=0.6)
    ax2.axhline(slope_overall, color="red", lw=1.5, linestyle="--",
                label=f"overall slope {slope_overall:.2f}°")
    ax2.set_ylabel("Slope (°)")
    ax2.set_xlabel("Consecutive camera pair")
    ax2.set_title("Pairwise slope between\nconsecutive camera positions", fontsize=10)
    ax2.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, pair_slopes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}°", ha="center", va="bottom", fontsize=8)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, max(pair_slopes) * 1.3 + 1)

    name = os.path.splitext(os.path.basename(glb_path))[0]
    plt.suptitle(f"{name} — terrain slope", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_png}\n")


def main():
    glb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_glbs")
    glbs = sorted(glob.glob(os.path.join(glb_dir, "*.glb")))

    if not glbs:
        print("No GLB files found in output_glbs/")
        return

    for glb_path in glbs:
        name = os.path.splitext(os.path.basename(glb_path))[0]
        out_png = os.path.join(glb_dir, f"{name}_slope.png")
        print(f"\n[{name}]")
        process_glb(glb_path, out_png)


if __name__ == "__main__":
    main()
