"""
Method 1 — ground-distance slope estimation (the general method).

Assumptions
-----------
* The GLB's +Y axis is aligned with gravity (+Y = down, VGGT/DA3 convention).
* geometry_0 is the terrain point cloud; geometry_1..N are camera frustums,
  with the camera centre at vertices[1]. Camera index 1..N follows the actual
  flight / capture order (see run_vggt_all.get_markers).

What it does
------------
For each camera it takes the road surface in a small vertical cylinder beneath
the camera (90th-percentile depth, to skip walls and stray noise), then fits a
line of ground-height vs. along-track distance. The slope is the angle of that
line. It reads the terrain directly, so — unlike the camera-altitude method —
it does NOT assume the drone kept a fixed height above the ground.

Conventions for this run
------------------------
* ALL frames are used (no outlier frames are dropped), per request.
* The sign is anchored to increasing image index (the direction the drone
  flew):  +  = terrain goes UPHILL as the flight progresses
          -  = terrain goes DOWNHILL as the flight progresses
"""

import os
import numpy as np
import trimesh
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

SEARCH_RADIUS_FRAC = 0.08   # base ground-search cylinder radius = this fraction of scene width
GROUND_PERCENTILE  = 90     # 90th-percentile depth = road surface (not wall top)
MIN_GROUND_PTS     = 50     # widen the cylinder until it holds at least this many points
MAX_RADIUS_GROWTH  = 10     # ...but never grow the radius beyond this multiple of the base


# ── geometry helpers ─────────────────────────────────────────────────────────

def load_glb(glb_path):
    """Return (terrain Nx3, cameras Mx3) with cameras in flight/capture order."""
    scene   = trimesh.load(glb_path)
    terrain = np.asarray(scene.geometry["geometry_0"].vertices, dtype=np.float64)
    terrain = terrain[np.isfinite(terrain).all(axis=1)]
    n_cams  = sum(1 for k in scene.geometry if k.startswith("geometry_") and k != "geometry_0")
    cameras = np.array([scene.geometry[f"geometry_{i}"].vertices[1]
                        for i in range(1, n_cams + 1)], dtype=np.float64)
    return terrain, cameras


def flight_direction(pts_xz, cameras_xz):
    """Principal horizontal direction, oriented from first camera to last."""
    _, _, Vt = np.linalg.svd(pts_xz - pts_xz.mean(0), full_matrices=False)
    fd = Vt[0]
    if (cameras_xz[-1] - cameras_xz[0]) @ fd < 0:   # point it the way the drone flew
        fd = -fd
    return fd


def ground_point_below(terrain, camera, radius):
    """Road surface beneath a camera: 90th-percentile-depth point in a vertical
    cylinder. The cylinder starts at `radius` and is widened (x1.5) until it holds
    at least MIN_GROUND_PTS points, so a camera whose footprint sits just off the
    densest terrain (e.g. forward-looking frames) still resolves a ground point
    instead of being dropped. Cameras already over dense terrain are unaffected —
    the loop exits on the first check."""
    xz_dist = np.linalg.norm(terrain[:, [0, 2]] - camera[[0, 2]], axis=1)
    max_radius = radius * MAX_RADIUS_GROWTH
    r = radius
    nearby = terrain[xz_dist < r]
    while len(nearby) < MIN_GROUND_PTS and r < max_radius:
        r *= 1.5
        nearby = terrain[xz_dist < r]
    if len(nearby) == 0:
        return None
    below = nearby[nearby[:, 1] > camera[1]]    # +Y = down, so "below" = larger Y
    if len(below) == 0:
        below = nearby
    y90 = np.percentile(below[:, 1], GROUND_PERCENTILE)
    return below[np.argmin(np.abs(below[:, 1] - y90))]


# ── Method 1 ─────────────────────────────────────────────────────────────────

def analyse(glb_path):
    """Run Method 1 on one GLB. Returns a result dict (all frames, signed)."""
    terrain, cameras = load_glb(glb_path)
    name = os.path.splitext(os.path.basename(glb_path))[0]
    return analyse_arrays(terrain, cameras, name)


def analyse_arrays(terrain, cameras, name="glb"):
    """Method 1 on pre-loaded arrays (terrain Nx3, cameras Mx3), so a caller can
    gravity-align the frame first (rotate terrain+cameras into a true-vertical
    frame) before measuring slope. Both arrays must share the same frame with
    +Y = down. Returns the same result dict as analyse()."""
    radius = np.ptp(terrain[:, [0, 2]]) * SEARCH_RADIUS_FRAC

    ground_pts, found = [], []
    for cam in cameras:
        pt = ground_point_below(terrain, cam, radius)
        found.append(pt is not None)
        ground_pts.append(pt if pt is not None else cam.copy())
    ground_pts = np.array(ground_pts)
    found      = np.array(found)

    if found.sum() < 2:
        return {"name": name, "ok": False, "n_cams": len(cameras)}

    # along-track coordinate, oriented by flight direction (image order)
    fd = flight_direction(cameras[:, [0, 2]], cameras[:, [0, 2]])
    s  = (ground_pts[:, [0, 2]] - ground_pts[:, [0, 2]].mean(0)) @ fd

    # overall slope: least-squares line of ground-Y vs along-track distance
    grad, intcpt = np.polyfit(s[found], ground_pts[found, 1], 1)
    yhat = grad * s[found] + intcpt
    ss_res = np.sum((ground_pts[found, 1] - yhat) ** 2)
    ss_tot = np.sum((ground_pts[found, 1] - ground_pts[found, 1].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    # +Y down: grad>0 → ground descends forward → downhill → negative signed slope
    overall_signed = -np.degrees(np.arctan(grad))
    direction      = "uphill" if overall_signed >= 0 else "downhill"

    # full 2-D terrain grade from a plane fit to the ground points — a cross-check
    # on the along-track line fit (which only sees the component along the flight
    # path). Only meaningful once the frame is gravity-aligned. `plane_slope_deg`
    # is the steepest grade of the surface; `aspect_deg` its downhill direction.
    plane_slope_deg, aspect_deg = plane_slope(ground_pts, found)

    # per-segment slopes between consecutive images (image order on x-axis)
    segments = []
    for i in range(len(cameras) - 1):
        a, b  = ground_pts[i], ground_pts[i + 1]
        horiz = np.linalg.norm(b[[0, 2]] - a[[0, 2]])
        dY    = b[1] - a[1]                              # +down
        seg   = -np.degrees(np.arctan2(dY, horiz)) if horiz > 1e-9 else float("nan")
        segments.append({
            "i": i + 1, "j": i + 2, "label": f"{i+1}→{i+2}",
            "signed_slope": seg, "horiz": horiz, "dY": dY,
            "ok": bool(found[i] and found[i + 1]),
        })

    return {
        "name": name, "ok": True, "n_cams": len(cameras),
        "n_ground_found": int(found.sum()),
        "overall_signed": overall_signed, "direction": direction, "r2": r2,
        "plane_slope_deg": plane_slope_deg, "aspect_deg": aspect_deg,
        "segments": segments,
        "terrain": terrain, "cameras": cameras,
        "ground_pts": ground_pts, "found": found, "fd": fd, "s": s,
    }


def plane_slope(ground_pts, found):
    """Terrain grade from a least-squares plane fit to the ground points, in the
    (gravity-aligned) +Y=down frame. Returns (max_slope_deg, aspect_deg):

      * max_slope_deg — steepest grade of the surface = angle of the plane normal
        from vertical. This is the full 2-D grade; the along-track `overall_signed`
        is its component along the flight path, so |overall_signed| <= max_slope_deg.
      * aspect_deg    — downhill direction in the horizontal (XZ) plane.

    A large gap between the two means the flight ran across the slope rather than
    up/down the fall line; NaN if fewer than 3 ground points."""
    g = ground_pts[found]
    if len(g) < 3:
        return float("nan"), float("nan")
    _, _, Vt = np.linalg.svd(g - g.mean(0), full_matrices=False)
    n = Vt[-1]                                          # unit plane normal
    n = n / (np.linalg.norm(n) + 1e-12)
    max_slope = float(np.degrees(np.arccos(min(1.0, abs(n[1])))))   # tilt from vertical
    aspect = (float(np.degrees(np.arctan2(n[2], n[0])))
              if abs(n[1]) < 0.9999 else float("nan"))
    return max_slope, aspect


# ── Method 2 (camera-altitude) — for side-by-side comparison only ─────────────

def analyse_method2(glb_path):
    """Camera-altitude slope (Method 2). Signed, image-order. For comparison."""
    _, cameras = load_glb(glb_path)
    if len(cameras) < 2:
        return None
    fd   = flight_direction(cameras[:, [0, 2]], cameras[:, [0, 2]])
    s    = (cameras[:, [0, 2]] - cameras[:, [0, 2]].mean(0)) @ fd
    grad = np.polyfit(s, cameras[:, 1], 1)[0]
    return -np.degrees(np.arctan(grad))     # +Y down → same sign convention


# ── per-segment plot ──────────────────────────────────────────────────────────

def plot_segments(result, out_png, title_extra="", final_signed=None, sign_note=""):
    """x-axis: image-index segment; y-axis: signed slope angle (+up / -down).

    The y-axis is capped to the bulk of the segments so a single near-vertical
    artifact (two closely-spaced cameras) doesn't flatten everything else;
    off-scale bars are drawn to the cap and labelled with their true value.

    `final_signed` overrides the displayed orientation (see plot_profile): when
    the up/down sign was anchored externally and opposes VGGT's, all values flip.
    """
    segs   = result["segments"]
    labels = [s["label"] for s in segs]
    vals   = [s["signed_slope"] for s in segs]
    ov     = result["overall_signed"]
    if final_signed is not None:
        if (final_signed < 0) != (ov < 0):
            vals = [-v for v in vals]
        ov = final_signed
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in vals]   # green up / red down

    finite = np.array([v for v in vals if np.isfinite(v)])
    absf   = np.abs(finite) if finite.size else np.array([1.0])
    cap = max(8.0, 2.5 * abs(ov), 1.5 * np.percentile(absf, 75))   # robust display cap

    fig, ax = plt.subplots(figsize=(10, 5.2))
    disp = [np.nan if not np.isfinite(v) else max(-cap, min(cap, v)) for v in vals]
    bars = ax.bar(range(len(labels)), disp, color=colors, edgecolor="white", width=0.7)
    ax.axhline(0, color="#444", lw=1)
    ax.axhline(ov, color="#264653", lw=1.6, linestyle="--",
               label=f"overall {ov:+.2f}°  (R²={result['r2']:.3f})")

    for b, v in zip(bars, vals):
        if not np.isfinite(v):
            continue
        clipped = abs(v) > cap
        y = max(-cap, min(cap, v))
        txt = (f"{v:+.0f}°↑" if v > 0 else f"{v:+.0f}°↓") if clipped else f"{v:+.1f}°"
        ax.text(b.get_x() + b.get_width() / 2,
                y + (0.2 if v >= 0 else -0.2), txt,
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8,
                fontweight="bold" if clipped else "normal",
                color="#9b2226" if clipped else "black")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Segment  (image index  i → i+1)")
    ax.set_ylabel("Slope angle (°)   + uphill / − downhill")
    ax.set_title(f"{result['name']}   |   per-segment slope (Method 1){title_extra}",
                 fontsize=11, fontweight="bold")
    if sign_note:
        ax.text(0.02, 0.02, sign_note, transform=ax.transAxes, fontsize=8,
                style="italic", color="#555", va="bottom")
    pad = cap * 0.18
    ax.set_ylim(min(-cap, ov) - pad, max(cap, ov) + pad)
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return out_png


def plot_profile(result, out_png, title_extra="", final_signed=None, sign_note=""):
    """The fit Method 1 actually performs: ground elevation vs along-track distance.

    x = along-track distance (m, in flight direction); y = elevation (= −Y, up).
    The fitted line's tilt IS the estimated slope; R² shows how cleanly the
    reconstruction encodes the terrain profile along the path.

    `final_signed` overrides the displayed orientation: VGGT's vertical axis is
    not gravity-locked, so when the up/down sign has been anchored externally
    (GPS or flight label) we flip the elevation axis to match, and annotate why.
    """
    s   = result["s"][result["found"]]
    elev = -result["ground_pts"][result["found"], 1]      # −Y so 'up' is up
    order = np.argsort(s); s = s[order]; elev = elev[order]

    raw_signed = result["overall_signed"]
    disp_signed = raw_signed
    if final_signed is not None:
        if (final_signed < 0) != (raw_signed < 0):        # external sign opposes VGGT's
            elev = -elev
        disp_signed = final_signed

    a, b = np.polyfit(s, elev, 1)                          # elevation slope per metre
    xs = np.linspace(s.min(), s.max(), 50)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.scatter(s, elev, c="dodgerblue", s=55, zorder=5, label="ground points")
    ax.plot(xs, a * xs + b, color="#264653", lw=2.2, linestyle="--",
            label=f"fit: {disp_signed:+.2f}°  (R²={result['r2']:.3f})")
    ax.set_xlabel("Along-track distance (m)  — flight direction →")
    ax.set_ylabel("Ground elevation (gravity-anchored, m)"
                  if final_signed is not None else "Ground elevation (m)  (= −Y, up)")
    ax.set_title(f"{result['name']}   |   Method 1 fit{title_extra}",
                 fontsize=11, fontweight="bold")
    if sign_note:
        ax.text(0.02, 0.02, sign_note, transform=ax.transAxes, fontsize=8,
                style="italic", color="#555", va="bottom")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return out_png


def plot_scene3d(result, out_png, title=None):
    """3-D view: terrain, cameras, gravity rays to the ground points, fitted plane."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.lines import Line2D

    terrain = result["terrain"]; cameras = result["cameras"]
    gp = result["ground_pts"]; found = result["found"]

    fig = plt.figure(figsize=(7.5, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    rng = np.random.default_rng(42)
    sub = rng.choice(len(terrain), min(30_000, len(terrain)), replace=False)
    p = terrain[sub]; hv = -p[:, 1]
    cols = plt.cm.terrain((hv - hv.min()) / max(hv.max() - hv.min(), 1e-9))[:, :3]
    ax.scatter(p[:, 0], p[:, 2], p[:, 1], c=cols, s=0.3, alpha=0.30, rasterized=True)

    for cam, g, ok in zip(cameras, gp, found):
        if ok:
            ax.plot([cam[0], g[0]], [cam[2], g[2]], [cam[1], g[1]], color="gold", lw=1.4)
            ax.scatter(g[0], g[2], g[1], c="dodgerblue", s=45, marker="o", zorder=6)
        ax.scatter(cam[0], cam[2], cam[1], c="red", s=55, marker="*", zorder=7)

    g_ok = gp[found]
    if len(g_ok) >= 3:
        c0 = g_ok.mean(0)
        _, _, Vt = np.linalg.svd(g_ok - c0, full_matrices=False)
        n = Vt[-1] * np.sign(Vt[-1, 1])
        sx = np.ptp(g_ok[:, 0]) + 0.2; sz = np.ptp(g_ok[:, 2]) + 0.2
        corners = [[c0[0]+dx, c0[2]+dz, c0[1] - (n[0]*dx + n[2]*dz)/(n[1]+1e-12)]
                   for dx, dz in [(-sx/2,-sz/2),(sx/2,-sz/2),(sx/2,sz/2),(-sx/2,sz/2)]]
        ax.add_collection3d(Poly3DCollection([[[c[0],c[1],c[2]] for c in corners]],
                            alpha=0.25, facecolor="cyan", edgecolor="cyan", lw=0.5))

    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_zlabel("Y  +down (m)")
    ax.invert_zaxis()
    ax.set_title(title or f"{result['name']}\n{result['overall_signed']:+.2f}° "
                 f"{result['direction']}  (R²={result['r2']:.3f})", fontsize=10, fontweight="bold")
    ax.view_init(elev=18, azim=-60)
    ax.legend(handles=[
        Line2D([0],[0], color="gold", lw=2, label="gravity rays"),
        Line2D([0],[0], marker="*", color="red", lw=0, ms=9, label="cameras"),
        Line2D([0],[0], marker="o", color="dodgerblue", lw=0, ms=7, label="ground points"),
        Line2D([0],[0], color="cyan", lw=4, alpha=0.4, label="fitted ground plane"),
    ], fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return out_png


if __name__ == "__main__":
    import sys, json
    for p in sys.argv[1:]:
        r = analyse(p)
        if not r["ok"]:
            print(f"{r['name']}: not enough ground points")
            continue
        print(f"{r['name']}: {r['overall_signed']:+.2f}° {r['direction']} "
              f"(R²={r['r2']:.3f}, {r['n_ground_found']}/{r['n_cams']} ground pts)")
