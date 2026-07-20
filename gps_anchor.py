"""
Gravity-true vertical anchor for Method 1 (historically "GPS anchor").

Method 1 reads slope from the VGGT point cloud, but its up/down SIGN depends on
VGGT's vertical axis matching true gravity — which VGGT does not guarantee (it has
no gravity sensor, and reconstructions carry an orientation/handedness ambiguity).

A per-frame (height, position) track gives a gravity-true reference: height vs.
horizontal distance along the flight fixes the uphill/downhill sign unambiguously
and gives a metric sanity-check slope. That track comes from one of two sources,
tried in order (see read_gps):

  * "sim ground-truth pose (filename)" — rendered/sim frames encode exact pose in
    the filename: null_<ts>_<idx>_<height>_<X/lon>_<Y/lat>_..._DONE0 …  (no sensor
    noise; this is ground truth, not GPS). Used by sagamore_0708 / N75E_0712.
  * "EXIF GPS" — real camera photos (e.g. Robinson: iPhone/DJI) carry lat/lon/alt
    in EXIF.

Pure-Python for the filename path; Pillow is imported lazily only for EXIF. No GPU.
"""
import os
import re
import glob
import numpy as np

IMG_EXTS = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")

# Filename-encoded GPS (drone-render exports):
#   null_<ts_ms>_<idx>_<alt_m>_<lon>_<lat>_..._DONE0 <date> <time>.png
# Field order after the leading token is: timestamp, index, altitude, lon, lat.
# This carries GPS with no EXIF and no image decode — parse it first (cheap, no
# Pillow needed), and only fall back to EXIF for real camera JPGs (Robinson).
_NAME_GPS = re.compile(
    r"^null_(\d+)_(\d+)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_"
)


def _ratio(x):
    """Pillow may return IFDRational / tuple / float — coerce to float."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(x[0]) / float(x[1])


def read_gps_from_name(image_path):
    """Return (lat_deg, lon_deg, alt_m) parsed from the filename, or None.

    Sanity-checks the values look like real WGS84 coordinates so we don't
    mis-parse an unrelated filename that happens to have underscores."""
    m = _NAME_GPS.match(os.path.basename(image_path))
    if not m:
        return None
    _, _, alt, lon, lat = (float(g) for g in m.groups())
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon, alt


def read_gps(image_path):
    """Return (lat_deg, lon_deg, alt_m) or None. Tries the filename encoding
    first (no image decode, no Pillow), then EXIF for real camera photos."""
    named = read_gps_from_name(image_path)
    if named is not None:
        return named
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS
        exif = Image.open(image_path).getexif()
    except Exception:
        return None
    if not exif:
        return None
    gps_ifd = exif.get_ifd(0x8825)          # GPSInfo IFD
    if not gps_ifd:
        return None
    g = {GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
    if "GPSLatitude" not in g or "GPSLongitude" not in g:
        return None

    def dms(vals):
        d, m, s = (_ratio(v) for v in vals)
        return d + m / 60.0 + s / 3600.0

    lat = dms(g["GPSLatitude"])
    if g.get("GPSLatitudeRef", "N") in ("S", b"S"):
        lat = -lat
    lon = dms(g["GPSLongitude"])
    if g.get("GPSLongitudeRef", "E") in ("W", b"W"):
        lon = -lon
    alt = _ratio(g["GPSAltitude"]) if "GPSAltitude" in g else float("nan")
    if g.get("GPSAltitudeRef", 0) in (1, b"\x01"):   # below sea level
        alt = -alt
    return lat, lon, alt


def _enu(lats, lons, lat0):
    """Equirectangular lat/lon -> local east/north metres (WGS84 scale at lat0)."""
    phi = np.radians(lat0)
    m_lat = 111132.92 - 559.82 * np.cos(2 * phi) + 1.175 * np.cos(4 * phi)
    m_lon = 111412.84 * np.cos(phi) - 93.5 * np.cos(3 * phi)
    east  = (lons - lons.mean()) * m_lon
    north = (lats - lats.mean()) * m_lat
    return east, north


def gps_track_slope(image_paths):
    """Signed slope of the GPS flight track (altitude vs along-track distance).

    Sign convention (matches Method 1 after this anchor):
        +  = drone/terrain rises as the flight progresses (uphill)
        -  = falls (downhill)
    Returns a dict, or {'ok': False} if fewer than 2 frames carry GPS+altitude.
    """
    # track how each fix was obtained so the anchor self-labels (sim pose vs GPS)
    raw = []
    for p in image_paths:
        named = read_gps_from_name(p)
        if named is not None and np.isfinite(named[2]):
            raw.append((p, named, "name"))
            continue
        g = read_gps(p)
        if g is not None and np.isfinite(g[2]):
            raw.append((p, g, "exif"))
    if len(raw) < 2:
        return {"ok": False, "n_with_gps": len(raw)}
    fixes = [(p, g) for p, g, _ in raw]
    src = "sim ground-truth pose" if sum(s == "name" for *_, s in raw) >= len(raw) / 2 else "EXIF GPS"

    lats = np.array([g[0] for _, g in fixes])
    lons = np.array([g[1] for _, g in fixes])
    alts = np.array([g[2] for _, g in fixes])

    east, north = _enu(lats, lons, float(lats.mean()))
    # along-track axis = principal horizontal direction, oriented first -> last frame
    H = np.column_stack([east, north])
    _, _, Vt = np.linalg.svd(H - H.mean(0), full_matrices=False)
    fd = Vt[0]
    if (H[-1] - H[0]) @ fd < 0:
        fd = -fd
    s = (H - H.mean(0)) @ fd

    grad, intat = np.polyfit(s, alts, 1)        # d(altitude)/d(horizontal)
    yhat = grad * s + intat
    ss_res = np.sum((alts - yhat) ** 2)
    ss_tot = np.sum((alts - alts.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    signed = np.degrees(np.arctan(grad))        # alt is true 'up' -> +grad = uphill
    return {
        "ok": True,
        "source": src,
        "n_with_gps": len(fixes),
        "signed_deg": float(signed),
        "direction": "uphill" if signed >= 0 else "downhill",
        "alt_gain_m": float(alts[-1] - alts[0]),
        "alt_total_change_m": float(alts.max() - alts.min()),
        "horiz_span_m": float(np.ptp(s)),
        "r2": float(r2),
        "alts": alts.tolist(),
        "s": s.tolist(),
        "grad": float(grad),
        "intercept": float(intat),
    }


def frame_positions(image_paths):
    """World positions for an ORDERED list of frames (same order as the GLB
    cameras). Returns (idx, P, source) where:

      * idx    — indices into image_paths of frames that carried a usable fix
      * P      — len(idx) x 3 array of METHOD-1-frame positions for those frames:
                 columns are [east, -up, north] metres (so +Y = down, matching
                 method1_slope's convention), mean-centred is not required.
      * source — "sim ground-truth pose" or "EXIF GPS"

    Returns (None, None, None) if fewer than 3 frames carry a fix (Kabsch needs
    at least 3 to fix a rotation)."""
    fixes, srcs = [], []
    for p in image_paths:
        named = read_gps_from_name(p)
        if named is not None and np.isfinite(named[2]):
            fixes.append(named); srcs.append("name"); continue
        g = read_gps(p)
        if g is not None and np.isfinite(g[2]):
            fixes.append(g); srcs.append("exif")
        else:
            fixes.append(None); srcs.append(None)

    idx = [i for i, f in enumerate(fixes) if f is not None]
    if len(idx) < 3:
        return None, None, None

    lats = np.array([fixes[i][0] for i in idx])
    lons = np.array([fixes[i][1] for i in idx])
    alts = np.array([fixes[i][2] for i in idx])
    east, north = _enu(lats, lons, float(lats.mean()))
    P = np.column_stack([east, -alts, north])      # +Y = down -> up is -alt
    src = ("sim ground-truth pose"
           if sum(srcs[i] == "name" for i in idx) >= len(idx) / 2 else "EXIF GPS")
    return np.array(idx), P, src


def _umeyama(A, B):
    """Least-squares similarity (s, R, t) mapping A -> B for matched Nx3 sets:
    minimises || s*R @ A_i + t - B_i ||.  Returns (R, s, t, sing) where `sing`
    are the singular values used (for a degeneracy check)."""
    muA, muB = A.mean(0), B.mean(0)
    A0, B0 = A - muA, B - muB
    H = (A0.T @ B0) / len(A)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))          # reflection guard
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    varA = (A0 ** 2).sum() / len(A)
    s = float((S * np.array([1.0, 1.0, d])).sum() / varA) if varA > 1e-12 else 1.0
    t = muB - s * R @ muA
    return R, s, t


def _rotation_from_up(up_vec):
    """Shortest-arc rotation mapping a GLB-frame up-vector onto method-1-frame
    world up = (0, -1, 0)  (+Y = down).

    Azimuth about the vertical is left arbitrary — it does NOT affect slope
    magnitude or the along-track sign, because the flight direction is re-derived
    from the (rotated) cameras. A single correct up-vector is therefore enough to
    put slope on a true-vertical footing, which is why a per-image gravity estimate
    (GeoCalib) or the drone IMU can rescue flights whose camera geometry is too
    degenerate for pose-based alignment (e.g. a straight-line / nadir pass)."""
    u = np.asarray(up_vec, dtype=np.float64)
    n = np.linalg.norm(u)
    if n < 1e-9:
        return np.eye(3)
    u = u / n
    t = np.array([0.0, -1.0, 0.0])
    v = np.cross(u, t)
    s = np.linalg.norm(v)
    c = float(np.dot(u, t))
    if s < 1e-9:                                    # already up, or exactly flipped
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def gravity_align(cameras_glb, image_paths, min_axis_ratio=0.05, max_resid_rel=0.5,
                  up_hint=None):
    """Recover the rotation that carries the VGGT/GLB frame into a gravity-true
    frame — the pipeline's central problem (VGGT returns geometry in camera-0's
    tilted frame up to a similarity, so slope read off it carries the camera tilt).

    Two gravity sources, tried in order:

      1. **Pose-based Umeyama** (primary) — align the GLB camera centres to the
         known ground-truth camera world positions (from the filenames). Needs a
         non-degenerate (non-collinear) camera constellation.
      2. **External up-vector** (`up_hint`, fallback) — a per-image up-vector (from
         GeoCalib) or the drone IMU/attitude, in the GLB frame. Used when the pose
         geometry is degenerate. One up-vector is enough to fix slope. Pass a single
         (3,) vector or an (N,3) array (averaged). This is the research-recommended
         rescue for straight-line / nadir flights.

    Args
      cameras_glb  Mx3 GLB camera centres, in GLB-camera order.
      image_paths  the ORDERED image list fed to VGGT (image_paths[i] <-> camera i).
      up_hint      optional GLB-frame up-vector(s) from GeoCalib/IMU (see above).

    Returns a dict always carrying the flight's degeneracy metrics
    (`collinearity` = 2nd/1st singular value of the camera spread, `planarity` =
    3rd/1st), plus on success {ok, R, scale, source, method, ...}. `ok` is False
    (with a `note`) when no gravity source resolves it — caller falls back to the
    un-aligned estimate. `scale` is metres per VGGT unit (None for the up-hint
    path, which does not observe scale)."""
    idx, P_world, src = frame_positions(image_paths)
    have_poses = idx is not None

    # Degeneracy metrics from the GLB camera constellation — reported ALWAYS so the
    # caller can gate confidence on flight geometry (Stage 5). collinearity < ~0.05
    # => cameras on a line (roll/gravity unrecoverable); planarity < ~0.02 => nearly
    # coplanar (low vertical parallax).
    A_all = np.asarray(cameras_glb, dtype=np.float64)
    degen = {}
    if len(A_all) >= 3:
        sv_all = np.linalg.svd(A_all - A_all.mean(0), compute_uv=False)
        degen = {"collinearity": round(float(sv_all[1] / max(sv_all[0], 1e-12)), 4),
                 "planarity": round(float(sv_all[2] / max(sv_all[0], 1e-12)), 4)}

    def _ret(d):
        d.update(degen)
        return d

    # ---- primary: pose-based Umeyama (needs non-degenerate cameras) ----
    pose_fail = None
    if not have_poses:
        pose_fail = "no pose track (need >=3 frames with a fix)"
    elif len(idx) > len(cameras_glb):
        pose_fail = "more pose fixes than GLB cameras (order mismatch)"
    else:
        A, B = A_all[idx], P_world
        sv = np.linalg.svd(A - A.mean(0), compute_uv=False)
        if sv[0] < 1e-9 or sv[1] / sv[0] < min_axis_ratio:
            pose_fail = (f"degenerate camera geometry (collinear, axis ratio "
                         f"{sv[1] / max(sv[0], 1e-12):.3f})")
        else:
            R, s, t = _umeyama(A, B)
            resid = A @ (s * R).T + t - B
            resid_m = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
            span = float(np.ptp(B[:, [0, 2]]))                 # horizontal baseline
            resid_rel = resid_m / span if span > 1e-9 else float("inf")
            if resid_rel > max_resid_rel:
                pose_fail = f"poor pose alignment (residual {resid_rel:.2f} of baseline)"
            else:
                return _ret({"ok": True, "R": R, "scale": round(float(s), 6),
                             "source": src, "method": "pose-umeyama",
                             "n_frames": int(len(idx)), "resid_m": round(resid_m, 3),
                             "resid_rel": round(resid_rel, 3),
                             "note": "gravity-aligned via known camera poses"})

    # ---- fallback: external up-vector (GeoCalib / IMU) rescues degenerate flights ----
    if up_hint is not None:
        up = np.asarray(up_hint, dtype=np.float64)
        up = up.mean(0) if up.ndim == 2 else up
        return _ret({"ok": True, "R": _rotation_from_up(up), "scale": None,
                     "source": "external up-vector (GeoCalib/IMU)", "method": "up-hint",
                     "n_frames": int(len(idx)) if have_poses else 0,
                     "note": f"gravity from up-vector hint (pose path unusable: {pose_fail})"})

    return _ret({"ok": False, "source": src if have_poses else None,
                 "n_frames": int(len(idx)) if have_poses else 0, "note": pose_fail})


def plot_track(gps, out_png, name=""):
    """GPS altitude vs along-track distance — the gravity-true profile + fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = np.array(gps["s"]); alt = np.array(gps["alts"])
    order = np.argsort(s); s = s[order]; alt = alt[order]
    xs = np.linspace(s.min(), s.max(), 50)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.scatter(s, alt, c="#c1121f", s=60, zorder=5, label="GPS fixes")
    ax.plot(xs, gps["grad"] * xs + gps["intercept"], color="#264653", lw=2.2, ls="--",
            label=f"fit: {gps['signed_deg']:+.2f}° {gps['direction']}  (R²={gps['r2']:.3f})")
    ax.set_xlabel("Along-track distance (m)  — flight direction →")
    ax.set_ylabel("GPS altitude (m, true gravity)")
    ax.set_title(f"{name}   |   GPS track slope (gravity-true)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return out_png


def gather_images(folder):
    """All images under `folder`, recursing into subfolders (marker1/, uphill/,
    …) so the GPS track is built from every frame, not just top-level ones."""
    imgs = []
    for ext in IMG_EXTS:
        imgs += glob.glob(os.path.join(folder, ext))
        imgs += glob.glob(os.path.join(folder, "**", ext), recursive=True)
    return sorted(set(imgs))


if __name__ == "__main__":
    import sys, json
    for folder in sys.argv[1:]:
        imgs = gather_images(folder)
        res = gps_track_slope(imgs)
        name = os.path.basename(os.path.normpath(folder))
        print(f"\n=== {name}  ({len(imgs)} images) ===")
        if not res["ok"]:
            print(f"  no usable GPS ({res['n_with_gps']} frames had GPS+alt)")
            continue
        print(f"  GPS track slope: {res['signed_deg']:+.2f}° {res['direction']}  "
              f"(R²={res['r2']:.3f}, {res['n_with_gps']} frames)")
        print(f"  altitude {res['alts'][0]:.1f} -> {res['alts'][-1]:.1f} m "
              f"(Δ {res['alt_gain_m']:+.2f} m over {res['horiz_span_m']:.1f} m horizontal)")
