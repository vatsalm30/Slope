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
