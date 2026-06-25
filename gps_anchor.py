"""
GPS sign-anchor for Method 1.

Method 1 reads slope from the VGGT point cloud, but its up/down SIGN depends on
VGGT's vertical axis matching true gravity — which VGGT does not guarantee (it has
no gravity sensor, and reconstructions carry an orientation/handedness ambiguity).

When the source photos carry GPS (lat, lon, altitude), the drone's own track gives
a gravity-true reference: altitude vs. horizontal distance along the flight. That
fixes the uphill/downhill sign unambiguously and gives a metric sanity-check slope.

This reads EXIF with Pillow only (cross-platform: works on the Mac and the A100),
no GPU, no extra dependencies.
"""
import os
import glob
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

IMG_EXTS = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")


def _ratio(x):
    """Pillow may return IFDRational / tuple / float — coerce to float."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(x[0]) / float(x[1])


def read_gps(image_path):
    """Return (lat_deg, lon_deg, alt_m) or None if the image has no GPS EXIF."""
    try:
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
    fixes = [(p, read_gps(p)) for p in image_paths]
    fixes = [(p, g) for p, g in fixes if g is not None and np.isfinite(g[2])]
    if len(fixes) < 2:
        return {"ok": False, "n_with_gps": len(fixes)}

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
    imgs = []
    for ext in IMG_EXTS:
        imgs += glob.glob(os.path.join(folder, ext))
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
