"""
run_pipeline.py — one pipeline over the whole `gdrive/` dataset.

For every flight folder (default: gdrive/raw/<flight>/ full of `null_..._DONE0 ….png`
frames), and for every image count k = 2 … N, it:

  1. reconstructs a 3-D scene with **VGGT** and with **DAV3** (Depth-Anything-3),
  2. extracts the ground point beneath each camera (Method 1),
  3. recovers gravity ("true down") — the crux of the whole problem — by aligning
     the reconstruction's camera centres to the ground-truth camera poses encoded
     in the frame filenames (Kabsch/Umeyama), with an optional per-image up-vector
     fallback (GeoCalib / IMU) for degenerate flights,
  4. computes the signed slope three ways in the gravity-aligned frame — along-track
     line fit, full 2-D plane fit, and a Horn DEM grid — and
  5. gates the result with a confidence flag (R², camera collinearity/planarity),

writing everything into a `process/`-style tree with all intermediate PNGs, both
GLBs, and the text/JSON outputs.

Research basis (why it's built this way):
  * VGGT/DAV3 return geometry only up to a similarity transform, in camera-0's
    tilted frame — so slope read directly off them can be off by tens of degrees.
    Gravity recovery (step 3) is what makes the number meaningful; the arctan math
    is trivial once "up" is correct.  (G3T / GeoCalib, ECCV'24 / Cornell'26.)
  * Sign (uphill/downhill) falls out of the gravity-aligned elevation change — no
    separate sign estimation.
  * Flight geometry can silently break it: collinear cameras / planar scenes make
    roll (hence gravity) unrecoverable — we detect and flag that (Nesbit &
    Hugenholtz 2019; degeneracy analyses).
  * Slope math: theta = arctan(dz/dd); plane fit theta = arctan(sqrt(a²+b²));
    Horn (1981) 3×3 DEM gradient. Scale cancels in the angle (dz/dd is a ratio),
    so metric scale is reported (from DAV3's metric depth) but not required.

Usage (nothing here runs automatically — you invoke it):

    # validate the machinery on ONE flight at ONE image count (fast):
    python run_pipeline.py --once
    python run_pipeline.py --once --flight Wsagamore_type3_1.9_forward_uphill_0712 --k 5 --backend vggt

    # analysis-only on a single existing GLB (no GPU, no models):
    python run_pipeline.py --once --from-glb

    # just list what would be processed:
    python run_pipeline.py --list

    # the full run: every flight, k = 2 … N, both backends:
    python run_pipeline.py

Flags: --root gdrive  --out output  --backend {vggt,dav3,both}  --from-glb
       --kmin 2  --kmax N  --kstep 1  --max-frames N  --flight NAME  --k K  --list --once
"""
import os
import sys
import glob
import re
import json
import math
import argparse

import numpy as np

SLOPE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SLOPE_DIR)

import method1_slope as m1
import gps_anchor
# run_vggt_all / run_dav3_all (torch + models) are imported lazily, only when a
# reconstruction is actually needed, so --list / --from-glb run without a GPU stack.

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG")
DEFAULT_ROOT = os.path.join(SLOPE_DIR, "gdrive")
DEFAULT_OUT = os.path.join(SLOPE_DIR, "output")
MIN_R2 = 0.90                 # research: R² < ~0.9 on a known site => escalate
DEM_GRID = 80                 # DEM raster resolution (cells per side)


# ── discovery / image ordering ───────────────────────────────────────────────

_IDX = re.compile(r"^null_\d+_(\d+)_")


def _frame_index(path):
    """Capture order from the filename (`null_<ts>_<idx>_…`); fallback to name."""
    m = _IDX.match(os.path.basename(path))
    return (0, int(m.group(1))) if m else (1, os.path.basename(path))


def gather_flight_images(flight_dir, max_frames=None):
    """All frames in a flight folder, in capture order. Non-recursive first; if the
    folder only holds subfolders, recurse. Optionally thin to `max_frames`, always
    keeping the first and last."""
    imgs = []
    for ext in IMG_EXTS:
        imgs += glob.glob(os.path.join(flight_dir, ext))
    if not imgs:
        for ext in IMG_EXTS:
            imgs += glob.glob(os.path.join(flight_dir, "**", ext), recursive=True)
    imgs = sorted(set(imgs), key=_frame_index)
    if max_frames and len(imgs) > max_frames:
        idx = sorted(set(np.linspace(0, len(imgs) - 1, max_frames).round().astype(int).tolist()))
        imgs = [imgs[i] for i in idx]
    return imgs


def find_flights(root):
    """Flight folders under `root`. Prefers `root/raw/<flight>`; if there is no
    `raw/`, treats each immediate subdir of `root` as a flight."""
    raw = os.path.join(root, "raw")
    base = raw if os.path.isdir(raw) else root
    flights = []
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        if os.path.isdir(p) and not d.startswith("."):
            flights.append((d, p))
    return flights


# ── Horn DEM slope (research §3) ──────────────────────────────────────────────

def rasterize_dem(pts_zup, grid=DEM_GRID):
    """Bin gravity-aligned points (Z = up) into a `grid`×`grid` DEM of mean cell
    height. Returns (dem, extent, (cellx, celly), coverage) or None if degenerate."""
    x, y, z = pts_zup[:, 0], pts_zup[:, 1], pts_zup[:, 2]
    xmin, xmax = np.percentile(x, [1, 99])
    ymin, ymax = np.percentile(y, [1, 99])
    if xmax - xmin < 1e-9 or ymax - ymin < 1e-9:
        return None
    keep = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    x, y, z = x[keep], y[keep], z[keep]
    ix = np.clip(((x - xmin) / (xmax - xmin) * (grid - 1)).astype(int), 0, grid - 1)
    iy = np.clip(((y - ymin) / (ymax - ymin) * (grid - 1)).astype(int), 0, grid - 1)
    sums = np.zeros((grid, grid)); cnt = np.zeros((grid, grid))
    np.add.at(sums, (iy, ix), z)
    np.add.at(cnt, (iy, ix), 1)
    dem = np.full((grid, grid), np.nan)
    dem[cnt > 0] = sums[cnt > 0] / cnt[cnt > 0]
    cellx = (xmax - xmin) / (grid - 1)
    celly = (ymax - ymin) / (grid - 1)
    return dem, (xmin, xmax, ymin, ymax), (cellx, celly), float((cnt > 0).mean())


def horn_slope_field(dem, cellx, celly):
    """Per-cell slope (deg) via Horn's 3×3 finite difference (GDAL/ArcGIS default).
    NaN where any of the 8 neighbours is missing."""
    z1, z2, z3 = dem[:-2, :-2], dem[:-2, 1:-1], dem[:-2, 2:]
    z4, z6 = dem[1:-1, :-2], dem[1:-1, 2:]
    z7, z8, z9 = dem[2:, :-2], dem[2:, 1:-1], dem[2:, 2:]
    dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * cellx)
    dzdy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8 * celly)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def horn_dem_slope(pts_zup, grid=DEM_GRID):
    """Mean/median Horn slope (deg) over a DEM of the gravity-aligned cloud, plus
    the DEM/slope fields for plotting. `pts_zup` has Z = up."""
    ras = rasterize_dem(pts_zup, grid)
    if ras is None:
        return None
    dem, extent, (cellx, celly), coverage = ras
    field = horn_slope_field(dem, cellx, celly)
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return None
    return {
        "mean_slope_deg": round(float(np.mean(finite)), 2),
        "median_slope_deg": round(float(np.median(finite)), 2),
        "coverage": round(coverage, 3),
        "cell_m": round(float((cellx + celly) / 2), 4),
        "_dem": dem, "_field": field, "_extent": extent,
    }


def plot_dem(horn, out_png, title=""):
    """DEM height (left) + Horn slope field (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dem, field, ext = horn["_dem"], horn["_field"], horn["_extent"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    im1 = ax1.imshow(dem, origin="lower", extent=ext, cmap="terrain", aspect="auto")
    ax1.set_title("DEM — elevation (m, gravity-aligned)", fontsize=10, fontweight="bold")
    fig.colorbar(im1, ax=ax1, shrink=0.8)
    im2 = ax2.imshow(field, origin="lower", extent=ext, cmap="magma", aspect="auto",
                     vmin=0, vmax=max(5.0, float(np.nanpercentile(field, 95))))
    ax2.set_title(f"Horn slope (deg)  mean {horn['mean_slope_deg']}°  cov {horn['coverage']}",
                  fontsize=10, fontweight="bold")
    fig.colorbar(im2, ax=ax2, shrink=0.8)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    for ax in (ax1, ax2):
        ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    return out_png


# ── one reconstruction → analysis ────────────────────────────────────────────

def analyse_reconstruction(glb_path, ordered_images, out_prefix, up_hint=None,
                           backend="vggt"):
    """Load a GLB, gravity-align it to the frame poses, and compute slope three ways
    (along-track line, 2-D plane, Horn DEM) with a confidence flag. Writes profile /
    segments / DEM PNGs. Returns a result dict (JSON-safe)."""
    terrain, cameras = m1.load_glb(glb_path)
    raw = m1.analyse_arrays(terrain, cameras, backend)     # tilted VGGT/DAV3 frame
    if not raw["ok"]:
        return {"ok": False, "reason": "not enough ground points beneath cameras"}

    align = gps_anchor.gravity_align(cameras, ordered_images, up_hint=up_hint)
    aligned = bool(align.get("ok"))
    if aligned:
        R = align["R"]
        terrain_a, cameras_a = terrain @ R.T, cameras @ R.T
        r = m1.analyse_arrays(terrain_a, cameras_a, backend)
    else:
        terrain_a, r = terrain, raw          # raw, tilted — flagged low-confidence

    # Horn DEM on the gravity-aligned ground points (convert +Y-down → Z-up)
    gp = r["ground_pts"][r["found"]]
    horn = None
    if len(gp) >= 12:
        pts_zup = np.column_stack([gp[:, 0], gp[:, 2], -gp[:, 1]])   # (east, north, up)
        horn = horn_dem_slope(pts_zup)

    # confidence (research §5): gravity present? + R² + degeneracy
    reasons = []
    level = "high"

    def demote(to, why):
        nonlocal level
        reasons.append(why)
        if {"high": 2, "medium": 1, "low": 0}[to] < {"high": 2, "medium": 1, "low": 0}[level]:
            level = to

    if not aligned:
        demote("low", f"no gravity reference ({align.get('note')}) — tilted-frame slope")
    if np.isfinite(r["r2"]) and r["r2"] < MIN_R2:
        demote("low" if r["r2"] < 0.5 else "medium", f"ground fit R²={r['r2']:.2f} < {MIN_R2}")
    col = align.get("collinearity")
    if not aligned and col is not None and col < 0.05:
        demote("low", f"near-collinear cameras (axis ratio {col:.3f})")
    if aligned and align.get("method") == "up-hint":
        demote("medium", "gravity from external up-vector, not camera poses")
    if not reasons:
        reasons.append("gravity-aligned, strong fit, non-degenerate geometry")

    # plots
    plot_signed = final = r["overall_signed"]
    note = align.get("note", "")
    m1.plot_profile(r, out_prefix + "_profile.png", final_signed=plot_signed if aligned else None,
                    sign_note=note)
    m1.plot_segments(r, out_prefix + "_segments.png", final_signed=plot_signed if aligned else None,
                     sign_note=note)
    if horn is not None:
        plot_dem(horn, out_prefix + "_dem.png", title=os.path.basename(out_prefix))

    # metric scale (DAV3 is already metric; VGGT gets scale from the pose Umeyama fit)
    scale = align.get("scale")
    run_m = rise_m = None
    if aligned and scale:
        s_found = r["s"][r["found"]]
        run_m = round(float(np.ptp(s_found)) * scale, 2)
        rise_m = round(run_m * math.tan(math.radians(final)), 2)

    return {
        "ok": True,
        "backend": backend,
        "n_cameras": r["n_cams"],
        "n_ground_found": r["n_ground_found"],
        # ---- signed slope, three independent estimators (deg) ----
        "along_track_deg": round(final, 2),                       # Method 1 line fit
        "along_track_r2": round(r["r2"], 4),
        "plane_slope_deg": round(r["plane_slope_deg"], 2) if np.isfinite(r["plane_slope_deg"]) else None,
        "aspect_deg": round(r["aspect_deg"], 1) if np.isfinite(r["aspect_deg"]) else None,
        "horn_dem_mean_deg": horn["mean_slope_deg"] if horn else None,
        "horn_dem_median_deg": horn["median_slope_deg"] if horn else None,
        "horn_coverage": horn["coverage"] if horn else None,
        "direction": "uphill" if final >= 0 else "downhill",
        # ---- raw (pre-alignment) value, for transparency ----
        "raw_vggt_frame_deg": round(raw["overall_signed"], 2),
        # ---- gravity ----
        "gravity_aligned": aligned,
        "gravity_method": align.get("method"),
        "gravity_source": align.get("source"),
        "align_resid_m": align.get("resid_m"),
        # ---- geometry degeneracy (research §3/§5) ----
        "collinearity": align.get("collinearity"),
        "planarity": align.get("planarity"),
        # ---- metric (from DAV3 depth / pose scale) ----
        "scale_m_per_unit": scale,
        "run_m": run_m,
        "rise_m": rise_m,
        # ---- confidence gate ----
        "confidence": level,
        "confidence_reasons": reasons,
        "segments": [
            {"label": s["label"],
             "signed_slope": round(s["signed_slope"], 2) if np.isfinite(s["signed_slope"]) else None,
             "horiz": round(s["horiz"], 3), "dY": round(s["dY"], 3)}
            for s in r["segments"]
        ],
    }


def _write_text_report(txt_path, flight, k, backend, rec):
    lines = [f"flight   : {flight}", f"images   : {k}", f"backend  : {backend}", ""]
    if not rec.get("ok"):
        lines.append(f"FAILED: {rec.get('reason')}")
    else:
        lines += [
            f"along-track slope : {rec['along_track_deg']:+.2f} deg {rec['direction']}  (R²={rec['along_track_r2']})",
            f"plane-fit grade   : {rec['plane_slope_deg']} deg (2-D max, aspect {rec['aspect_deg']})",
            f"Horn DEM grade    : {rec['horn_dem_mean_deg']} deg mean / {rec['horn_dem_median_deg']} median (cov {rec['horn_coverage']})",
            f"raw VGGT-frame    : {rec['raw_vggt_frame_deg']:+.2f} deg (pre-gravity, tilted)",
            f"gravity           : aligned={rec['gravity_aligned']} via {rec['gravity_method']} ({rec['gravity_source']}), resid {rec['align_resid_m']} m",
            f"geometry          : collinearity={rec['collinearity']} planarity={rec['planarity']}",
            f"metric            : scale={rec['scale_m_per_unit']} m/unit, run={rec['run_m']} m, rise={rec['rise_m']} m",
            f"CONFIDENCE        : {rec['confidence'].upper()} — {'; '.join(rec['confidence_reasons'])}",
        ]
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── one (flight, k) across the requested backends ─────────────────────────────

def run_one(models, flight, images, k, out_root, backends, from_glb, up_hint=None):
    """Reconstruct (or reuse) + analyse the first k images for each backend. Writes
    into out_root/process/<flight>/k<NN>/. Returns {backend: record}."""
    subset = images[:k]
    kdir = os.path.join(out_root, "process", flight, f"k{k:02d}")
    os.makedirs(kdir, exist_ok=True)
    with open(os.path.join(kdir, "input.txt"), "w") as f:
        f.write("\n".join(os.path.basename(p) for p in subset) + "\n")

    out = {}
    for backend in backends:
        glb_path = os.path.join(kdir, f"{backend}.glb")
        if not from_glb and not os.path.exists(glb_path):
            if backend == "vggt":
                import run_vggt_all as rv
                rv.run_glb(models["vggt"], subset, glb_path)
            elif backend == "dav3":
                import run_dav3_all as rd
                rd.run_glb(models["dav3"], subset, glb_path)
        if not os.path.exists(glb_path):
            out[backend] = {"ok": False, "reason": "no GLB (reconstruction skipped/failed)"}
            _write_text_report(os.path.join(kdir, f"output_{backend}.txt"), flight, k, backend, out[backend])
            continue
        rec = analyse_reconstruction(glb_path, subset, os.path.join(kdir, backend),
                                     up_hint=up_hint, backend=backend)
        rec.update({"flight": flight, "k": k, "glb": os.path.relpath(glb_path, out_root)})
        with open(os.path.join(kdir, f"{backend}_slope.json"), "w") as f:
            json.dump(rec, f, indent=2)
        _write_text_report(os.path.join(kdir, f"output_{backend}.txt"), flight, k, backend, rec)
        out[backend] = rec
        tag = (f"{rec['along_track_deg']:+.2f}° {rec['direction']} [{rec['confidence']}]"
               if rec.get("ok") else f"FAIL: {rec.get('reason')}")
        print(f"    [{flight} k={k:02d} {backend}] {tag}")
    return out


# ── model loading ─────────────────────────────────────────────────────────────

def load_models(backends):
    """Load only the requested reconstruction models once (GPU)."""
    models = {}
    if "vggt" in backends:
        import run_vggt_all as rv
        import torch
        print(f"Loading VGGT ({rv.device}, {rv.dtype})...")
        m = rv.VGGT()
        m.load_state_dict(torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            map_location=rv.device))
        models["vggt"] = m.eval().to(rv.device)
    if "dav3" in backends:
        import run_dav3_all as rd
        print(f"Loading DAV3 ({rd.DA3_MODEL_ID})...")
        models["dav3"] = rd.DepthAnything3.from_pretrained(rd.DA3_MODEL_ID).to(rd.device).eval()
    print("Models ready.\n")
    return models


# ── full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(root, out_root, backends, from_glb, kmin, kmax, kstep, max_frames,
                 flight_filter=None):
    flights = find_flights(root)
    if flight_filter:
        flights = [(n, p) for n, p in flights if flight_filter in n]
    if not flights:
        print(f"No flights found under {root}")
        return

    models = {} if from_glb else load_models(backends)
    summary = []
    for name, path in flights:
        images = gather_flight_images(path, max_frames=max_frames)
        N = len(images)
        if N < 2:
            print(f"[{name}] {N} image(s) — need >=2, skipping")
            continue
        top = min(kmax, N) if kmax else N
        ks = list(range(max(2, kmin), top + 1, kstep))
        print(f"\n[{name}] {N} images → k = {ks}")
        per_k = {}
        for k in ks:
            per_k[k] = run_one(models, name, images, k, out_root, backends, from_glb)
        # per-flight convergence table: slope(k) for each backend
        conv = {b: [{"k": k, "along_track_deg": per_k[k][b].get("along_track_deg"),
                     "confidence": per_k[k][b].get("confidence")}
                    for k in ks if per_k[k].get(b, {}).get("ok")]
                for b in backends}
        rec = {"flight": name, "n_images": N, "ks": ks, "convergence": conv}
        fdir = os.path.join(out_root, "process", name)
        os.makedirs(fdir, exist_ok=True)
        with open(os.path.join(fdir, f"{name}_summary.json"), "w") as f:
            json.dump(rec, f, indent=2)
        summary.append(rec)

    with open(os.path.join(out_root, "pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Per-flight outputs in {os.path.join(out_root, 'process')}")
    print(f"Overall summary → {os.path.join(out_root, 'pipeline_summary.json')}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="dataset root (expects <root>/raw/<flight>/)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output root (a process/ tree is written here)")
    ap.add_argument("--backend", choices=["vggt", "dav3", "both"], default="both")
    ap.add_argument("--from-glb", action="store_true",
                    help="skip reconstruction; analyse GLBs already in the output tree (no GPU)")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=None, help="cap the image count (default = all N)")
    ap.add_argument("--kstep", type=int, default=1, help="step for k (2,3,4… by default)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="thin each flight to at most N frames before k-sweeping (GPU/time guard)")
    ap.add_argument("--flight", default=None, help="only flights whose name contains this")
    ap.add_argument("--k", type=int, default=None, help="single image count for --once")
    ap.add_argument("--list", action="store_true", help="list discovered flights + image counts and exit")
    ap.add_argument("--once", action="store_true",
                    help="validation: run ONE flight at ONE k (first match, k=--k or 2) and stop")
    args = ap.parse_args()

    backends = ["vggt", "dav3"] if args.backend == "both" else [args.backend]

    if args.list:
        flights = find_flights(args.root)
        if args.flight:
            flights = [(n, p) for n, p in flights if args.flight in n]
        print(f"{len(flights)} flight(s) under {args.root}:")
        for n, p in flights:
            imgs = gather_flight_images(p, max_frames=args.max_frames)
            print(f"  {n:55s} {len(imgs):>4d} images")
        return

    if args.once:
        flights = find_flights(args.root)
        if args.flight:
            flights = [(n, p) for n, p in flights if args.flight in n]
        if not flights:
            print(f"No flight found under {args.root}" + (f" matching '{args.flight}'" if args.flight else ""))
            return
        name, path = flights[0]
        images = gather_flight_images(path, max_frames=args.max_frames)
        k = args.k or max(2, args.kmin)
        if len(images) < k:
            print(f"[{name}] only {len(images)} images, need >= {k}")
            return
        models = {} if args.from_glb else load_models(backends)
        print(f"[once] {name}  k={k}  backends={backends}  from_glb={args.from_glb}")
        run_one(models, name, images, k, args.out, backends, args.from_glb)
        print(f"\nOutput → {os.path.join(args.out, 'process', name, f'k{k:02d}')}")
        return

    run_pipeline(args.root, args.out, backends, args.from_glb,
                 args.kmin, args.kmax, args.kstep, args.max_frames, args.flight)


if __name__ == "__main__":
    main()
