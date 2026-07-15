"""
Robinson datasets — reconstruct with VGGT, then apply Method 1 (ground-distance).

RUN THIS ON THE A100 MACHINE (the VGGT reconstruction needs the GPU).
Everything downstream (Method 1, plots) is light and also runs here.

Usage
-----
    # one dataset folder of images (uses ALL images, sorted by filename = order 1,2,3,…):
    python run_robinson.py /path/to/robinson/dataset_A

    # or point at a parent folder that holds several dataset subfolders:
    python run_robinson.py /path/to/robinson_unzipped --multi

    # marker-style layout (subfolders marker1/, marker2/, … each with frames):
    python run_robinson.py /path/to/dataset --markers

Outputs (per dataset)
---------------------
    output_glbs/vggt/robinson_<name>.glb
    results/robinson_<name>_method1_segments.png
    results/robinson_<name>_method1.json   (signed slope + per-segment table)

Notes
-----
* Uses ALL images for the estimate, as requested (1, 2, 3, …).
* Slope sign:  +  = terrain goes UPHILL as the flight progresses
              -  = terrain goes DOWNHILL as the flight progresses
* If a dataset's reconstruction comes back nearly flat with low R² (as the
  fixed-altitude / level flights did), that means the flight geometry did not
  give VGGT enough vertical parallax — not that the terrain is flat. Prefer
  terrain-following / oblique flights for Method 1.
"""
import os, sys, glob, json, argparse
import numpy as np

SLOPE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SLOPE_DIR)

import math
import method1_slope as m1
import gps_anchor
# run_vggt_all (torch + vggt) is imported lazily, only when reconstruction is
# actually needed — so --from-glb / analysis-only runs on machines without a GPU stack.

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG")

# GPS altitude only reflects terrain on terrain-following / oblique flights; on a
# fixed-altitude hold it is flat and its "slope" is meaningless. Trust the GPS
# track for sign/magnitude only when its along-track fit clears this R².
GPS_MIN_R2 = 0.3


def _subsample(paths, max_frames):
    """Evenly thin `paths` (already in order) down to at most max_frames, always
    keeping the first and last. VGGT ingests every frame at once, so hundreds of
    frames would OOM the GPU — 20-40 evenly-spaced views span the track just as
    well for a global slope fit."""
    if not max_frames or len(paths) <= max_frames:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [paths[i] for i in idx]


def gather_images(folder, markers=False, max_frames=None):
    """Return image paths in capture order (1, 2, 3, …)."""
    if markers:
        import run_vggt_all as rv
        subs = sorted(
            (d for d in os.listdir(folder)
             if os.path.isdir(os.path.join(folder, d)) and not d.startswith(".")),
            key=lambda x: int("".join(c for c in x if c.isdigit()) or 0),
        )
        imgs = [rv.pick_best_image(os.path.join(folder, s)) for s in subs]
        return [i for i in imgs if i]
    imgs = []
    for ext in IMG_EXTS:
        imgs += glob.glob(os.path.join(folder, ext))
    return _subsample(sorted(set(imgs)), max_frames)


def _images_folder_for(name):
    """Best guess at the source images for a robinson_<name>.glb (for GPS)."""
    cand = os.path.join(SLOPE_DIR, "robinson copy", name)
    return cand if os.path.isdir(cand) else None


def _resolve_sign(r, gps, label):
    """Decide the final signed slope. Sign comes from GPS if available, else the
    folder label ('uphill'/'downhill'). Magnitude comes from the GPS track when
    Method 1's ground-fit is weak (R²<0.5), otherwise from Method 1 — because
    VGGT's vertical axis is not gravity-locked, so its raw sign is not trusted."""
    label_sign = (+1 if "up" in label.lower()
                  else -1 if "down" in label.lower() else None)
    m1_mag = abs(r["overall_signed"])

    # GPS is only a valid gravity reference when the flight actually gained/lost
    # altitude with the terrain. Fixed-altitude holds (all sagamore flights) give
    # a flat, meaningless GPS "slope" — gate it out via the along-track R².
    gps_usable = bool(gps and gps.get("ok")
                      and np.isfinite(gps.get("r2", np.nan))
                      and gps["r2"] >= GPS_MIN_R2)

    if gps_usable:
        src = gps.get("source", "GPS")          # "sim ground-truth pose" or "EXIF GPS"
        sign = 1.0 if gps["signed_deg"] >= 0 else -1.0
        if r["r2"] < 0.5:
            return gps["signed_deg"], f"{src} track", src
        return math.copysign(m1_mag, sign), f"Method 1 magnitude, {src} sign", src
    if label_sign is not None:
        return label_sign * m1_mag, "Method 1 magnitude, folder-label sign", "folder label"
    return r["overall_signed"], "Method 1 (sign unverified)", "VGGT (unverified)"


def analyze_glb(glb_path, name, n_images=None, images_folder=None, prefix="robinson"):
    """Method 1 + GPS sign-anchor + plots + JSON for one GLB (no GPU)."""
    res_dir = os.path.join(SLOPE_DIR, "results")
    os.makedirs(res_dir, exist_ok=True)

    r = m1.analyse(glb_path)
    if not r["ok"]:
        print(f"  !! {name}: not enough ground points beneath cameras.")
        return None
    if n_images is None:
        n_images = r["n_cams"]

    # GPS sign-anchor from the source photos, if we can find them
    if images_folder is None:
        images_folder = _images_folder_for(name)
    gps = None
    if images_folder:
        gps = gps_anchor.gps_track_slope(gps_anchor.gather_images(images_folder))

    final_signed, basis, sign_source = _resolve_sign(r, gps, name)
    final_dir = "uphill" if final_signed >= 0 else "downhill"

    # If the FINAL magnitude comes from Method 1, orient its plots to the FINAL
    # (externally-anchored) sign so plots/titles/table agree. If the FINAL came
    # from GPS instead (Method 1 was unreliable), leave Method 1 plots raw so they
    # honestly show what Method 1 produced.
    if "Method 1 magnitude" in basis:
        plot_signed = final_signed
        note = f"orientation anchored to {sign_source} (VGGT vertical axis is not gravity-locked)"
    else:
        plot_signed, note = None, ""

    out_png = os.path.join(res_dir, f"{prefix}_{name}_method1_segments.png")
    m1.plot_segments(r, out_png, final_signed=plot_signed, sign_note=note)
    m1.plot_profile(r, os.path.join(res_dir, f"{prefix}_{name}_method1_profile.png"),
                    final_signed=plot_signed, sign_note=note)
    if gps and gps.get("ok"):
        gps_anchor.plot_track(gps, os.path.join(res_dir, f"{prefix}_{name}_gps_track.png"), name)

    rec = {
        "dataset": name, "glb": os.path.basename(glb_path),
        "n_images": n_images, "n_ground_found": r["n_ground_found"],
        # raw Method 1 (VGGT frame) — kept for transparency
        "method1_signed_deg": round(r["overall_signed"], 2),
        "method1_r2": round(r["r2"], 4),
        # gravity-true anchor track, if available (EXIF GPS or sim ground-truth pose)
        "gps_available": bool(gps and gps.get("ok")),
        "anchor_source": gps.get("source") if (gps and gps.get("ok")) else None,
        "gps_signed_deg": round(gps["signed_deg"], 2) if gps and gps.get("ok") else None,
        "gps_r2": round(gps["r2"], 4) if gps and gps.get("ok") else None,
        # final reported estimate
        "final_signed_deg": round(final_signed, 2),
        "direction": final_dir,
        "sign_source": sign_source,
        "estimate_basis": basis,
        "segments": [
            {"label": s["label"],
             "signed_slope": round(s["signed_slope"], 2)
             if np.isfinite(s["signed_slope"]) else None,
             "horiz_m": round(s["horiz"], 3), "dY_m": round(s["dY"], 3)}
            for s in r["segments"]
        ],
    }
    with open(os.path.join(res_dir, f"{prefix}_{name}_method1.json"), "w") as f:
        json.dump(rec, f, indent=2)

    print(f"  Method 1 (VGGT): {r['overall_signed']:+.2f}° (R²={r['r2']:.3f}, {r['n_ground_found']} ground pts)")
    if gps and gps.get("ok"):
        print(f"  {gps['source']:<15s}: {gps['signed_deg']:+.2f}° {gps['direction']} "
              f"(R²={gps['r2']:.3f}, {gps['n_with_gps']} frames)")
    else:
        print(f"  anchor track   : none")
    print(f"  → FINAL        : {final_signed:+.2f}° {final_dir}   [{basis}]")
    return rec


def process_dataset(model, folder, markers=False, prefix="robinson", max_frames=None):
    name = os.path.basename(os.path.normpath(folder))
    images = gather_images(folder, markers=markers, max_frames=max_frames)
    if len(images) < 2:
        print(f"  !! {name}: found {len(images)} images, need >=2. Skipping.")
        return None

    import run_vggt_all as rv
    glb_dir = os.path.join(SLOPE_DIR, "output_glbs", "vggt")
    os.makedirs(glb_dir, exist_ok=True)
    glb_path = os.path.join(glb_dir, f"{prefix}_{name}.glb")
    print(f"\n[{name}] reconstructing {len(images)} images with VGGT → {os.path.basename(glb_path)}")
    rv.run_glb(model, images, glb_path)          # all selected images at once

    return analyze_glb(glb_path, name, n_images=len(images), images_folder=folder, prefix=prefix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="dataset folder, or parent folder with --multi")
    ap.add_argument("--multi",    action="store_true", help="path holds several dataset subfolders")
    ap.add_argument("--markers",  action="store_true", help="datasets use marker1/.. subfolders")
    ap.add_argument("--prefix",   default="robinson",
                    help="output name prefix (e.g. sagamore, N75E). Default: robinson")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="evenly subsample continuous flights to at most N frames before VGGT "
                         "(avoids GPU OOM on hundreds of frames). Ignored with --markers.")
    ap.add_argument("--from-glb", action="store_true",
                    help="skip VGGT; re-run Method 1 on existing output_glbs/vggt/<prefix>_*.glb (no GPU)")
    args = ap.parse_args()

    # --from-glb: analysis only, no model — re-derive slopes/plots from existing GLBs
    if args.from_glb:
        glb_dir = os.path.join(SLOPE_DIR, "output_glbs", "vggt")
        glbs = sorted(glob.glob(os.path.join(glb_dir, f"{args.prefix}_*.glb")))
        if not glbs:
            print(f"No {args.prefix}_*.glb found in {glb_dir}")
            return
        summary = []
        for g in glbs:
            name = os.path.splitext(os.path.basename(g))[0][len(args.prefix) + 1:]
            print(f"\n[{name}] re-analyzing {os.path.basename(g)} (no reconstruction)")
            rec = analyze_glb(g, name, prefix=args.prefix)
            if rec:
                summary.append(rec)
        out = os.path.join(SLOPE_DIR, "results", f"{args.prefix}_summary.json")
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nDone. Summary → {out}")
        return

    import run_vggt_all as rv
    print(f"Device: {rv.device}, dtype: {rv.dtype}")
    print("Loading VGGT model...")
    model = rv.VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    import torch
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, map_location=rv.device))
    model.eval().to(rv.device)
    print("Model ready.")

    if args.multi:
        folders = sorted(
            os.path.join(args.path, d) for d in os.listdir(args.path)
            if os.path.isdir(os.path.join(args.path, d)) and not d.startswith(".")
        )
    else:
        folders = [args.path]

    summary = []
    for folder in folders:
        rec = process_dataset(model, folder, markers=args.markers,
                              prefix=args.prefix, max_frames=args.max_frames)
        if rec:
            summary.append(rec)

    out = os.path.join(SLOPE_DIR, "results", f"{args.prefix}_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll done. Summary → {out}")


if __name__ == "__main__":
    main()
