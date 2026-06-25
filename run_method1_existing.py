"""
Run Method 1 on the existing reconstructions (no heavy compute) and produce:
  results/<scenario>_method1_segments.png   per-segment slope plots
  results/method1_summary.json              numbers for the slide deck

Uses the 'all views' GLB for each scenario (highest k = every marker used).
"""
import os, glob, json, re
import numpy as np
import method1_slope as m1

SLOPE_DIR = os.path.dirname(os.path.abspath(__file__))
GLB_DIR   = os.path.join(SLOPE_DIR, "output_glbs", "vggt")
OUT_DIR   = os.path.join(SLOPE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

SCENARIOS = [
    ("fixed_altitude_uphill",        "Previous dataset (Method-1 home)"),
    ("fixed_altitude_downhill",      "Previous dataset (Method-1 home)"),
    ("fixed_distance2ground_uphill",  "Method-2 dataset (cross-applied)"),
    ("fixed_distance2ground_downhill","Method-2 dataset (cross-applied)"),
]


def all_views_glb(scenario):
    fs = glob.glob(os.path.join(GLB_DIR, f"{scenario}_k*.glb"))
    if not fs:
        return None
    return max(fs, key=lambda f: int(re.search(r"_k(\d+)\.glb", f).group(1)))


def main():
    summary = []
    for scenario, group in SCENARIOS:
        glb = all_views_glb(scenario)
        if glb is None:
            print(f"!! no GLB for {scenario}")
            continue

        r  = m1.analyse(glb)
        m2 = m1.analyse_method2(glb)
        if not r["ok"]:
            print(f"!! {scenario}: not enough ground points")
            continue

        out_png = os.path.join(OUT_DIR, f"{scenario}_method1_segments.png")
        extra = f"   [M2 camera-alt: {m2:+.2f}°]" if m2 is not None else ""
        m1.plot_segments(r, out_png, title_extra=extra)

        rec = {
            "scenario": scenario, "group": group,
            "glb": os.path.basename(glb), "n_cams": r["n_cams"],
            "n_ground_found": r["n_ground_found"],
            "method1_signed": round(r["overall_signed"], 2),
            "method1_direction": r["direction"],
            "method1_r2": round(r["r2"], 4),
            "method2_signed": round(m2, 2) if m2 is not None else None,
            "segments": [
                {"label": s["label"],
                 "signed_slope": round(s["signed_slope"], 2)
                 if np.isfinite(s["signed_slope"]) else None,
                 "horiz_m": round(s["horiz"], 3),
                 "dY_m": round(s["dY"], 3)}
                for s in r["segments"]
            ],
            "segment_plot": os.path.basename(out_png),
        }
        summary.append(rec)

        print(f"\n[{scenario}]  ({r['n_cams']} views, {r['n_ground_found']} ground pts)")
        print(f"  Method 1 overall: {r['overall_signed']:+.2f}° {r['direction']}  (R²={r['r2']:.3f})")
        if m2 is not None:
            print(f"  Method 2 overall: {m2:+.2f}°  (camera-altitude, comparison)")
        print(f"  plot → {os.path.basename(out_png)}")

    # Original Happy Hollow reconstruction (the first Method-1 demonstration)
    hh = os.path.join(SLOPE_DIR, "VGGT_11_views.glb")
    if os.path.exists(hh):
        r = m1.analyse(hh)
        if r["ok"]:
            out_png = os.path.join(OUT_DIR, "happy_hollow_method1_segments.png")
            m1.plot_segments(r, out_png)
            summary.insert(0, {
                "scenario": "happy_hollow_11views", "group": "Original Method-1 demo",
                "glb": "VGGT_11_views.glb", "n_cams": r["n_cams"],
                "n_ground_found": r["n_ground_found"],
                "method1_signed": round(r["overall_signed"], 2),
                "method1_direction": r["direction"],
                "method1_r2": round(r["r2"], 4), "method2_signed": None,
                "segments": [
                    {"label": s["label"],
                     "signed_slope": round(s["signed_slope"], 2)
                     if np.isfinite(s["signed_slope"]) else None,
                     "horiz_m": round(s["horiz"], 3), "dY_m": round(s["dY"], 3)}
                    for s in r["segments"]
                ],
                "segment_plot": os.path.basename(out_png),
            })
            print(f"\n[happy_hollow_11views]  ({r['n_cams']} views)")
            print(f"  Method 1 overall: {r['overall_signed']:+.2f}° {r['direction']}  (R²={r['r2']:.3f})")
            print(f"  plot → {os.path.basename(out_png)}")

    with open(os.path.join(OUT_DIR, "method1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {os.path.join(OUT_DIR, 'method1_summary.json')}")


if __name__ == "__main__":
    main()
