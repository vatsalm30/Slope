#!/usr/bin/env bash
# One-shot: apply the gravity-alignment fix, regenerate the corrected N75E_type3
# numbers + plots (no GPU), and build the PowerPoint.  Safe to re-run.
cd /mnt/sdb/vatsal/Slope || { echo "cd /mnt/sdb/vatsal/Slope failed"; exit 1; }

cat > gravity_fix.patch <<'PATCH_EOF'
diff --git a/RUNBOOK_sagamore_N75E.md b/RUNBOOK_sagamore_N75E.md
index 67b72f9..a1fbb38 100644
--- a/RUNBOOK_sagamore_N75E.md
+++ b/RUNBOOK_sagamore_N75E.md
@@ -8,6 +8,18 @@ Note on the anchor: these are rendered/sim frames — there is **no GPS**. Each
 filename carries exact **ground-truth pose** (`null_<ts>_<idx>_<height>_<X>_<Y>_…`),
 which is a better gravity reference than GPS. The code reads it directly.

+**Gravity alignment (new).** VGGT reconstructs only up to a similarity transform and
+exports the scene in camera-0's frame, so its `+Y` is *not* gravity — the raw Method 1
+slope is measured in a tilted frame and its **magnitude is wrong** (e.g. a true 25°
+slope read as ~65°). The pipeline now Umeyama-aligns the GLB camera centres to the
+ground-truth poses, recovers the VGGT→gravity rotation, rotates the terrain into a
+true-vertical frame, and only then measures slope. This corrects **both magnitude and
+sign** from the poses. Each result reports `method1_signed_deg` (gravity-aligned),
+`method1_raw_vggt_deg` (the old tilted value, for transparency), and `gravity_aligned`
+/ `align_resid_m` (a small residual, ~<0.1 m, means a trustworthy alignment). If the
+camera constellation is collinear/degenerate, alignment is skipped and the run falls
+back to the old folder-label-sign behaviour (see `align_note`).
+
 VGGT-1B weights auto-download from HuggingFace on first run.

 ## 1. Sagamore — 6 marker flights (1 sharp frame per marker)
@@ -18,13 +30,16 @@ python run_robinson.py sagamore_0708 --multi --markers --prefix sagamore

 Processes all 6 flights. Each reconstructs from 5–12 images (one per marker).

-- Pose **height is flat** on every sagamore flight (fixed-altitude & terrain-
-  following both hold height), so the sign comes from the folder label
-  (`_uphill`/`_downhill`) and the **magnitude comes from VGGT Method 1**.
+- Pose **height is flat** on every sagamore flight, but the full 2-D camera
+  positions (east/north) are enough to gravity-align, so magnitude **and** sign
+  now come from the pose-aligned Method 1 — not the folder label. The label is
+  only a fallback if the markers are collinear (alignment then skips).
 - Ground truth is in the folder name: `slope_4…` ≈ 4°, `slope_25…` ≈ 25°.
-  Compare `final_signed_deg` against that.
-- Watch `method1_r2` in each JSON — fixed-altitude flights can rebuild nearly flat
-  (low R²). If so, that flight's geometry didn't give VGGT enough parallax.
+  Compare `final_signed_deg` (should now be close) against that; `method1_raw_vggt_deg`
+  is the old tilted value and will still look inflated.
+- Watch `method1_r2` and `align_resid_m` in each JSON. Fixed-altitude flights can
+  rebuild nearly flat (low R²) — if so, that flight's geometry didn't give VGGT
+  enough parallax. A large `align_resid_m` means the pose alignment itself is weak.

 ## 2. N75E — 5 continuous flights (32 evenly-spaced frames each)

@@ -52,14 +67,19 @@ folder label, magnitude from VGGT.

 ## Re-run analysis only (no GPU, after reconstructions exist)

-Tweak Method 1 / plots without re-reconstructing:
+Tweak Method 1 / plots without re-reconstructing. **Pass the dataset root + the same
+`--multi`/`--markers`/`--max-frames` flags** so the analysis can find the source
+images and gravity-align (the poses live in the filenames):

 ```bash
-python run_robinson.py --from-glb --prefix sagamore
-python run_robinson.py --from-glb --prefix N75E
-python run_robinson.py --from-glb --prefix N75E_type3
+python run_robinson.py sagamore_0708 --from-glb --multi --markers --prefix sagamore
+python run_robinson.py N75E_0712 --from-glb --multi --max-frames 32 --prefix N75E
+python run_robinson.py N75E_0712/type3_front_N75E --from-glb --multi --markers --prefix N75E_type3
 ```

+(Omitting the path still works but skips gravity alignment — you'd get the old
+raw-VGGT-frame magnitudes.)
+
 ## Outputs (per flight, in `results/`)

 - `<prefix>_<flight>_method1.json`      — final signed slope + per-segment table
diff --git a/build_type3_slides.py b/build_type3_slides.py
new file mode 100644
index 0000000..3cf8ebb
--- /dev/null
+++ b/build_type3_slides.py
@@ -0,0 +1,222 @@
+"""
+Build a PowerPoint summarising gravity-aligned Method 1 slope results for one run
+(any --prefix: N75E_type3, sagamore, N75E, …).
+
+Reads:   results/<prefix>_summary.json  and the per-flight PNGs that
+         run_robinson.py wrote (<prefix>_<flight>_method1_profile.png / _segments.png).
+Writes:  slides/<prefix>_report.pptx
+
+    python build_type3_slides.py                 # default prefix N75E_type3
+    python build_type3_slides.py sagamore        # any other run
+
+Run it AFTER the analysis (ideally the gravity-aligned `--from-glb` pass) so the
+numbers and plots it embeds are the corrected ones. Requires python-pptx
+(`pip install python-pptx`).
+"""
+import os, sys, json
+from pptx import Presentation
+from pptx.util import Inches, Pt
+from pptx.dml.color import RGBColor
+from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
+from pptx.enum.shapes import MSO_SHAPE
+
+SLOPE_DIR = os.path.dirname(os.path.abspath(__file__))
+RES = os.path.join(SLOPE_DIR, "results")
+OUT_DIR = os.path.join(SLOPE_DIR, "slides")
+os.makedirs(OUT_DIR, exist_ok=True)
+
+PREFIX = sys.argv[1] if len(sys.argv) > 1 else "N75E_type3"
+
+NAVY  = RGBColor(0x26, 0x46, 0x53)
+TEAL  = RGBColor(0x2A, 0x9D, 0x8F)
+RUST  = RGBColor(0xE7, 0x6F, 0x51)
+GREY  = RGBColor(0x55, 0x55, 0x55)
+WHITE = RGBColor(0xFF, 0xFF, 0xFF)
+INK   = RGBColor(0x22, 0x22, 0x22)
+
+prs = Presentation()
+prs.slide_width  = Inches(13.333)
+prs.slide_height = Inches(7.5)
+SW, SH = prs.slide_width, prs.slide_height
+BLANK = prs.slide_layouts[6]
+
+
+# ── helpers ───────────────────────────────────────────────────────────────────
+
+def _txt(slide, l, t, w, h):
+    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
+    tf = tb.text_frame; tf.word_wrap = True
+    return tf
+
+def band(slide, color=NAVY, h=1.0):
+    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, Inches(h))
+    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background()
+    s.shadow.inherit = False
+    return s
+
+def header(slide, title, sub=None, color=NAVY):
+    band(slide, color, 1.15 if sub else 0.95)
+    tf = _txt(slide, 0.5, 0.16, 12.3, 0.8)
+    p = tf.paragraphs[0]; p.text = title
+    p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = WHITE
+    if sub:
+        tf2 = _txt(slide, 0.5, 0.98, 12.3, 0.45)
+        p2 = tf2.paragraphs[0]; p2.text = sub
+        p2.font.size = Pt(14); p2.font.color.rgb = RGBColor(0xDD, 0xE6, 0xE8)
+
+def bullet(tf, text, size=16, bold=False, color=INK, first=False, level=0):
+    p = tf.paragraphs[0] if first else tf.add_paragraph()
+    p.text = text; p.level = level
+    p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color
+    return p
+
+def add_image_fit(slide, path, l, t, max_w, max_h):
+    """Add image scaled to fit within (max_w, max_h) inches, keeping aspect."""
+    if not (path and os.path.exists(path)):
+        tf = _txt(slide, l, t + max_h / 2 - 0.3, max_w, 0.6)
+        p = tf.paragraphs[0]; p.text = "[plot not found — run the analysis first]"
+        p.font.size = Pt(12); p.font.italic = True; p.font.color.rgb = GREY
+        p.alignment = PP_ALIGN.CENTER
+        return
+    from PIL import Image
+    iw, ih = Image.open(path).size
+    ar = iw / ih
+    w, h = max_w, max_w / ar
+    if h > max_h:
+        h, w = max_h, max_h * ar
+    l2 = l + (max_w - w) / 2
+    t2 = t + (max_h - h) / 2
+    slide.shapes.add_picture(path, Inches(l2), Inches(t2), Inches(w), Inches(h))
+
+def fmt(v, suf="", nd=2):
+    return f"{v:+.{nd}f}{suf}" if isinstance(v, (int, float)) else "—"
+
+
+# ── load ──────────────────────────────────────────────────────────────────────
+
+summary_path = os.path.join(RES, f"{PREFIX}_summary.json")
+if not os.path.exists(summary_path):
+    sys.exit(f"No summary at {summary_path} — run run_robinson.py --prefix {PREFIX} first.")
+summary = json.load(open(summary_path))
+if not summary:
+    sys.exit(f"{summary_path} is empty.")
+
+any_aligned = any(r.get("gravity_aligned") for r in summary)
+
+
+# ── title slide ───────────────────────────────────────────────────────────────
+
+s = prs.slides.add_slide(BLANK)
+band(s, NAVY, 7.5)                      # full navy background (slide height, inches)
+tf = _txt(s, 0.9, 2.4, 11.5, 1.6)
+p = tf.paragraphs[0]; p.text = f"{PREFIX} — Slope Estimation"
+p.font.size = Pt(40); p.font.bold = True; p.font.color.rgb = WHITE
+p2 = tf.add_paragraph()
+p2.text = ("VGGT reconstruction  →  Method 1 (ground-distance)  →  "
+           "gravity-aligned to ground-truth camera poses")
+p2.font.size = Pt(18); p2.font.color.rgb = TEAL
+p3 = tf.add_paragraph()
+p3.text = f"{len(summary)} flight(s):  " + ",  ".join(r["dataset"] for r in summary)
+p3.font.size = Pt(14); p3.font.color.rgb = RGBColor(0xCC, 0xD5, 0xD7)
+
+
+# ── method slide ──────────────────────────────────────────────────────────────
+
+s = prs.slides.add_slide(BLANK)
+header(s, "Method", "Why the estimate is trustworthy")
+tf = _txt(s, 0.6, 1.35, 12.1, 5.7)
+bullet(tf, "1.  VGGT reconstructs a 3-D point cloud + camera poses from the frames.",
+       size=17, bold=True, color=NAVY, first=True)
+bullet(tf, "2.  Method 1 reads the road surface beneath each camera and fits ground "
+           "elevation vs. along-track distance — the line's tilt is the slope.", size=15)
+bullet(tf, "3.  Gravity alignment (key fix):", size=17, bold=True, color=NAVY)
+bullet(tf, "VGGT's reconstruction is only defined up to a similarity transform and is "
+           "exported in camera-0's frame — its vertical axis is NOT gravity.", size=14, level=1)
+bullet(tf, "So the raw slope MAGNITUDE is measured in a tilted frame and can be far off "
+           "(a true 25° slope can read ~65°).", size=14, level=1, color=RUST)
+bullet(tf, "Fix: align the VGGT camera centres to the ground-truth camera world positions "
+           "(encoded in each frame's filename), recover the missing rotation, rotate the "
+           "terrain into a true-vertical frame, then measure slope.", size=14, level=1, color=TEAL)
+bullet(tf, "This corrects both the magnitude and the sign from one physical anchor. A small "
+           "alignment residual (align_resid_m) confirms the fit is trustworthy.", size=14, level=1)
+
+
+# ── per-flight slides ─────────────────────────────────────────────────────────
+
+for r in summary:
+    name = r["dataset"]
+    final = r.get("final_signed_deg")
+    direction = r.get("direction", "")
+    s = prs.slides.add_slide(BLANK)
+    hue = TEAL if (isinstance(final, (int, float)) and final >= 0) else RUST
+    header(s, f"{name}", f"Final slope  {fmt(final,'°')}  {direction}", color=hue)
+
+    # plots: profile (left) + segments (right)
+    prof = os.path.join(RES, f"{PREFIX}_{name}_method1_profile.png")
+    segs = os.path.join(RES, f"{PREFIX}_{name}_method1_segments.png")
+    add_image_fit(s, prof, 0.4, 1.35, 6.2, 3.9)
+    add_image_fit(s, segs, 6.8, 1.35, 6.2, 3.9)
+
+    # stats strip
+    tf = _txt(s, 0.5, 5.45, 12.3, 1.9)
+    aligned = r.get("gravity_aligned")
+    bullet(tf, f"Final (gravity-aligned): {fmt(final,'°')} {direction}"
+               f"    |    Method 1 R² = {fmt(r.get('method1_r2'),'',3)}"
+               f"    |    ground points = {r.get('n_ground_found','—')}",
+           size=15, bold=True, color=NAVY, first=True)
+    if aligned:
+        bullet(tf, f"Gravity alignment: {r.get('align_source')} — residual "
+                   f"{r.get('align_resid_m')} m over {r.get('align_frames')} frames  "
+                   f"(raw VGGT-frame value was {fmt(r.get('method1_raw_vggt_deg'),'°')}, "
+                   f"in a tilted frame).", size=13, color=GREY)
+    else:
+        bullet(tf, f"Gravity alignment NOT applied ({r.get('align_note')}). "
+                   f"Falling back to: {r.get('estimate_basis')}. Magnitude in VGGT's "
+                   f"tilted frame — treat with caution.", size=13, color=RUST)
+
+
+# ── summary table slide ───────────────────────────────────────────────────────
+
+s = prs.slides.add_slide(BLANK)
+header(s, "Summary", "Gravity-aligned vs. raw VGGT-frame")
+cols = ["Flight", "Final (aligned)", "Direction", "Raw VGGT", "R²", "Aligned?", "Resid (m)"]
+rows = len(summary) + 1
+tbl = s.shapes.add_table(rows, len(cols), Inches(0.6), Inches(1.5),
+                         Inches(12.1), Inches(0.5 + 0.55 * len(summary))).table
+for j, c in enumerate(cols):
+    cell = tbl.cell(0, j); cell.text = c
+    cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
+    para = cell.text_frame.paragraphs[0]
+    para.font.size = Pt(13); para.font.bold = True; para.font.color.rgb = WHITE
+for i, r in enumerate(summary, start=1):
+    vals = [
+        r["dataset"],
+        fmt(r.get("final_signed_deg"), "°"),
+        r.get("direction", "—"),
+        fmt(r.get("method1_raw_vggt_deg"), "°"),
+        fmt(r.get("method1_r2"), "", 3),
+        "yes" if r.get("gravity_aligned") else "no",
+        str(r.get("align_resid_m", "—")),
+    ]
+    for j, v in enumerate(vals):
+        cell = tbl.cell(i, j); cell.text = str(v)
+        para = cell.text_frame.paragraphs[0]
+        para.font.size = Pt(12); para.font.color.rgb = INK
+        if j == 1:
+            para.font.bold = True
+            para.font.color.rgb = TEAL if str(v).startswith("+") else RUST
+
+note = _txt(s, 0.6, 1.6 + 0.55 * len(summary) + 0.4, 12.1, 1.4)
+bullet(note, "Final = gravity-aligned Method 1 (magnitude + sign from ground-truth poses). "
+             "Raw VGGT = pre-alignment value in the tilted reconstruction frame, shown for "
+             "transparency. A small residual means the pose alignment is reliable.",
+       size=13, color=GREY, first=True)
+if not any_aligned:
+    bullet(note, "⚠  No flight in this run was gravity-aligned — numbers are raw VGGT-frame. "
+                 "Re-run with the dataset path + flags so the poses can be read.",
+           size=13, bold=True, color=RUST)
+
+
+out = os.path.join(OUT_DIR, f"{PREFIX}_report.pptx")
+prs.save(out)
+print(f"Saved → {out}   ({len(prs.slides)} slides)")
diff --git a/gps_anchor.py b/gps_anchor.py
index 77e4e23..0cd34ac 100644
--- a/gps_anchor.py
+++ b/gps_anchor.py
@@ -164,6 +164,111 @@ def gps_track_slope(image_paths):
     }


+def frame_positions(image_paths):
+    """World positions for an ORDERED list of frames (same order as the GLB
+    cameras). Returns (idx, P, source) where:
+
+      * idx    — indices into image_paths of frames that carried a usable fix
+      * P      — len(idx) x 3 array of METHOD-1-frame positions for those frames:
+                 columns are [east, -up, north] metres (so +Y = down, matching
+                 method1_slope's convention), mean-centred is not required.
+      * source — "sim ground-truth pose" or "EXIF GPS"
+
+    Returns (None, None, None) if fewer than 3 frames carry a fix (Kabsch needs
+    at least 3 to fix a rotation)."""
+    fixes, srcs = [], []
+    for p in image_paths:
+        named = read_gps_from_name(p)
+        if named is not None and np.isfinite(named[2]):
+            fixes.append(named); srcs.append("name"); continue
+        g = read_gps(p)
+        if g is not None and np.isfinite(g[2]):
+            fixes.append(g); srcs.append("exif")
+        else:
+            fixes.append(None); srcs.append(None)
+
+    idx = [i for i, f in enumerate(fixes) if f is not None]
+    if len(idx) < 3:
+        return None, None, None
+
+    lats = np.array([fixes[i][0] for i in idx])
+    lons = np.array([fixes[i][1] for i in idx])
+    alts = np.array([fixes[i][2] for i in idx])
+    east, north = _enu(lats, lons, float(lats.mean()))
+    P = np.column_stack([east, -alts, north])      # +Y = down -> up is -alt
+    src = ("sim ground-truth pose"
+           if sum(srcs[i] == "name" for i in idx) >= len(idx) / 2 else "EXIF GPS")
+    return np.array(idx), P, src
+
+
+def _umeyama(A, B):
+    """Least-squares similarity (s, R, t) mapping A -> B for matched Nx3 sets:
+    minimises || s*R @ A_i + t - B_i ||.  Returns (R, s, t, sing) where `sing`
+    are the singular values used (for a degeneracy check)."""
+    muA, muB = A.mean(0), B.mean(0)
+    A0, B0 = A - muA, B - muB
+    H = (A0.T @ B0) / len(A)
+    U, S, Vt = np.linalg.svd(H)
+    d = np.sign(np.linalg.det(Vt.T @ U.T))          # reflection guard
+    D = np.diag([1.0, 1.0, d])
+    R = Vt.T @ D @ U.T
+    varA = (A0 ** 2).sum() / len(A)
+    s = float((S * np.array([1.0, 1.0, d])).sum() / varA) if varA > 1e-12 else 1.0
+    t = muB - s * R @ muA
+    return R, s, t
+
+
+def gravity_align(cameras_glb, image_paths, min_axis_ratio=0.05, max_resid_rel=0.5):
+    """Recover the rotation that carries the VGGT/GLB frame into a gravity-true
+    frame, using the known ground-truth camera world positions.
+
+    VGGT reconstructs only up to a similarity transform, and the exported scene
+    sits in camera-0's frame — its +Y is NOT gravity. But every sim frame's
+    filename encodes the true camera world position, so aligning the GLB camera
+    centres to those true positions recovers the missing rotation. Rotating the
+    terrain by it puts slope back on a true-vertical footing.
+
+    Args
+      cameras_glb  Mx3 GLB camera centres, in GLB-camera order.
+      image_paths  the ORDERED image list fed to VGGT (image_paths[i] <-> camera i).
+
+    Returns a dict:
+      {ok, R (3x3), source, n_frames, resid_m, resid_rel, note}
+    `ok` is False (with a `note`) when there is no pose track, too few frames,
+    a collinear/degenerate camera constellation, or a large alignment residual —
+    in those cases the caller should fall back to the un-aligned estimate."""
+    idx, P_world, src = frame_positions(image_paths)
+    if idx is None:
+        return {"ok": False, "note": "no pose track (need >=3 frames with a fix)"}
+    if len(idx) > len(cameras_glb):
+        return {"ok": False, "note": "more pose fixes than GLB cameras (order mismatch)"}
+
+    A = np.asarray(cameras_glb, dtype=np.float64)[idx]      # GLB centres, matched
+    B = P_world                                             # true world centres
+
+    # collinearity guard: the GLB camera constellation must span >=2 dimensions
+    # for a rotation to be determined (coplanar is fine; a single line is not).
+    sv = np.linalg.svd(A - A.mean(0), compute_uv=False)
+    if sv[0] < 1e-9 or sv[1] / sv[0] < min_axis_ratio:
+        return {"ok": False, "source": src, "n_frames": int(len(idx)),
+                "note": f"degenerate camera geometry (collinear, axis ratio "
+                        f"{sv[1] / max(sv[0], 1e-12):.3f})"}
+
+    R, s, t = _umeyama(A, B)
+    resid = A @ (s * R).T + t - B
+    resid_m = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
+    span = float(np.ptp(B[:, [0, 2]]))                      # horizontal baseline
+    resid_rel = resid_m / span if span > 1e-9 else float("inf")
+    if resid_rel > max_resid_rel:
+        return {"ok": False, "source": src, "n_frames": int(len(idx)),
+                "resid_m": round(resid_m, 3), "resid_rel": round(resid_rel, 3),
+                "note": f"poor pose alignment (residual {resid_rel:.2f} of baseline)"}
+
+    return {"ok": True, "R": R, "source": src, "n_frames": int(len(idx)),
+            "resid_m": round(resid_m, 3), "resid_rel": round(resid_rel, 3),
+            "note": "gravity-aligned via known camera poses"}
+
+
 def plot_track(gps, out_png, name=""):
     """GPS altitude vs along-track distance — the gravity-true profile + fit."""
     import matplotlib
diff --git a/method1_slope.py b/method1_slope.py
index 77706c8..f2f9bc5 100644
--- a/method1_slope.py
+++ b/method1_slope.py
@@ -89,7 +89,14 @@ def analyse(glb_path):
     """Run Method 1 on one GLB. Returns a result dict (all frames, signed)."""
     terrain, cameras = load_glb(glb_path)
     name = os.path.splitext(os.path.basename(glb_path))[0]
+    return analyse_arrays(terrain, cameras, name)

+
+def analyse_arrays(terrain, cameras, name="glb"):
+    """Method 1 on pre-loaded arrays (terrain Nx3, cameras Mx3), so a caller can
+    gravity-align the frame first (rotate terrain+cameras into a true-vertical
+    frame) before measuring slope. Both arrays must share the same frame with
+    +Y = down. Returns the same result dict as analyse()."""
     radius = np.ptp(terrain[:, [0, 2]]) * SEARCH_RADIUS_FRAC

     ground_pts, found = [], []
diff --git a/run_robinson.py b/run_robinson.py
index 126f9e9..5f460c1 100644
--- a/run_robinson.py
+++ b/run_robinson.py
@@ -113,37 +113,61 @@ def _resolve_sign(r, gps, label):
     return r["overall_signed"], "Method 1 (sign unverified)", "VGGT (unverified)"


-def analyze_glb(glb_path, name, n_images=None, images_folder=None, prefix="robinson"):
-    """Method 1 + GPS sign-anchor + plots + JSON for one GLB (no GPU)."""
+def analyze_glb(glb_path, name, n_images=None, images_folder=None, prefix="robinson",
+                ordered_images=None):
+    """Method 1 + pose gravity-anchor + plots + JSON for one GLB (no GPU).
+
+    When `ordered_images` (the ORDERED image list fed to VGGT, one per GLB camera)
+    is given and carries ground-truth poses, the GLB frame is gravity-aligned to
+    those poses before Method 1 runs — this corrects BOTH the slope magnitude and
+    sign, because VGGT's raw vertical axis is not gravity-locked. Otherwise it
+    falls back to the raw VGGT-frame estimate with the sign anchored externally
+    (GPS track / folder label), as before."""
     res_dir = os.path.join(SLOPE_DIR, "results")
     os.makedirs(res_dir, exist_ok=True)

-    r = m1.analyse(glb_path)
-    if not r["ok"]:
+    terrain, cameras = m1.load_glb(glb_path)
+    raw = m1.analyse_arrays(terrain, cameras, name)      # VGGT-frame (un-aligned)
+    if not raw["ok"]:
         print(f"  !! {name}: not enough ground points beneath cameras.")
         return None
     if n_images is None:
-        n_images = r["n_cams"]
+        n_images = raw["n_cams"]
+
+    # Gravity-align the GLB frame to the known camera poses, if we can.
+    align = {"ok": False, "note": "no ordered image list supplied"}
+    if ordered_images:
+        align = gps_anchor.gravity_align(cameras, ordered_images)
+    aligned = bool(align.get("ok"))
+    if aligned:
+        R = align["R"]
+        r = m1.analyse_arrays(terrain @ R.T, cameras @ R.T, name)
+    else:
+        r = raw

-    # GPS sign-anchor from the source photos, if we can find them
+    # GPS/pose track (altitude-vs-distance) — still computed for its plot and as a
+    # secondary sanity reference. Sign anchoring below only uses it when NOT aligned.
     if images_folder is None:
         images_folder = _images_folder_for(name)
     gps = None
     if images_folder:
         gps = gps_anchor.gps_track_slope(gps_anchor.gather_images(images_folder))

-    final_signed, basis, sign_source = _resolve_sign(r, gps, name)
-    final_dir = "uphill" if final_signed >= 0 else "downhill"
-
-    # If the FINAL magnitude comes from Method 1, orient its plots to the FINAL
-    # (externally-anchored) sign so plots/titles/table agree. If the FINAL came
-    # from GPS instead (Method 1 was unreliable), leave Method 1 plots raw so they
-    # honestly show what Method 1 produced.
-    if "Method 1 magnitude" in basis:
-        plot_signed = final_signed
-        note = f"orientation anchored to {sign_source} (VGGT vertical axis is not gravity-locked)"
+    if aligned:
+        # gravity-aligned Method 1 gives a trustworthy signed slope on its own.
+        final_signed = r["overall_signed"]
+        sign_source = align["source"]
+        basis = f"Method 1, gravity-aligned via {sign_source}"
+        plot_signed = None          # arrays already in a true-vertical frame
+        note = f"gravity-aligned to {sign_source} poses (resid {align['resid_m']} m, {align['n_frames']} frames)"
     else:
-        plot_signed, note = None, ""
+        final_signed, basis, sign_source = _resolve_sign(r, gps, name)
+        if "Method 1 magnitude" in basis:
+            plot_signed = final_signed
+            note = f"orientation anchored to {sign_source} (VGGT vertical axis is not gravity-locked)"
+        else:
+            plot_signed, note = None, ""
+    final_dir = "uphill" if final_signed >= 0 else "downhill"

     out_png = os.path.join(res_dir, f"{prefix}_{name}_method1_segments.png")
     m1.plot_segments(r, out_png, final_signed=plot_signed, sign_note=note)
@@ -155,9 +179,17 @@ def analyze_glb(glb_path, name, n_images=None, images_folder=None, prefix="robin
     rec = {
         "dataset": name, "glb": os.path.basename(glb_path),
         "n_images": n_images, "n_ground_found": r["n_ground_found"],
-        # raw Method 1 (VGGT frame) — kept for transparency
+        # Method 1 result used for the final estimate (gravity-aligned when possible)
         "method1_signed_deg": round(r["overall_signed"], 2),
         "method1_r2": round(r["r2"], 4),
+        # raw VGGT-frame Method 1 (before gravity alignment) — kept for transparency
+        "method1_raw_vggt_deg": round(raw["overall_signed"], 2),
+        # gravity alignment from known camera poses
+        "gravity_aligned": aligned,
+        "align_source": align.get("source"),
+        "align_resid_m": align.get("resid_m"),
+        "align_frames": align.get("n_frames"),
+        "align_note": align.get("note"),
         # gravity-true anchor track, if available (EXIF GPS or sim ground-truth pose)
         "gps_available": bool(gps and gps.get("ok")),
         "anchor_source": gps.get("source") if (gps and gps.get("ok")) else None,
@@ -179,12 +211,15 @@ def analyze_glb(glb_path, name, n_images=None, images_folder=None, prefix="robin
     with open(os.path.join(res_dir, f"{prefix}_{name}_method1.json"), "w") as f:
         json.dump(rec, f, indent=2)

-    print(f"  Method 1 (VGGT): {r['overall_signed']:+.2f}° (R²={r['r2']:.3f}, {r['n_ground_found']} ground pts)")
+    print(f"  Method 1 (raw VGGT frame): {raw['overall_signed']:+.2f}° (R²={raw['r2']:.3f}, {raw['n_ground_found']} ground pts)")
+    if aligned:
+        print(f"  Method 1 (gravity-aligned): {r['overall_signed']:+.2f}° (R²={r['r2']:.3f})  "
+              f"[{align['source']}, resid {align['resid_m']} m over {align['n_frames']} frames]")
+    else:
+        print(f"  gravity align  : skipped — {align.get('note')}")
     if gps and gps.get("ok"):
         print(f"  {gps['source']:<15s}: {gps['signed_deg']:+.2f}° {gps['direction']} "
               f"(R²={gps['r2']:.3f}, {gps['n_with_gps']} frames)")
-    else:
-        print(f"  anchor track   : none")
     print(f"  → FINAL        : {final_signed:+.2f}° {final_dir}   [{basis}]")
     return rec

@@ -203,7 +238,8 @@ def process_dataset(model, folder, markers=False, prefix="robinson", max_frames=
     print(f"\n[{name}] reconstructing {len(images)} images with VGGT → {os.path.basename(glb_path)}")
     rv.run_glb(model, images, glb_path)          # all selected images at once

-    return analyze_glb(glb_path, name, n_images=len(images), images_folder=folder, prefix=prefix)
+    return analyze_glb(glb_path, name, n_images=len(images), images_folder=folder,
+                       prefix=prefix, ordered_images=images)


 def main():
@@ -230,8 +266,21 @@ def main():
         summary = []
         for g in glbs:
             name = os.path.splitext(os.path.basename(g))[0][len(args.prefix) + 1:]
-            print(f"\n[{name}] re-analyzing {os.path.basename(g)} (no reconstruction)")
-            rec = analyze_glb(g, name, prefix=args.prefix)
+            # Locate the source images so we can gravity-align without the GPU.
+            # Pass the dataset root as the positional path (same --multi/--markers/
+            # --max-frames flags used for the reconstruction run).
+            flight_folder = None
+            if args.path:
+                cand = os.path.join(args.path, name) if args.multi else args.path
+                if os.path.isdir(cand):
+                    flight_folder = cand
+            ordered = (gather_images(flight_folder, markers=args.markers,
+                                     max_frames=args.max_frames)
+                       if flight_folder else None)
+            print(f"\n[{name}] re-analyzing {os.path.basename(g)} (no reconstruction)"
+                  + ("" if flight_folder else "  [no source images → gravity align skipped]"))
+            rec = analyze_glb(g, name, images_folder=flight_folder, prefix=args.prefix,
+                              ordered_images=ordered)
             if rec:
                 summary.append(rec)
         out = os.path.join(SLOPE_DIR, "results", f"{args.prefix}_summary.json")
PATCH_EOF

# ---- apply the fix (idempotent) ----
if git apply --check gravity_fix.patch 2>/dev/null; then
    git apply gravity_fix.patch && echo ">>> PATCH APPLIED"
elif grep -q "def gravity_align" gps_anchor.py 2>/dev/null \
     && grep -q "def analyse_arrays" method1_slope.py 2>/dev/null; then
    echo ">>> Fix already present — skipping patch."
else
    echo ">>> ERROR: patch did not apply and fix not present. Send this output to Claude."
    exit 1
fi

# ---- regenerate corrected numbers + plots (reuses existing GLBs, no GPU) ----
echo ">>> Re-analyzing N75E_type3 with gravity alignment..."
python run_robinson.py N75E_0712/type3_front_N75E --from-glb --multi --markers --prefix N75E_type3 \
    || { echo ">>> analysis failed"; exit 1; }

# ---- build the deck ----
python -c "import pptx" 2>/dev/null || pip install python-pptx
python build_type3_slides.py N75E_type3 || { echo ">>> slide build failed"; exit 1; }

echo ">>> DONE. Deck at: /mnt/sdb/vatsal/Slope/slides/N75E_type3_report.pptx"
