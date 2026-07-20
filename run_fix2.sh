#!/usr/bin/env bash
# Incremental algo update (research-driven): up-vector gravity fallback, metric
# scale, 2-D plane-grade cross-check, degeneracy metrics + confidence gating.
# Patches the 3 python files and refreshes the deck builder, then re-runs.
# Safe to re-run. Assumes the earlier gravity fix (run_fix.sh) was already applied.
cd /mnt/sdb/vatsal/Slope || { echo "cd /mnt/sdb/vatsal/Slope failed"; exit 1; }

cat > algo_update.patch <<'PATCH2_EOF'
diff --git a/gps_anchor.py b/gps_anchor.py
index 0cd34ac..5c737b9 100644
--- a/gps_anchor.py
+++ b/gps_anchor.py
@@ -218,55 +218,115 @@ def _umeyama(A, B):
     return R, s, t
 
 
-def gravity_align(cameras_glb, image_paths, min_axis_ratio=0.05, max_resid_rel=0.5):
+def _rotation_from_up(up_vec):
+    """Shortest-arc rotation mapping a GLB-frame up-vector onto method-1-frame
+    world up = (0, -1, 0)  (+Y = down).
+
+    Azimuth about the vertical is left arbitrary — it does NOT affect slope
+    magnitude or the along-track sign, because the flight direction is re-derived
+    from the (rotated) cameras. A single correct up-vector is therefore enough to
+    put slope on a true-vertical footing, which is why a per-image gravity estimate
+    (GeoCalib) or the drone IMU can rescue flights whose camera geometry is too
+    degenerate for pose-based alignment (e.g. a straight-line / nadir pass)."""
+    u = np.asarray(up_vec, dtype=np.float64)
+    n = np.linalg.norm(u)
+    if n < 1e-9:
+        return np.eye(3)
+    u = u / n
+    t = np.array([0.0, -1.0, 0.0])
+    v = np.cross(u, t)
+    s = np.linalg.norm(v)
+    c = float(np.dot(u, t))
+    if s < 1e-9:                                    # already up, or exactly flipped
+        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
+    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
+    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
+
+
+def gravity_align(cameras_glb, image_paths, min_axis_ratio=0.05, max_resid_rel=0.5,
+                  up_hint=None):
     """Recover the rotation that carries the VGGT/GLB frame into a gravity-true
-    frame, using the known ground-truth camera world positions.
+    frame — the pipeline's central problem (VGGT returns geometry in camera-0's
+    tilted frame up to a similarity, so slope read off it carries the camera tilt).
 
-    VGGT reconstructs only up to a similarity transform, and the exported scene
-    sits in camera-0's frame — its +Y is NOT gravity. But every sim frame's
-    filename encodes the true camera world position, so aligning the GLB camera
-    centres to those true positions recovers the missing rotation. Rotating the
-    terrain by it puts slope back on a true-vertical footing.
+    Two gravity sources, tried in order:
+
+      1. **Pose-based Umeyama** (primary) — align the GLB camera centres to the
+         known ground-truth camera world positions (from the filenames). Needs a
+         non-degenerate (non-collinear) camera constellation.
+      2. **External up-vector** (`up_hint`, fallback) — a per-image up-vector (from
+         GeoCalib) or the drone IMU/attitude, in the GLB frame. Used when the pose
+         geometry is degenerate. One up-vector is enough to fix slope. Pass a single
+         (3,) vector or an (N,3) array (averaged). This is the research-recommended
+         rescue for straight-line / nadir flights.
 
     Args
       cameras_glb  Mx3 GLB camera centres, in GLB-camera order.
       image_paths  the ORDERED image list fed to VGGT (image_paths[i] <-> camera i).
-
-    Returns a dict:
-      {ok, R (3x3), source, n_frames, resid_m, resid_rel, note}
-    `ok` is False (with a `note`) when there is no pose track, too few frames,
-    a collinear/degenerate camera constellation, or a large alignment residual —
-    in those cases the caller should fall back to the un-aligned estimate."""
+      up_hint      optional GLB-frame up-vector(s) from GeoCalib/IMU (see above).
+
+    Returns a dict always carrying the flight's degeneracy metrics
+    (`collinearity` = 2nd/1st singular value of the camera spread, `planarity` =
+    3rd/1st), plus on success {ok, R, scale, source, method, ...}. `ok` is False
+    (with a `note`) when no gravity source resolves it — caller falls back to the
+    un-aligned estimate. `scale` is metres per VGGT unit (None for the up-hint
+    path, which does not observe scale)."""
     idx, P_world, src = frame_positions(image_paths)
-    if idx is None:
-        return {"ok": False, "note": "no pose track (need >=3 frames with a fix)"}
-    if len(idx) > len(cameras_glb):
-        return {"ok": False, "note": "more pose fixes than GLB cameras (order mismatch)"}
-
-    A = np.asarray(cameras_glb, dtype=np.float64)[idx]      # GLB centres, matched
-    B = P_world                                             # true world centres
-
-    # collinearity guard: the GLB camera constellation must span >=2 dimensions
-    # for a rotation to be determined (coplanar is fine; a single line is not).
-    sv = np.linalg.svd(A - A.mean(0), compute_uv=False)
-    if sv[0] < 1e-9 or sv[1] / sv[0] < min_axis_ratio:
-        return {"ok": False, "source": src, "n_frames": int(len(idx)),
-                "note": f"degenerate camera geometry (collinear, axis ratio "
-                        f"{sv[1] / max(sv[0], 1e-12):.3f})"}
-
-    R, s, t = _umeyama(A, B)
-    resid = A @ (s * R).T + t - B
-    resid_m = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
-    span = float(np.ptp(B[:, [0, 2]]))                      # horizontal baseline
-    resid_rel = resid_m / span if span > 1e-9 else float("inf")
-    if resid_rel > max_resid_rel:
-        return {"ok": False, "source": src, "n_frames": int(len(idx)),
-                "resid_m": round(resid_m, 3), "resid_rel": round(resid_rel, 3),
-                "note": f"poor pose alignment (residual {resid_rel:.2f} of baseline)"}
-
-    return {"ok": True, "R": R, "source": src, "n_frames": int(len(idx)),
-            "resid_m": round(resid_m, 3), "resid_rel": round(resid_rel, 3),
-            "note": "gravity-aligned via known camera poses"}
+    have_poses = idx is not None
+
+    # Degeneracy metrics from the GLB camera constellation — reported ALWAYS so the
+    # caller can gate confidence on flight geometry (Stage 5). collinearity < ~0.05
+    # => cameras on a line (roll/gravity unrecoverable); planarity < ~0.02 => nearly
+    # coplanar (low vertical parallax).
+    A_all = np.asarray(cameras_glb, dtype=np.float64)
+    degen = {}
+    if len(A_all) >= 3:
+        sv_all = np.linalg.svd(A_all - A_all.mean(0), compute_uv=False)
+        degen = {"collinearity": round(float(sv_all[1] / max(sv_all[0], 1e-12)), 4),
+                 "planarity": round(float(sv_all[2] / max(sv_all[0], 1e-12)), 4)}
+
+    def _ret(d):
+        d.update(degen)
+        return d
+
+    # ---- primary: pose-based Umeyama (needs non-degenerate cameras) ----
+    pose_fail = None
+    if not have_poses:
+        pose_fail = "no pose track (need >=3 frames with a fix)"
+    elif len(idx) > len(cameras_glb):
+        pose_fail = "more pose fixes than GLB cameras (order mismatch)"
+    else:
+        A, B = A_all[idx], P_world
+        sv = np.linalg.svd(A - A.mean(0), compute_uv=False)
+        if sv[0] < 1e-9 or sv[1] / sv[0] < min_axis_ratio:
+            pose_fail = (f"degenerate camera geometry (collinear, axis ratio "
+                         f"{sv[1] / max(sv[0], 1e-12):.3f})")
+        else:
+            R, s, t = _umeyama(A, B)
+            resid = A @ (s * R).T + t - B
+            resid_m = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
+            span = float(np.ptp(B[:, [0, 2]]))                 # horizontal baseline
+            resid_rel = resid_m / span if span > 1e-9 else float("inf")
+            if resid_rel > max_resid_rel:
+                pose_fail = f"poor pose alignment (residual {resid_rel:.2f} of baseline)"
+            else:
+                return _ret({"ok": True, "R": R, "scale": round(float(s), 6),
+                             "source": src, "method": "pose-umeyama",
+                             "n_frames": int(len(idx)), "resid_m": round(resid_m, 3),
+                             "resid_rel": round(resid_rel, 3),
+                             "note": "gravity-aligned via known camera poses"})
+
+    # ---- fallback: external up-vector (GeoCalib / IMU) rescues degenerate flights ----
+    if up_hint is not None:
+        up = np.asarray(up_hint, dtype=np.float64)
+        up = up.mean(0) if up.ndim == 2 else up
+        return _ret({"ok": True, "R": _rotation_from_up(up), "scale": None,
+                     "source": "external up-vector (GeoCalib/IMU)", "method": "up-hint",
+                     "n_frames": int(len(idx)) if have_poses else 0,
+                     "note": f"gravity from up-vector hint (pose path unusable: {pose_fail})"})
+
+    return _ret({"ok": False, "source": src if have_poses else None,
+                 "n_frames": int(len(idx)) if have_poses else 0, "note": pose_fail})
 
 
 def plot_track(gps, out_png, name=""):
diff --git a/method1_slope.py b/method1_slope.py
index f2f9bc5..acea5a9 100644
--- a/method1_slope.py
+++ b/method1_slope.py
@@ -125,6 +125,12 @@ def analyse_arrays(terrain, cameras, name="glb"):
     overall_signed = -np.degrees(np.arctan(grad))
     direction      = "uphill" if overall_signed >= 0 else "downhill"
 
+    # full 2-D terrain grade from a plane fit to the ground points — a cross-check
+    # on the along-track line fit (which only sees the component along the flight
+    # path). Only meaningful once the frame is gravity-aligned. `plane_slope_deg`
+    # is the steepest grade of the surface; `aspect_deg` its downhill direction.
+    plane_slope_deg, aspect_deg = plane_slope(ground_pts, found)
+
     # per-segment slopes between consecutive images (image order on x-axis)
     segments = []
     for i in range(len(cameras) - 1):
@@ -142,12 +148,36 @@ def analyse_arrays(terrain, cameras, name="glb"):
         "name": name, "ok": True, "n_cams": len(cameras),
         "n_ground_found": int(found.sum()),
         "overall_signed": overall_signed, "direction": direction, "r2": r2,
+        "plane_slope_deg": plane_slope_deg, "aspect_deg": aspect_deg,
         "segments": segments,
         "terrain": terrain, "cameras": cameras,
         "ground_pts": ground_pts, "found": found, "fd": fd, "s": s,
     }
 
 
+def plane_slope(ground_pts, found):
+    """Terrain grade from a least-squares plane fit to the ground points, in the
+    (gravity-aligned) +Y=down frame. Returns (max_slope_deg, aspect_deg):
+
+      * max_slope_deg — steepest grade of the surface = angle of the plane normal
+        from vertical. This is the full 2-D grade; the along-track `overall_signed`
+        is its component along the flight path, so |overall_signed| <= max_slope_deg.
+      * aspect_deg    — downhill direction in the horizontal (XZ) plane.
+
+    A large gap between the two means the flight ran across the slope rather than
+    up/down the fall line; NaN if fewer than 3 ground points."""
+    g = ground_pts[found]
+    if len(g) < 3:
+        return float("nan"), float("nan")
+    _, _, Vt = np.linalg.svd(g - g.mean(0), full_matrices=False)
+    n = Vt[-1]                                          # unit plane normal
+    n = n / (np.linalg.norm(n) + 1e-12)
+    max_slope = float(np.degrees(np.arccos(min(1.0, abs(n[1])))))   # tilt from vertical
+    aspect = (float(np.degrees(np.arctan2(n[2], n[0])))
+              if abs(n[1]) < 0.9999 else float("nan"))
+    return max_slope, aspect
+
+
 # ── Method 2 (camera-altitude) — for side-by-side comparison only ─────────────
 
 def analyse_method2(glb_path):
diff --git a/run_robinson.py b/run_robinson.py
index 5f460c1..c595d50 100644
--- a/run_robinson.py
+++ b/run_robinson.py
@@ -113,8 +113,46 @@ def _resolve_sign(r, gps, label):
     return r["overall_signed"], "Method 1 (sign unverified)", "VGGT (unverified)"
 
 
+def _assess_confidence(r, align, aligned):
+    """Combine the research's confidence signals into one flag + reasons (Stage 5).
+
+    Note on geometry metrics: a low *planarity* is expected and harmless for any
+    fixed-altitude flight (its cameras are coplanar), so it is reported but NOT
+    scored. *Collinearity* only counts against confidence when it actually
+    prevented alignment — if an up-vector hint rescued a collinear flight, the
+    slope is fine. Returns ("high"|"medium"|"low", [reasons])."""
+    reasons = []
+    r2 = r.get("r2")
+    weak_fit = r2 is not None and np.isfinite(r2) and r2 < 0.5
+
+    if not aligned:
+        reasons.append("no gravity reference — slope measured in VGGT's tilted frame")
+        col = align.get("collinearity")
+        if col is not None and col < 0.05:
+            reasons.append(f"near-collinear cameras (axis ratio {col:.3f}) — pose gravity unrecoverable")
+        if weak_fit:
+            reasons.append(f"weak ground fit (R²={r2:.2f})")
+        return "low", reasons
+
+    level = "high"
+    if weak_fit:
+        reasons.append(f"weak ground fit (R²={r2:.2f})")
+        level = "low"
+    if align.get("method") == "up-hint":
+        reasons.append("gravity from external up-vector (GeoCalib/IMU), not camera poses")
+        level = "medium" if level == "high" else level
+    rr = align.get("resid_rel")
+    if rr is not None and rr > 0.15:
+        reasons.append(f"moderate pose-alignment residual ({rr:.2f} of baseline)")
+        level = "medium" if level == "high" else level
+
+    if not reasons:
+        reasons.append("gravity-aligned, strong fit, non-degenerate geometry")
+    return level, reasons
+
+
 def analyze_glb(glb_path, name, n_images=None, images_folder=None, prefix="robinson",
-                ordered_images=None):
+                ordered_images=None, up_hint=None):
     """Method 1 + pose gravity-anchor + plots + JSON for one GLB (no GPU).
 
     When `ordered_images` (the ORDERED image list fed to VGGT, one per GLB camera)
@@ -134,10 +172,12 @@ def analyze_glb(glb_path, name, n_images=None, images_folder=None, prefix="robin
     if n_images is None:
         n_images = raw["n_cams"]
 
-    # Gravity-align the GLB frame to the known camera poses, if we can.
+    # Gravity-align the GLB frame to the known camera poses, if we can. `up_hint`
+    # (a per-image up-vector from GeoCalib or the drone IMU, in the GLB frame) is an
+    # optional fallback that rescues degenerate/collinear flights the poses can't.
     align = {"ok": False, "note": "no ordered image list supplied"}
     if ordered_images:
-        align = gps_anchor.gravity_align(cameras, ordered_images)
+        align = gps_anchor.gravity_align(cameras, ordered_images, up_hint=up_hint)
     aligned = bool(align.get("ok"))
     if aligned:
         R = align["R"]
@@ -169,6 +209,19 @@ def analyze_glb(glb_path, name, n_images=None, images_folder=None, prefix="robin
             plot_signed, note = None, ""
     final_dir = "uphill" if final_signed >= 0 else "downhill"
 
+    confidence, conf_reasons = _assess_confidence(r, align, aligned)
+
+    # Metric rise/run — the slope ANGLE is scale-invariant, but with a metric scale
+    # (metres per VGGT unit, from the pose Umeyama fit) we can also report the true
+    # elevation change and horizontal run. Only when pose-aligned (up-hint path has
+    # no scale). Scale/gravity are separable: gravity fixes the angle, scale meters.
+    run_m = rise_m = None
+    scale = align.get("scale")
+    if aligned and scale:
+        s_found = r["s"][r["found"]]
+        run_m = round(float(np.ptp(s_found)) * scale, 2)
+        rise_m = round(run_m * math.tan(math.radians(final_signed)), 2)
+
     out_png = os.path.join(res_dir, f"{prefix}_{name}_method1_segments.png")
     m1.plot_segments(r, out_png, final_signed=plot_signed, sign_note=note)
     m1.plot_profile(r, os.path.join(res_dir, f"{prefix}_{name}_method1_profile.png"),
@@ -184,12 +237,26 @@ def analyze_glb(glb_path, name, n_images=None, images_folder=None, prefix="robin
         "method1_r2": round(r["r2"], 4),
         # raw VGGT-frame Method 1 (before gravity alignment) — kept for transparency
         "method1_raw_vggt_deg": round(raw["overall_signed"], 2),
-        # gravity alignment from known camera poses
+        # full 2-D terrain grade (plane fit) — cross-check on the along-track value
+        "plane_slope_deg": round(r["plane_slope_deg"], 2) if np.isfinite(r.get("plane_slope_deg", float("nan"))) else None,
+        "aspect_deg": round(r["aspect_deg"], 1) if np.isfinite(r.get("aspect_deg", float("nan"))) else None,
+        # gravity alignment from known camera poses (or external up-vector)
         "gravity_aligned": aligned,
+        "align_method": align.get("method"),
         "align_source": align.get("source"),
         "align_resid_m": align.get("resid_m"),
         "align_frames": align.get("n_frames"),
         "align_note": align.get("note"),
+        # flight-geometry degeneracy metrics (Stage 5 gating)
+        "collinearity": align.get("collinearity"),
+        "planarity": align.get("planarity"),
+        # metric scale + rise/run (None unless pose-aligned with a metric fit)
+        "scale_m_per_unit": scale,
+        "run_m": run_m,
+        "rise_m": rise_m,
+        # consolidated confidence flag + reasons
+        "confidence": confidence,
+        "confidence_reasons": conf_reasons,
         # gravity-true anchor track, if available (EXIF GPS or sim ground-truth pose)
         "gps_available": bool(gps and gps.get("ok")),
         "anchor_source": gps.get("source") if (gps and gps.get("ok")) else None,
@@ -220,7 +287,11 @@ def analyze_glb(glb_path, name, n_images=None, images_folder=None, prefix="robin
     if gps and gps.get("ok"):
         print(f"  {gps['source']:<15s}: {gps['signed_deg']:+.2f}° {gps['direction']} "
               f"(R²={gps['r2']:.3f}, {gps['n_with_gps']} frames)")
+    if np.isfinite(r.get("plane_slope_deg", float("nan"))):
+        extra = f" ({run_m} m run, {rise_m} m rise)" if run_m is not None else ""
+        print(f"  plane grade    : {r['plane_slope_deg']:.2f}° max (2-D cross-check){extra}")
     print(f"  → FINAL        : {final_signed:+.2f}° {final_dir}   [{basis}]")
+    print(f"  confidence     : {confidence.upper()} — {'; '.join(conf_reasons)}")
     return rec
 
 
PATCH2_EOF

# ---- apply the algo patch to the 3 python files (idempotent) ----
if git apply --check algo_update.patch 2>/dev/null; then
    git apply algo_update.patch && echo ">>> ALGO PATCH APPLIED"
elif grep -q "_rotation_from_up" gps_anchor.py 2>/dev/null \
     && grep -q "_assess_confidence" run_robinson.py 2>/dev/null; then
    echo ">>> Algo update already present — skipping patch."
else
    echo ">>> ERROR: algo patch did not apply and update not present. Send this output to Claude."
    exit 1
fi

# ---- overwrite the deck builder with the current version (robust, no context match) ----
cat > build_type3_slides.py <<'BUILD_EOF'
"""
Build a PowerPoint summarising gravity-aligned Method 1 slope results for one run
(any --prefix: N75E_type3, sagamore, N75E, …).

Reads:   results/<prefix>_summary.json  and the per-flight PNGs that
         run_robinson.py wrote (<prefix>_<flight>_method1_profile.png / _segments.png).
Writes:  slides/<prefix>_report.pptx

    python build_type3_slides.py                 # default prefix N75E_type3
    python build_type3_slides.py sagamore        # any other run

Run it AFTER the analysis (ideally the gravity-aligned `--from-glb` pass) so the
numbers and plots it embeds are the corrected ones. Requires python-pptx
(`pip install python-pptx`).
"""
import os, sys, json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

SLOPE_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(SLOPE_DIR, "results")
OUT_DIR = os.path.join(SLOPE_DIR, "slides")
os.makedirs(OUT_DIR, exist_ok=True)

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "N75E_type3"

NAVY  = RGBColor(0x26, 0x46, 0x53)
TEAL  = RGBColor(0x2A, 0x9D, 0x8F)
RUST  = RGBColor(0xE7, 0x6F, 0x51)
GREY  = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK   = RGBColor(0x22, 0x22, 0x22)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


# ── helpers ───────────────────────────────────────────────────────────────────

def _txt(slide, l, t, w, h):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    return tf

def band(slide, color=NAVY, h=1.0):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background()
    s.shadow.inherit = False
    return s

def header(slide, title, sub=None, color=NAVY):
    band(slide, color, 1.15 if sub else 0.95)
    tf = _txt(slide, 0.5, 0.16, 12.3, 0.8)
    p = tf.paragraphs[0]; p.text = title
    p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = WHITE
    if sub:
        tf2 = _txt(slide, 0.5, 0.98, 12.3, 0.45)
        p2 = tf2.paragraphs[0]; p2.text = sub
        p2.font.size = Pt(14); p2.font.color.rgb = RGBColor(0xDD, 0xE6, 0xE8)

def bullet(tf, text, size=16, bold=False, color=INK, first=False, level=0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.text = text; p.level = level
    p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color
    return p

def add_image_fit(slide, path, l, t, max_w, max_h):
    """Add image scaled to fit within (max_w, max_h) inches, keeping aspect."""
    if not (path and os.path.exists(path)):
        tf = _txt(slide, l, t + max_h / 2 - 0.3, max_w, 0.6)
        p = tf.paragraphs[0]; p.text = "[plot not found — run the analysis first]"
        p.font.size = Pt(12); p.font.italic = True; p.font.color.rgb = GREY
        p.alignment = PP_ALIGN.CENTER
        return
    from PIL import Image
    iw, ih = Image.open(path).size
    ar = iw / ih
    w, h = max_w, max_w / ar
    if h > max_h:
        h, w = max_h, max_h * ar
    l2 = l + (max_w - w) / 2
    t2 = t + (max_h - h) / 2
    slide.shapes.add_picture(path, Inches(l2), Inches(t2), Inches(w), Inches(h))

def fmt(v, suf="", nd=2):
    return f"{v:+.{nd}f}{suf}" if isinstance(v, (int, float)) else "—"


# ── load ──────────────────────────────────────────────────────────────────────

summary_path = os.path.join(RES, f"{PREFIX}_summary.json")
if not os.path.exists(summary_path):
    sys.exit(f"No summary at {summary_path} — run run_robinson.py --prefix {PREFIX} first.")
summary = json.load(open(summary_path))
if not summary:
    sys.exit(f"{summary_path} is empty.")

any_aligned = any(r.get("gravity_aligned") for r in summary)


# ── title slide ───────────────────────────────────────────────────────────────

s = prs.slides.add_slide(BLANK)
band(s, NAVY, 7.5)                      # full navy background (slide height, inches)
tf = _txt(s, 0.9, 2.4, 11.5, 1.6)
p = tf.paragraphs[0]; p.text = f"{PREFIX} — Slope Estimation"
p.font.size = Pt(40); p.font.bold = True; p.font.color.rgb = WHITE
p2 = tf.add_paragraph()
p2.text = ("VGGT reconstruction  →  Method 1 (ground-distance)  →  "
           "gravity-aligned to ground-truth camera poses")
p2.font.size = Pt(18); p2.font.color.rgb = TEAL
p3 = tf.add_paragraph()
p3.text = f"{len(summary)} flight(s):  " + ",  ".join(r["dataset"] for r in summary)
p3.font.size = Pt(14); p3.font.color.rgb = RGBColor(0xCC, 0xD5, 0xD7)


# ── method slide ──────────────────────────────────────────────────────────────

s = prs.slides.add_slide(BLANK)
header(s, "Method", "Why the estimate is trustworthy")
tf = _txt(s, 0.6, 1.35, 12.1, 5.7)
bullet(tf, "1.  VGGT reconstructs a 3-D point cloud + camera poses from the frames.",
       size=17, bold=True, color=NAVY, first=True)
bullet(tf, "2.  Method 1 reads the road surface beneath each camera and fits ground "
           "elevation vs. along-track distance — the line's tilt is the slope.", size=15)
bullet(tf, "3.  Gravity alignment (key fix):", size=17, bold=True, color=NAVY)
bullet(tf, "VGGT's reconstruction is only defined up to a similarity transform and is "
           "exported in camera-0's frame — its vertical axis is NOT gravity.", size=14, level=1)
bullet(tf, "So the raw slope MAGNITUDE is measured in a tilted frame and can be far off "
           "(a true 25° slope can read ~65°).", size=14, level=1, color=RUST)
bullet(tf, "Fix: align the VGGT camera centres to the ground-truth camera world positions "
           "(encoded in each frame's filename), recover the missing rotation, rotate the "
           "terrain into a true-vertical frame, then measure slope.", size=14, level=1, color=TEAL)
bullet(tf, "This corrects both the magnitude and the sign from one physical anchor. A small "
           "alignment residual (align_resid_m) confirms the fit is trustworthy.", size=14, level=1)


# ── per-flight slides ─────────────────────────────────────────────────────────

for r in summary:
    name = r["dataset"]
    final = r.get("final_signed_deg")
    direction = r.get("direction", "")
    s = prs.slides.add_slide(BLANK)
    hue = TEAL if (isinstance(final, (int, float)) and final >= 0) else RUST
    header(s, f"{name}", f"Final slope  {fmt(final,'°')}  {direction}", color=hue)

    # plots: profile (left) + segments (right)
    prof = os.path.join(RES, f"{PREFIX}_{name}_method1_profile.png")
    segs = os.path.join(RES, f"{PREFIX}_{name}_method1_segments.png")
    add_image_fit(s, prof, 0.4, 1.35, 6.2, 3.9)
    add_image_fit(s, segs, 6.8, 1.35, 6.2, 3.9)

    # stats strip
    tf = _txt(s, 0.5, 5.4, 12.3, 2.0)
    aligned = r.get("gravity_aligned")
    conf = (r.get("confidence") or "").lower()
    conf_color = {"high": TEAL, "medium": RGBColor(0xC9, 0x9A, 0x2E), "low": RUST}.get(conf, GREY)
    # honest label: only say "gravity-aligned" when it actually was
    label = "Final (gravity-aligned)" if aligned else "Final (raw VGGT frame, UNCORRECTED)"
    line1 = (f"{label}: {fmt(final,'°')} {direction}    |    R² = {fmt(r.get('method1_r2'),'',3)}"
             f"    |    ground pts = {r.get('n_ground_found','—')}")
    if r.get("plane_slope_deg") is not None:
        line1 += f"    |    2-D max grade = {fmt(r.get('plane_slope_deg'),'°')}"
    bullet(tf, line1, size=14, bold=True, color=NAVY, first=True)
    if conf:
        bullet(tf, f"Confidence: {conf.upper()} — " + "; ".join(r.get("confidence_reasons", [])),
               size=12, bold=True, color=conf_color)
    if aligned:
        rm = f" ({r.get('rise_m')} m rise / {r.get('run_m')} m run)" if r.get("run_m") else ""
        bullet(tf, f"Gravity: {r.get('align_source')} via {r.get('align_method')} — residual "
                   f"{r.get('align_resid_m')} m; raw VGGT value was {fmt(r.get('method1_raw_vggt_deg'),'°')}"
                   f" in the tilted frame{rm}.", size=12, color=GREY)
    else:
        bullet(tf, f"Gravity alignment NOT applied ({r.get('align_note')}). Number is in VGGT's "
                   f"tilted frame with sign from the folder label — treat with caution.",
               size=12, color=RUST)


# ── summary table slide ───────────────────────────────────────────────────────

s = prs.slides.add_slide(BLANK)
header(s, "Summary", "Gravity-aligned vs. raw VGGT-frame")
cols = ["Flight", "Final", "Dir", "Raw VGGT", "R²", "2-D grade", "Aligned?", "Confidence"]
rows = len(summary) + 1
tbl = s.shapes.add_table(rows, len(cols), Inches(0.6), Inches(1.5),
                         Inches(12.1), Inches(0.5 + 0.55 * len(summary))).table
for j, c in enumerate(cols):
    cell = tbl.cell(0, j); cell.text = c
    cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
    para = cell.text_frame.paragraphs[0]
    para.font.size = Pt(12); para.font.bold = True; para.font.color.rgb = WHITE
conf_rgb = {"high": TEAL, "medium": RGBColor(0xC9, 0x9A, 0x2E), "low": RUST}
for i, r in enumerate(summary, start=1):
    conf = (r.get("confidence") or "—")
    vals = [
        r["dataset"],
        fmt(r.get("final_signed_deg"), "°"),
        r.get("direction", "—"),
        fmt(r.get("method1_raw_vggt_deg"), "°"),
        fmt(r.get("method1_r2"), "", 3),
        fmt(r.get("plane_slope_deg"), "°"),
        "yes" if r.get("gravity_aligned") else "no",
        conf.upper(),
    ]
    for j, v in enumerate(vals):
        cell = tbl.cell(i, j); cell.text = str(v)
        para = cell.text_frame.paragraphs[0]
        para.font.size = Pt(11); para.font.color.rgb = INK
        if j == 1:
            para.font.bold = True
            para.font.color.rgb = TEAL if str(v).startswith("+") else RUST
        if j == 7:
            para.font.bold = True
            para.font.color.rgb = conf_rgb.get(conf.lower(), GREY)

note = _txt(s, 0.6, 1.6 + 0.55 * len(summary) + 0.4, 12.1, 1.4)
bullet(note, "Final = gravity-aligned Method 1 (magnitude + sign from ground-truth poses). "
             "Raw VGGT = pre-alignment value in the tilted reconstruction frame, shown for "
             "transparency. A small residual means the pose alignment is reliable.",
       size=13, color=GREY, first=True)
if not any_aligned:
    bullet(note, "⚠  No flight in this run was gravity-aligned — numbers are raw VGGT-frame. "
                 "Re-run with the dataset path + flags so the poses can be read.",
           size=13, bold=True, color=RUST)


out = os.path.join(OUT_DIR, f"{PREFIX}_report.pptx")
prs.save(out)
print(f"Saved → {out}   ({len(prs.slides)} slides)")
BUILD_EOF
echo ">>> deck builder refreshed"

# ---- regenerate corrected numbers + plots (no GPU) and rebuild the deck ----
echo ">>> Re-analyzing N75E_type3..."
python run_robinson.py N75E_0712/type3_front_N75E --from-glb --multi --markers --prefix N75E_type3 \
    || { echo ">>> analysis failed"; exit 1; }
python -c "import pptx" 2>/dev/null || pip install python-pptx
python build_type3_slides.py N75E_type3 || { echo ">>> slide build failed"; exit 1; }

echo ">>> DONE. Deck at: /mnt/sdb/vatsal/Slope/slides/N75E_type3_report.pptx"
echo ">>> For the sagamore before/after story (alignment should FIRE there):"
echo ">>>   python run_robinson.py sagamore_0708 --from-glb --multi --markers --prefix sagamore"
echo ">>>   python build_type3_slides.py sagamore"
