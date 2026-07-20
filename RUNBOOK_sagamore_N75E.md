# Runbook — slopes for `sagamore_0708` and `N75E_0712`

The only GPU-intensive step is the VGGT reconstruction. Run everything below on the
**A100** (in this repo dir, with the torch/vggt env active). Method 1, the pose
anchor and plots are light and run in the same commands.

Note on the anchor: these are rendered/sim frames — there is **no GPS**. Each frame's
filename carries exact **ground-truth pose** (`null_<ts>_<idx>_<height>_<X>_<Y>_…`),
which is a better gravity reference than GPS. The code reads it directly.

**Gravity alignment (new).** VGGT reconstructs only up to a similarity transform and
exports the scene in camera-0's frame, so its `+Y` is *not* gravity — the raw Method 1
slope is measured in a tilted frame and its **magnitude is wrong** (e.g. a true 25°
slope read as ~65°). The pipeline now Umeyama-aligns the GLB camera centres to the
ground-truth poses, recovers the VGGT→gravity rotation, rotates the terrain into a
true-vertical frame, and only then measures slope. This corrects **both magnitude and
sign** from the poses. Each result reports `method1_signed_deg` (gravity-aligned),
`method1_raw_vggt_deg` (the old tilted value, for transparency), and `gravity_aligned`
/ `align_resid_m` (a small residual, ~<0.1 m, means a trustworthy alignment). If the
camera constellation is collinear/degenerate, alignment is skipped and the run falls
back to the old folder-label-sign behaviour (see `align_note`).

VGGT-1B weights auto-download from HuggingFace on first run.

## 1. Sagamore — 6 marker flights (1 sharp frame per marker)

```bash
python run_robinson.py sagamore_0708 --multi --markers --prefix sagamore
```

Processes all 6 flights. Each reconstructs from 5–12 images (one per marker).

- Pose **height is flat** on every sagamore flight, but the full 2-D camera
  positions (east/north) are enough to gravity-align, so magnitude **and** sign
  now come from the pose-aligned Method 1 — not the folder label. The label is
  only a fallback if the markers are collinear (alignment then skips).
- Ground truth is in the folder name: `slope_4…` ≈ 4°, `slope_25…` ≈ 25°.
  Compare `final_signed_deg` (should now be close) against that; `method1_raw_vggt_deg`
  is the old tilted value and will still look inflated.
- Watch `method1_r2` and `align_resid_m` in each JSON. Fixed-altitude flights can
  rebuild nearly flat (low R²) — if so, that flight's geometry didn't give VGGT
  enough parallax. A large `align_resid_m` means the pose alignment itself is weak.

## 2. N75E — 5 continuous flights (32 evenly-spaced frames each)

```bash
python run_robinson.py N75E_0712 --multi --max-frames 32 --prefix N75E
```

Processes `type1_front`, `type2_bird`, `type2_front`, `type2_front_harrison`,
`type2_front_run2_harrison`. `type3_front_N75E` has no top-level frames and is
auto-skipped here (handled in step 3).

- `type1_front` / `type2_front`: real terrain signal. Pose baseline −3.27° / −2.83°
  (R² 0.80 / 0.52). VGGT Method 1 should corroborate; sign anchored to pose height.
- `bird` (top-down) and both `harrison` flights (**different site**) have flat pose
  height → low-confidence, included per request.

## 3. N75E type3 — nested uphill/downhill markers

```bash
python run_robinson.py N75E_0712/type3_front_N75E --multi --markers --prefix N75E_type3
```

Reconstructs `uphill/` and `downhill/` (5 markers each). Flat GPS → sign from
folder label, magnitude from VGGT.

## Re-run analysis only (no GPU, after reconstructions exist)

Tweak Method 1 / plots without re-reconstructing. **Pass the dataset root + the same
`--multi`/`--markers`/`--max-frames` flags** so the analysis can find the source
images and gravity-align (the poses live in the filenames):

```bash
python run_robinson.py sagamore_0708 --from-glb --multi --markers --prefix sagamore
python run_robinson.py N75E_0712 --from-glb --multi --max-frames 32 --prefix N75E
python run_robinson.py N75E_0712/type3_front_N75E --from-glb --multi --markers --prefix N75E_type3
```

(Omitting the path still works but skips gravity alignment — you'd get the old
raw-VGGT-frame magnitudes.)

## Outputs (per flight, in `results/`)

- `<prefix>_<flight>_method1.json`      — final signed slope + per-segment table
- `<prefix>_<flight>_method1_profile.png`, `..._segments.png`
- `<prefix>_<flight>_gps_track.png`     — pose-height vs distance; only when the pose
                                          anchor is usable (N75E front flights)
- `<prefix>_summary.json`               — all flights for that run
- `output_glbs/vggt/<prefix>_<flight>.glb`

## Method notes (research-backed)

Recovering true **down** (gravity) is the crux — VGGT returns geometry in camera-0's
tilted frame, so slope read directly off it can be off by tens of degrees. The
pipeline resolves gravity in this priority:

1. **Pose Umeyama** (default) — align GLB camera centres to the filename poses.
   Needs a non-collinear camera constellation.
2. **External up-vector** (`up_hint` → `gravity_align`) — a per-image up-vector from
   **GeoCalib** (ECCV 2024, single-image up + intrinsics) or the drone **IMU/attitude**.
   One up-vector is enough to fix slope, so this rescues the straight-line / nadir
   flights that pose-alignment can't (e.g. N75E_type3). Not wired to a model yet —
   pass `up_hint` when you have GeoCalib/IMU up-vectors in the GLB frame.

**Acquisition matters more than the model.** Straight-line, fixed-altitude, nadir-only
flights are near-collinear/planar — a degeneracy that destroys roll (hence gravity and
slope). Per UAV-SfM literature (Nesbit & Hugenholtz 2019), fly a **cross-hatch/double
grid with oblique passes (20–35° off-nadir) and varied altitude**; that adds the
parallax needed to constrain gravity. Each result now reports `collinearity` (2nd/1st
camera singular value; <0.05 ⇒ degenerate) so you can reject/refly.

**Scale vs gravity are separable:** gravity fixes the *angle* and sign; metric scale
(from the pose Umeyama fit) only adds `rise_m`/`run_m`. For a slope angle you don't
need meters.

**New JSON fields:** `plane_slope_deg`/`aspect_deg` (full 2-D grade from a plane fit,
a cross-check on the along-track value), `collinearity`/`planarity`, `align_method`
(`pose-umeyama` | `up-hint`), `scale_m_per_unit`, `run_m`/`rise_m`, and a consolidated
`confidence` (high/medium/low) + `confidence_reasons`. Gate on `confidence`.

Not a drop-in for photogrammetry: on aerial blocks VGGT still lags COLMAP/MVS — treat
the pure-image slope as an estimate with an attached confidence, and keep a classical
SfM+MVS path as the accuracy reference for anything survey-grade.

## Notes / gotchas

- `sagamore_0708/slope_4_fixed_altitude_uphill/` contains a stray `_` folder; in
  `--markers` mode it becomes an extra "marker 0". Delete it for a cleaner run, or
  leave it (it contributes one frame).
- A gravity-true pose baseline for all flights (computed with no GPU) is already in
  `results/pose_preview_sagamore_N75E.json`.
- Pose is read from the **filename** (`null_<ts>_<idx>_<height>_<X>_<Y>_…`) — there
  is no GPS/EXIF. Handled in `gps_anchor.py` (kept that filename for the shared
  Robinson-EXIF path; JSON now includes an `anchor_source` field that self-labels
  each result as "sim ground-truth pose" or "EXIF GPS").
