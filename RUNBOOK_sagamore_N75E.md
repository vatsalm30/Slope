# Runbook — slopes for `sagamore_0708` and `N75E_0712`

The only GPU-intensive step is the VGGT reconstruction. Run everything below on the
**A100** (in this repo dir, with the torch/vggt env active). Method 1, the pose
anchor and plots are light and run in the same commands.

Note on the anchor: these are rendered/sim frames — there is **no GPS**. Each frame's
filename carries exact **ground-truth pose** (`null_<ts>_<idx>_<height>_<X>_<Y>_…`),
which is a better gravity reference than GPS. The code reads it directly.

VGGT-1B weights auto-download from HuggingFace on first run.

## 1. Sagamore — 6 marker flights (1 sharp frame per marker)

```bash
python run_robinson.py sagamore_0708 --multi --markers --prefix sagamore
```

Processes all 6 flights. Each reconstructs from 5–12 images (one per marker).

- Pose **height is flat** on every sagamore flight (fixed-altitude & terrain-
  following both hold height), so the sign comes from the folder label
  (`_uphill`/`_downhill`) and the **magnitude comes from VGGT Method 1**.
- Ground truth is in the folder name: `slope_4…` ≈ 4°, `slope_25…` ≈ 25°.
  Compare `final_signed_deg` against that.
- Watch `method1_r2` in each JSON — fixed-altitude flights can rebuild nearly flat
  (low R²). If so, that flight's geometry didn't give VGGT enough parallax.

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

Tweak Method 1 / plots without re-reconstructing:

```bash
python run_robinson.py --from-glb --prefix sagamore
python run_robinson.py --from-glb --prefix N75E
python run_robinson.py --from-glb --prefix N75E_type3
```

## Outputs (per flight, in `results/`)

- `<prefix>_<flight>_method1.json`      — final signed slope + per-segment table
- `<prefix>_<flight>_method1_profile.png`, `..._segments.png`
- `<prefix>_<flight>_gps_track.png`     — pose-height vs distance; only when the pose
                                          anchor is usable (N75E front flights)
- `<prefix>_summary.json`               — all flights for that run
- `output_glbs/vggt/<prefix>_<flight>.glb`

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
