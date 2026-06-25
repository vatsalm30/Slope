import sys, numpy as np
sys.path.insert(0, "/mnt/sdb/vatsal/Slope")
import method1_slope as m1

GLB = "/mnt/sdb/vatsal/Slope/output_glbs/vggt"
for nm in ["robinson_downhill", "robinson_uphill"]:
    terrain, cameras = m1.load_glb(f"{GLB}/{nm}.glb")
    sw = float(np.ptp(terrain[:, [0, 2]]))
    print(f"\n=== {nm} ===")
    print(f"  terrain pts {len(terrain)}   scene width {sw:.3f}")
    print(f"  terrain Y [{terrain[:,1].min():.3f}, {terrain[:,1].max():.3f}]  (+Y = down)")
    print(f"  camera  Y {np.round(cameras[:,1],3).tolist()}")
    for frac in [0.08, 0.15, 0.25, 0.40, 0.60]:
        r = sw * frac
        cnt = [int((np.linalg.norm(terrain[:,[0,2]] - c[[0,2]], axis=1) < r).sum())
                for c in cameras]
        print(f"  radius {frac:.2f} ({r:.2f} m): pts/cam {cnt}")