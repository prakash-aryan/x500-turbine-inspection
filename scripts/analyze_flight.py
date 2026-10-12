"""Analyse an inspection flight: coverage from the actual photos, defect check, fused 3D point cloud.

    uv run python scripts/analyze_flight.py RUN_DIR [--gsd-limit 8]
Writes RUN_DIR/analysis/{report.json, coverage.npz, defects/*.jpg, model.ply}.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turbine_inspection import analysis  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--gsd-limit", type=float, default=8.0, help="mm/px counted as sharp enough")
ap.add_argument("--voxel", type=float, default=0.05)
a = ap.parse_args()
out = Path(a.run_dir) / "analysis"
out.mkdir(parents=True, exist_ok=True)
photos = analysis.load_photos(str(Path(a.run_dir) / "photos"))
print(f"{len(photos)} photos")
samples = dict(np.load(ROOT / "assets" / "plan" / "coverage_samples.npz"))
cov, views, best = analysis.coverage_from_photos(photos, samples, a.gsd_limit)
np.savez(out / "coverage.npz", points=samples["points"], labels=samples["labels"], views=views, gsd=best)
gt = json.load(open(ROOT / "assets" / "scene" / "defects_ground_truth.json"))
defects = analysis.defect_check(photos, gt["defects"], crop_dir=str(out / "defects"))
pts, cols = analysis.fuse_point_cloud(photos, voxel=a.voxel)
analysis.write_ply(str(out / "model.ply"), pts, cols)
report = {"photos": len(photos), "coverage": cov, "defects": defects, "model_points": int(len(pts))}
json.dump(report, open(out / "report.json", "w"), indent=2)
print(json.dumps(report, indent=2))
