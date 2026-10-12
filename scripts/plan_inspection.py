"""Plan the inspection flight around the turbine and estimate photo coverage.

    python scripts/plan_inspection.py --out assets/plan [--standoff 9] [--step 4]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turbine_inspection.coverage import (CameraSpec, Planner, PlannerConfig, coverage,  # noqa: E402
                                         route_json, route_stats)
from turbine_inspection.turbine import TurbineSpec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="assets/plan")
ap.add_argument("--standoff", type=float, default=9.0)
ap.add_argument("--step", type=float, default=4.0)
ap.add_argument("--hfov", type=float, default=50.0)
a = ap.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
spec = TurbineSpec()
cfg = PlannerConfig(standoff=a.standoff, step=a.step)
planner = Planner(spec, cfg)
route = planner.plan()
stats = route_stats(route, cfg)
min_clear = min(planner.clearance([w.x, w.y, w.z]) for w in route if w.segment not in ("takeoff", "land"))
report, detail = coverage(route, spec, CameraSpec(hfov_deg=a.hfov))
(out / "waypoints.json").write_text(json.dumps(route_json(route), indent=1))
(out / "coverage.json").write_text(json.dumps({"route": stats, "min_clearance_m": round(min_clear, 2),
                                              "coverage": report}, indent=2))
np.savez_compressed(out / "coverage_samples.npz", **{k: v for k, v in detail.items()})
print(json.dumps(stats))
print("min clearance to the turbine on the route:", round(min_clear, 2), "m")
for k, v in report.items():
    print(f"  {k:8s} coverage {v['coverage']*100:5.1f}%  median GSD {v['median_gsd_mm']} mm/px  views {v['mean_views']}")
