"""Build the wind turbine inspection scene (USD), its textures and the defect ground truth.

    python scripts/build_scene.py --out assets/scene [--hdri /path/to/sky.hdr] [--no-farm]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turbine_inspection.scene import build_scene  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="assets/scene")
ap.add_argument("--hdri", default=None, help="equirectangular sky image for the dome light (optional)")
ap.add_argument("--no-farm", action="store_true", help="only the inspected turbine, no neighbours")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
res = build_scene(a.out, hdri=a.hdri, farm=not a.no_farm, seed=a.seed)
print("scene:", res["usd"])
print("ground truth:", res["ground_truth"])
for d in res["defects"]:
    print(f"  {d.id} {d.kind:22s} {d.component:8s} centroid={d.centroid} size={d.size_m}")
