"""Build separate convex collision meshes from the original CAD visual meshes.

Requires coacd 1.0.14, trimesh and numpy. The threshold is in metres.
"""
from pathlib import Path
import copy
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import coacd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'urdf' / 'moresense_doge.urdf'
OUT = Path(os.environ['COACD_OUTPUT_DIR']) if 'COACD_OUTPUT_DIR' in os.environ else ROOT / 'meshes' / 'coacd_05mm'
EXCLUDED = {'gearing.stl', 'MOTOR_1723_3.stl', 'pcb.stl'}
THRESHOLD = 0.0005

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(SOURCE)
    names = sorted({Path(m.attrib['filename']).name for m in tree.findall('.//visual/geometry/mesh')} - EXCLUDED)
    coacd.set_log_level('error')
    report = {}
    selected = names if len(sys.argv) == 1 else [n for n in names if Path(n).stem in sys.argv[1:]]
    for name in selected:
        directory = OUT / Path(name).stem
        directory.mkdir(exist_ok=True)
        marker = directory / 'complete.json'
        if marker.exists():
            report[name] = json.loads(marker.read_text())
            print('cached', name, report[name]['hulls'], flush=True)
            continue
        t = time.monotonic()
        mesh = trimesh.load_mesh(ROOT / 'meshes' / name, process=True)
        assert isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) and np.isfinite(mesh.vertices).all()
        # A mesh already proven convex needs no approximation or decomposition.
        already_convex = bool(mesh.is_convex and mesh.is_watertight)
        parts = [(mesh.vertices, mesh.faces)] if already_convex else coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=THRESHOLD, real_metric=True, seed=0)
        assert parts
        repaired = []
        for i, (vertices, faces) in enumerate(parts):
            hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            if not (len(hull.faces) and hull.is_watertight and hull.is_convex and hull.volume > 0):
                # CoACD occasionally emits triangles with numerical degeneracies.
                # Re-hull the same points; fail explicitly if no valid 3D solid exists.
                try:
                    hull = hull.convex_hull
                    repaired.append(i)
                except Exception as exc:
                    raise RuntimeError(f'{name} hull {i}: cannot form convex solid') from exc
                if not (len(hull.faces) and hull.is_watertight and hull.is_convex and hull.volume > 0):
                    raise RuntimeError(f'{name} hull {i}: invalid after re-hull; volume={hull.volume}, watertight={hull.is_watertight}, convex={hull.is_convex}')
            hull.export(directory / f'{i:04d}.stl')
        report[name] = {'hulls': len(parts), 'method': 'exact_convex' if already_convex else 'CoACD', 'repaired_hulls': repaired, 'source_bounds_m': mesh.bounds.tolist(), 'source_watertight': bool(mesh.is_watertight), 'seconds': round(time.monotonic()-t, 1)}
        marker.write_text(json.dumps(report[name], indent=2))
        print('done', name, report[name], flush=True)
    if len(selected) != len(names):
        return
    total = 0
    for link in tree.findall('link'):
        for collision in link.findall('collision'):
            link.remove(collision)
        for visual in link.findall('visual'):
            mesh = visual.find('geometry/mesh')
            if mesh is None:
                continue
            name = Path(mesh.attrib['filename']).name
            if name in EXCLUDED:
                continue
            assert mesh.attrib.get('scale', '1 1 1') == '1 1 1'
            directory = OUT / Path(name).stem
            for i in range(report[name]['hulls']):
                collision = ET.SubElement(link, 'collision', {'name': f'coacd_{total:05d}'})
                origin = visual.find('origin')
                if origin is not None:
                    collision.append(copy.deepcopy(origin))
                geometry = ET.SubElement(collision, 'geometry')
                ET.SubElement(geometry, 'mesh', {'filename': f'../meshes/coacd_05mm/{Path(name).stem}/{i:04d}.stl'})
                total += 1
    ET.indent(tree, space='  ')
    tree.write(ROOT / 'urdf' / 'moresense_doge_coacd.urdf', encoding='utf-8', xml_declaration=True)
    (ROOT / 'coacd_report.json').write_text(json.dumps({'threshold_m': THRESHOLD, 'real_metric': True, 'excluded_internal_meshes': sorted(EXCLUDED), 'collision_instances': total, 'parts': report}, indent=2))
    print('collision instances', total, flush=True)

if __name__ == '__main__':
    main()
