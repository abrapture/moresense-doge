"""Create a separate URDF with hand-designed, link-local primitive collision geometry.

Usage: python generate_simple_collisions.py [proposal.json] [output.urdf]
Lengths are metres, orientations are URDF roll/pitch/yaw radians. Visual geometry,
inertia and joint definitions are copied from the original unchanged.
"""
from pathlib import Path
import argparse
import copy
import json
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'urdf' / 'moresense_doge.urdf'


def vector(value, length, label, positive=False):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f'{label}: expected {length} values')
    result = [float(v) for v in value]
    if not all(math.isfinite(v) and (not positive or v > 0) for v in result):
        raise ValueError(f'{label}: invalid values {value}')
    return result


def fmt(values):
    return ' '.join(f'{v:.12g}' for v in values)


def strip_collisions(root):
    result = copy.deepcopy(root)
    for link in result.findall('link'):
        for collision in link.findall('collision'):
            link.remove(collision)
    # Whitespace indentation changes on serialisation; normalise before comparison.
    ET.indent(result, space='  ')
    return ET.tostring(result, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('proposal', nargs='?', type=Path, default=ROOT / 'simple_collision_proposals.json')
    parser.add_argument('output', nargs='?', type=Path, default=ROOT / 'urdf' / 'moresense_doge_simple.urdf')
    args = parser.parse_args()
    source = SOURCE.resolve()
    output = args.output.resolve()
    if output == source:
        raise ValueError('Refusing to overwrite source URDF')
    data = json.loads(args.proposal.read_text(encoding='utf-8'))
    entries = data['links']
    if not isinstance(entries, dict):
        raise ValueError('links must be an object')
    tree = ET.parse(source)
    root = tree.getroot()
    original = strip_collisions(root)
    links = {link.attrib['name']: link for link in root.findall('link')}
    if set(entries) != set(links):
        raise ValueError(f'Link mismatch: missing={set(links)-set(entries)}, unknown={set(entries)-set(links)}')
    count = 0
    for name, link in links.items():
        for collision in link.findall('collision'):
            link.remove(collision)
        shapes = entries[name]
        if not isinstance(shapes, list):
            raise ValueError(f'{name}: collision entries must be a list')
        for index, shape in enumerate(shapes):
            typ = shape['type']
            xyz = vector(shape['xyz'], 3, f'{name} xyz')
            rpy = vector(shape.get('rpy', [0, 0, 0]), 3, f'{name} rpy')
            if typ == 'cylinder' and not name.endswith('_knee_link'):
                raise ValueError(f'{name}: cylinders are reserved for the feet on knee links')
            prefix = 'foot' if typ == 'cylinder' else 'simple'
            collision = ET.SubElement(link, 'collision', name=f'{prefix}_{name}_{index:02d}')
            ET.SubElement(collision, 'origin', xyz=fmt(xyz), rpy=fmt(rpy))
            geom = ET.SubElement(collision, 'geometry')
            if typ == 'box':
                size = vector(shape['size'], 3, f'{name} size', positive=True)
                ET.SubElement(geom, 'box', size=fmt(size))
            elif typ == 'cylinder':
                radius = vector([shape['radius']], 1, f'{name} radius', positive=True)[0]
                length = vector([shape['length']], 1, f'{name} length', positive=True)[0]
                ET.SubElement(geom, 'cylinder', radius=fmt([radius]), length=fmt([length]))
            else:
                raise ValueError(f'{name}: unsupported shape {typ!r}')
            count += 1
    if strip_collisions(root) != original:
        raise AssertionError('Non-collision URDF content changed')
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space='  ')
    tree.write(output, encoding='utf-8', xml_declaration=True)
    reloaded = ET.parse(output).getroot()
    if strip_collisions(reloaded) != original or len(reloaded.findall('.//collision')) != count:
        raise AssertionError('Saved URDF failed integrity check')
    print(f'{output}: {len(links)} links, {count} collision primitives; joints, inertia and visuals preserved')


if __name__ == '__main__':
    main()
