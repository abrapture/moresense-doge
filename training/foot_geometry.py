"""Distances of the calibrated lower-leg collision shapes to a horizontal floor.

These are geometric distances, not shape-pair contact forces. Body link poses
must come from the physics backend, with quaternions in Isaac Lab 3 xyzw order.
"""
import math
import xml.etree.ElementTree as ET

import torch


def _vec(value, default):
    values = [float(v) for v in (value if value is not None else default).split()]
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError('Invalid collision origin or box size')
    return values


def _rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
            [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
            [-sp, cp*sr, cp*cr]]


def _quaternion_matrix(q):
    # xyzw; normalize to avoid geometry errors from accumulated roundoff.
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), dim=-1).reshape(*q.shape[:-1], 3, 3)


class FootGeometry:
    """Load four knee-link collision shapes and map them to body indices.

    ``heights(body_link_pose_w)`` returns (feet, boxes), each shaped [N, 4],
    with signed minimum Z above a world-horizontal plane at z=0, in metres.
    Boxes is the minimum of all lower-leg boxes per leg. Negative is penetration.
    """
    def __init__(self, urdf_path, body_names):
        root = ET.parse(urdf_path).getroot()
        links = {link.get('name'): link for link in root.findall('link')}
        self.legs = []
        for i in range(1, 5):
            name = f'leg_{i}_knee_link'
            if name not in links or body_names.count(name) != 1:
                raise ValueError(f'Missing or ambiguous body {name}')
            shapes = []
            for collision in links[name].findall('collision'):
                origin = collision.find('origin')
                xyz = _vec(origin.get('xyz') if origin is not None else None, '0 0 0')
                rpy = _vec(origin.get('rpy') if origin is not None else None, '0 0 0')
                geometry = collision.find('geometry')
                if geometry is None:
                    raise ValueError(f'Missing geometry for {name}')
                cylinder, box = geometry.find('cylinder'), geometry.find('box')
                if cylinder is not None:
                    dims = [float(cylinder.get('radius')), float(cylinder.get('length'))]
                    kind = 'foot'
                    if not collision.get('name', '').startswith('foot_'):
                        raise ValueError(f'Unexpected cylinder in {name}')
                elif box is not None:
                    dims = _vec(box.get('size'), '')
                    kind = 'box'
                else:
                    raise ValueError(f'Unsupported collision geometry in {name}')
                if not all(math.isfinite(v) and v > 0 for v in dims):
                    raise ValueError(f'Invalid collision dimensions in {name}')
                shapes.append((kind, xyz, _rpy_matrix(rpy), dims))
            if sum(s[0] == 'foot' for s in shapes) != 1 or not any(s[0] == 'box' for s in shapes):
                raise ValueError(f'Expected one foot cylinder and at least one box in {name}')
            self.legs.append((body_names.index(name), shapes))

    def heights(self, body_link_pose_w):
        if body_link_pose_w.ndim != 3 or body_link_pose_w.shape[-1] != 7:
            raise ValueError('Expected [num_envs, num_bodies, 7] body link poses')
        feet, boxes = [], []
        for index, shapes in self.legs:
            pose = body_link_pose_w[:, index, :]
            rotation = _quaternion_matrix(pose[:, 3:])
            foot_heights, box_heights = [], []
            for kind, xyz, local_rotation, dims in shapes:
                center = pose[:, :3] + rotation @ pose.new_tensor(xyz)
                world_rotation = rotation @ pose.new_tensor(local_rotation)
                if kind == 'foot':
                    axis_z = world_rotation[:, 2, 2].clamp(-1, 1)
                    extent = axis_z.abs()*dims[1]/2 + dims[0]*torch.sqrt((1-axis_z.square()).clamp_min(0))
                    foot_heights.append(center[:, 2] - extent)
                else:
                    extent = (world_rotation[:, 2, :].abs() * pose.new_tensor(dims)).sum(dim=-1)/2
                    box_heights.append(center[:, 2] - extent)
            feet.append(torch.stack(foot_heights, dim=1).min(dim=1).values)
            boxes.append(torch.stack(box_heights, dim=1).min(dim=1).values)
        return torch.stack(feet, dim=1), torch.stack(boxes, dim=1)
