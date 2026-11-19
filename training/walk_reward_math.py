"""Foot geometry and soft rewards; lengths in metres, quaternions xyzw."""
import math
import xml.etree.ElementTree as ET
import torch


def quaternion_matrix(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def _rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return torch.tensor([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                         [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                         [-sp, cp*sr, cp*cr]], dtype=torch.float64)


def _xyz(element, attribute, default):
    v = [float(x) for x in (element.get(attribute, default) if element is not None else default).split()]
    if len(v) != 3 or not all(math.isfinite(x) for x in v):
        raise ValueError('Expected three finite URDF values')
    return v


def nominal_feet_from_urdf(path, positions):
    """FK foot cylinder centers relative to base_link in saved stance.

    Returns (4,3) nominal centers and (4,3) local offsets in leg_1..4 order.
    """
    root = ET.parse(path).getroot()
    links = {e.get('name'): e for e in root.findall('link')}
    parents = {}
    for joint in root.findall('joint'):
        child, parent = joint.find('child'), joint.find('parent')
        if child is None or parent is None or child.get('link') in parents:
            raise ValueError('Invalid or multiply-parented URDF joint')
        parents[child.get('link')] = (parent.get('link'), joint)
    centers, local = [], []
    for leg in range(1, 5):
        name = f'leg_{leg}_knee_link'
        foots = [c for c in links[name].findall('collision') if c.get('name', '').startswith('foot_') and c.find('geometry/cylinder') is not None]
        if len(foots) != 1:
            raise ValueError(f'{name} requires one labelled cylinder')
        xyz = torch.tensor(_xyz(foots[0].find('origin'), 'xyz', '0 0 0'), dtype=torch.float64)
        local.append(xyz)
        chain, cursor = [], name
        while cursor != 'base_link':
            if cursor not in parents or len(chain) > len(parents):
                raise ValueError(f'{name} disconnected from base_link')
            cursor, joint = parents[cursor]
            chain.append(joint)
        r = torch.eye(3, dtype=torch.float64)
        p = torch.zeros(3, dtype=torch.float64)
        for joint in reversed(chain):
            origin = joint.find('origin')
            p = p + r @ torch.tensor(_xyz(origin, 'xyz', '0 0 0'), dtype=torch.float64)
            r = r @ _rpy_matrix(_xyz(origin, 'rpy', '0 0 0'))
            kind = joint.get('type')
            if kind in ('revolute', 'continuous'):
                name_j = joint.get('name')
                if name_j not in positions:
                    raise ValueError(f'Missing stance position: {name_j}')
                angle = float(positions[name_j])
                if not math.isfinite(angle):
                    raise ValueError(f'Nonfinite stance position: {name_j}')
                axis = torch.tensor(_xyz(joint.find('axis'), 'xyz', '1 0 0'), dtype=torch.float64)
                axis_norm = torch.linalg.vector_norm(axis)
                if axis_norm < 1e-9:
                    raise ValueError(f'Invalid joint axis: {name_j}')
                x, y, z = axis / axis_norm
                K = torch.stack((x*0, -z, y, z, x*0, -x, -y, x, x*0)).reshape(3, 3)
                r = r @ (torch.eye(3, dtype=torch.float64) + math.sin(angle)*K + (1-math.cos(angle))*(K @ K))
            elif kind != 'fixed':
                raise ValueError(f'Unsupported joint type: {kind}')
        centers.append(p + r @ xyz)
    return torch.stack(centers).float(), torch.stack(local).float()


def foot_centers_in_base(root_pose, body_link_pose, body_indices, local_centers):
    """Return [N,4,3] cylinder centers in current base frame."""
    if root_pose.ndim != 2 or root_pose.shape[-1] != 7:
        raise ValueError('Expected root poses [N,7]')
    poses = body_link_pose[:, body_indices, :]
    rot_legs = quaternion_matrix(poses[..., 3:])
    local = local_centers.to(device=poses.device, dtype=poses.dtype)
    world = poses[..., :3] + (rot_legs @ local[None, :, :, None]).squeeze(-1)
    rot_root = quaternion_matrix(root_pose[:, 3:])
    return (rot_root.transpose(-1, -2)[:, None] @ (world - root_pose[:, None, :3])[..., None]).squeeze(-1)


def expanded_footprint(nominal, width=0.010, length=0.010):
    if nominal.shape != (4, 3) or width < 0 or length < 0:
        raise ValueError('Expected four stance feet and nonnegative offsets')
    result = nominal.clone()
    result[:, 0] += torch.sign(nominal[:, 0]) * width
    result[:, 1] += torch.sign(nominal[:, 1]) * length
    return result


def footprint_penalty(feet, target, command_y, *, stand_deadband=0.008, move_deadband=0.025, scale=0.05):
    """Per-robot soft XY penalty: walking gets a wider corridor."""
    stand = command_y.abs() < 1e-5
    deadband = torch.where(stand, stand_deadband, move_deadband)[:, None, None]
    excess = ((feet[..., :2] - target[None, :, :2]).abs() - deadband).clamp_min(0)
    return (excess / scale).square().mean(dim=(1, 2)).clamp_max(4.0)


def symmetry_penalty(feet, target, command_y):
    """Left/right residuals vs nominal, keeping measured mechanical asymmetry."""
    err = feet[..., :2] - target[None, :, :2]
    residual = torch.stack((err[:, 0, 0] + err[:, 2, 0],
                            err[:, 1, 0] + err[:, 3, 0],
                            err[:, 0, 1] - err[:, 2, 1],
                            err[:, 1, 1] - err[:, 3, 1]), dim=-1)
    deadband = torch.where(command_y.abs() < 1e-5, 0.015, 0.05)[:, None]
    return (((residual.abs()-deadband).clamp_min(0) / 0.05).square().mean(dim=-1)).clamp_max(4.0)


def heading_error(root_quat_xyzw, reference_quat_xyzw):
    """Signed world heading deviation with local +Y as forward."""
    direction = quaternion_matrix(root_quat_xyzw)[..., :2, 1]
    reference = quaternion_matrix(reference_quat_xyzw)[..., :2, 1]
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-6)
    reference = reference / torch.linalg.vector_norm(reference, dim=-1, keepdim=True).clamp_min(1e-6)
    cross = reference[..., 0] * direction[..., 1] - reference[..., 1] * direction[..., 0]
    dot = (reference * direction).sum(dim=-1)
    return torch.atan2(cross, dot)


def terminal_fall_penalty(terminated, weight):
    """Once per actual termination, no dt scaling; timeouts excluded by caller."""
    if weight < 0 or not math.isfinite(weight):
        raise ValueError('Fall penalty weight must be finite and nonnegative')
    return terminated.to(dtype=torch.float32) * weight
