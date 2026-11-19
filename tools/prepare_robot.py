"""Prepare this Onshape export without inventing measured dynamics.

Requires numpy. Original CAD zero positions and root frame are preserved.
"""
from pathlib import Path
import argparse
import collections
import copy
import json
import math
import re
import shutil
import struct
import xml.etree.ElementTree as ET
import numpy as np

CHAINS = {
    'leg_1': ['revolute_1', 'revolute_1_8', 'revolute_2_10'],
    'leg_2': ['revolute_1_11', 'revolute_1_9', 'revolute_2_4'],
    'leg_3': ['revolute_1_7', 'revolute_2_6', 'revolute_2_5'],
    'leg_4': ['revolute_1_3', 'revolute_2_2', 'revolute_2_1'],
}
STAGES = ['shoulder', 'hip', 'knee']
SERVO_MESHES = {'gearing', 'MOTOR_1723_3', 'motor_body', 'motor_bottom_cover',
                'Motor_upper_Cover', 'pcb', 'roll1', 'roll2'}

def vec(s):
    return np.array([float(x) for x in s.split()])

def fmt(x):
    return ' '.join(f'{float(v):.12g}' for v in np.asarray(x).flat)

def rotation(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])

def rpy(R):
    p = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    if abs(math.cos(p)) > 1e-8:
        return np.array([math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])])
    return np.array([math.atan2(-R[1, 2], R[1, 1]), p, 0.0])

def transform(e):
    T = np.eye(4)
    if e is not None:
        T[:3, :3] = rotation(vec(e.get('rpy', '0 0 0')))
        T[:3, 3] = vec(e.get('xyz', '0 0 0'))
    return T

def inverse(T):
    I = np.eye(4)
    I[:3, :3] = T[:3, :3].T
    I[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return I

def put_origin(parent, T):
    e = parent.find('origin')
    if e is None:
        e = ET.SubElement(parent, 'origin')
    e.set('xyz', fmt(T[:3, 3]))
    e.set('rpy', fmt(rpy(T[:3, :3])))
    assert np.max(np.abs(rotation(rpy(T[:3, :3])) - T[:3, :3])) < 1e-7
    return e

def tensor(e):
    return np.array([[float(e.get('ixx')), float(e.get('ixy')), float(e.get('ixz'))],
                     [float(e.get('ixy')), float(e.get('iyy')), float(e.get('iyz'))],
                     [float(e.get('ixz')), float(e.get('iyz')), float(e.get('izz'))]])

def tensor_attrs(I):
    return {k: f'{I[a,b]:.12g}' for k, a, b in
            [('ixx',0,0), ('ixy',0,1), ('ixz',0,2), ('iyy',1,1), ('iyz',1,2), ('izz',2,2)]}

def stl_vertices(path):
    b = Path(path).read_bytes()
    n = struct.unpack_from('<I', b, 80)[0]
    assert len(b) == 84 + 50*n
    dtype = np.dtype([('normal','<f4',(3,)), ('v','<f4',(3,3)), ('attr','<u2')])
    return np.frombuffer(b, dtype=dtype, offset=84, count=n)['v'].astype(float)

def forward(root, qs=None):
    joints = root.findall('joint')
    children = {j.find('child').get('link') for j in joints}
    names = {l.get('name') for l in root.findall('link')}
    roots = names - children
    assert len(roots) == 1, roots
    T = {next(iter(roots)): np.eye(4)}
    pending = list(joints)
    while pending:
        count = len(pending)
        for j in pending[:]:
            p = j.find('parent').get('link')
            c = j.find('child').get('link')
            assert p in names and c in names
            if p not in T:
                continue
            assert c not in T, ('multiple parents', c)
            J = transform(j.find('origin'))
            q = (qs or {}).get(j.get('name'), 0.0)
            if j.get('type') != 'fixed' and q:
                a = vec(j.find('axis').get('xyz')); a = a / np.linalg.norm(a)
                K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
                U = np.eye(4); U[:3,:3] = np.eye(3) + math.sin(q)*K + (1-math.cos(q))*(K@K)
                J = J @ U
            T[c] = T[p] @ J
            pending.remove(j)
        assert len(pending) < count, 'cycle or disconnected graph'
    assert len(T) == len(names)
    return T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--parameters', type=Path, required=True)
    ap.add_argument('--collision-proposals', type=Path)
    args = ap.parse_args()
    cfg = json.loads(args.parameters.read_text(encoding='utf-8'))
    source = ET.parse(args.source).getroot()
    links = {l.get('name'): l for l in source.findall('link')}
    joints = {j.get('name'): j for j in source.findall('joint')}
    W = forward(source)
    keep = {n for seq in CHAINS.values() for n in seq}
    moving = {n for n,j in joints.items() if j.get('type') != 'fixed'}
    locked = moving - keep
    assert len(keep) == 12 and len(locked) == 12 and len(links) == 118
    # Verify that every removed DoF drives only one terminal shaft/disc.
    by_parent = collections.defaultdict(list)
    for j in joints.values():
        by_parent[j.find('parent').get('link')].append(j)
    for n in locked:
        child = joints[n].find('child').get('link')
        assert re.fullmatch(r'roll[12](?:_\d+)?', child), (n, child)
        assert not by_parent[child], (n, 'not terminal')

    leader = {n:n for n in links}
    def find(n):
        if leader[n] != n:
            leader[n] = find(leader[n])
        return leader[n]
    for n,j in joints.items():
        if j.get('type') == 'fixed':
            leader[find(j.find('child').get('link'))] = find(j.find('parent').get('link'))
    couplings = {}
    for n in sorted(locked):
        leaf = joints[n].find('child').get('link')
        match = re.fullmatch(r'(roll[12])(_\d+)?', leaf)
        suffix = match.group(2) or ''
        opposite = ('roll2' if match.group(1) == 'roll1' else 'roll1') + suffix
        output_group = find(opposite)
        housing_group = find('motor_upper_cover'+suffix)
        candidates = [a for a in keep if
            {find(joints[a].find('parent').get('link')),find(joints[a].find('child').get('link'))}
            == {output_group,housing_group}]
        assert len(candidates) == 1, (n,candidates)
        active = candidates[0]
        active_axis = W[joints[active].find('child').get('link')][:3,:3] @ vec(joints[active].find('axis').get('xyz'))
        leaf_axis = W[leaf][:3,:3] @ vec(joints[n].find('axis').get('xyz'))
        direction = 1 if find(joints[active].find('child').get('link')) == output_group else -1
        alignment = float(active_axis @ leaf_axis)
        assert abs(alignment) > 0.9999999
        multiplier = direction * (1 if alignment > 0 else -1)
        couplings[n] = {'active_source_joint':active, 'multiplier':multiplier,
            'terminal_disc':leaf, 'rigid_with':opposite,
            'assumption':'Opposite support disc follows the same U-bracket as the structural output horn.'}
        # This models both horn attachment faces as one rigid bracket, rather than
        # leaving a support disc fixed to its servo housing.
        leader[find(leaf)] = find(opposite)
    groups = collections.defaultdict(list)
    for n in links:
        groups[find(n)].append(n)
    assert len(groups) == 13
    anchors = {find('root'): ('base_link', 'root')}
    joint_names = {}
    for leg, seq in CHAINS.items():
        for stage, n in zip(STAGES, seq):
            child = joints[n].find('child').get('link')
            anchors[find(child)] = (f'{leg}_{stage}_link', child)
            joint_names[n] = f'{leg}_{stage}_joint'
    assert len(anchors) == 13
    rel = {n: inverse(W[anchors[find(n)][1]]) @ W[n] for n in links}

    meshes_dir = args.source.parent.parent / 'meshes'
    meshes = {p.name: stl_vertices(p) for p in meshes_dir.glob('*.stl')}
    mass_rows = []
    old_masses = {}
    mesh_for_link = {}
    for n,l in links.items():
        mesh = l.find('visual/geometry/mesh')
        mesh_for_link[n] = Path(mesh.get('filename')).name if mesh is not None else None
        m = l.find('inertial/mass')
        old_masses[n] = float(m.get('value')) if m is not None else 0.0
    servo_volume_mass = sum(old_masses[n] for n in links if n in
        ['gearing','motor_1723_3','motor_body','motor_bottom_cover','motor_upper_cover','pcb','roll1','roll2'])
    missing = []
    masses = {}
    group_masses = {}
    for mg in cfg.get('mass_groups', []):
        total_old = sum(old_masses[n] for n in mg['links'])
        assert total_old > 0 and mg['total_mass_g'] > 0
        for n in mg['links']:
            assert n not in group_masses, ('overlapping mass groups', n)
            group_masses[n] = (mg['total_mass_g'] * old_masses[n] / total_old, mg['label'])
    for n,l in links.items():
        if n == 'root':
            masses[n] = 0.0
            continue
        mesh = mesh_for_link[n]
        stem = Path(mesh).stem
        mass_g = cfg.get('mass_g_by_link', {}).get(n)
        policy = 'configured_link_mass'
        if mass_g is None:
            mass_g = cfg.get('mass_g_by_mesh', {}).get(mesh)
            policy = 'configured_mesh_mass'
        if mass_g is None and n in group_masses:
            mass_g, label = group_masses[n]
            policy = 'group_total_distributed_by_CAD_volume_APPROXIMATION:'+label
        if mass_g is None and stem in SERVO_MESHES and cfg.get('servo_total_mass_g') is not None:
            mass_g = cfg['servo_total_mass_g'] * old_masses[n] / servo_volume_mass
            policy = 'servo_total_distributed_by_CAD_volume_APPROXIMATION'
        if mass_g is None and n == 'part_1' and cfg.get('battery_mass_g') is not None:
            mass_g = cfg['battery_mass_g']
            policy = 'user_battery_mass'
        if mass_g is None and stem not in SERVO_MESHES and n != 'part_1' and cfg.get('effective_print_density_kg_m3') is not None:
            mass_g = old_masses[n] * cfg['effective_print_density_kg_m3'] * 1000.0
            policy = 'user_effective_print_density_APPROXIMATION'
        if mass_g is None:
            masses[n] = old_masses[n]
            policy = 'UNCONFIRMED_original_density_1_kg_m3'
            missing.append(n)
        else:
            assert math.isfinite(mass_g) and mass_g > 0, (n, mass_g)
            masses[n] = mass_g / 1000.0
        mass_rows.append({'source_link':n, 'mesh':mesh, 'merged_link':anchors[find(n)][0],
                          'source_mass_kg':old_masses[n], 'mass_kg':masses[n], 'policy':policy})

    payloads = []
    for p in cfg.get('added_payloads', []):
        ref = p['reference_source_link']
        assert ref in links
        size = np.array(p['size_m'], dtype=float)
        mass = float(p['mass_g']) / 1000.0
        assert size.shape == (3,) and np.isfinite(size).all() and (size > 0).all()
        assert math.isfinite(mass) and mass > 0
        offset = np.eye(4)
        offset[:3,3] = np.array(p['center_in_source_visual_m'], dtype=float)
        offset[:3,:3] = rotation(p.get('rpy_in_source_visual_rad', [0,0,0]))
        visual = links[ref].find('visual')
        assert visual is not None
        local = transform(visual.find('origin')) @ offset
        merged_T = rel[ref] @ local
        world_T = W[ref] @ local
        # A uniform cuboid is the stated approximation for each added payload.
        a,b,c = size
        inertia = mass/12 * np.diag([b*b+c*c, a*a+c*c, a*a+b*b])
        entry = {'config':p, 'mass':mass, 'size':size, 'inertia':inertia,
            'group':find(ref), 'merged_T':merged_T, 'world_T':world_T}
        payloads.append(entry)
        mass_rows.append({'source_link':'payload:'+p['name'], 'mesh':None,
            'merged_link':anchors[find(ref)][0], 'source_mass_kg':0, 'mass_kg':mass,
            'policy':'user_added_payload_uniform_cuboid_APPROXIMATION',
            'reference_source_link':ref, 'center_in_merged_link_m':merged_T[:3,3].tolist(),
            'size_m':size.tolist(), 'notes':p.get('notes')})

    out = ET.Element('robot', name='moresense_doge')
    out.append(ET.Comment('STRUCTURE PREVIEW: unconfirmed dynamics. Read status.json before simulation.'))
    proposals = json.loads(args.collision_proposals.read_text(encoding='utf-8')) if args.collision_proposals and args.collision_proposals.exists() else {}
    collision_rows = []
    aggregate = []
    for group, members in groups.items():
        new_name, anchor = anchors[group]
        link = ET.SubElement(out, 'link', name=new_name)
        components = []
        for n in members:
            I = links[n].find('inertial')
            if I is None:
                continue
            T = rel[n] @ transform(I.find('origin'))
            Ii = tensor(I.find('inertia')) * masses[n] / old_masses[n]
            components.append((masses[n], T[:3,3], T[:3,:3] @ Ii @ T[:3,:3].T))
        group_payloads = [p for p in payloads if p['group'] == group]
        for p in group_payloads:
            T=p['merged_T']
            components.append((p['mass'],T[:3,3],T[:3,:3]@p['inertia']@T[:3,:3].T))
        M = sum(m for m,p,I in components)
        c = sum((m*p for m,p,I in components), np.zeros(3)) / M
        I = sum((Ii + m*((d:=p-c)@d*np.eye(3)-np.outer(d,d)) for m,p,Ii in components), np.zeros((3,3)))
        assert np.linalg.eigvalsh(I).min() > 0
        ev = np.linalg.eigvalsh(I)
        assert ev[2] <= ev[0]+ev[1]+1e-10
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'origin', xyz=fmt(c), rpy='0 0 0')
        ET.SubElement(inertial, 'mass', value=f'{M:.12g}')
        ET.SubElement(inertial, 'inertia', **tensor_attrs(I))
        aggregate.append({'link':new_name, 'mass_kg':M, 'com':c.tolist(), 'inertia_eigenvalues':ev.tolist(), 'members':members,
            'added_payloads':[p['config']['name'] for p in group_payloads]})
        for p in group_payloads:
            if not p['config'].get('visual', True):
                continue
            visual=ET.SubElement(link,'visual',name=p['config']['name'])
            put_origin(visual,p['merged_T'])
            geometry=ET.SubElement(visual,'geometry')
            ET.SubElement(geometry,'box',size=fmt(p['size']))
            mat=ET.SubElement(visual,'material',name=p['config']['name']+'_material')
            ET.SubElement(mat,'color',rgba=fmt(p['config'].get('rgba',[0.05,0.4,0.15,1])))
        for n in members:
            for visual in links[n].findall('visual'):
                v = copy.deepcopy(visual)
                T = rel[n] @ transform(v.find('origin'))
                put_origin(v, T)
                mesh = v.find('geometry/mesh')
                if mesh is not None:
                    filename = Path(mesh.get('filename')).name
                    mesh.set('filename', '../meshes/' + filename)
                    mat = v.find('material')
                    if mat is not None:
                        mat.set('name', n + '_material')
                link.append(v)
            mesh_name = mesh_for_link[n]
            if mesh_name is None:
                continue
            stem = Path(mesh_name).stem
            # No battery/electronics/internal motor or small shaft colliders.
            if stem in SERVO_MESHES and stem not in {'motor_body','motor_bottom_cover','Motor_upper_Cover'}:
                continue
            if stem in {'motor_bottom_cover','Motor_upper_Cover'}:
                continue  # Included in the combined motor housing box below.
            if n == 'part_1':
                continue
            v = links[n].find('visual')
            Tv = rel[n] @ transform(v.find('origin'))
            scale = vec(v.find('geometry/mesh').get('scale', '1 1 1'))
            points = meshes[mesh_name].reshape(-1,3) * scale
            if stem == 'motor_body':
                suffix = n.removeprefix('motor_body')
                housing_points = [points]
                for other in ['motor_upper_cover'+suffix, 'motor_bottom_cover'+suffix]:
                    assert find(other) == group
                    ov = links[other].find('visual')
                    O = inverse(Tv) @ rel[other] @ transform(ov.find('origin'))
                    op = meshes[mesh_for_link[other]].reshape(-1,3)
                    op = op * vec(ov.find('geometry/mesh').get('scale','1 1 1'))
                    housing_points.append(op @ O[:3,:3].T + O[:3,3])
                points = np.concatenate(housing_points)
            shapes = proposals.get(mesh_name)
            if not shapes:
                # Mesh-local AABBs for large structural parts; no PCA/sign ambiguity.
                lo,hi = points.min(0), points.max(0)
                shapes = [{'type':'box','center':((lo+hi)/2).tolist(),'size':(hi-lo).tolist(), 'rpy':[0,0,0]}]
            for k,shape in enumerate(shapes):
                C = np.eye(4)
                C[:3,3] = shape['center']
                C[:3,:3] = rotation(shape.get('rpy',[0,0,0]))
                collision = ET.SubElement(link, 'collision', name=f'{n}_proxy_{k}')
                put_origin(collision, Tv @ C)
                geometry = ET.SubElement(collision, 'geometry')
                if shape['type'] == 'box':
                    ET.SubElement(geometry, 'box', size=fmt(shape['size']))
                elif shape['type'] == 'cylinder':
                    ET.SubElement(geometry, 'cylinder', radius=f"{shape['radius']:.12g}", length=f"{shape['length']:.12g}")
                else:
                    raise ValueError(shape)
                collision_rows.append({'merged_link':new_name, 'source_link':n, 'mesh':mesh_name, **shape})
    for old_name in [n for seq in CHAINS.values() for n in seq]:
        original = joints[old_name]
        child = original.find('child').get('link')
        parent = original.find('parent').get('link')
        parent_name, parent_anchor = anchors[find(parent)]
        child_name, child_anchor = anchors[find(child)]
        assert child_anchor == child
        new_name = joint_names[old_name]
        limits = cfg.get('joint_limits', {}).get(new_name)
        joint = ET.SubElement(out, 'joint', name=new_name, type='revolute' if limits else 'continuous')
        put_origin(joint, inverse(W[parent_anchor]) @ W[child])
        ET.SubElement(joint, 'parent', link=parent_name)
        ET.SubElement(joint, 'child', link=child_name)
        axis = vec(original.find('axis').get('xyz'))
        assert abs(np.linalg.norm(axis)-1)<1e-8
        ET.SubElement(joint, 'axis', xyz=fmt(axis))
        if limits:
            for field in ['lower_rad','upper_rad','effort_nm','velocity_rad_s']:
                assert limits.get(field) is not None, (new_name,field)
            assert limits['lower_rad'] < limits['upper_rad']
            assert limits['effort_nm'] > 0 and limits['velocity_rad_s'] > 0
            ET.SubElement(joint, 'limit', lower=str(limits['lower_rad']), upper=str(limits['upper_rad']),
                          effort=str(limits['effort_nm']), velocity=str(limits['velocity_rad_s']))
        elif cfg.get('servo_effort_nm') is not None and cfg.get('servo_velocity_rad_s') is not None:
            ET.SubElement(joint, 'limit', effort=str(cfg['servo_effort_nm']), velocity=str(cfg['servo_velocity_rad_s']))

    # Meaningful invariants: retained kinematics, all visuals, total mass, full spatial inertia.
    rng = np.random.default_rng(129)
    max_fk = 0.0
    max_structure_fk = 0.0
    max_support_translation = 0.0
    max_support_rotation = 0.0
    terminal_discs = {c['terminal_disc'] for c in couplings.values()}
    for trial in range(12):
        old_q = {n:float(rng.uniform(-0.7,0.7)) for n in keep} if trial else {}
        old_q.update({n:c['multiplier']*old_q.get(c['active_source_joint'],0) for n,c in couplings.items()})
        old_W = forward(source, old_q)
        new_W = forward(out, {joint_names[n]:v for n,v in old_q.items() if n in keep})
        for n in links:
            merged, anchor = anchors[find(n)]
            error = old_W[n]-new_W[merged]@rel[n]
            max_fk = max(max_fk, float(np.max(np.abs(error))))
            if n in terminal_discs:
                max_support_translation = max(max_support_translation,float(np.linalg.norm(error[:3,3])))
                max_support_rotation = max(max_support_rotation,float(np.max(np.abs(error[:3,:3]))))
            else:
                max_structure_fk = max(max_structure_fk,float(np.max(np.abs(error))))
    assert max_structure_fk < 1e-8, max_structure_fk
    # Source opposite shaft axes have export rounding: up to 1.8um lateral and
    # 7.4urad angular mismatch. Exact rigid support follows the bracket instead.
    assert max_support_translation < 1e-5, max_support_translation
    assert max_support_rotation < 2e-5, max_support_rotation
    added_visual_count=sum(p['config'].get('visual',True) for p in payloads)
    assert len(out.findall('.//visual')) == len(source.findall('.//visual')) + added_visual_count
    def spatial_mass(root, W, replacement=None, extra_payloads=None):
        comps = []
        for l in root.findall('link'):
            n=l.get('name'); ie=l.find('inertial')
            if ie is None: continue
            m=float(ie.find('mass').get('value'))
            scale=1.0
            if replacement is not None:
                scale=replacement[n]/m; m=replacement[n]
            T=W[n]@transform(ie.find('origin')); R=T[:3,:3]; p=T[:3,3]
            I=R@tensor(ie.find('inertia'))@R.T*scale
            comps.append((m,p,I))
        for payload in extra_payloads or []:
            T=payload['world_T'];R=T[:3,:3]
            comps.append((payload['mass'],T[:3,3],R@payload['inertia']@R.T))
        M=sum(m for m,p,I in comps)
        c=sum((m*p for m,p,I in comps),np.zeros(3))/M
        I=sum((Ii+m*((d:=p-c)@d*np.eye(3)-np.outer(d,d)) for m,p,Ii in comps),np.zeros((3,3)))
        return M,c,I
    A=spatial_mass(source,W,masses,payloads);B=spatial_mass(out,forward(out))
    assert abs(A[0]-B[0])<1e-9
    assert np.max(np.abs(A[1]-B[1]))<1e-9
    assert np.max(np.abs(A[2]-B[2]))<1e-9
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'urdf').mkdir(exist_ok=True)
    (args.output/'meshes').mkdir(exist_ok=True)
    for p in meshes_dir.glob('*.stl'):
        dest=args.output/'meshes'/p.name
        if p.resolve() != dest.resolve():
            shutil.copy2(p,dest)
    masses_complete = not missing
    limits_complete = all(cfg.get('joint_limits',{}).get(n) for n in joint_names.values())
    ready = masses_complete and limits_complete and cfg.get('cad_zero_and_neutral_pose_confirmed',False)
    filename = 'moresense_doge.urdf' if masses_complete else 'moresense_structure_preview.urdf'
    state = ('PARAMETERS_COMPLETE_NOT_SIMULATION_VALIDATED' if ready else
             'IMPORT_READY_CALIBRATION_PENDING' if masses_complete else 'DRAFT_MASSES_PENDING')
    status = {'status':state,
        'urdf_file':'urdf/'+filename,
        'mass_parameters_complete':masses_complete,'joint_limits_complete':limits_complete,
        'parameter_notes':cfg.get('notes'), 'training_validation_complete':False,
        'cad_zero_and_neutral_pose_confirmed':cfg.get('cad_zero_and_neutral_pose_confirmed',False),
        'mass_kg':A[0] if masses_complete else None, 'current_xml_mass_kg':A[0],
        'missing_mass_links':missing, 'source_mass_kg':sum(old_masses.values()),
        'links':13,'moving_joints':12,'collisions':len(collision_rows),
        'added_payload_mass_kg':sum(p['mass'] for p in payloads),
        'added_payloads':[{'name':p['config']['name'],'mass_kg':p['mass'],
            'merged_link':anchors[p['group']][0],'center_in_merged_link_m':p['merged_T'][:3,3].tolist(),
            'size_m':p['size'].tolist(),'visual':p['config'].get('visual',True),
            'notes':p['config'].get('notes')} for p in payloads],
        'retained_joints':joint_names, 'removed_independent_terminal_joints':sorted(locked),
        'support_disc_couplings':couplings,
        'chains':CHAINS, 'root_frame':'Original CAD root; +Z up. No front/left assumption.',
        'self_collision':False, 'simulation_validation_performed':False,
        'validation':{'structure_FK_max_matrix_error_12_poses':max_structure_fk,
            'all_FK_max_matrix_error_12_poses':max_fk,
            'support_disc_max_translation_error_m':max_support_translation,
            'support_disc_max_rotation_matrix_error':max_support_rotation,
            'support_disc_error_reason':'Original opposite shaft axes have CAD export rounding; exact rigid coupling removes this inconsistency.',
            'original_visuals_preserved':117, 'added_visuals':added_visual_count,
            'mass_sum_preserved':True,'center_of_mass_preserved':True,'full_inertia_preserved':True,
            'positive_definite_inertias':True},
        'notes':['Collision proxies approximate CAD surfaces; check contacts and swept clearance in Isaac.',
            'All original joints were continuous with no actuator or angular limits.',
            'CAD zero is asymmetric and is not a calibrated neutral standing pose.',
            'Support discs are rigidly attached to the matching structural output bracket; confirm attachment on hardware.',
            'Servo total mass distribution by CAD volume is an explicit approximation when used.',
            'Self collision must stay disabled until proxy clearances have been checked.']}
    out[0].text = ' '+status['status']+'; see status.json. CAD frame and zero positions preserved. '
    stale_name = 'moresense_structure_preview.urdf' if masses_complete else 'moresense_doge.urdf'
    stale = args.output/'urdf'/stale_name
    if stale.exists():
        stale.unlink()  # Only the known alternative generated URDF, never user/source files.
    ET.indent(out, space='  ')
    ET.ElementTree(out).write(args.output/'urdf'/filename, encoding='utf-8',xml_declaration=True)
    for name,data in [('status',status),('mass_audit',mass_rows),('rigid_bodies',aggregate),('collision_proxies',collision_rows)]:
        (args.output/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'parameters.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    mapping={'source_link_to_merged_link':{n:anchors[find(n)][0] for n in links},
        'source_link_to_merged_transform':{n:rel[n].tolist() for n in links},
        'retained_joint_renames':joint_names}
    (args.output/'source_mapping.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
