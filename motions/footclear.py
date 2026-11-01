#!/usr/bin/env python3
# 本ファイルは構成を説明するための骨組みである。報酬の重み・接触条件・終了条件の値・参照生成のパラメータは非公開(None のプレースホルダ)で、このままでは学習結果を再現できない。© 2026 okakei. All rights reserved.
# This file is a skeleton that describes the structure only. Reward weights, contact conditions, termination values and reference-generation parameters are withheld (None placeholders), so the learned results cannot be reproduced as is. © 2026 okakei. All rights reserved.
"""Foot-contact and wheelchair-clearance correction for G1 reference motions.

Usage:
  python footclear.py build --urdf g1_29dof_rev_1_0.urdf --input <dir with npz> --output <dir>
  python footclear.py verify <dir>
Input npz names: stand_input.npz, sit_input.npz (rename to your own clips).

NumPy-only G1 FK/IK and conservative, analytic wheelchair clearance.

NPZ order is URDF movable-joint breadth-first order, verified against source
motion poses. Sphere/OBB and sphere/cylinder distances are analytic. Tubes are
enclosed by capsules; wheel assemblies by cylinders. Positive distances are
therefore certified lower bounds for the generated chair, not point samples.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

# 学習・生成に用いた値は非公開。None のままでは動かない。
SPHERES = None   # 足裏の接触球(足首座標系、4 点)
RADIUS = None    # 接触球の半径
FOOT_VISUAL_CORNERS = None  # 足の外形ボックスの 8 頂点(足首座標系)
POS_TOL = None   # 順運動学・IK の位置許容 [m]
ANG_TOL = None   # 順運動学・IK の角度許容 [rad]
CLEARANCE_MIN = None   # 足と部材の最小距離の要求 [m]
GROUND_MIN = None; GROUND_MAX = None   # 足裏の接地高さの許容 [m]
VISUAL_MIN = None   # 足の外形ボックスと部材の最小距離 [m]
HEIGHT_RAMP = None   # 骨盤高さの補正を掛ける範囲 (z0, z1) [m]
ANKLE_Z = None   # 平置きの足首高さ [m]
IK_ITERS = None; IK_ORI_W = None; IK_DAMP = None   # 脚 IK の反復数、姿勢の重み、減衰
IK_STEP_MAX = None; IK_LIMIT_MARGIN = None; IK_CONV = None   # 更新量の上限、関節限界の余裕、収束判定
ANKLES = [18, 19]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    a = np.asarray(angle)
    return (np.eye(3) + np.sin(a)[..., None, None] * k
            + (1 - np.cos(a))[..., None, None] * (k @ k))


def qmat(q):
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1).reshape(q.shape[:-1]+(3, 3))


def matq(m):
    """Stable eigenvector conversion, wxyz, continuous hemisphere over time."""
    out = np.empty(m.shape[:-2] + (4,))
    for idx in np.ndindex(m.shape[:-2]):
        r = m[idx]
        xx, yy, zz = np.diag(r)
        k = np.array([[xx-yy-zz, r[0,1]+r[1,0], r[0,2]+r[2,0], r[2,1]-r[1,2]],
                      [r[0,1]+r[1,0], yy-xx-zz, r[1,2]+r[2,1], r[0,2]-r[2,0]],
                      [r[0,2]+r[2,0], r[1,2]+r[2,1], zz-xx-yy, r[1,0]-r[0,1]],
                      [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], xx+yy+zz]]) / 3
        _, v = np.linalg.eigh(k)
        out[idx] = v[[3, 0, 1, 2], -1]
    if out.ndim >= 3:
        for t in range(1, len(out)):
            flip = (out[t] * out[t-1]).sum(-1) < 0
            out[t, flip] *= -1
    return out


def rotvec(r):
    v = np.stack([r[...,2,1]-r[...,1,2], r[...,0,2]-r[...,2,0],
                  r[...,1,0]-r[...,0,1]], -1) * .5
    theta = np.arccos(np.clip((np.trace(r, axis1=-2, axis2=-1)-1)/2, -1, 1))
    return v * np.where(theta < 1e-6, 1., theta / np.maximum(np.sin(theta), 1e-12))[..., None]


def make_model(urdf):
    root = ET.parse(urdf).getroot()
    movable = [j for j in root.findall('joint') if j.get('type') != 'fixed']
    bodies, joints = ['pelvis'], []
    for parent in bodies:
        for j in movable:
            if j.find('parent').get('link') != parent:
                continue
            ori = j.find('origin')
            xyz = np.fromstring(ori.get('xyz', '0 0 0'), sep=' ')
            rpy = np.fromstring(ori.get('rpy', '0 0 0'), sep=' ')
            rotation = (axis_rotation([0,0,1], rpy[2]) @ axis_rotation([0,1,0], rpy[1])
                        @ axis_rotation([1,0,0], rpy[0]))
            limit = j.find('limit')
            child = j.find('child').get('link')
            joints.append(dict(name=j.get('name'), parent=bodies.index(parent), child=child,
                               xyz=xyz.tolist(), rotation=rotation.tolist(),
                               axis=np.fromstring(j.find('axis').get('xyz'), sep=' ').tolist(),
                               lower=float(limit.get('lower')), upper=float(limit.get('upper')),
                               velocity=float(limit.get('velocity'))))
            bodies.append(child)
    if len(joints) != 29 or bodies[18:20] != ['left_ankle_roll_link', 'right_ankle_roll_link']:
        raise ValueError('URDF is not the verified G1 BFS articulation')
    by_child = {j.find('child').get('link'):j for j in root.findall('joint')}
    masses = []
    for link in root.findall('link'):
        inertial = link.find('inertial')
        if inertial is None: continue
        mass = float(inertial.find('mass').get('value'))
        center = np.fromstring(inertial.find('origin').get('xyz','0 0 0'),sep=' ')
        name = link.get('name')
        while name not in bodies:
            joint = by_child[name]
            if joint.get('type') != 'fixed': raise ValueError('unmapped movable link')
            origin = joint.find('origin')
            if origin is None: origin = ET.Element('origin')
            xyz = np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            rpy = np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
            rotation = axis_rotation([0,0,1],rpy[2])@axis_rotation([0,1,0],rpy[1])@axis_rotation([1,0,0],rpy[0])
            center = xyz+rotation@center
            name = joint.find('parent').get('link')
        masses.append(dict(body=bodies.index(name),mass=mass,center=center.tolist()))
    return dict(source='g1_29dof_rev_1_0.urdf', urdf_sha256=digest(urdf), bodies=bodies, joints=joints,masses=masses)


def fk(model, root_p, root_r, q, jacobian=False):
    n = len(q)
    pos = np.zeros((n, 30, 3))
    rot = np.zeros((n, 30, 3, 3))
    pos[:, 0], rot[:, 0] = root_p, root_r
    axes = np.zeros((n, 29, 3))
    for i, j in enumerate(model['joints']):
        parent = j['parent']
        pos[:, i+1] = pos[:, parent] + np.einsum('nij,j->ni', rot[:, parent], j['xyz'])
        before = rot[:, parent] @ np.asarray(j['rotation'])
        axes[:, i] = np.einsum('nij,j->ni', before, j['axis'])
        rot[:, i+1] = before @ axis_rotation(j['axis'], q[:, i])
    return (pos, rot, axes) if jacobian else (pos, rot)


def chair_parts(path=None):
    """車いすの衝突形状を読む。公開版の JSON は部材名と形状種別だけで、寸法は null。"""
    path = Path(path) if path else Path(__file__).with_name('chair_parts.json')
    return json.loads(path.read_text(encoding='utf-8'))['parts']


def point_sdf(points, part):
    p = np.asarray(points)
    if part['kind'] == 'box':
        q = np.abs((p-np.array(part['center'])) @ np.array(part['rotation']))-part['half']
        return np.linalg.norm(np.maximum(q,0), axis=-1) + np.minimum(q.max(-1),0)
    if part['kind'] == 'capsule':
        a, b = np.array(part['a']), np.array(part['b'])
        ab = b-a
        t = np.clip(((p-a)*ab).sum(-1)/(ab@ab),0,1)
        return np.linalg.norm(p-a-t[...,None]*ab,axis=-1)-part['radius']
    q = p-np.array(part['center'])
    d = np.stack([np.linalg.norm(q[...,[0,2]],axis=-1)-part['radius'],
                  np.abs(q[...,1])-part['half']],-1)
    return np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(-1),0)


def sphere_centers(pos, rot):
    return pos[:, ANKLES, None, :] + np.einsum('nfij,sj->nfsi', rot[:, ANKLES], SPHERES)


def clearance(pos, rot, parts):
    centers = sphere_centers(pos, rot)
    return np.stack([point_sdf(centers, p)-RADIUS for p in parts], axis=-1)


def visual_aabb_lower_bound(pos, rot, parts):
    # Supplied visual STL extents, with a conservative lower z extent.
    corners = np.asarray(FOOT_VISUAL_CORNERS)
    vertices = pos[:,ANKLES,None,:] + np.einsum('nfij,sj->nfsi',rot[:,ANKLES],corners)
    lo,hi = vertices.min(2),vertices.max(2)
    bounds = []
    for p in parts:
        if p['kind'] == 'box':
            half = np.abs(p['rotation']) @ np.array(p['half'])
            a,b = np.array(p['center'])-half,np.array(p['center'])+half
        elif p['kind'] == 'capsule':
            a,b = np.minimum(p['a'],p['b'])-p['radius'],np.maximum(p['a'],p['b'])+p['radius']
        else:
            half = np.array([p['radius'],p['half'],p['radius']])
            a,b = np.array(p['center'])-half,np.array(p['center'])+half
        bounds.append(np.linalg.norm(np.maximum(np.maximum(a-hi,lo-b),0),axis=-1))
    return np.stack(bounds,-1)


def solve_legs(model, root_p, root_r, q0, targets):
    q = q0.copy()
    lower = np.array([j['lower'] for j in model['joints']])
    upper = np.array([j['upper'] for j in model['joints']])
    chains = [[0,3,6,9,13,17], [1,4,7,10,14,18]]
    for iteration in range(IK_ITERS):
        p, r, axes = fk(model, root_p, root_r, q, True)
        maximum = 0.
        for foot, chain in enumerate(chains):
            bi = ANKLES[foot]
            er = rotvec(np.swapaxes(r[:,bi],-1,-2))  # target rotation = world identity
            ep = targets[:,foot]-p[:,bi]
            maximum = max(maximum, float(np.abs(ep).max()), float(np.abs(er).max()))
            jp = np.cross(axes[:,chain], p[:,bi,None,:]-p[:,np.array(chain)+1])
            j = np.concatenate([jp.transpose(0,2,1), IK_ORI_W*axes[:,chain].transpose(0,2,1)], axis=1)
            err = np.concatenate([ep,IK_ORI_W*er],axis=1)
            jt = j.transpose(0,2,1)
            step = np.linalg.solve(jt@j + IK_DAMP*np.eye(6), (jt@err[...,None]))[...,0]
            step *= np.minimum(1., IK_STEP_MAX/np.maximum(np.max(np.abs(step),axis=1),1e-12))[:,None]
            q[:,chain] = np.clip(q[:,chain]+step, lower[chain]+IK_LIMIT_MARGIN, upper[chain]-IK_LIMIT_MARGIN)
        if maximum < IK_CONV:
            break
    p, r = fk(model,root_p,root_r,q)
    pe = np.linalg.norm(p[:,ANKLES]-targets,axis=-1)
    re = np.linalg.norm(rotvec(r[:,ANKLES]),axis=-1)
    print(f'IK iterations={iteration+1} position_max_m={pe.max():.9g} angle_max_rad={re.max():.9g}', flush=True)
    if pe.max() > POS_TOL or re.max() > ANG_TOL:
        idx = np.unravel_index(pe.argmax(),pe.shape)
        raise ValueError(f'IK infeasible frame/foot={idx}, pos={pe.max()}, ori={re.max()}')
    return q,p,r


def derivatives(q, pos, rot, fps):
    joint_v = np.gradient(q, 1/fps, axis=0, edge_order=2)
    linear = np.gradient(pos, 1/fps, axis=0, edge_order=2)
    dr = np.gradient(rot, 1/fps, axis=0, edge_order=2)
    w = dr @ np.swapaxes(rot,-1,-2)
    angular = np.stack([w[...,2,1]-w[...,1,2],w[...,0,2]-w[...,2,0],w[...,1,0]-w[...,0,1]],-1)/2
    return joint_v,linear,angular


def center_of_mass(model,pos,rot):
    total = sum(m['mass'] for m in model['masses'])
    return sum(m['mass']*(pos[:,m['body']]+np.einsum('nij,j->ni',rot[:,m['body']],m['center']))
               for m in model['masses'])/total


def support_margin(com, pos, rot):
    """Signed distance to convex hull of projected sphere contact points."""
    result = []
    for c,feet in zip(com,sphere_centers(pos,rot)):
        points = sorted(set(map(tuple,feet.reshape(-1,3)[:,:2])))
        def cross(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        def half(seq):
            hull=[]
            for p in seq:
                while len(hull)>=2 and cross(hull[-2],hull[-1],p)<=0: hull.pop()
                hull.append(p)
            return hull
        hull=half(points)[:-1]+half(points[::-1])[:-1]
        result.append(min(cross(a,b,c)/np.linalg.norm(np.array(b)-a)
                          for a,b in zip(hull,hull[1:]+hull[:1])))
    return np.array(result)


def summarize(data, model, parts, require=True):
    q = data['joint_pos'].astype(float)
    pos = data['body_pos_w'].astype(float)
    rot = qmat(data['body_quat_w'])
    fp,fr = fk(model,pos[:,0],rot[:,0],q)
    fk_error = float(np.linalg.norm(fp-pos,axis=-1).max())
    ori_error = float(np.linalg.norm(rotvec(fr @ np.swapaxes(rot,-1,-2)),axis=-1).max())
    distances = clearance(pos,rot,parts)
    worst = np.unravel_index(distances.argmin(),distances.shape)
    low = np.array([j['lower'] for j in model['joints']])
    high = np.array([j['upper'] for j in model['joints']])
    limits_ok = bool(np.all(q >= low-1e-6) and np.all(q <= high+1e-6))
    ground = sphere_centers(pos,rot)[...,2]-RADIUS
    visual_bound = float(visual_aabb_lower_bound(pos,rot,parts).min())
    com = center_of_mass(model,pos,rot)
    margin = support_margin(com,pos,rot)
    per_part = []
    for i, part in enumerate(parts):
        d = distances[...,i]
        w = np.unravel_index(d.argmin(),d.shape)
        hits = np.flatnonzero(d.min(axis=(1,2)) < 0)
        per_part.append(dict(name=part['name'], minimum_m=float(d.min()), worst_frame=int(w[0]),
                             worst_foot=int(w[1]), intersect_frames=hits.tolist()))
    result = dict(frames=len(q), fps=float(data['fps'][0]),
                  clearance_min_m=float(distances.min()), worst_frame=int(worst[0]),
                  worst_foot=int(worst[1]), worst_sphere=int(worst[2]), worst_part=parts[worst[3]]['name'],
                  fk_position_error_m=fk_error, fk_angle_error_rad=ori_error,
                  visual_aabb_clearance_lower_bound_m=visual_bound,
                  com_start=com[0].tolist(),com_end=com[-1].tolist(),
                  static_support_margin_first50_min_m=float(margin[:50].min()),
                  static_support_margin_last50_min_m=float(margin[-50:].min()),
                  joint_limits_ok=limits_ok, ground_min_m=float(ground.min()), ground_max_m=float(ground.max()),
                  pelvis_start=pos[0,0].tolist(), pelvis_end=pos[-1,0].tolist(), per_part=per_part,
                  joint_ranges=[dict(name=j['name'], min=float(q[:,i].min()), max=float(q[:,i].max()),
                                     lower=j['lower'],upper=j['upper']) for i,j in enumerate(model['joints'])])
    print(f"GEOMETRY min_clearance_m={result['clearance_min_m']:.9f} part={result['worst_part']} "
          f"frame={result['worst_frame']} FK_m={fk_error:.3g} JOINT_LIMITS={'PASS' if limits_ok else 'FAIL'}", flush=True)
    if require:
        if not np.isfinite(np.concatenate([a.ravel() for a in data.values()])).all():
            raise ValueError('nonfinite motion')
        if distances.min() < CLEARANCE_MIN or not limits_ok or fk_error > POS_TOL or ori_error > ANG_TOL:
            raise ValueError('geometry / limits / FK validation failed')
        if ground.min() < -GROUND_MIN or ground.max() > GROUND_MAX:
            raise ValueError('feet not grounded')
        if visual_bound < VISUAL_MIN:
            raise ValueError('visual foot bounding box clearance below the requirement')
        for key, value in zip(['joint_vel','body_lin_vel_w','body_ang_vel_w'],
                              derivatives(q,pos,rot,result['fps'])):
            if not np.allclose(data[key],value,atol=ANG_TOL,rtol=1e-4):
                raise ValueError(f'inconsistent derivative: {key}')
        if np.any(np.abs(data['joint_vel']) > np.array([j['velocity'] for j in model['joints']])+1e-5):
            raise ValueError('URDF joint velocity limit exceeded')
    return result


def build(args):
    out = Path(args.output)
    model = make_model(args.urdf)
    parts = chair_parts()
    pending = {'model.json':(json.dumps(model,indent=2)+'\n').encode(),
               'chair_parts.json':(json.dumps(parts,indent=2)+'\n').encode()}
    report = dict(method='flat planted feet, leg IK, pelvis z ramp; no added frames', files={})
    for name in ['stand_input','sit_input']:
        src = Path(args.input)/(name+'.npz')
        with np.load(src,allow_pickle=False) as archive:
            old = {k:archive[k].copy() for k in archive.files}
        base = summarize(old,model,parts,require=False)
        if base['fk_position_error_m'] > POS_TOL or base['fk_angle_error_rad'] > ANG_TOL:
            raise ValueError('URDF / NPZ order mismatch')
        root_p = old['body_pos_w'][:,0].astype(float).copy()
        root_r = qmat(old['body_quat_w'][:,0])
        # Height-based smooth ramp preserves the seated pelvis and all timing.
        phase = np.clip((root_p[:,2]-HEIGHT_RAMP[0])/(HEIGHT_RAMP[1]-HEIGHT_RAMP[0]),0,1)
        phase = phase*phase*(3-2*phase)
        root_p[:,2] -= args.drop*phase
        root_p[:,1] += args.pelvis_y_shift*phase
        targets = np.broadcast_to(np.array([[args.foot_x,args.half_stance,ANKLE_Z],
                                            [args.foot_x,-args.half_stance,ANKLE_Z]]), (len(root_p),2,3)).copy()
        q,pos,rot = solve_legs(model,root_p,root_r,old['joint_pos'].astype(float),targets)
        quat = matq(rot)
        # Quantize poses first, then differentiate exactly the saved motion.
        result = dict(fps=old['fps'],joint_pos=q.astype('float32'),body_pos_w=pos.astype('float32'),
                      body_quat_w=quat.astype('float32'))
        values = derivatives(result['joint_pos'].astype(float),result['body_pos_w'].astype(float),
                             qmat(result['body_quat_w']),float(result['fps'][0]))
        for key,value in zip(['joint_vel','body_lin_vel_w','body_ang_vel_w'],values):
            result[key] = value.astype('float32')
        corrected = summarize(result,model,parts)
        delta_z = result['body_pos_w'][:,0,2]-old['body_pos_w'][:,0,2]
        corrected.update(pelvis_delta_z_min_m=float(delta_z.min()),pelvis_delta_z_max_m=float(delta_z.max()),
                         max_joint_change_rad=float(np.abs(result['joint_pos']-old['joint_pos']).max()),
                         max_joint_step_rad=float(np.abs(np.diff(result['joint_pos'],axis=0)).max()))
        foot_delta = result['body_pos_w'][:,ANKLES]-old['body_pos_w'][:,ANKLES]
        corrected['ankle_delta_xyz_min_m'] = foot_delta.min(axis=(0,1)).tolist()
        corrected['ankle_delta_xyz_max_m'] = foot_delta.max(axis=(0,1)).tolist()
        velocity_limits = np.array([j['velocity'] for j in model['joints']])
        corrected['joint_velocity_limit_ratio_max'] = float((np.abs(result['joint_vel'])/velocity_limits).max())
        filename = name+'_footclear.npz'
        buffer = io.BytesIO()
        np.savez(buffer,**result)
        pending[filename] = buffer.getvalue()
        report['files'][name] = dict(input_sha256=digest(src), output_sha256=hashlib.sha256(pending[filename]).hexdigest(),
                                     baseline=base,corrected=corrected)
        print(f"VALIDATED={filename} PELVIS_DELTA_Z=[{delta_z.min():.6f},{delta_z.max():.6f}]",flush=True)
    report['parameters'] = dict(foot_x=args.foot_x,half_stance=args.half_stance,ankle_z=ANKLE_Z,pelvis_drop=args.drop,
                                pelvis_y_shift=args.pelvis_y_shift)
    pending['validation.json'] = (json.dumps(report,indent=2)+'\n').encode()
    # Do not publish either motion if any validation failed; manifest is written last.
    out.mkdir(parents=True,exist_ok=True)
    for filename,content in pending.items():
        (out/filename).write_bytes(content)
        print(f'SAVED={out/filename}',flush=True)
    print('BUILD_PASS',flush=True)


def verify(args):
    directory = Path(args.directory)
    model = json.loads((directory/'model.json').read_text(encoding='utf-8'))
    parts = json.loads((directory/'chair_parts.json').read_text(encoding='utf-8'))
    report = json.loads((directory/'validation.json').read_text(encoding='utf-8'))
    for name, meta in report['files'].items():
        src = directory/(name+'_footclear.npz')
        if digest(src) != meta['output_sha256']:
            raise ValueError(f'NPZ hash mismatch: {src}')
        with np.load(src,allow_pickle=False) as data:
            summarize({k:data[k] for k in data.files},model,parts)
    print('VERIFY_PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('build')
    p.add_argument('--urdf',required=True)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--foot-x',type=float,default=None)  # 値は非公開
    p.add_argument('--half-stance',type=float,default=None)  # 値は非公開
    p.add_argument('--drop',type=float,default=None)  # 値は非公開
    p.add_argument('--pelvis-y-shift',type=float,default=None)  # 値は非公開
    p.set_defaults(func=build)
    p = sub.add_parser('verify')
    p.add_argument('directory')
    p.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
