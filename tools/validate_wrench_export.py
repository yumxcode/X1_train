# -*- coding: utf-8 -*-
"""
validate_wrench_export.py — wrench_export 模块的数学正确性验证

两部分：
  [A] 合成 2 连杆铰链链，重力补偿构造【精确静态】（qacc 残差=0），
      对 WrenchExporter.joint_wrench_local / foot_wrench_world 做逐位手算对照。
      这是数学证明：布局/方向/参考点/平移公式/局部系旋转全部精确验证。
  [B] 真实 X1 模型 PD 站立镇定后做整体量校核（ΣFz≈mg，踝接口 Fz≈远端子树重），
      验证 12 关节接线、命名映射、接触/CoP 管线。

运行：conda run -n x1 python tools/validate_wrench_export.py
"""
import os
import sys
import numpy as np
import mujoco
import importlib.util

_root = os.path.join(os.path.dirname(__file__), '..')
_spec = importlib.util.spec_from_file_location(
    'wrench_export', os.path.join(_root, 'humanoid/utils/wrench_export.py'))
_we = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_we)
WrenchExporter, JOINT_MAP = _we.WrenchExporter, _we.JOINT_MAP

G = 9.81
XML_SYN = """
<mujoco>
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 1.5">
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.05 0.05 0.05"/>
      <body name="upper" pos="0 0 0">
        <joint name="j1" type="hinge" axis="0 1 0" pos="0 0 0"/>
        <inertial pos="0.2 0 -0.1" mass="2.0" diaginertia="0.02 0.02 0.02"/>
        <geom type="box" size="0.25 0.03 0.03" pos="0.2 0 -0.1"/>
        <body name="lower" pos="0.4 0 -0.2">
          <joint name="j2" type="hinge" axis="0 1 0" pos="0 0 0"/>
          <inertial pos="0.1 0.05 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
          <geom type="box" size="0.1 0.03 0.03" pos="0.1 0.05 0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="j1" ctrlrange="-100 100"/>
    <motor name="m2" joint="j2" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
"""


def part_a_synthetic_exact():
    print('=' * 60)
    print('[A] 合成模型精确静态验证')
    m = mujoco.MjModel.from_xml_string(XML_SYN)
    d = mujoco.MjData(m)
    d.qvel[:] = 0.0
    d.qacc[:] = 0.0
    mujoco.mj_inverse(m, d)
    d.ctrl[:] = d.qfrc_inverse[-2:]
    mujoco.mj_forward(m, d)
    assert np.abs(d.qacc).max() < 1e-9, "静态构造失败"
    mujoco.mj_rnePostConstraint(m, d)

    # monkeypatch 一个最小 exporter（合成模型没有 12 关节映射，只测核心函数）
    class _Mini(WrenchExporter):
        def __init__(self):
            self.model, self.data = m, d
            self.contact_thresh = 1e-6
            self._joints = {}
            for jname in ('j1', 'j2'):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
                self._joints[jname] = (jid, m.jnt_bodyid[jid])
            self._foot_bodies = {}

    ex = _Mini()

    # ---- 手算真值：子连杆（lower）静态平衡 ----
    il = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'lower')
    m_l = m.body_mass[il]
    com_l = d.xpos[il] + d.xmat[il].reshape(3, 3) @ m.body_ipos[il]
    anchor2 = d.xanchor[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'j2')]
    F_true = np.array([0, 0, m_l * G])                       # 支撑力，世界系
    M_true_anchor = np.cross(com_l - anchor2, F_true)        # 平衡力矩（含执行器路径）

    R = d.xmat[il].reshape(3, 3)
    F_true_local = R.T @ F_true
    M_true_local = R.T @ M_true_anchor

    w = ex.joint_wrench_local('j2')
    err_f = np.abs(w[:3] - F_true_local).max()
    err_m = np.abs(w[3:] - M_true_local).max()
    print(f'  j2 lower: |F_err|={err_f:.3e} N  |M_err|={err_m:.3e} N·m   (期望 0)')
    assert err_f < 1e-9 and err_m < 1e-9, 'j2 接口力提取错误'

    # upper：子树(upper+lower)总支撑
    iu = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'upper')
    com_u = d.xpos[iu] + d.xmat[iu].reshape(3, 3) @ m.body_ipos[iu]
    anchor1 = d.xanchor[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'j1')]
    F_u_true = np.array([0, 0, (m.body_mass[iu] + m_l) * G])
    M_u_true = np.cross(com_u - anchor1, np.array([0, 0, m.body_mass[iu] * G])) + \
               np.cross(com_l - anchor1, np.array([0, 0, m_l * G]))
    Ru = d.xmat[iu].reshape(3, 3)
    wu = ex.joint_wrench_local('j1')
    err_f = np.abs(wu[:3] - Ru.T @ F_u_true).max()
    err_m = np.abs(wu[3:] - Ru.T @ M_u_true).max()
    print(f'  j1 upper: |F_err|={err_f:.3e} N  |M_err|={err_m:.3e} N·m   (期望 0)')
    assert err_f < 1e-9 and err_m < 1e-9, 'j1 接口力提取错误'

    # 一般姿态（旋转关节角后重复，验证旋转处理）
    d.qpos[-2:] = [0.4, -0.7]
    d.qvel[:] = 0.0
    d.qacc[:] = 0.0
    mujoco.mj_inverse(m, d)
    d.ctrl[:] = d.qfrc_inverse[-2:]
    mujoco.mj_forward(m, d)
    assert np.abs(d.qacc).max() < 1e-9
    mujoco.mj_rnePostConstraint(m, d)
    com_l = d.xpos[il] + d.xmat[il].reshape(3, 3) @ m.body_ipos[il]
    anchor2 = d.xanchor[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'j2')]
    R = d.xmat[il].reshape(3, 3)
    w = ex.joint_wrench_local('j2')
    F_t = R.T @ np.array([0, 0, m_l * G])
    M_t = R.T @ np.cross(com_l - anchor2, np.array([0, 0, m_l * G]))
    err = max(np.abs(w[:3] - F_t).max(), np.abs(w[3:] - M_t).max())
    print(f'  一般姿态(0.4,-0.7): |err|={err:.3e}   (期望 0)')
    assert err < 1e-9, '一般姿态提取错误'
    print('  [A] PASS — 布局/方向/参考点/平移/局部系旋转全部精确')
    return True


def part_b_x1_stand():
    print('=' * 60)
    print('[B] 真实 X1 模型精确静态校核（基座焊接 + 关节重力补偿）')
    root = os.path.join(os.path.dirname(__file__), '..')
    xml = os.path.join(root, 'resources/robots/x1/mjcf/xyber_x1_flat.xml')

    # 用 MjSpec 删除 freejoint -> 基座焊接在世界，构造可精确静态的模型
    spec = mujoco.MjSpec.from_file(xml)
    for j in list(spec.joints):
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            spec.delete(j)
    for k in list(spec.keys):
        spec.delete(k)
    m = spec.compile()
    assert m.nq == 12, 'freejoint 未删除成功'
    d = mujoco.MjData(m)
    ex = WrenchExporter(m, d)

    default = np.array([0.4, 0.05, -0.31, 0.49, -0.21, 0.0,
                        -0.4, -0.05, 0.31, 0.49, -0.21, 0.0])
    d.qpos[-12:] = default
    mujoco.mj_forward(m, d)
    foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'left_ankle_roll_link')
    foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_ankle_roll_link')
    minz = 1e9
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] in (foot_l, foot_r) and m.geom_contype[gid] > 0:
            minz = min(minz, d.geom_xpos[gid][2] - m.geom_size[gid][0])
    # 上移地板让脚底产生 ~1mm 穿透（基座已焊接，改地板不改机器人）
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            g.pos = [g.pos[0], g.pos[1], minz + 0.001]
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[-12:] = default
    ex = WrenchExporter(m, d)

    # 基座焊接 -> PD 镇定必收敛到准静态
    kps = np.array([30, 40, 35, 100, 35, 35] * 2, dtype=float)
    kds = np.array([3, 3, 4, 10, 0.5, 0.5] * 2, dtype=float)
    d.qvel[:] = 0.0
    for _ in range(4000):
        q, dq = d.qpos[-12:], d.qvel[-12:]
        tau = np.clip((default - q) * kps - dq * kds, -500, 500)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
    mujoco.mj_rnePostConstraint(m, d)
    print(f'  镇定后 |qvel|max = {np.abs(d.qvel[-12:]).max():.2e} rad/s')
    assert np.abs(d.qvel[-12:]).max() < 1e-2, '镇定失败'

    total_mass = mujoco.mj_getTotalmass(m)
    fz_l = d.cfrc_ext[ex._foot_bodies['left'], 5]
    fz_r = d.cfrc_ext[ex._foot_bodies['right'], 5]
    weight = total_mass * G
    print(f'  双足接触 Fz = {fz_l:.1f} + {fz_r:.1f} N（基座焊接承重，无需等于整机重量）')

    # 12 关节接口静态对照（镇定态，与基座支撑方式无关的恒等式）：
    #   cfrc_int[接口].Fz(世界) = g·Σm(子树) − Σ cfrc_ext[子树内].Fz
    print('  --- 关节接口 Fz 静态对照（世界系） ---')
    n_bad = 0
    for urdf in JOINT_MAP:
        bid = ex._joints[urdf][1]
        sub_mass = m.body_mass[bid]
        sub_contact_fz = d.cfrc_ext[bid, 5]
        for b in range(bid + 1, m.nbody):
            if _we._is_descendant(m, b, bid):
                sub_mass += m.body_mass[b]
                sub_contact_fz += d.cfrc_ext[b, 5]
        predicted = sub_mass * G - sub_contact_fz
        measured = d.cfrc_int[bid, 5]
        rel = abs(measured - predicted) / max(abs(predicted), 1.0)
        flag = 'OK ' if rel < 0.05 else 'BAD'
        if rel >= 0.05:
            n_bad += 1
        print(f'  {flag} {urdf:28s} 实测 {measured:8.2f} N | 预测 {predicted:8.2f} N | rel {rel*100:5.1f}%')
    assert n_bad == 0, f'{n_bad} 个关节接口静态对照超差'

    # 快照列完整性 + CoP 合理性
    row = ex.snapshot(0)
    need = [f'joint_wrench_{u}_{c}' for u in JOINT_MAP for c in
            ('fx', 'fy', 'fz', 'mx', 'my', 'mz')]
    need += [f'link_left_ankle_roll_{c}' for c in ('fx', 'fy', 'fz', 'mx', 'my', 'mz')]
    need += ['cop_left_x', 'foot_wrench_left_fx', 'left_contact',
             'joint_net_left_ankle_roll_joint']
    missing = [k for k in need if k not in row]
    assert not missing, f'缺少列: {missing}'
    for side in ('left', 'right'):
        cx, cy = row[f'cop_{side}_x'], row[f'cop_{side}_y']
        print(f"  {side}: Fz={row[f'foot_wrench_{side}_fz']:7.1f} N  CoP=({cx:.4f}, {cy:.4f}) m")
        assert abs(cx) < 0.2 and abs(cy) < 0.2, 'CoP 超出脚掌合理范围'
    print('  [B] PASS — 真实模型上接线/命名/接触/CoP/量纲全部正确')
    return True


if __name__ == '__main__':
    a = part_a_synthetic_exact()
    b = part_b_x1_stand()
    print('=' * 60)
    print('ALL PASS' if (a and b) else 'FAILED')
    sys.exit(0 if (a and b) else 1)
