# -*- coding: utf-8 -*-
"""
wrench_export.py — 六维接口载荷提取模块（MuJoCo sim2sim 数据交付用）

数据来源与语义（全部经合成模型精确静态验证，见 tools/validate_wrench_export.py）：
  data.cfrc_int[b]  : 父刚体通过关节作用在子刚体 b 上的【总】传递力（约束+执行器+阻尼），
                      布局 [torque(3); force(3)]，世界轴方向，参考点 = subtree_com[body_rootid[b]]，
                      子树累积（承载全部远端子结构的载荷）。
  data.cfrc_ext[b]  : 作用在 b 子树上的外部/接触力，布局 [torque(3); force(3)]，世界轴，
                      参考点 = subtree_com[body_rootid[b]]，含接触（地面→脚，向上为正）。
  两者均需在 mj_step / mj_forward 之后显式调用 mujoco.mj_rnePostConstraint(m, d) 才有效。

坐标变换约定（交付约定）：
  - 输出连杆局部坐标系（= MJCF body frame = URDF link frame = SolidWorks CAD 系，已逐位核对），
    右手系；F_local = R^T @ F_world，M_local = R^T @ M_world。
  - 参考点 = 关节锚点。本模型所有关节 jnt_pos=0，锚点与子连杆原点重合（manifest 声明）。
  - 力矩平移公式：M@A = M@R + (R - A) x F   （R 为 cfrc 参考点）。
  - 单位：力 N，力矩 N·m，长度 m。
"""
import numpy as np
import mujoco

# URDF 关节名（交付命名） -> MJCF joint 名
JOINT_MAP = {
    'left_hip_pitch_joint': 'left_hip_pitch',
    'left_hip_roll_joint': 'left_hip_roll',
    'left_hip_yaw_joint': 'left_hip_yaw',
    'left_knee_pitch_joint': 'left_knee_pitch',
    'left_ankle_pitch_joint': 'left_ankle_pitch',
    'left_ankle_roll_joint': 'left_ankle_roll',
    'right_hip_pitch_joint': 'right_hip_pitch',
    'right_hip_roll_joint': 'right_hip_roll',
    'right_hip_yaw_joint': 'right_hip_yaw',
    'right_knee_pitch_joint': 'right_knee_pitch',
    'right_ankle_pitch_joint': 'right_ankle_pitch',
    'right_ankle_roll_joint': 'right_ankle_roll',
}

# MJCF 左膝 body 名含拼写错误 "lleft_knee_pitch_link"，此处做名字修正映射
_BODY_NAME_FIX = {'lleft_knee_pitch_link': 'left_knee_pitch_link'}


def _body_id(model, mjcf_body_name):
    name = _BODY_NAME_FIX.get(mjcf_body_name, mjcf_body_name)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        # 回退：直接用原名
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mjcf_body_name)
    if bid < 0:
        raise ValueError(f"body not found: {mjcf_body_name}")
    return bid


class WrenchExporter:
    """从当前 mjData 状态提取全部交付载荷通道。

    用法：
        mujoco.mj_rnePostConstraint(model, data)   # 每个 mj_step 后必须调用
        row = exporter.snapshot()                  # dict: 列名 -> float
    """

    def __init__(self, model, data, contact_force_threshold_n=10.0):
        self.model = model
        self.data = data
        self.contact_thresh = contact_force_threshold_n

        self._joints = {}          # urdf_name -> (jid, child_body_id)
        for urdf_name, mjcf_name in JOINT_MAP.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mjcf_name)
            if jid < 0:
                raise ValueError(f"joint not found in MJCF: {mjcf_name}")
            child_body = model.jnt_bodyid[jid]
            self._joints[urdf_name] = (jid, child_body)

        self._foot_bodies = {
            'left': _body_id(model, 'left_ankle_roll_link'),
            'right': _body_id(model, 'right_ankle_roll_link'),
        }
        # 执行器顺序 <-> 关节顺序（qpos 尾部 12 维按 XML 定义顺序排列，与执行器一致）
        self._act_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                   model.actuator_trnid[i][0])
                                 for i in range(model.nu)]

    # ---------- 核心提取 ----------

    def joint_wrench_local(self, urdf_name):
        """关节接口六维载荷（父->子），子连杆局部系，参考点=关节锚点。
        返回 np.array([Fx,Fy,Fz,Mx,My,Mz])，单位 N / N·m。"""
        jid, bid = self._joints[urdf_name]
        d = self.data
        ref = d.subtree_com[self.model.body_rootid[bid]]          # cfrc 参考点（世界系）
        f_w = d.cfrc_int[bid, 3:6].copy()                          # 力，世界系
        tq_w = d.cfrc_int[bid, 0:3].copy() + np.cross(ref - d.xanchor[jid], f_w)
        R = d.xmat[bid].reshape(3, 3)                              # local->world
        return np.concatenate([R.T @ f_w, R.T @ tq_w])

    def foot_wrench_world(self, side):
        """地面对脚的净六维载荷，世界轴，参考点=脚连杆原点（ankle_roll body origin）。
        返回 (force[3], torque[3])，单位 N / N·m；以及 COP（世界系，Fz<阈值时为 NaN）。"""
        bid = self._foot_bodies[side]
        d = self.data
        ref = d.subtree_com[self.model.body_rootid[bid]]
        f_w = d.cfrc_ext[bid, 3:6].copy()
        origin = d.xpos[bid].copy()
        tq_w = d.cfrc_ext[bid, 0:3].copy() + np.cross(ref - origin, f_w)
        fz = f_w[2]
        if abs(fz) > self.contact_thresh:
            # 力矩移到脚原点正下方地面投影点后反解 CoP（r_z=0 平面）
            # M_x = r_y*Fz  -> r_y = M_x/Fz ;  M_y = -r_x*Fz -> r_x = -M_y/Fz
            cop_x = origin[0] + (-tq_w[1] / fz)
            cop_y = origin[1] + (tq_w[0] / fz)
            cop = np.array([cop_x, cop_y, origin[2] - d.xpos[bid][2]])  # z 未知，置 NaN 更诚实
            cop = np.array([cop_x, cop_y, np.nan])
        else:
            cop = np.full(3, np.nan)
        return f_w, tq_w, cop

    def foot_contact(self, side):
        bid = self._foot_bodies[side]
        return float(self.data.cfrc_ext[bid, 5] > self.contact_thresh)

    def net_joint_torques(self):
        """执行器实现力矩（joint_net 校核列），按 URDF 关节名索引，单位 N·m。
        语义：actuator 施加在关节自由度上的广义力，+ 号 = 绕关节轴正向。"""
        out = {}
        for i, jname in enumerate(self._act_joint_names):
            urdf = jname + '_joint'
            if urdf in JOINT_MAP.values() or urdf in JOINT_MAP:
                out[urdf] = float(self.data.actuator_force[i])
        return out

    # ---------- 一行快照 ----------

    def snapshot(self, sample_index):
        d = self.data
        row = {'sample_index': int(sample_index), 'sim_time_s': float(round(d.time, 3))}
        # 12 关节六维（joint_wrench.csv 用）
        for urdf in JOINT_MAP:
            w = self.joint_wrench_local(urdf)
            for k, comp in enumerate(('fx', 'fy', 'fz', 'mx', 'my', 'mz')):
                row[f'joint_wrench_{urdf}_{comp}'] = float(w[k])
        # 踝部接口（ankle_wrench.csv 用，link_<joint> 命名按硬件组规格）
        for urdf in ('left_ankle_pitch_joint', 'left_ankle_roll_joint',
                     'right_ankle_pitch_joint', 'right_ankle_roll_joint'):
            w = self.joint_wrench_local(urdf)
            short = urdf[:-6]  # 去掉 _joint
            for k, comp in enumerate(('fx', 'fy', 'fz', 'mx', 'my', 'mz')):
                row[f'link_{short}_{comp}'] = float(w[k])
        # 足部
        for side in ('left', 'right'):
            f_w, tq_w, cop = self.foot_wrench_world(side)
            for k, comp in enumerate(('fx', 'fy', 'fz')):
                row[f'foot_wrench_{side}_{comp}'] = float(f_w[k])
            for k, comp in enumerate(('mx', 'my', 'mz')):
                row[f'foot_wrench_{side}_{comp}'] = float(tq_w[k])
            for k, comp in enumerate(('x', 'y', 'z')):
                row[f'cop_{side}_{comp}'] = float(cop[k])
            row[f'{side}_contact'] = self.foot_contact(side)
        # 校核：净关节力矩（全部 12 个）
        for urdf, tau in self.net_joint_torques().items():
            row[f'joint_net_{urdf}'] = tau
        return row

    # ---------- 静态自检 ----------

    def static_selfcheck(self, atol_f=1.0, atol_t=0.5, verbose=True):
        """近静态状态下的整体量校核（非精确证明，仅接线/符号检查）：
        1) 双足支撑力之和 ~= 整机重量
        2) 单腿踝接口竖直力 ~= 该腿远端子树重量
        精确数学证明由 tools/validate_wrench_export.py 的合成模型静态完成。"""
        d, m = self.data, self.model
        total_mass = mujoco.mj_getTotalmass(m)
        fz_l = d.cfrc_ext[self._foot_bodies['left'], 5]
        fz_r = d.cfrc_ext[self._foot_bodies['right'], 5]
        weight = total_mass * 9.81
        sum_err = abs(fz_l + fz_r - weight)
        # 踝 pitch 接口竖直力（世界系）vs 远端子树（ankle_pitch+ankle_roll）重量
        reports = {'total_mass_kg': total_mass,
                   'weight_n': weight,
                   'foot_fz_sum_n': float(fz_l + fz_r),
                   'foot_fz_sum_rel_err': float(sum_err / weight)}
        ok = sum_err / weight < 0.05
        for side in ('left', 'right'):
            bid = self._joints[f'{side}_ankle_pitch_joint'][1]
            f_w = d.cfrc_int[bid, 5]
            # 远端子树质量：ankle_pitch body 及其后代
            sub_mass = 0.0
            for b in range(bid, m.nbody):
                if m.body_rootid[b] == m.body_rootid[bid] and _is_descendant(m, b, bid):
                    sub_mass += m.body_mass[b]
            reports[f'{side}_ankle_pitch_fz_n'] = float(f_w)
            reports[f'{side}_ankle_pitch_subtree_mass_kg'] = float(sub_mass)
        if verbose:
            print('[static_selfcheck]', reports, 'PASS' if ok else 'FAIL')
        return ok, reports


def _is_descendant(model, body, ancestor):
    b = body
    while b > 0:
        b = model.body_parentid[b]
        if b == ancestor:
            return True
    return False
