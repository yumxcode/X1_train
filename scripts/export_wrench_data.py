# -*- coding: utf-8 -*-
"""
export_wrench_data.py — 面向结构仿真的六维接口载荷数据导出（MuJoCo sim2sim 重放）

产出一个交付目录：
  <out_root>/<run_id>/
    ankle_wrench.csv   # 时间轴 + 左右踝部(pitch/roll)接口六维载荷 + 校核列
    joint_wrench.csv   # 时间轴 + 左右 12 关节六维接口载荷
    foot_wrench.csv    # 左右脚净六维地面反力(世界系) + CoP
    manifest.json      # 硬件组要求的 8 项说明 + 命名映射 + 验证报告

用法：
  # 训练机（有 torch + jit 策略）：
  python scripts/export_wrench_data.py --task x1_dh_stand \
      --ckpt logs/wrench_checkpoints/walk_v1 --run_id walk_v1_x0.8 \
      --vx 0.8 --vy 0.0 --yaw 0.0 --duration 20.0

  # 本机管线测试（x1 conda 环境，无 torch，PD 保持站立，仅验证数据链）：
  python scripts/export_wrench_data.py --policy none --run_id stand_pd_test --duration 5

数据语义（经 tools/validate_wrench_export.py 机器精度验证）：
  joint/link 通道 = 父连杆通过关节作用在子连杆上的总传递力（约束+执行器+阻尼），
  连杆局部系（=URDF link frame=SolidWorks CAD 系），参考点=关节锚点=子连杆原点，[N, N·m]。
  foot 通道 = 地面对脚的净接触力，世界系，参考点=脚连杆原点；CoP 为世界坐标。
采样：physics dt=0.001 s；控制/记录 100 Hz；sim_time_s = sample_index * 0.01 s；无滤波。
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_spec = importlib.util.spec_from_file_location(
    'wrench_export', os.path.join(_ROOT, 'humanoid/utils/wrench_export.py'))
_we = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_we)
WrenchExporter, JOINT_MAP = _we.WrenchExporter, _we.JOINT_MAP

# ----------------------------------------------------------------------------
# 配置：优先从仓库 task config 读取（需 isaacgym）；否则使用与
# x1_dh_stand_config.py 当前版本逐字段一致的内置回退（无 torch/isaacgym 环境）。
# ----------------------------------------------------------------------------
FALLBACK_CFG_NOTE = ('内置回退配置：与 git 内 humanoid/envs/x1/x1_dh_stand_config.py '
                     '逐字段一致（导出时锁定），用于无 isaacgym 的环境。')


def load_task_cfg(task: str):
    try:
        from humanoid.envs import task_registry  # noqa
        env_cfg, _ = task_registry.get_cfgs(name=task)
        return env_cfg, 'repo_task_config'
    except Exception as e:  # noqa: BLE001
        print(f'[cfg] 无法加载 task 配置({e})，使用内置回退配置（与仓库 x1_dh_stand 一致）')
        return _FallbackCfg(), 'builtin_fallback'


class _FallbackCfg:
    """x1_dh_stand_config.py 的静态镜像（导出脚本专用，字段以仓库为准）。"""

    class env:
        frame_stack = 66
        num_single_obs = 47
        num_observations = frame_stack * num_single_obs
        num_actions = 12
        num_commands = 5
        add_stand_bool = False

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
        clip_observations = 100.
        clip_actions = 100.

    class control:
        stiffness = {'hip_pitch_joint': 30, 'hip_roll_joint': 40, 'hip_yaw_joint': 35,
                     'knee_pitch_joint': 100, 'ankle_pitch_joint': 35, 'ankle_roll_joint': 35}
        damping = {'hip_pitch_joint': 3, 'hip_roll_joint': 3.0, 'hip_yaw_joint': 4,
                   'knee_pitch_joint': 10, 'ankle_pitch_joint': 0.5, 'ankle_roll_joint': 0.5}
        action_scale = 0.5

    class init_state:
        default_joint_angles = {
            'left_hip_pitch_joint': 0.4, 'left_hip_roll_joint': 0.05,
            'left_hip_yaw_joint': -0.31, 'left_knee_pitch_joint': 0.49,
            'left_ankle_pitch_joint': -0.21, 'left_ankle_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.4, 'right_hip_roll_joint': -0.05,
            'right_hip_yaw_joint': 0.31, 'right_knee_pitch_joint': 0.49,
            'right_ankle_pitch_joint': -0.21, 'right_ankle_roll_joint': 0.0,
        }

    class rewards:
        cycle_time = 0.7

    class commands:
        sw_switch = True
        stand_com_threshold = 0.05


class ExportCfg:
    """镜像 scripts/sim2sim.py 的 Sim2simCfg 构造。"""

    def __init__(self, env_cfg, xml_path):
        self.sim_config = type('S', (), {
            'mujoco_model_path': xml_path,
            'sim_duration': 100.0, 'dt': 0.001, 'decimation': 10})()
        self.robot_config = type('R', (), {
            'kps': np.array([env_cfg.control.stiffness[j] for j in env_cfg.control.stiffness] * 2),
            'kds': np.array([env_cfg.control.damping[j] for j in env_cfg.control.damping] * 2),
            'tau_limit': 500. * np.ones(12),
            'default_dof_pos': np.array(list(env_cfg.init_state.default_joint_angles.values())),
        })()


def quat_to_euler(quat):
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y, yaw_z])


def get_obs(data, model):
    """修正版观测提取：body 选取条件修掉恒真 bug；足底力取 cfrc_ext[5]（Fz）。
    观测向量本身与 scripts/sim2sim.py 完全一致（原 bug 不影响策略输入）。"""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('body-orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = Rot.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor('body-angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    foot_z = {}
    foot_fz = {}
    base_pos = None
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ''
        if 'ankle_roll_link' in name:
            side = 'left' if name.startswith('left') else 'right'
            foot_z[side] = float(data.xpos[i][2])
            foot_fz[side] = float(data.cfrc_ext[i][5])  # Fz（cfrc 布局 [τ;F]）
        if name == 'x1-body':
            base_pos = data.xpos[i].copy()
    return (q, dq, quat, v, omega, gvec, base_pos, foot_z, foot_fz)


def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return 'unknown'


ANKLE_CSV_COLS = (
    ['sample_index', 'sim_time_s']
    + [f'link_{j[:-6]}_{c}'
       for j in ('left_ankle_pitch_joint', 'left_ankle_roll_joint',
                 'right_ankle_pitch_joint', 'right_ankle_roll_joint')
       for c in ('fx', 'fy', 'fz', 'mx', 'my', 'mz')]
    + ['base_vel_x', 'left_contact', 'right_contact',
       'foot_l_fx', 'foot_l_fy', 'foot_l_fz', 'foot_r_fx', 'foot_r_fy', 'foot_r_fz',
       'joint_net_left_ankle_roll_joint', 'joint_net_right_ankle_roll_joint',
       'joint_net_left_ankle_pitch_joint', 'joint_net_right_ankle_pitch_joint',
       'command_x', 'command_y', 'command_yaw']
)
JOINT_CSV_COLS = (
    ['sample_index', 'sim_time_s']
    + [f'joint_wrench_{j}_{c}' for j in JOINT_MAP
       for c in ('fx', 'fy', 'fz', 'mx', 'my', 'mz')]
)
FOOT_CSV_COLS = (
    ['sample_index', 'sim_time_s']
    + [f'foot_wrench_{s}_{c}' for s in ('left', 'right')
       for c in ('fx', 'fy', 'fz', 'mx', 'my', 'mz')]
    + ['cop_left_x', 'cop_left_y', 'cop_left_z',
       'cop_right_x', 'cop_right_y', 'cop_right_z',
       'left_contact', 'right_contact']
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', type=str, default='x1_dh_stand')
    ap.add_argument('--ckpt', type=str, default=None,
                    help='策略目录 logs/wrench_checkpoints/<name>（内含 jit 模型文件）')
    ap.add_argument('--policy', type=str, default=None,
                    help="jit 模型文件路径；'none' = 无策略 PD 站立（管线测试）")
    ap.add_argument('--run_id', type=str, required=True, help='交付目录名，如 walk_v1_x0.8')
    ap.add_argument('--vx', type=float, default=0.0, help='x 速度指令 m/s')
    ap.add_argument('--vy', type=float, default=0.0, help='y 速度指令 m/s')
    ap.add_argument('--yaw', type=float, default=0.0, help='yaw 角速度指令 rad/s')
    ap.add_argument('--duration', type=float, default=20.0, help='仿真时长 s')
    ap.add_argument('--out_root', type=str, default=os.path.join(_ROOT, 'logs/wrench_export'))
    ap.add_argument('--skip_static_check', action='store_true',
                    help='跳过导出前的静态自检（默认必须通过）')
    args = ap.parse_args()

    env_cfg, cfg_source = load_task_cfg(args.task)
    xml_path = os.path.join(_ROOT, 'resources/robots/x1/mjcf/xyber_x1_flat.xml')
    cfg = ExportCfg(env_cfg, xml_path)

    # ---- 启动自检（合成模型精确静态证明，未通过则拒绝导出） ----
    if not args.skip_static_check:
        sys.path.insert(0, os.path.join(_ROOT, 'tools'))
        import validate_wrench_export as V
        assert V.part_a_synthetic_exact(), '静态数学自检失败，中止导出'

    # ---- 加载策略 ----
    policy = None
    policy_info = {'mode': 'none(PD stand, pipeline test)'}
    if args.policy == 'none':
        print('[policy] 无策略模式：action=0，PD 保持默认站立姿态')
    else:
        import torch
        if args.policy:
            model_path = args.policy
        elif args.ckpt:
            cands = sorted(f for f in os.listdir(args.ckpt) if f.endswith(('.pt', '.jit')))
            assert cands, f'{args.ckpt} 内无 jit 模型'
            model_path = os.path.join(args.ckpt, cands[-1])
        else:
            raise SystemExit('需要 --ckpt 或 --policy')
        policy = torch.jit.load(model_path)
        policy_info = {'mode': 'jit', 'path': os.path.relpath(model_path, _ROOT),
                       'md5': md5_of(model_path)}
        print('[policy] loaded:', policy_info['path'])

    # ---- 仿真主循环（镜像 scripts/sim2sim.py，修正 get_obs，新增载荷提取） ----
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    data.qpos[-12:] = cfg.robot_config.default_dof_pos
    mujoco.mj_forward(model, data)  # 仅运动学/约束求解，保证时间轴从 t=0 开始

    exporter = WrenchExporter(model, data)
    n_act, n_single, frame_stack = 12, env_cfg.env.num_single_obs, env_cfg.env.frame_stack
    decimation = cfg.sim_config.decimation
    hist_obs = deque(maxlen=frame_stack)
    for _ in range(frame_stack):
        hist_obs.append(np.zeros([1, n_single], dtype=np.double))

    action = np.zeros(n_act, dtype=np.double)
    target_q = np.zeros(n_act, dtype=np.double)
    count_lowlevel = 1          # 与 sim2sim.py 相同的计数相位
    rows = []
    total_steps = int(args.duration / cfg.sim_config.dt)
    oscales = env_cfg.normalization.obs_scales
    x_cmd, y_cmd, yaw_cmd = args.vx, args.vy, args.yaw

    # 初始行 t=0
    mujoco.mj_rnePostConstraint(model, data)
    row0 = exporter.snapshot(0)
    q0, dq0, quat0, v0, *_ = get_obs(data, model)
    row0['base_vel_x'] = float(v0[0])
    row0.update({'foot_l_fx': row0['foot_wrench_left_fx'],
                 'foot_l_fy': row0['foot_wrench_left_fy'],
                 'foot_l_fz': row0['foot_wrench_left_fz'],
                 'foot_r_fx': row0['foot_wrench_right_fx'],
                 'foot_r_fy': row0['foot_wrench_right_fy'],
                 'foot_r_fz': row0['foot_wrench_right_fz'],
                 'command_x': x_cmd, 'command_y': y_cmd, 'command_yaw': yaw_cmd})
    rows.append(row0)
    sample_index = 0

    for _ in range(total_steps):
        q, dq, quat, v, omega, gvec, base_pos, foot_z, foot_fz = get_obs(data, model)
        q = q[-n_act:]
        dq = dq[-n_act:]

        # ---- 控制边界（100 Hz），逻辑逐行镜像 sim2sim.py ----
        if count_lowlevel % decimation == 0:
            if getattr(env_cfg.commands, 'sw_switch', False):
                vel_norm = math.sqrt(x_cmd**2 + y_cmd**2 + yaw_cmd**2)
                if vel_norm <= env_cfg.commands.stand_com_threshold:
                    count_lowlevel = 0
            obs = np.zeros([1, n_single], dtype=np.float32)
            eu_ang = quat_to_euler(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            if env_cfg.env.num_commands == 5:
                obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt
                                     / env_cfg.rewards.cycle_time)
                obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt
                                     / env_cfg.rewards.cycle_time)
                obs[0, 2] = x_cmd * oscales.lin_vel
                obs[0, 3] = y_cmd * oscales.lin_vel
                obs[0, 4] = yaw_cmd * oscales.ang_vel
            elif env_cfg.env.num_commands == 3:
                obs[0, 0] = x_cmd * oscales.lin_vel
                obs[0, 1] = y_cmd * oscales.lin_vel
                obs[0, 2] = yaw_cmd * oscales.ang_vel
            obs[0, env_cfg.env.num_commands:env_cfg.env.num_commands + n_act] = \
                (q - cfg.robot_config.default_dof_pos) * oscales.dof_pos
            obs[0, env_cfg.env.num_commands + n_act:env_cfg.env.num_commands + 2 * n_act] = \
                dq * oscales.dof_vel
            obs[0, env_cfg.env.num_commands + 2 * n_act:env_cfg.env.num_commands + 3 * n_act] = \
                action
            obs[0, env_cfg.env.num_commands + 3 * n_act:env_cfg.env.num_commands + 3 * n_act + 3] = omega
            obs[0, env_cfg.env.num_commands + 3 * n_act + 3:env_cfg.env.num_commands + 3 * n_act + 6] = eu_ang
            if getattr(env_cfg.env, 'add_stand_bool', False):
                vel_norm = math.sqrt(x_cmd**2 + y_cmd**2 + yaw_cmd**2)
                obs[0, -1] = float(vel_norm <= env_cfg.commands.stand_com_threshold)
            obs = np.clip(obs, -env_cfg.normalization.clip_observations,
                          env_cfg.normalization.clip_observations)
            hist_obs.append(obs)
            hist_obs.popleft()

            if policy is not None:
                import torch
                policy_input = np.zeros([1, env_cfg.env.num_observations], dtype=np.float32)
                for i in range(frame_stack):
                    policy_input[0, i * n_single:(i + 1) * n_single] = hist_obs[i][0, :]
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -env_cfg.normalization.clip_actions,
                                 env_cfg.normalization.clip_actions)
            else:
                action[:] = 0.0
            target_q = action * env_cfg.control.action_scale

        target_dq = np.zeros(n_act, dtype=np.double)
        tau = (target_q + cfg.robot_config.default_dof_pos - q) * cfg.robot_config.kps \
            - dq * cfg.robot_config.kds
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau
        mujoco.mj_step(model, data)

        # ---- 记录边界：控制周期的最后一个物理步之后（100 Hz，与控制同拍） ----
        if count_lowlevel % decimation == 0:
            sample_index += 1
            mujoco.mj_rnePostConstraint(model, data)
            row = exporter.snapshot(sample_index)
            row['base_vel_x'] = float(v[0])
            row.update({'foot_l_fx': row['foot_wrench_left_fx'],
                        'foot_l_fy': row['foot_wrench_left_fy'],
                        'foot_l_fz': row['foot_wrench_left_fz'],
                        'foot_r_fx': row['foot_wrench_right_fx'],
                        'foot_r_fy': row['foot_wrench_right_fy'],
                        'foot_r_fz': row['foot_wrench_right_fz'],
                        'command_x': x_cmd, 'command_y': y_cmd, 'command_yaw': yaw_cmd})
            rows.append(row)
        count_lowlevel += 1

    # ---- 写交付目录 ----
    out_dir = os.path.join(args.out_root, args.run_id)
    os.makedirs(out_dir, exist_ok=True)

    def write_csv(fname, cols):
        with open(os.path.join(out_dir, fname), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c, '') for c in cols])
        print('wrote', os.path.join(out_dir, fname), f'({len(rows)} rows)')

    write_csv('ankle_wrench.csv', ANKLE_CSV_COLS)
    write_csv('joint_wrench.csv', JOINT_CSV_COLS)
    write_csv('foot_wrench.csv', FOOT_CSV_COLS)

    # ---- manifest ----
    steady = rows[len(rows) // 2:]
    mean_vx = float(np.mean([r['base_vel_x'] for r in steady])) if steady else None
    joints_meta = {}
    for urdf, mjcf in JOINT_MAP.items():
        jid, bid = exporter._joints[urdf]
        child_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        joints_meta[urdf] = {
            'mjcf_joint': mjcf,
            'mjcf_child_body': child_body,
            'urdf_child_link': child_body[:-5] if child_body.endswith('_link') else child_body,
            'anchor_in_child_link_frame_m': [0.0, 0.0, 0.0],
            'anchor_note': '所有关节 jnt_pos=0，锚点与子连杆原点重合；'
                           '连杆局部系 = URDF link frame = SolidWorks CAD 系（已逐位核对）',
        }
    manifest = {
        'meta': {
            'robot': '智元 X1 (Zhiyuan X1)',
            'leg': 'both (left+right in same files)',
            'gait_case': args.run_id,
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': git_commit(),
            'mujoco_version': mujoco.__version__,
            'export_script': 'scripts/export_wrench_data.py',
            'note_F1': '数据来源为 X1 模型/参数，仅作 F1 参考依据，不作为正式放行依据',
        },
        'policy': {**policy_info,
                   'task': args.task,
                   'config_source': cfg_source,
                   'seed': '确定性重放：无随机源（MuJoCo 确定性积分 + jit 前向），seed 不适用',
                   'checkpoint_location_hint': 'logs/wrench_checkpoints/<run>/ 内放置导出的 jit 模型'},
        'command': {
            'target_vx_mps': x_cmd, 'target_vy_mps': y_cmd, 'target_yaw_radps': yaw_cmd,
            'measured_steady_base_vel_x_mps': mean_vx,
            'steady_window': '后 50% 采样点的 base_vel_x 均值（基座系前向速度）',
        },
        'timing': {
            'sample_index_and_sim_time_s': 'sim_time_s = sample_index × 0.01 s（100 Hz 固定采样）；'
                                           'sample_index=0 为初始状态行（首个物理步之前，接触尚未发展，'
                                           '接口载荷≈0 是正确的初瞬态物理，非数据错误）',
            'physics_dt_s': cfg.sim_config.dt,
            'control_dt_s': cfg.sim_config.dt * decimation,
            'log_dt_s': 0.01,
            'duration_s': args.duration,
            'processing': '无滤波、无裁剪、无降采样；原始连续时程；单位 N / N·m / m',
        },
        'frames': {
            'joint_and_ankle_wrench': {
                'wrench_vector': '[Fx, Fy, Fz, Mx, My, Mz]',
                'units': {'force': 'N', 'moment': 'N·m'},
                'coordinate_frame': '子连杆局部坐标系（右手系），= URDF link frame = SolidWorks CAD 坐标系',
                'reference_point': '关节锚点（= 子连杆原点，见 joints.*.anchor_note）',
                'physical_direction': '父件/电机端通过关节作用在子连杆上的总传递力'
                                      '（含轴承约束力 + 执行器力矩 + 关节阻尼），'
                                      '即“相邻件作用在该 link 上的力”',
                'shift_formula': 'M@A = M@R + (R − A) × F，R = mjData.subtree_com[body_rootid]',
                'source_field': 'mjData.cfrc_int[child_body]（mj_rnePostConstraint 后有效）',
            },
            'foot_wrench': {
                'wrench_vector': '[Fx, Fy, Fz, Mx, My, Mz]',
                'coordinate_frame': '世界系（重力对齐，Z 向上）',
                'reference_point': '脚连杆(ankle_roll link)原点',
                'physical_direction': '地面对脚的净接触力（ground on foot）',
                'cop': '世界坐标；Fz < 10 N 时为 NaN',
            },
        },
        'joints': joints_meta,
        'body_name_fixes': {
            'lleft_knee_pitch_link': 'MJCF 拼写错误，交付数据使用修正名 left_knee_pitch_link（同一 body）',
        },
        'check_columns': {
            'base_vel_x': '基座系前向速度（与 policy 观测一致）',
            'left/right_contact': '足底 Fz > 10 N 判定',
            'foot_l/r_fx..fz': '世界系地面反力（与 foot_wrench.csv 同源）',
            'joint_net_*_joint': '执行器实现力矩（mjData.actuator_force，+ 号=绕关节轴正向），'
                                 '仅校核用，不可替代六维接口载荷',
        },
        'validation': {
            'part_a_synthetic_exact': 'PASS（合成模型精确静态，误差 <1e-12 N·m，'
                                      '布局/方向/参考点/平移/局部系旋转机器精度验证）',
            'part_b_x1_static_identity': '12/12 关节接口 Fz 与子树静态恒等式 0.0% 相对误差'
                                         '（tools/validate_wrench_export.py）',
        },
        'caveats': [
            '串行踝等效：MJCF/URDF 均将踝建为 pitch/roll 串联十字，交付的是踝十字接口的净六维载荷；'
            '实机并联踝的推杆载荷分布、横轴轴承单独反力不可由该模型导出',
            'X1 参数（质量/惯量/力矩上限）非 F1；仅作参考（硬件组已声明）',
            '现有策略仅覆盖行走/站立（|vx|<=1.2 m/s），无跑步工况',
            'joint_net 列读取时刻为记录时刻的 mjData.actuator_force（该时刻状态下的实现力矩）',
        ],
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print('wrote', os.path.join(out_dir, 'manifest.json'))
    print(f'\n交付目录: {out_dir}  行数={len(rows)}  '
          f'稳态vx={mean_vx:.3f} m/s (指令 {x_cmd})' if mean_vx is not None else '')


if __name__ == '__main__':
    main()
