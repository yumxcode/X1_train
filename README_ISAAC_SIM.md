# X1 训练 pipeline 迁移：Isaac Gym → Isaac Sim (IsaacLab)

本仓库在保留原 Isaac Gym 版本（`humanoid/`）的基础上，新增了基于 **Isaac Sim 5.0 + IsaacLab 2.2** 的训练管线（`humanoid_lab/`），用于在 Gradmotion 平台的 IsaacSim 镜像上训练 X1 站立/行走策略。

## 目录结构

```
humanoid_lab/
├── algo/                     # 从原版复制的 PPO（去掉 wandb 依赖）
│   ├── actor_critic_dh.py    # teacher-student 网络（长历史 CNN + 状态估计器）
│   ├── dh_ppo.py             # DHPPO
│   ├── dh_on_policy_runner.py# 训练 runner（tensorboard + checkpoint，兼容原格式）
│   └── rollout_storage.py
├── envs/
│   ├── x1/
│   │   ├── x1_env_cfg.py     # IsaacLab DirectRLEnvCfg（scene/sim/URDF/地形 + 全部原版超参）
│   │   └── x1_dh_stand_env.py# Direct workflow 环境实现（obs/reward/gait/domain-rand 全逻辑迁移）
│   └── vec_env_adapter.py    # DirectRLEnv -> rsl_rl 风格 VecEnv 适配器
├── scripts/
│   ├── train_lab.py          # 训练入口（AppLauncher -> env -> DHOnPolicyRunner）
│   └── utils.py              # class_to_dict / get_load_path
└── utils/torch_utils.py      # 原 isaacgym.torch_utils 函数的纯 torch 实现
```

## 训练命令

```bash
python humanoid_lab/scripts/train_lab.py --headless --num_envs 4096 --run_name xxx
# 恢复训练
python humanoid_lab/scripts/train_lab.py --headless --resume --load_run -1 --checkpoint -1
```

模型与 tensorboard 保存在 `logs/x1_dh_stand/exported_data/<date><run_name>/model_<iter>.pt`（与原版及 Gradmotion 扫描路径一致）。

## 关键实现映射（Isaac Gym → IsaacLab）

| Isaac Gym（原版） | IsaacLab（本实现） |
|---|---|
| `gym.create_sim` | `SimulationContext`（DirectRLEnv 内部） |
| trimesh 地形（flat/rough/slope 混合） | `TerrainImporter` + HF 地形生成器（比例与原版对应） |
| `create_env` + URDF actor | `Articulation` + `UrdfFileCfg`（运行时 URDF→USD，进程内缓存） |
| `acquire/refresh_*_tensor` | `ArticulationData` / `ContactSensor.data.net_forces_w` |
| `set_dof_actuation_force_tensor`（每 1ms 重算 PD） | `_apply_action` 每 substep 调用，`set_joint_effort_target` |
| `set_dof_state_tensor_indexed` | `write_joint_state_to_sim(pos, vel, env_ids)` |
| `set_actor_root_state_tensor[_indexed]` | `write_root_state_to_sim` / `write_root_velocity_to_sim` |
| 关节 armature/摩擦/阻尼随机化 | `write_joint_{armature,friction_coefficient,damping}_to_sim` |
| base/link 质量随机化 | `root_physx_view.get/set_masses` |
| 四元数 xyzw | IsaacLab 为 wxyz，属性层做转换，内部逻辑保持原 xyzw 惯例 |

## 已记录的工程假设（与原版的已知差异）

1. **求解器内关节摩擦/阻尼保持 0**：原版对 URDF 默认值 0 做“乘法随机化”，实际效果恒 0；电机摩擦改由力矩计算中的显式 Coulomb/viscous 项模拟（同原版）。armature 随机化为直接赋值，予以保留。
2. **机器人-地形物理摩擦随机化暂未启用**（Phase 1）：privileged obs 中的 `env_frictions` 仍为随机值（保持观测维度与语义），物理侧使用固定地形材质（0.6/0.6/0）。
3. **COM 位移随机化暂未写回物理**（Phase 1）：观测维度保持不变。
4. **地形**：原版 20×20 trimesh 混合地形以 IsaacLab HF 生成器近似（flat 0.3 / rough 0.3 / slope±0.2，尺寸 8m、网格 0.1m、curriculum 关闭、初始 level≤5）。
5. **PD 控制频率**：与原版一致——物理 1kHz、policy 100Hz（decimation 10），每个物理子步重算显式 PD 力矩。
