# X1 → 结构仿真载荷数据交付说明

**用途**：6DUUF1 小腿支架边界条件建立、F1 全下肢载荷库构建及早期结构筛查。
数据由 MuJoCo sim2sim 重放已有稳定行走策略生成，**不新增训练**。

## 1. 交付物

每个工况一个目录 `logs/wrench_export/<run_id>/`：

| 文件 | 内容 |
|---|---|
| `ankle_wrench.csv` | 时间轴 + 左右踝 pitch/roll 接口六维载荷（`link_<side>_ankle_<pitch/roll>_{fx..mz}`）+ 校核列（base_vel_x、接触标志、足底力、joint_net 踝力矩、指令） |
| `joint_wrench.csv` | 时间轴 + 左右 12 关节六维接口载荷（`joint_wrench_<joint>_{fx..mz}`，joint 名 = URDF 名，含 `_joint` 后缀） |
| `foot_wrench.csv` | 左右脚净六维地面反力（世界系，参考点=脚连杆原点）+ CoP（世界坐标，Fz<10 N 时为 NaN） |
| `manifest.json` | 硬件组要求的 8 项说明 + URDF↔MJCF 命名映射 + 验证报告 |

- 采样：**100 Hz 固定**，`sim_time_s = sample_index × 0.01 s`；physics dt = 0.001 s；无滤波/裁剪/降采样。
- 单位：力 N，力矩 N·m，长度 m。
- 载荷向量：`[Fx, Fy, Fz, Mx, My, Mz]`，**原始连续时程**（未按分量取极值拼装）。

## 2. 载荷物理语义（施加到 CAD 前必读）

| 项 | 约定 |
|---|---|
| 物理方向 | **父件/电机端通过关节作用在子连杆上的总传递力**（含轴承约束力 + 执行器力矩 + 关节阻尼），即"相邻件作用在该 link 上的力"。反作用力取负号即可 |
| 坐标系 | **子连杆局部坐标系（右手系）= URDF link frame = SolidWorks CAD 坐标系**（已与 URDF 逐位核对） |
| 参考点 | **关节锚点**。本 MJCF 所有关节 `jnt_pos=0`，锚点与子连杆原点重合 → **参考点即连杆原点，变换为恒等**（manifest 中每关节亦有 `anchor_in_child_link_frame_m` 字段） |
| 踝十字 | 踝 pitch 与 roll 两轴**同点**（URDF/MJCF 均为原点重合），即实机踝轴承/横轴中心。6DUUF1 直接使用 `link_<side>_ankle_pitch_*`（小腿支架↔踝十字接口）与 `link_<side>_ankle_roll_*`（踝十字↔足接口）可交叉校核 |
| 足部 | 世界系（Z 向上），地面对脚；CoP 为地面反力合力作用点世界坐标 |

## 3. 数据正确性验证（随 manifest 附带）

1. **合成模型精确静态证明**：重力补偿构造精确静态（qacc 残差=0），接口力/力矩与自由体平衡手算值误差 < 1e-12（`tools/validate_wrench_export.py` Part A）。
2. **真实 X1 静态恒等式校核**：12/12 关节接口 Fz 与"子树重量−子树内接触力"恒等式相对误差 0.0%（Part B）。
3. MuJoCo 数据语义（cfrc_int/cfrc_ext 布局 `[torque; force]`、参考点 `subtree_com[body_rootid]`、平移公式 `M@A = M@R + (R−A)×F`）经上述两步实证，非文档转述。

## 4. 已知限制（必须书面知会硬件组）

1. **串行踝等效**：MJCF 与 URDF 均将踝建为 pitch/roll 串联十字。交付的是**踝十字接口净六维载荷**（可直接作边界条件）；实机并联踝的**推杆载荷分布、横轴轴承单独反力不可由该模型导出**。
2. **X1 ≠ F1**：质量/惯量/执行器参数均为 X1；仅作参考，不作 F1 放行依据（硬件组已知）。
3. **无跑步工况**：现有策略仅覆盖行走/站立（|vx| ≤ 1.2 m/s）。
4. `joint_net_*` 为执行器实现力矩（actuator_force，+ = 绕关节轴正向），仅校核用。
5. `sample_index=0` 行为初始状态（接触未发展，载荷≈0 是正确初瞬态）。

## 5. 生成流程

```bash
# 训练机（含 torch + jit 策略，见 logs/wrench_checkpoints/README.md）
python scripts/export_wrench_data.py --task x1_dh_stand \
    --ckpt logs/wrench_checkpoints/<策略名> --run_id <工况名> \
    --vx 0.8 --vy 0.0 --yaw 0.0 --duration 20.0

# 管线自检（无需 torch：conda run -n x1）
python scripts/export_wrench_data.py --policy none --run_id stand_pd_test --duration 3
python tools/validate_wrench_export.py
```

## 6. 顺带发现的原有代码问题（不影响本交付，建议另行修复）

`scripts/sim2sim.py get_obs()` 存在两个恒真条件 bug（`'5_link' or ...`）与
`cfrc_ext[i][2]` 索引错误（实为 Mz 非 Fz），导致其 `base_height` 实为右脚 z、
`foot_forcez_*` 实为 Mz。**该 bug 不影响策略观测向量**（obs 未使用这些量），故本
导出脚本重放同一策略时步态不受影响；但用原 sim2sim.py 生成的历史诊断图/数据中
`base_height`、`foot_forcez_*` 两列无效。
