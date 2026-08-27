# X1 IsaacSim 行走训练 + Sim2Sim 严格通过标准（v1.0）

> 本标准在训练启动前固定，禁止事后放宽。所有阈值基于以下标定：
> - v1 基线（TASK_20260827_024，1000 iters）：mean_reward≈7.7，mean_episode_length≈307/2400 步，rew_tracking_lin_vel≈0.08（图表口径 ≈ 每秒奖励率，量级上限=scale 值）
> - 理论上限：正奖励 scale 总和 ≈ 16.1/步（含 stand_still），episode 上限 2400 步（24 s @100 Hz）
> - 训练指令范围：vx∈[-0.4,1.2]、vy∈[-0.4,0.4]、wz∈[-0.6,0.6]，步态周期 0.7 s

## A. 训练通过标准（在最终训练任务的平台图表上评估，取 last-100-iter 均值）

| ID | 指标 | 阈值（严格） | 依据 |
|----|------|--------------|------|
| T1 | Train/mean_reward | ≥ 200 | 收敛策略 ≈ 0.08–0.13/步 × 2400 步；200 对应 ≈52% 理论上限且 ≈26× v1 |
| T2 | Train/mean_episode_length | ≥ 2100 / 2400 | 绝大多数 episode 以超时结束而非摔倒（v1 仅 307） |
| T3 | Episode/rew_tracking_lin_vel | ≥ 1.20 | scale=1.8 的 67%，对应平均速度误差 err² ≤ 25·ln(1.8/1.2)≈0.4 → err≈0.63 m/s 量级以下 |
| T4 | Episode/rew_ref_joint_pos | ≥ 1.40 | scale=2.2 的 64%，步态跟踪成型 |
| T5 | Episode/rew_collision | ≥ -0.005 | 躯干触地率近零 |
| T6 | 稳定性 | last-100 mean_reward ≥ 0.9 × 全程最大 | 末端无崩塌 |

## B. Sim2Sim（Mujoco）通过标准 —— 每个 trial 必须全部满足

Trial 定义（各 24 s，1 kHz 物理 / 100 Hz 策略，与训练 episode 等长）：
- `forward`：0–2 s 站立 → 2–20 s 前进 vx=1.0 → 20–24 s 站立
- `omni`：0–2 s 站立 → 2–8 s (0.5, 0.3, 0) → 8–14 s (0.5, -0.3, 0) → 14–19 s (0, 0, 0.5) → 19–24 s (0.3, 0, -0.4)
- `max`：0–2 s 站立 → 2–21 s 前进 vx=1.2 → 21–24 s 站立

| ID | 指标 | 阈值（严格） |
|----|------|--------------|
| S1 | 无跌倒 | 3 个 trial 全程存活；终止条件（躯干接触力 >1 N、\|roll\|>1.5、\|pitch\|>1.5、根高 <0.40 m）从未触发 |
| S2 | 速度跟踪（活跃窗口，切换后 1 s 宽限剔除） | \|Δvx\| 中位数 ≤ 0.25 m/s（\|vx_cmd\|≥0.5 段）；\|Δvy\| 中位数 ≤ 0.20 m/s（\|vy_cmd\|≥0.2 段）；\|Δwz\| 均值 ≤ 0.30 rad/s（\|wz_cmd\|≥0.2 段） |
| S3 | 躯干高度 | 活跃窗口均值 ∈ [0.52, 0.70] m（目标 0.61）；全程最小 > 0.45 m |
| S4 | 力矩安全 | 指令力矩永不超出执行器 ctrlrange（硬性）；饱和步比例（\|τ\|≥0.98·限幅）≤ 0.5% |
| S5 | 位移达成 | forward：净前进位移 ≥ 0.70×∫vx_cmd dt = 12.6 m；max：≥ 0.65×∫vx_cmd dt ≈ 14.8 m |
| S6 | 视频交付 | 3 个 mp4（每 trial 一个），时长 ≥ 20 s，分辨率 ≥ 640×480，帧率 ≥ 24 fps，机器人可见行走（根位置行程 > 2 m） |

## C. 判定与交付

- **总判定 = A(6/6) ∧ B(6/6)**，任一子项 FAIL 即总 FAIL。
- sim2sim 脚本（`humanoid_lab/scripts/sim2sim_eval.py`）自动计算 B 部分并输出 `verdict.json` + 控制台 PASS/FAIL 表；exit code：0=全过，3=指标未过，4=视频渲染失败。
- 交付物：训练曲线（平台图表导出）、3 个行走 mp4（SDK 自动上传至任务页 `videoUrl`）、指标打包 `logs/x1_dh_stand/gm_play/sim2sim_metrics.pt`、verdict.json。
- 若训练标准未达而曲线仍上升：允许追加训练迭代（resume），不允许降低阈值。
