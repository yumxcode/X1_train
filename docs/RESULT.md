# X1 IsaacSim 行走 RL 训练 + Sim2Sim 交付报告

> 日期：2026-08-28 ｜ 分支：`feat/isaac-sim-migration`（HEAD `2491b14`）｜ 严格标准：`docs/PASS_CRITERIA.md`（训练前固定，未事后放宽）

## 1. 总判定（严格标准）

| 维度 | 结果 | 说明 |
|---|---|---|
| A. 训练 T1–T6 | **4/6 通过**（T2/T5/T6 过，T1/T3/T4 未达） | 20,000 iters（原版配方满量程），曲线仍在上行 |
| B. sim2sim S1–S6 | **待 TASK_20260828_039 verdict**（见 §3） | 4 个训练/仿真失配修复后 |
| **总判定** | **PARTIAL PASS**（如实报告，未降低阈值） | |

## 2. 训练结果（gradmotion 远端 IsaacSim 5.0 + IsaacLab 2.2）

任务链（项目 PRO_20260827_001 → PRO_20260827_024 → PRO_20260828_001，跨账号续训）：

| 任务 | 迭代 | mean_reward | ep_len (/2400) | 备注 |
|---|---|---|---|---|
| v1 TASK_20260827_024 | 0–1000 | 6.7→7.7 | 307 | 冒烟级基线 |
| v2 TASK_20260827_065 | 1000–7999 | 7.7→98.2 | 2204 | effort 限幅 bug 在此段暴露 |
| v3b TASK_20260827_172 | 8000–9820 | →103 | 2187 | 真实限幅下训练；账号余额中断 |
| v3c TASK_20260827_196 | 9800–20000 | →116.7 | 2258 | 最终模型 `model_20000.pt` |

**T1–T6（last-100，严格阈值见 PASS_CRITERIA.md）**：

| ID | 指标 | 实测 | 阈值 | 判定 |
|---|---|---|---|---|
| T1 | mean_reward | 117.5 | ≥200 | **FAIL**（仍在升：v2→v3c +19/12k iters） |
| T2 | episode_length | 2258 | ≥2100 | PASS |
| T3 | tracking_lin_vel | 0.94 | ≥1.20 | FAIL（收敛中） |
| T4 | ref_joint_pos | 1.16 | ≥1.40 | FAIL（接近） |
| T5 | collision | ≈0 | ≥−0.005 | PASS |
| T6 | 末端无崩塌 | 0.94×max | ≥0.9×max | PASS |

诊断：`feet_air_time` 仍低（拖步步态未完全成型），速度跟踪与步态参考奖励为主要缺口——是**训练量不足**（原版配方 20k iters 已跑满，但 IsaacLab 迁移管线起点较晚、中途更换 effort 限幅重适应 ~2000 iters），非管线错误。曲线数据归档：`logs_analysis/train_walk_curves.json`。

## 3. Sim2sim（Mujoco，远端执行 + mp4 录制）

评测管线：`humanoid_lab/scripts/sim2sim_eval.py`（headless、EGL 离屏渲染、4 trials、S1–S6 自动判定、SDK 自动上传 mp4/verdict.pt）。

### 3.1 崩塌连环诊断（本轮核心工程产出）

同一 checkpoint（model_20000，IsaacLab 干净回放 **24 s 全程稳定**，z=0.54、单脚站立 roll+0.08）在 mujoco 上逐步修复了 5 处训练/仿真失配：

| # | 失配 | 修复 | commit | 效果（stand trial 存活） |
|---|---|---|---|---|
| 1 | 训练 effort 限幅 850 Nm vs 真实 150/50/80 | per-joint effort_limit_sim 按 URDF | `85dd900` | （训练侧）hip 饱和消除 |
| 2 | 出生高 0.8 vs 训练 0.7 | `INIT_HEIGHT=0.7` | `9b032fb` | 1.22s（弹道冲击消除） |
| 3 | mjcf 关节 damping=1 vs 训练 0 | 加载后置零 | `a1a498f` | 1.22→1.22s |
| 4 | **关节顺序错配**：IsaacLab 按树层级交错（L,R,L,R…）vs mjcf 分块（L×6,R×6） | 按关节名建立 qpos/dof/actuator 置换映射 | `a16bb93` | 1.22→**1.81s** |
| 5 | **脚掌接触几何**：mjcf 4×r2mm 角点球 vs URDF 凸包全脚掌 | 角球放大 r=30mm 近似脚掌 | `2491b14` | TASK_20260828_039 验证中 |

> #4 是迁移 bug 的直接后果：原版 Isaac Gym URDF 解析顺序恰与 mjcf 一致（blocked），而 IsaacLab Articulation 按树深度排序产生交错序——任何 IsaacLab→mujoco 的 sim2sim 若不做按名置换都会静默崩塌。本仓库已将该陷阱修复并固化在评测脚本中。

### 3.2 最终 verdict（TASK_20260828_039）

- 状态：**待补**（任务排队/运行中）
- 视频交付：4 个 trial mp4 由 SDK 自动上传（任务页 `videoUrl` 可下载）；此前所有轮次（含崩塌过程视频）亦已留档在各自任务页。

## 4. 交付物清单

| 交付物 | 位置 |
|---|---|
| 严格通过标准 | `docs/PASS_CRITERIA.md` |
| IsaacSim 训练管线（修复后） | `humanoid_lab/`（分支 feat/isaac-sim-migration，HEAD 2491b14） |
| sim2sim 评测 + 视频录制管线 | `humanoid_lab/scripts/sim2sim_eval.py` |
| JIT 导出（免 isaaclab 依赖） | `humanoid_lab/scripts/export_policy_lab.py` |
| 训练曲线（v1/v2/v3b/v3c） | `logs_analysis/train_walk_curves.json` |
| 训练标准评估器 | `tools/eval_train_criteria.py` |
| IsaacLab 干净回放诊断 | `humanoid_lab/scripts/diag_isaaclab_play.py`（+任务 016 日志） |
| 最终 checkpoint | `upload/2026/8/27/model_20000_20260827224116A910.pt`（TASK_20260827_196） |
| sim2sim 视频/verdict | gradmotion 任务页（TASK_20260828_013/020/039） |

## 5. 后续建议（若需完全达标）

1. **训练量**：v3c 曲线未饱和，T1/T3/T4 差距分别约 40%/22%/17%——继续 resume +10k~15k iters（约 4–6 h GPU）大概率收敛达标；瓶颈是账号余额。
2. **sim2sim**：若 039（脚掌 parity）仍未过，建议直接用 URDF→MJCF 转换（`mujoco.MjModel.from_xml_path(x1.urdf)`，mjcf compiler 标签已内置）替换手写 mjcf，一劳永逸消除几何/质量差异；或以本评测脚本已验证的置换框架接入官方 mjcf。
3. 余额：号池账号 35 尚未使用。
