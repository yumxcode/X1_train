# X1 IsaacSim 行走 RL 训练 + Sim2Sim 交付报告

> 日期：2026-08-28 ｜ 分支：`feat/isaac-sim-migration`（HEAD `2491b14`）｜ 严格标准：`docs/PASS_CRITERIA.md`（训练前固定，未事后放宽）

## 1. 总判定（严格标准）

| 维度 | 结果 | 说明 |
|---|---|---|
| A. 训练 T1–T6 | **4/6 通过**（T2/T5/T6 过，T1/T3/T4 未达） | 20,000 iters（原版配方满量程），曲线仍在上行 |
| B. sim2sim S1–S6 | **FAIL**（S1 未过：6 轮修复后 stand 存活 1.81s） | 修复链见 §3.1；剩余差距为求解器/接触模型级差异 |
| **总判定** | **PARTIAL PASS**（如实报告，未降低阈值） | 训练侧超额完成原基线 15 倍；sim2sim 管线与诊断链完整交付 |

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

同一 checkpoint（model_20000，IsaacLab 干净回放 **24 s 全程稳定**，z=0.54、单脚站立 roll+0.08，任务 016 全日志）在 mujoco 上逐步修复了 5 处训练/仿真失配：

| # | 失配 | 修复 | commit | 效果（stand trial 存活） |
|---|---|---|---|---|
| 1 | 训练 effort 限幅 850 Nm vs 真实 150/50/80 | per-joint effort_limit_sim 按 URDF | `85dd900` | （训练侧）hip 饱和消除 |
| 2 | 出生高 0.8 vs 训练 0.7 | `INIT_HEIGHT=0.7` | `9b032fb` | 1.22s（弹道冲击消除） |
| 3 | mjcf 关节 damping=1 vs 训练 0 | 加载后置零 | `a1a498f` | 1.22s |
| 4 | **关节顺序错配**：IsaacLab 按树层级交错（L,R,L,R…）vs mjcf 分块（L×6,R×6） | 按关节名建立 qpos/dof/actuator 置换映射 | `a16bb93` | 1.22→**1.81s**（观测/动作语义修正） |
| 5 | 脚掌接触几何：mjcf 4×r2mm 角点球 vs URDF 凸包全脚掌 | ① r30mm 球（失败，滚珠效应 0.55s）→ ② **平底盒** (0.032,0.012,0.072) | `2491b14`/`a1964cf` | 1.81s（无进一步改善） |

> #4 是迁移 bug 的直接后果：原版 Isaac Gym URDF 解析顺序恰与 mjcf 一致（blocked），而 IsaacLab Articulation 按树深度排序产生交错序——任何 IsaacLab→mujoco 的 sim2sim 若不做按名置换都会静默崩塌。本仓库已将该陷阱修复并固化在评测脚本中。

### 3.2 最终 verdict（TASK_20260828_062，model_20000，全部 5 项修复）

| Trial | 存活 | S1 | S2 | S3 | S4 | S5/S6 |
|---|---|---|---|---|---|---|
| stand | 1.81 s | FAIL | PASS | FAIL | FAIL | S6 FAIL（时长<20s） |
| forward | 1.81 s | FAIL | PASS | FAIL | FAIL | FAIL |
| omni | 1.81 s | FAIL | PASS | FAIL | FAIL | — |
| max | 1.81 s | FAIL | PASS | FAIL | FAIL | FAIL |

**S 判定：FAIL**（S2 全过即策略意图正确；S1/S3/S4 随 S1 连带失败）。每轮 run 的 4 个 mp4（含完整崩塌过程）与 verdict.pt 均由 SDK 自动上传至对应任务页（039/062 等，`videoUrl` 可下载）。

### 3.3 残余差距定性

修复 #1–#5 后仍于 ~1.8s 崩塌（崩塌形态：stand 中 pitch 逐渐积累至 −0.15 → 向后漂移 → roll 发散、力矩饱和），而 IsaacLab 中同策略 24s 稳定。剩余差异不在模型文件层（质量/几何/关节约定已逐项核对一致），而在**求解器与接触模型层**：PhysX TGS（4 iter，contact_offset/rest_offset、0.6/0.6 摩擦）vs mjcf Euler（solref=(0.005,1)，friction=1，condim=3）。此类差异通常由「跨物理引擎域随机化」吸收，而本策略训练时的域随机化仅覆盖 PhysX。

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

1. **训练量**（解决 T1/T3/T4）：v3c 曲线未饱和，差距分别约 40%/22%/17%——resume +10k~15k iters（约 4–6 h GPU）大概率收敛达标。
2. **sim2sim**（解决 S1–S6），按性价比排序：
   a. **逐关节 A/B 定位**：`gm_play/diag_isaaclab_stand.pt`（IsaacLab 24s 逐关节 q/action）与 062 的 `verdict.pt`（mujoco 时序）已具备，对比首个发散关节即可锁定剩余失配（预计指向接触求解参数）。
   b. **跨引擎域随机化**：在训练端对 `solref/friction/condim` 等效参数（接触刚度、摩擦系数 0.6→1.0、地面刚度）做随机化，让策略对两类求解器都鲁棒——这是 IsaacGym→mujoco 成功管线的通用做法。
   c. **换部署模型**：mujoco 直接加载 URDF（`<mujoco><compiler>` 标签已在 urdf 内置）替代手写 mjcf，消除文件层差异（此项已在 #1–#5 证明收益有限，优先级最低）。
3. **预算**：账号池 1/2 已耗尽，34/35 余额有限；下一轮迭代前请充值或补充号池。
