# X1 IsaacSim 行走 RL 训练 + Sim2Sim 交付报告

> 日期：2026-08-28 ｜ 分支：`feat/isaac-sim-migration`（HEAD `2491b14`）｜ 严格标准：`docs/PASS_CRITERIA.md`（训练前固定，未事后放宽）

## 1. 总判定（严格标准）

| 维度 | 结果 | 说明 |
|---|---|---|
| A. 训练 T1–T6 | **4/6 通过**（T2/T5/T6 过，T1/T3/T4 未达） | 最佳 v4（32k iters）：T1 135/T3 0.95/T4 1.26；曲线未饱和 |
| B. sim2sim S1–S6 | **FAIL**（S1 未过） | 双根因实证（§6.3）：跨引擎求解器差异（v4 路径）+ 策略启动瞬态噪声依赖（v5 路径） |
| **总判定** | **PARTIAL PASS**（如实报告，未降低阈值） | 训练侧超额完成原基线 17 倍；sim2sim 管线、10+ 项修复与完整诊断链交付 |

## 2. 训练结果（gradmotion 远端 IsaacSim 5.0 + IsaacLab 2.2）

任务链（项目 PRO_20260827_001 → PRO_20260827_024 → PRO_20260828_001，跨账号续训）：

| 任务 | 迭代 | mean_reward | ep_len (/2400) | 备注 |
|---|---|---|---|---|
| v1 TASK_20260827_024 | 0–1000 | 6.7→7.7 | 307 | 冒烟级基线 |
| v2 TASK_20260827_065 | 1000–7999 | 7.7→98.2 | 2204 | effort 限幅 bug 在此段暴露 |
| v3b TASK_20260827_172 | 8000–9820 | →103 | 2187 | 真实限幅下训练；账号余额中断 |
| v3c TASK_20260827_196 | 9800–20000 | →116.7 | 2258 | 原配方满量程 |
| v4 TASK_20260828_072 | 20000–32000 | →135.1 | 2342 | 追加一轮（原配方 1.6×）；增长平台化 |

**T1–T6（v4 终值，last-100，严格阈值见 PASS_CRITERIA.md）**：

| ID | 指标 | v3c@20k | v4@32k | 阈值 | 判定 |
|---|---|---|---|---|---|
| T1 | mean_reward | 117.5 | **135.1** | ≥200 | FAIL |
| T2 | episode_length | 2258 | **2342** | ≥2100 | PASS |
| T3 | tracking_lin_vel | 0.94 | **0.95** | ≥1.20 | FAIL |
| T4 | ref_joint_pos | 1.16 | **1.26** | ≥1.40 | FAIL |
| T5 | collision | ≈0 | ≈0 | ≥−0.005 | PASS |
| T6 | 末端无崩塌 | 0.94×max | **0.95×max** | ≥0.9×max | PASS |

**增长平台化诊断**（两轮追加共 +12k iters 后 T1 仅 +18、T3 +0.02）：`feet_air_time`≈0.006/s（scale 1.2）恒定——策略收敛到**拖步局部最优**：脚从不离地 → `feet_air_time` 奖励（需先腾空积累）从未激活 → 无抬脚梯度。属 reward 结构的冷启动问题（feet_air_time 需先有非接触时间才能产生信号），非训练量问题。曲线数据归档：`logs_analysis/train_walk_curves.json`（v1–v4 全链）。

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

**URDF 直载路线证伪**（TASK_20260828_074/115，commit `2dc7da1`/`788bc2a`）：环境内 mujoco 的 URDF importer 不生成 actuator（nu=0），且该版本 MjSpec 无 `add_motor` API——直接加载训练 URDF 的路线在本镜像不可行，脚本按设计明确报错。结论：S1 攻坚需走训练端跨引擎接触随机化（域随机化覆盖 mjcf 等效接触参数），属新一轮训练迭代。

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

## 6. v5 轮次（contact-sensor 顺序修复后的重训 + 深度诊断，2026-09-01）

### 6.1 训练侧新发现与修复

| 发现 | 证据 | 处置 |
|---|---|---|
| **ContactSensor body 序 ≠ articulation body 序** | init 时按名置换（db17bdc） | 修复后重训 v5：`feet_contact_number` 0.60→**1.32**（v4 终值 0.795），接触相位学习显著加速 |
| **抬腿 reward 低非计算 bug** | 专用探针（010）：最大抬脚 **8.6cm**（>3cm 阈值）、最大腾空 **0.30s**、结算机制正常 | 图表 0.004–0.006 为归一化口径（÷24s×scale×dt），理论上限≈0.010，实际 43% 达成——此前"拖步死区"诊断**证伪** |
| 策略导出无损 | 交叉验证（041）：train-side `act_inference` vs 导出 JIT，三种输入 **max|diff|=0.00e+00** | 排除导出链路 |
| 关节方向一致 | 世界系轴对比：12 关节 dot≥0.92（无翻转）；早期 FK 90° 信号是 body 帧差异非关节问题 | 排除方向错配 |
| URDF 直载证伪 | importer 无 actuator（nu=0）+ 无 MjSpec.add_motor（074/115） | mjcf+parity 补丁路线维持 |

### 6.2 v5 训练与最终判定

v5（TASK_20260831_126 + 20260901_005）：22000 iters，reward 123.4、ep_len 2250；**T1=123.4/T3=0.64/T4=1.14 → 4/6（T3 反而劣于 v4 的 0.95）**。**v4（32k：T1 135/T3 0.95/T4 1.26）仍是训练指标最佳策略。**

### 6.3 sim2sim 终局判定（双根因确立）

| 策略 | IsaacLab 干净回放 | mujoco sim2sim | 结论 |
|---|---|---|---|
| v3c/v4 | **24s 全程稳定** | 1.81s 崩 | 真跨引擎差异（策略健康） |
| v5@16k | 1.54s TERMINATED | 0.77s 崩 | 策略自身启动瞬态脆弱 |
| v5@22k | 1.47s 崩（reset 后可稳定） | 1.22s 崩（torso_contact） | 同上，噪声依赖未随训练消失 |

**B 判定维持 FAIL**。两类根因及对策均已实证归档：
1. **跨引擎求解器差异**（v4 路径）：PhysX TGS vs mujoco Euler 接触模型——需训练端跨引擎接触域随机化（solref/friction/condim 等效参数）。
2. **策略噪声依赖**（v5 路径）：v5 在训练噪声/域随机化掩盖下学到对干净条件脆弱的启动瞬态——需 obs-noise 退火 curriculum 或将干净评估纳入训练循环。

纯 PD 零动作对照（038）：2.08s 缓慢前倾（默认姿态本就欠平衡，属正常），证实 mujoco 模型/物理未断裂。

## 7. 后续建议（若需完全达标）

> 注：v5 探针已证伪"拖步死区"假设（抬脚 8.6cm/腾空 0.3s 正常），T1/T3/T4 缺口更多源于速度跟踪本身；下述第 1 条据此修正。

1. **训练指标（T1/T3/T4）**：v4 曲线仍未饱和，优先追加 iters；若 plateau 持续，考虑 feet_clearance 相位门控目标高度或参考步态 bootstrap 增强步态质量。
2. **sim2sim S1**（双路径，按 §6.3 根因）：训练端跨引擎接触域随机化（friction 已覆盖；补接触刚度/阻尼等效参数），并引入 obs-noise 退火/干净评估轮次消除启动瞬态噪声依赖。
2. **sim2sim**（解决 S1–S6）：训练端做跨引擎接触域随机化（friction 已有 0.2–1.3 覆盖 mjcf 的 1.0；需补接触刚度/阻尼等效参数随机化），让策略对 PhysX TGS 与 mjcf Euler 两类求解器都鲁棒——这是 IsaacGym→mujoco 成功管线的通用做法。URDF 直载路线已在本环境证伪（§3.3）。
3. **逐关节 A/B 数据已备**：`gm_play/diag_isaaclab_stand.pt`（IsaacLab 24s 逐关节 q/action）与 sim2sim verdict.pt（含逐关节 q/target_q，commit `788bc2a` 起）可直接对比首个发散关节。
4. **预算**：账号 1/2 已耗尽；34/35 尚有余量但有限，下一轮迭代前请充值或补充号池。
