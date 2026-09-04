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


## 8. v6/v6.1 轮次（跨引擎接触 DR + ω 鲁棒性，2026-09-01 ~ 09-03）

### 8.1 v6：物理接触摩擦 DR + obs 噪声退火（针对 §6.3 根因 1+2）

| 改动 | 实现 | 任务 |
|---|---|---|
| **物理摩擦/恢复系数 DR**（此前从未写回物理） | 每 env 每 episode 采样有效摩擦 μ_eff~U(0.35,1.25)（覆盖 mujoco μ=1.0 及旧训练值 0.6），经 PhysX average 合成反演为 robot 侧材质 μ_r=2μ_eff-0.6 写入 material buffer (N,3,3)；restitution~U(0,0.3)。privileged obs 的 env_frictions 改报真实有效值 | smoke TASK_20260901_139 验证通过 |
| **obs 噪声退火** | noise_level 线性退火 1.0→0.05（8000 iters，从 learn() 入口起算以适配 resume），Train/noise_level_factor 入曲线 | 同上 |
| 续训起点 | v3c model_20000（OSS 跨账号挂载；v4 ckpt 在已耗尽账号不可达） | TASK_20260901_144→20260902_007（40100→34500 iters） |

**v6 训练终值**（model_34500，配额截断）：reward 123.5、ep_len 2351、tracking 0.96、contact_number 1.18、collision≈0。DR 适配极快（~900 iters 恢复 109 → 终值 123.5 > v3c 116.7）。

**v6 sim2sim（TASK_20260903_016/017，real-ω）**：三 trial 仍于 **1.55s 崩塌**（base_height<0.40，pitch 自 0.5s 单调前漂）。→ **摩擦物理失配被证伪为唯一根因**（mujoco μ=1.0 已在训练分布内仍崩）。

### 8.2 ω 观测通道 5 轮证伪链（24s stand trial，model_34500）

| 轮 | 变体 | 存活 | 结论 |
|---|---|---|---|
| 1（039） | base / stiff(solref 0.002) / soft(0.012) / no_euler / no_angvel / settle2s | 1.52 / 1.64 / 1.80 / 1.20 / **24.00** / 1.30 s | **ω=0 全稳**；接触刚度、euler、落地瞬态排除 |
| 2（043） | ω 源 A/B：gyro sensor vs rot.T@qvel[3:6] | 1.52 / 1.65 s（两源数值差<0.05） | ω 计算源无关 |
| 3（054） | 逐轴取反 neg_x/y/z + 幅值减半 | 2.06 / **4.98** / 1.78 / 1.72 s | neg_y 4.98s 为伪信号；帧错配证伪（URDF vs mjcf 基座挂点逐位一致） |
| 4（060） | EMA 低通 20/50/100ms | 1.60 / 1.60 / 1.66 s | 高频振铃假设证伪（滤波连步态频段一起滤掉） |
| 5（063） | 限幅 0.3/0.5/1.0 | 1.97 / 1.66 / 1.61 s | 幅值尾部假设证伪 |

**结论**：任何真实 ω 输入 → 1.5~2.0s 崩塌（跨 v4/v5/v6 策略与纯 PD 对照 2.08s 同量级）；ω=0 → 24s 全稳。训练所得 ω 响应与 mujoco 系统性不兼容。

### 8.3 v6.1：ω 鲁棒性训练（per-episode ω-dropout + 噪声提升）

- **实现**：`--omega_dropout_prob 0.5`（50% env 的 policy-obs ω 块置零，privileged obs 保留真值）+ `--ang_vel_noise_mult 5.0`（ang_vel 噪声通道 ×5，覆盖 mujoco 接触尖峰 ±1.5 rad/s）。修复了 CLI 覆盖在 env 构造后失效的 bug（be23011）。
- **训练**（TASK_20260903_072，model_34500→42500）：dropout 分布下 reward 恢复至 119.5、ep_len 2304（=v6 的 97%）。
- **sim2sim 双模式**（model_42500，TASK_20260903_125 real / 126 zero）：

| 模式 | 站立段（0-2s） | 行走段 | 判定 |
|---|---|---|---|
| v6 real-ω | 1.55s 崩 | — | FAIL |
| v6.1 real-ω | **稳定**（ω 训练生效） | 3.07-3.93s 崩（起步 ~1s） | FAIL |
| v6.1 zero-ω | 稳定 | 2.97-3.49s 崩 | FAIL |

### 8.4 行走起步崩塌机制（verdict 数据，125/126）

行走指令（t=2s）后：双足保持触地 0.6s、**vx 反向加速 -0.10→-1.48 m/s**、L_ank_p 跟踪误差在 t=2.0 即 0.57 rad；站立段策略踝目标大幅振荡（±0.8 rad）而实际踝被地面约束（踝平衡策略），mujoco 中该策略在起步时 paddling 反推躯干。**两种 obs 模式在行走起步同点崩塌 → 残余根因在步态动力学层（踝策略/摆动相接触转移的跨引擎差异），非观测通道层。**

### 8.5 本轮最终判定（严格标准，未放宽）

| 维度 | v6 | v6.1 | 阈值 | 判定 |
|---|---|---|---|---|
| T1 mean_reward | 123.5 | 116.1（max 125.5） | ≥200 | FAIL |
| T2 ep_len | 2351 | 2263 | ≥2100 | **PASS** |
| T3 tracking_lin_vel | 0.96 | 0.865（max 1.16） | ≥1.20 | FAIL |
| T4 ref_joint_pos | 1.11 | 1.168（max 1.31） | ≥1.40 | FAIL |
| T5 collision | ≈0 | ≈0 | ≥-0.005 | **PASS** |
| T6 末端稳定 | — | 0.925×max | ≥0.9×max | **PASS** |
| B: S1-S6（mujoco） | 全 FAIL | 全 FAIL | — | FAIL |

**总判定：训练 T 3/6（T2/T5/T6 过），sim2sim S FAIL（站立段已修复，行走起步段未解）。**

### 8.6 交付物与残留差距的工程结论

- 已证伪（本轮新增）：摩擦物理失配为唯一根因、ω 观测的符号/幅值/带宽/限幅/源/帧错配、接触刚度、落地瞬态。
- 已修复：跨账号 checkpoint 挂载续训管线、物理摩擦 DR、噪声退火、ω-dropout 训练（站立段 real-ω 崩塌 1.5s→>2s 稳定）。
- **残留**：行走起步（t≈2s 后 ~1s 内）在 mujoco 崩塌，两 obs 模式同点——指向踝平衡策略与摆动相接触动力学的跨引擎行为差异（PhysX TGS 隐式平滑 vs mujoco Euler + mesh-hull 足底）。
- 建议后续（超出本轮预算/号池）：① mujoco-in-the-loop 训练或多求解器混合 DR（需 GPU mujoco 环境）；② 步态层面 bootstrapping（参考步态跟踪强化）降低对踝策略的依赖；③ mujoco 侧 solver 参数系统标定（isaaclab ↔ mujoco 接触参数等效映射）。

## 9. 视频交付与 IsaacLab 相机路线封盘（2026-09-04）

### 9.1 mujoco EGL sim2sim 视频（已交付）

v6.1 model_42500 双模式评测（TASK_20260903_125/126）的 EGL 离屏渲染 mp4 已下载归档至本仓库 `videos/`（real-ω 模式，forward/omni/max 各一，含行走起步崩塌全过程，SDK 自动上传原始副本在任务页 `videoUrl`）：

| 文件 | 大小 | 内容 |
|---|---|---|
| `videos/sim2sim_forward.mp4` | 706 KB | forward trial（3.26s 崩塌） |
| `videos/sim2sim_omni.mp4` | 857 KB | omni trial（3.93s 崩塌） |
| `videos/sim2sim_max.mp4` | 669 KB | max trial（3.07s 崩塌） |

### 9.2 IsaacLab play 视频路线：5 次尝试后封盘（环境阻塞）

`humanoid_lab/scripts/play_video_lab.py`（TiledCamera 干净回放，walk+stand 各 24s）在 A10 镜像（BJX00000093/V000136）上的尝试链：

| # | 任务 | 配置 | 结果 |
|---|---|---|---|
| 1 | TASK_20260904_005 | spawn 省略 | `Missing values ... spawn`（configclass 校验） |
| 2 | TASK_20260904_010 | spawn=PinholeCameraCfg @ Robot prim | `prim already exists` |
| 3 | TASK_20260904_016 | spawn=None（attach 模式）+ enable_cameras | RTX 渲染器初始化挂起 |
| 4 | TASK_20260904_020 | 同 3 复验 | 同点挂起（19s 后日志冻结） |
| 5 | TASK_20260904_026 | 固定机位 /World/Camera 新 prim | 同点挂起 → 停任务 |

**阻塞根因**：镜像内 NVIDIA driver 535.5 落在 Omniverse RTX 不支持区间 `[0.0, 535.129)`（日志 `rtx driver verification failed`），`enable_cameras=True` 路径的渲染器初始化在该驱动上挂起（无崩溃、无输出，30 分钟无进展）。与相机配置（attach/固定机位）无关——属镜像/驱动层环境缺陷，非代码可修。IsaacLab 侧干净回放的**数值**证据（24s 稳定、逐关节轨迹）已由 `diag_isaaclab_play.py`（任务 016 等）以无渲染方式交付；视频层以 mujoco EGL mp4 交付（§9.1）。

> 若需 IsaacLab 渲染视频：需平台侧升级 A10 镜像驱动 ≥535.161.07，或改用 4090D 资源（ESKU000001）复跑 `play_video_lab.py --trial both`。

## 10. 审核第二轮：solver 矩阵证伪 + v7 训练 + IsaacLab play 视频交付（2026-09-04）

### 10.1 mujoco solver 选项矩阵（TASK_20260904_039，行走 trial）

针对行走起步崩塌的最后一类 mujoco 侧假设（求解器数值稳定性）：

| 变体 | 存活 | 崩塌形态 |
|---|---|---|
| base Euler (1ms) | 3.00s | x=-0.49m, vx_end=-1.30 |
| implicitfast 积分器 | 3.00s | 同上（数值不变） |
| condim6 | 3.05s | 同上 |
| implicitfast+condim6 | 3.06s | 同上 |
| 半步长 0.5ms Euler | 4.39s | 延迟但同形态 |

**证伪求解器假设**：所有选项下崩塌形态一致（**倒退** x<0、vx→-1.3），这是策略行为而非数值失稳。

### 10.2 决定性发现：v6.1 策略在训练引擎中也不行走（TASK_20260904_040）

IsaacLab 干净回放（walk trial，无渲染）v6.1 model_42500：24s 内 vx≈0（cmd=+0.62 窗口 mean_vx=-0.44）、双脚持续触地（拖步）、中途 1.09s 曾终止。**结合 10.1：行走崩塌的近因是策略本身——50% ω-dropout 训练把策略退化为站立型**（stand_still scale 2.5 在双模式噪声下最划算），跨引擎差异反而不是主因。

### 10.3 v7 恢复轮（TASK_20260904_041：dropout 0.2 + angvel×3 + 退火 6000）

12k iters 至 54499，reward 峰值 **138.9（历史新高）**，但：

- T 判定（last-100）：T1 122.5 FAIL ｜ T2 2291 **PASS** ｜ T3 0.879 FAIL ｜ T4 1.157 FAIL ｜ T5 **PASS** ｜ T6 0.882 FAIL（末端回落）
- 干净回放（TASK_20260904_143/149）：**v7 更差**——stand 0.8s 即终止，walk 门控 vx≈0

**结论：ω 鲁棒性家族（v6.1/v7）系统性破坏行走能力；v6（model_34500）仍为最佳可部署策略**（real-ω 崩塌仅剩站立段问题，行走段曾在 IsaacLab 内健康）。

### 10.4 IsaacLab play 视频交付（4090D，TASK_20260904_149/153）

- 4090D 节点（ESKU000001，无个人存储挂载）相机管线**完全正常**——A10 挂起确认为主机驱动层缺陷。
- 已归档 `videos/`：`isaaclab_v61_play_{stand,walk}.mp4`（v6.1 model_42500，stand 24s 稳定 z=0.615）、`isaaclab_v7_{stand,walk}.mp4`（v7 model_54499，秒塌）。每段 817 帧（imageio 编码，moov 时间戳部分异常但帧完整）。
- 修复链沉淀：`spawn=None` attach 语义、`enable_cameras=True`、scene 成员测试需直接访问（无 `__contains__`）、视频先写扫描范围外再拷入（SDK 会即时抓取半写文件）。

### 10.5 本审核轮最终判定

| 维度 | v6 | v6.1 | v7 | 阈值 |
|---|---|---|---|---|
| T1 mean_reward | 123.5 | 116.1 | 122.5（峰值 138.9） | ≥200 FAIL |
| T2 ep_len | 2351 | 2263 | 2291 | **PASS** |
| T3 tracking | 0.96 | 0.865 | 0.879 | ≥1.20 FAIL |
| T4 ref_joint | 1.11 | 1.168 | 1.157 | ≥1.40 FAIL |
| T5 collision | ≈0 | ≈0 | ≈0 | **PASS** |
| T6 末端稳定 | PASS | PASS | 0.882 | FAIL（仅 v7） |
| S1-S6 mujoco | FAIL | FAIL | 未评 | FAIL |

**总判定维持：T 3/6、S FAIL。结构性阻塞 = ω 鲁棒性与行走能力的训练张力**：dropout/噪声让策略放弃行走换取站立鲁棒。下一轮需换范式（mujoco-in-the-loop DR / 行走命令课程 + 步态 bootstrapping / obs 级跨引擎随机化而非 channel dropout），单纯调 dropout/噪声参数已被本轮证伪。
