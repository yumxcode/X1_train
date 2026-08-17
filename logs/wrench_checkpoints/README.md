# 策略放置说明（logs/wrench_checkpoints/）

本目录用于存放**已导出的稳定行走 jit 策略**，供 `scripts/export_wrench_data.py` 重放生成载荷数据。

## 放置格式

```
logs/wrench_checkpoints/
  <策略名>/                  # 例如 walk_v1、walk_seed5_iter12000
    *.pt / *.jit             # export_policy_dh.py 导出的 jit 模型（每目录一个）
```

- 从训练机 `log/exported_policies/<date>/` 拷贝导出的 jit 文件到上面的目录即可。
- 目录名会作为 `--ckpt logs/wrench_checkpoints/<策略名>` 的参数。

## 导出数据（在含 torch 的环境中运行）

```bash
# 例：0.8 m/s 前向行走，20 秒
python scripts/export_wrench_data.py --task x1_dh_stand \
    --ckpt logs/wrench_checkpoints/walk_v1 \
    --run_id walk_v1_x0.8 --vx 0.8 --vy 0.0 --yaw 0.0 --duration 20.0
```

输出在 `logs/wrench_export/<run_id>/`：`ankle_wrench.csv`、`joint_wrench.csv`、
`foot_wrench.csv`、`manifest.json`。

每个速度工况换一个 `--run_id` 重复执行（如 `walk_v1_x0.4`、`walk_v1_x1.0`、侧移、转向）。
建议工况集：vx ∈ {0.4, 0.8, 1.2}，vy ∈ {±0.3}，yaw ∈ {±0.5}，按已有策略覆盖范围取。
