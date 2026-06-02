# SAC + KF + B-spline 最小验证项目

这是一个纯 Python 最小验证系统，用来测试：

- Soft Actor-Critic (SAC) 是否能在连续 2D 导航环境中学到从起点到目标的控制策略。
- SAC actor 输出 raw action `u_t` 后，环境内部使用 Kalman Filter 在线平滑为 executed action `a_exec`。
- 环境实际执行 `a_exec`，但 replay buffer / SAC 更新仍然使用传入 `env.step()` 的 raw action `u_t`。
- observation 使用 19 维 augmented state，把原始导航观测和 KF/动作平滑器的上一执行动作拼接起来。
- 训练后使用 B-spline 对执行轨迹做可视化平滑，不反向影响训练环境。

项目不使用 ROS，适合先在 Ubuntu 22.04 + Python 3.10+ 上做最小可运行验证。

## 方法逻辑

本项目对应 Scheme B：

1. SAC policy 输出 raw action：
   `u_t = [vx_raw, vy_raw]`
2. `env.step(u_t)` 接收 raw action。
3. 环境内部匀速 Kalman Filter 根据 `u_t` 输出 filtered/executed action：
   `a_exec = [vx_smooth, vy_smooth]`
4. agent 位置更新只使用 `a_exec`：
   `p_{t+1} = p_t + a_exec * max_speed * dt`
5. stable-baselines3 replay buffer 记录的是传入环境的 raw action `u_t`，不是 filtered action。
6. observation 是 19 维 augmented state：
   原始环境观测 `base_obs(17)` + 上一时刻 executed action `(2)`。

因此 critic 学到的是 `Q(s_aug, u)`：raw action `u` 经过 KF 和环境动力学后的长期回报。奖励函数中没有显式 action smoothness penalty，动作平滑来自 KF-in-the-loop。

当前 KF 是匀速/随机游走动作滤波器，内部状态为 `x=[vx, vy]^T`，不再使用 `[vx, vy, ax, ay]^T` 匀加速模型。默认参数为 `process_noise_std=0.15`、`measurement_noise_std=0.30`，比早期版本更轻，避免把高频 raw action 过度平均到接近零速度。`prev_exec_delta` 仍会记录到 `info` 和 TensorBoard，用于诊断动作平滑效果，但不再作为 policy observation 的一部分。

注意：当前 observation 维度已经从旧版本的 21 维改为 19 维，旧模型文件与新环境输入层不兼容，需要重新训练后再评估。

当前 reward 在原始 dense goal progress 基础上加入了两项导航 shaping：

- 近障碍安全距离惩罚：进入障碍物表面外的安全 margin 时，按接近程度连续惩罚。
- heading-to-goal 奖励：执行动作方向与目标方向一致时给小奖励。
- 加权 goal progress 奖励：放大 `previous_distance - current_distance`，强化持续靠近目标。
- 距离惩罚：按当前剩余距离给小惩罚，避免长时间停在远离目标的位置。
- anti-stall 惩罚：离目标较远、执行速度很小且目标进展不足时惩罚停滞。
- timeout 剩余距离惩罚：超时结束时按剩余距离额外扣分。

reward 中仍然不加入 action smoothness penalty。

## 安装

如果你已经有 miniconda 环境 `rl_env`，不要再新建 `.venv`。直接激活已有环境，并确认当前 Python 能 import PyTorch：

```bash
cd sac_kf_bspline_minivalidation
conda activate rl_env
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
python -m pip install -r requirements.txt
```

`requirements.txt` 里故意没有写 `torch`，会复用你在 `rl_env` 中已经安装好的 PyTorch。当前项目只会额外安装缺失的轻量依赖：`gymnasium`、`stable-baselines3`、`numpy`、`matplotlib`、`scipy`、`tqdm`、`tensorboard`。

如果 `pip install -r requirements.txt` 仍然开始下载很大的 `torch` 包，通常说明当前终端没有真正进入 `rl_env`，或者该环境里的 PyTorch 版本太旧。先检查：

```bash
which python
python -c "import torch; print(torch.__version__)"
```

## 训练

默认训练现在使用 APF warm-up + SAC 训练：

1. APF policy 先跑 warm-up episodes，把 transition 写入 SAC replay buffer。
2. 如果 `--use-kf 1`，APF warm-up 和 SAC 训练都从一开始使用 KF 平滑 raw action。
3. 如果 `--use-kf 0`，APF warm-up 和 SAC 训练都全程不使用 KF，方便做 ablation。

```bash
python -m train.simple_env --total-steps 100000 --use-kf 1 --seed 0
```

等价的显式命令：

```bash
python -m train.simple_env \
  --total-steps 100000 \
  --use-kf 1 \
  --apf-warmup-episodes 1000 \
  --seed 0
```

默认模型保存到：

```text
outputs/models/sac_kf_nav.zip
```

关闭 KF 做 ablation：

```bash
python -m train.simple_env --total-steps 100000 --use-kf 0 --apf-warmup-episodes 1000 --seed 0
```

复杂迷宫环境训练入口：

```bash
python -m train.complex_env \
  --total-steps 500000 \
  --use-kf 1 \
  --apf-warmup-episodes 2000 \
  --seed 0
```

常用训练参数：

- `--apf-warmup-episodes`：APF warm-up episode 数，默认 `1000`。
- `--buffer-size`：SAC replay buffer 大小，默认 `500000`。
- `--tb-log-freq`：自定义 TensorBoard 指标写入频率，默认 `100` step。
- `--show-progress`：是否显示 APF/tqdm 进度条，默认 `1`。
 
## TensorBoard

训练时会写入 SB3 原生日志和自定义 `SummaryWriter` 指标：

```bash
tensorboard --logdir outputs/logs
```

可查看的关键 tags 包括：

- `apf/episode_return`
- `apf/state_value_mean`
- `env/distance_to_goal`
- `env/goal_progress`
- `action/exec_norm`
- `action/raw_delta`
- `action/exec_delta`
- `action/smoothing_ratio`
- `reward/progress_reward`
- `reward/heading_reward`
- `reward/obstacle_penalty`
- `reward/distance_penalty`
- `reward/stall_penalty`
- `reward/timeout_distance_penalty`
- `env/min_obstacle_signed_distance`
- `curriculum/use_kf`

## 评估与可视化

```bash
python -m evaluate.simple_env --model-path outputs/models/sac_kf_nav.zip --use-kf 1
```

复杂迷宫环境评估入口：

```bash
python -m evaluate.complex_env --model-path outputs/models/sac_kf_complex_nav.zip --use-kf 1
```

评估会打印：

- average return
- average steps
- success rate
- collision rate
- out-of-bounds rate
- timeout rate
- mean raw action delta
- mean executed action delta
- smoothing ratio = mean_exec_delta / mean_raw_delta

## 输出图像

评估后生成：

```text
outputs/figures/path_kf_bspline.png
outputs/figures/eval_reward_curve.png
outputs/figures/eval_step_distance.png
outputs/figures/action_smoothing_comparison.png
```

图像含义：

- `path_kf_bspline.png`：地图、障碍物、起点、目标、实际执行轨迹和 B-spline 平滑轨迹。
- `eval_reward_curve.png`：单条评估 episode 的 step reward 和 cumulative reward。
- `eval_step_distance.png`：单条评估 episode 中 distance-to-goal 随 step 的变化。
- `action_smoothing_comparison.png`：`vx_raw` vs `vx_exec`、`vy_raw` vs `vy_exec`、`||delta raw||` vs `||delta exec||`。

## 项目结构

```text
sac_kf_bspline_minivalidation/
├── requirements.txt
├── README.md
├── train/
│   ├── __init__.py
│   ├── simple_env.py
│   └── complex_env.py
├── evaluate/
│   ├── __init__.py
│   ├── simple_env.py
│   └── complex_env.py
├── envs/
│   ├── __init__.py
│   ├── continuous_nav_env.py
│   └── complex_nav_env.py
├── filters/
│   ├── __init__.py
│   └── kalman_action_smoother.py
├── utils/
│   ├── __init__.py
│   ├── apf.py
│   ├── apf_complex.py
│   ├── plotting.py
│   ├── spline.py
│   └── training_callbacks.py
└── outputs/
    ├── models/
    ├── logs/
    └── figures/
```

## 已知限制

- 这是最小验证系统，不是完整机器人动力学仿真。
- 当前 agent 是点质量机器人，使用二维速度控制。
- 当前障碍物为固定圆形障碍物，start/goal 默认固定。
- B-spline 只用于训练后的轨迹展示，不影响环境执行或训练。
- APF 只用于 replay buffer warm-up 和可视化诊断，不作为 SAC reward 或 critic target。
- 后续需要加入 non-holonomic dynamics、真实控制器、传感器噪声和更严格的安全约束。
