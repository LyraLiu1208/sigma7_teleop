# sigma7_teleop

这个仓库已经整理成单仓库迁移形态，目标是在 Linux 上只需要 `git clone` 当前仓库，再做最少量环境安装，就可以运行 MuJoCo 侧和 Sigma7 侧入口。

## 现在已经内置到仓库里的内容

- `src/stiffness_copilot_mujoco/`: 原先在外部仓库里的 MuJoCo/控制器/数据集代码
- `configs/`、`models/`: 运行所需配置和模型
- `third_party/mujoco_menagerie/franka_emika_panda/`: Franka 资产
- `third_party/dinov3/` 和 `checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`: 训练链路需要的本地 DINOv3 代码和权重
- `scripts/`: 当前项目自身脚本，加上少量被直接调用的 MuJoCo 辅助脚本

## 仍然无法完全内置的外部依赖

Sigma7 硬件发送端依赖 Force Dimension / Sigma.7 的 Linux SDK。当前仓库里没有 Linux 版动态库，所以 Linux 机器上仍然需要你单独安装该 SDK，并设置：

```bash
export SIGMA7_SDK_ROOT=/path/to/force-dimension-sdk
```

这是目前唯一无法仅靠仓库内容解决的关键外部依赖。

## Linux 快速开始

默认推荐 `CPU runtime` 环境。它适合你当前的 Linux 侧 `screening`、残差策略推理和 MuJoCo viewer，不依赖 CUDA 驱动匹配细节。

```bash
git clone <your-remote-url>
cd sigma7_teleop
bash scripts/setup_linux.sh
source .venv/bin/activate
python scripts/doctor_linux.py
```

## Linux 环境分层

这个仓库现在明确分成两套 Linux 环境：

- `bash scripts/setup_linux_cpu.sh`
  - 用途：`screening`、残差策略推理、MuJoCo viewer、日常部署
  - Torch：官方 CPU wheel
  - 特点：最稳定，不依赖 NVIDIA 驱动/CUDA 兼容性
- `bash scripts/setup_linux_gpu.sh`
  - 用途：训练，或者未来明确要做 GPU 推理时再用
  - Torch：官方 `cu124` wheel
  - 特点：要求本机驱动和 CUDA 兼容

为了兼容旧文档，`bash scripts/setup_linux.sh` 现在等价于：

```bash
bash scripts/setup_linux_cpu.sh
```

### CPU runtime

```bash
rm -rf .venv
bash scripts/setup_linux_cpu.sh
source .venv/bin/activate
python scripts/doctor_linux.py
```

### GPU training

```bash
rm -rf .venv
bash scripts/setup_linux_gpu.sh
source .venv/bin/activate
python scripts/doctor_linux.py
```

如果你需要覆盖默认版本，也可以显式指定：

```bash
TORCH_VERSION=2.6.0 \
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
bash scripts/setup_linux_cpu.sh
```

```bash
TORCH_VERSION=2.6.0 \
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
bash scripts/setup_linux_gpu.sh
```

如果 `bash scripts/setup_linux.sh` 在 `python3 -m venv` 或 `ensurepip` 处失败，先安装：

```bash
sudo apt update
sudo apt install -y python3-venv
```

然后重跑：

```bash
rm -rf .venv
bash scripts/setup_linux_cpu.sh
source .venv/bin/activate
python scripts/doctor_linux.py
```

## 推荐部署方式：Mac 连 Sigma7，Linux 跑仿真和 viewer

这是你当前最省时间、也最符合现有代码结构的部署方式：

- `Mac` 保留现成的 Sigma7 SDK 和硬件连接，只负责发送 UDP 位姿
- `Linux` 负责运行 `run_sigma7_screening_episode.py`、残差策略推理、MuJoCo 仿真和本地 viewer
- `Linux` 不需要安装 Sigma7 SDK，除非你还想把 sender 也迁过去

### 1. 先把当前仓库推到远端

如果当前仓库还没有远端：

```bash
git remote add origin <your-remote-url>
git push -u origin main
```

### 2. Linux 机器拉代码并准备环境

```bash
git clone <your-remote-url>
cd sigma7_teleop
bash scripts/setup_linux_cpu.sh
source .venv/bin/activate
python scripts/doctor_linux.py
```

### 3. 把 policy 文件也放到 Linux

例如你可以把它放到仓库内的 `artifacts/models/vision_residual_bc/`：

```bash
mkdir -p artifacts/models/vision_residual_bc
cp /path/to/circle_calibrated_v1_track_a_c600_53ep_frozen_train_val_split_dinov3_residual_bc_policy_6d.npz artifacts/models/vision_residual_bc/
```

### 4. Linux 上启动 screening episode

把 `192.168.1.50` 替换成 Linux 自己的局域网 IP。viewer 会直接开在 Linux 本机屏幕上。

```bash
cd /path/to/sigma7_teleop
source .venv/bin/activate

python scripts/run_sigma7_screening_episode.py \
  --participant p01 \
  --scene circle \
  --controller residual \
  --episode-id 0 \
  --policy /path/to/sigma7_teleop/artifacts/models/vision_residual_bc/circle_calibrated_v1_track_a_c600_53ep_frozen_train_val_split_dinov3_residual_bc_policy_6d.npz \
  --packet-host 0.0.0.0 \
  --packet-port 5005
```

### 5. Mac 上只启动 Sigma7 UDP sender

把 `192.168.1.50` 替换成 Linux 的局域网 IP：

```bash
cd /Users/lyra/Desktop/MasterThesis/sigma7_teleop

python scripts/run_sigma7_pose_udp_sender.py \
  --host 192.168.1.50 \
  --port 5005
```

### 6. 网络检查

- Mac 和 Linux 需要在同一个局域网里
- 优先用有线网络
- Linux 防火墙需要允许 UDP `5005`
- Linux 必须有图形桌面环境，因为 MuJoCo viewer 是本地打开的
- 如果你只做这种“Mac 控硬件、Linux 跑 viewer”的分机方案，Linux 不需要 `SIGMA7_SDK_ROOT`

如果你要构建 Sigma7 UDP sender：

```bash
export SIGMA7_SDK_ROOT=/path/to/force-dimension-sdk
cmake -S tools/sigma7_pose_udp_sender -B tools/build
cmake --build tools/build -j
```

## 常用入口

- MuJoCo + Sigma7 联合启动：`python scripts/run_sigma7_teleop_stack.py`
- 只启动 MuJoCo viewer：`python scripts/run_sigma7_mujoco_live_teleop.py`
- 只启动 Sigma7 UDP sender：`python scripts/run_sigma7_pose_udp_sender.py`
- 采集一条 BC episode：`python scripts/collect_sigma7_residual_bc_episode.py ...`
- 构建场景级数据集：`python scripts/build_sigma7_residual_bc_dataset.py --scene <scene>`
- 训练 6D frozen DINOv3 residual BC：`python scripts/train_sigma7_frozen_dinov3_residual_bc_6d.py --scene <scene>`

## Git 准备

本目录现在适合直接作为独立 Git 仓库使用。推荐流程：

```bash
git init -b main
git add .
git commit -m "Prepare Linux migration"
git remote add origin <your-remote-url>
git push -u origin main
```

`.gitignore` 已经排除了 `artifacts/`、macOS SDK 镜像、构建产物和缓存，避免把无关大文件推上去。
