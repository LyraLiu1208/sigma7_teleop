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

```bash
git clone <your-remote-url>
cd sigma7_teleop
bash scripts/setup_linux.sh
source .venv/bin/activate
python scripts/doctor_linux.py
```

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
