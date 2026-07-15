# MASAC Colab Runtime

该目录用于在 Colab 中运行 MASAC 多智能体 DMP 训练。目录仅保留训练所需的最小源码闭包，不包含历史实验结果、MAPPO/HARL/MARL 等无关工程目录。

## 目录结构

- `scripts/train_masac_multi_agent_dmp.py`：MASAC 训练入口。
- `MASAC/`：MASAC 算法、buffer 与实验配置。
- `net/`：MASAC actor/critic 网络结构。
- `Environment/`：多智能体 DMP 环境与单智能体基础环境。
- `Controller/`：DMP 控制器。
- `Entity/`：动力学模型、传感器与障碍物定义。
- `requirements_colab.txt`：Colab 运行依赖，不覆盖 Colab 自带 PyTorch。
- `run_masac_colab.sh`：命令行启动脚本。
- `colab_runner.ipynb`：Colab 交互式运行 notebook。

## Colab 使用流程

1. 将整个 `MASAC` 文件夹上传或复制到 Colab 的 `/content/MASAC`。
2. 在 Colab 中进入目录：

```bash
cd /content/MASAC
```

3. 安装依赖：

```bash
pip install -r requirements_colab.txt
```

4. 运行最小编译检查：

```bash
python -m py_compile scripts/train_masac_multi_agent_dmp.py MASAC/MASAC.py MASAC/config.py MASAC/Buffer.py net/masac.py Environment/multi_agent_dmp_env.py Environment/single_agent_dmp_env.py Controller/dmp_rl.py Entity/KinematicModel.py Entity/sensors.py Entity/static_obstacles.py Entity/dynamic_obstacles.py Entity/obstacle_generators.py
```

5. 启动训练：

```bash
bash run_masac_colab.sh
```

如果需要将训练结果直接写入 Google Drive，可先挂载 Drive，并设置输出目录：

```bash
MASAC_OUTPUT_ROOT=/content/drive/MyDrive/MASAC_runs bash run_masac_colab.sh
```

也可以临时调整训练步数：

```bash
MASAC_TOTAL_STEPS=10000 MASAC_START_STEPS=1000 bash run_masac_colab.sh
```

训练结果包括 `config.json`、`metrics.csv`、TensorBoard 日志以及 `models/` 下的 checkpoint。

## 前向传播耗时诊断

如果需要定位训练慢的主要网络模块，可启用 forward profiling：

```bash
python scripts/train_masac_multi_agent_dmp.py \
  --device auto \
  --output-root /content/MASAC_profile \
  --total-steps 8000 \
  --start-steps 5000 \
  --batch-size 128 \
  --learn-interval 4 \
  --updates-per-step 1 \
  --disable-tensorboard \
  --profile-forward \
  --profile-forward-interval 500 \
  --profile-forward-topk 20
```

该模式会统计 MASAC 网络中关键模块的前向传播耗时，包括 `ObservationEncoder`、`MultiheadAttention`、`GRU`、`MASACActor`、`MASACCritic` 和 `_CentralizedQBranch` 等。统计结果会输出到控制台，并保存为：

```text
forward_profile.csv
```

注意：profiling 会进行 CUDA 同步计时，会降低训练速度，因此只建议短程诊断时开启，正式训练时关闭。
