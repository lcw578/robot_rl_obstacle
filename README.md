# AUBO i5 番茄采摘仿真

在 MuJoCo 中训练 AUBO i5 六轴机械臂接近番茄果梗。HOPP 给出抓取点、采摘区和工具 X 轴约束，HAF 将启发式旋转与 SAC 策略融合，MuJoCo mocap/weld 执行七维末端动作。任务覆盖从 HOME 姿态到抓取位姿的接近过程，并检查机械臂自碰撞以及与车体、地面和植物的碰撞。

## 环境配置

训练与评估入口使用 Linux 的进程资源保护。以下命令从仓库根目录运行。MuJoCo 图形查看器需要可用的桌面图形会话。

```bash
git clone https://github.com/lcw578/robot_rl_obstacle.git
cd robot_rl_obstacle
conda env create -f environment.yml
conda activate robot_rl_obstacle
```

环境文件指定 Python 3.10。依赖版本参照开发环境锁定，其中 MuJoCo 为 3.12.0，PyTorch 为 2.5.1，Gymnasium 为 1.2.3，Stable-Baselines3 为 2.8.0。已有 Python 3.10 虚拟环境时，也可以直接安装依赖。 [Conda 环境文件](environment.yml) · [Python 依赖](requirements.txt)

```bash
python -m pip install -r requirements.txt
```

## 查看、训练和评估

### 查看场景

```bash
python drl/inspect_paper_scene.py
python drl/inspect_paper_scene.py --view
```

第一条在终端输出观测形状、奖励和碰撞反馈。第二条打开 MuJoCo 窗口。加 `--seed 1000000` 可指定模型编号，模型清单列出了可用编号。 [数据清单](datasets/paper500_approach_v3/manifest.json)

### 训练

```bash
python drl/train_whole_robot_approach.py --steps 1024 --output runs/paper_smoke
```

这条命令用于短训检查。正式训练可通过 `--steps` 设置采样步数上限，或通过 `--episodes` 额外限制完整回合数。默认使用 CPU。PyTorch 检测到可用 CUDA 时，可加 `--device cuda`。每次运行使用新的输出目录。训练保存 `model.zip`、`replay_buffer.pkl`、配置和场景快照，`report.json` 记录实际步数、完整回合数和训练锚点结果。

### 评估和回归用例

```bash
python drl/evaluate_paper_dataset.py --policy runs/paper_smoke/model.zip --output runs/paper_smoke_test
PYTHONPATH=drl python -m unittest discover -s tests
```

评估从训练目录读取与 `model.zip` 配套的 `config.yaml` 和 `scene/`，遍历独立的 100 个测试模型，将成功率、位姿误差、动作步数和路径增量写入 `test_report.json`。正式评估请把命令中的策略路径替换为正式训练输出。

### 仿真参数和数据

默认配置设定工作区、HOME、控制周期、碰撞及成功阈值。数据清单路径相对于仓库根目录，`--config`、`--policy` 和 `--output` 路径相对于终端当前目录，导出场景的网格路径相对于 `scene.xml` 所在目录。 [默认仿真配置](config/paper_simulation.yaml) · [番茄生成配置](config/tomato_cluster.yaml) · [夹具 TCP 配置](config/gripper.yaml)

更新植物生成规则后，可把新数据集生成到独立目录。

```bash
python drl/prepare_paper_dataset.py --output runs/datasets/custom
```

要使用新数据集，在训练配置的 `paper_training.dataset_manifest` 中填入 `runs/datasets/custom/manifest.json`。

## 目录

| 路径 | 内容 |
|---|---|
| `drl/` | 仿真环境、HOPP、HAF、HER-SAC，以及查看、训练、评估和数据生成入口 |
| `config/` | 仿真、番茄几何和夹具 TCP 参数 |
| `assets/robot_for_tomato_harvest/` | 整车 MJCF、网格和来源说明 |
| `datasets/paper500_approach_v3/` | 500 个冻结模型、400/100 划分清单及生成配置 |
| `tests/` | 回归用例 |

训练和评估示例将产物写入 `runs/`。Git 忽略该目录和本地 `archive/`。

## 论文仿真与 AUBO i5 适配

| 论文内容 | 当前实现 |
|---|---|
| §2.2 植物几何 | 一段圆柱主茎、两段圆柱果梗、长方体果穗，保留 7 个关键点。500 个模型由本项目参数化生成，沿用原有编号及 400/100 划分。论文使用温室 RGB-D 数据重建。 |
| §2.3 HOPP | 按式 (1)–(5) 计算，抓取点距主茎轴线 10 mm，采摘半球半径 150 mm。论文基座系的前方映射到 AUBO 基座系的 +Y，所以半球朝向 −Y。 |
| Algorithm 1 与 §2.5.1 场景随机化 | 青框标出 400 × 400 × 200 mm 的抓取点采样区，中心为 AUBO 基座系的 (0, 460, 720) mm。果梗和果穗随形态自然伸展。训练每回合从打乱的 400 个模型中选一个，对整株施加刚体变换并重采 Pg。本地采用随机转轴和 0–20° 转角。HOME 碰撞或式 (4) 选侧退化时重抽同一模型的位姿，合计最多尝试 32 个位姿。查看器与独立评估使用各编号固定的基准位姿。 |
| 式 (14) 观测与 HER | 18 项物理观测加一项回合时钟，两个六维目标和三维 HAF 场景参考点，默认 Actor 输入共 34 维。HER 重标记同步更新位置目标、姿态目标、目标距离和目标法向，HAF 场景参考点保留真实番茄的 Pg。目标几何和奖励计算保留双精度，网络特征按 SB3 预处理使用单精度。 |
| 式 (15)、式 (21)–(25) 与表 1 | 七维动作包含每轴最大 50 mm 的平移请求和单位四元数。预热、推理和 SAC 更新共用 HAF，SLERP 衰减系数为 2。每回合最多 50 步，SB3 的 HER 参数 k=4 对应约 80% 的虚拟样本，每条虚拟样本各取一个未来目标。默认经验池 100,000，训练入口的步数预算可配置。 |
| MuJoCo 控制与碰撞 | 使用现有六轴 i5、120 mm 工具 TCP、HOME 和整车几何。完整位姿请求按当前控制周期插补给 mocap，MuJoCo weld 推进实际状态。每个物理子步检查自碰撞、机械臂与车体或地面的碰撞，以及植物碰撞，沿用相邻安装件排除。碰撞回退本步并结束回合，惩罚为 −100。关节越限和数值异常也结束回合。 |
| §2.5.1 独立评估 | 每个保留模型固定初始姿态，计分一次。以 15 mm、0.05 rad 及全程无碰撞为成功条件，按论文的误差变化阈值判断停滞。保留轨迹的 TCP 路程按物理子步累计，回退步计零。路径增量为累计路程减起点至 Pg 的直线距离，保留符号并纳入所有完成初始化的模型。初始化失败仍计入成功率分母，各连续指标注明有效样本数。 |

式 (16) 的近距离位置项取负，式 (17) 计算工具 X 轴与 HOPP 法向的夹角，式 (18) 使用实际关节位移的欧氏范数。成功阈值使用原文公式的 15 mm 和 0.05 rad。控制周期、焊接参数、阻尼、重力补偿、网络宽度和 HER 虚拟目标构造采用本项目实现。

图 5 将策略原始动作写入回放。本项目按照整机接口约定，回放和双 Q 使用送入环境的融合动作，四元数统一符号。SAC 熵基于融合前的 tanh 高斯潜变量。当 HAF 完全确定旋转时，熵只计算平移分量。HER 在整个回合保留真实场景的参考点。式 (13) 的目标熵沿用表 1 列出的数值 −1。新版观测及回放接口需要重新训练对应策略。

工具质量和惯量以参考 MJCF 为准。训练报告记录编译后的夹具质量、TCP、执行器参数、模型哈希、数据来源和实际采样范围。显式设置 `gravity_compensation: false` 会清除机械臂与夹具本体上继承的重力补偿。

## 参考资料

- Li, Y., et al. (2024). *Peduncle collision-free grasping based on deep reinforcement learning for tomato harvesting robot*. Computers and Electronics in Agriculture, 216, 108488. [原论文 DOI](https://doi.org/10.1016/j.compag.2023.108488)
- 整车 MJCF 和网格的模型说明。 [模型来源](assets/robot_for_tomato_harvest/README.md)
