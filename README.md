# IOTA 5201 Mid-Term Assignment: Implementing Q-Learning
Name：Faxiang Yan，ID：50051798，本目录为期中项目的代码库，包含两组 Q-learning 实验以及消融实验。运行脚本会重新生成 `results/` 中的 CSV、`report/assets/` 中的 SVG 图，并把结果写入最终报告 `report/index.html`。

## 运行方法

需要 Python 3.10 或更新版本；仅使用标准库，在本目录执行：

```sh
python3 code/run_experiments.py
```

这条命令依次运行 C→F 和 A→C 两组实验，打印最终 Q 表，并重新生成所有 CSV、两张 SVG 收敛图及 HTML 报告中的结果表格。也可以只运行其中一组：

```sh
python3 code/run_experiments.py --config config/baseline.json
python3 code/run_experiments.py --config config/start_A_goal_C.json
```

三组消融实验：

```sh
python3 code/run_ablations.py
```

消融实验的设置写在 `config/ablations.json`，只研究 C→F 环境，并保持奖励、学习率、折扣因子和状态图不变：

- 随机种子重复：种子 `0, 1, 2, 3, 42`，每次 10,000 回合；报告成功率、最后 500 回合平均回报和步数、最终 Q 表误差、最少动作访问次数，并给出均值和样本标准差。
- 训练预算：固定种子 `42`，比较 `500、1000、2000、5000、10000` 回合。
- 探索率：固定种子 `42` 和 10,000 回合，比较当前的 `0.4→0.1/12000`、固定 `ε=0.1`、以及 `0.4→0.01/2000` 的快速衰减。

这里的 `final_q_max_abs_error` 是最终 Q 表与独立动态规划参考 Q* 的最大绝对差。Q* 只用于这个小型确定性环境的离线诊断，不参与 Q-learning 更新。成功率、回报和步数反映起点策略表现；动作访问次数和 Q* 误差用于识别“成功到达目标但没有充分学习整张表”的情况。

修改 JSON 中的 `start_state` 和 `target_state` 即可自定义起点、目标；也可在命令行覆盖单个配置，例如：

```sh
python3 code/run_experiments.py --config config/baseline.json --start B --target E
```

直接用浏览器打开 `report/index.html` 即可阅读，无需本地服务器或外部服务。

## 实验设置

两组实验均将所有合法 `Q(s,a)` 初始化为 `0.0`，学习率 `α=0.2`，折扣因子 `γ=0.9`，训练 `10,000` 回合，随机种子 `42`。ε-greedy 的 ε 在前 `12,000` 回合从 `0.4` 线性下降到 `0.1`，所以本次训练最后一个回合约为 `0.15`；继续训练到第 12,000 回合后才达到 `0.1`。每回合至多 `30` 步。

## 部分文件说明

| 文件 | 用途和后续需要填写的内容 |
| --- | --- |
| `code/environment.py` | 校验配置并定义状态、合法动作、确定性转移、奖励和终止规则。 |
| `code/q_learning.py` | Q 表初始化、ε-greedy 选动作、Q-learning 更新及逐回合记录。 |
| `code/run_experiments.py` | 运行配置、打印最终 Q 表、输出 CSV、收敛图和 case-study 热力图，并更新 HTML 中的结果表格。 |
| `code/run_ablations.py` | 运行三组 C→F 消融实验，计算 Q* 误差、成功率、回报、步数和动作覆盖率，输出 CSV/SVG，并更新 HTML 报告。 |
| `config/baseline.json` | C→F 实验的图结构、奖励、终止规则与训练参数。 |
| `config/ablations.json` | 三组消融实验的随机种子、训练预算和探索率策略。 |
| `report/index.html` | 离线 HTML 报告。 |
| `report/assets/baseline_convergence.svg` | 从 `results/baseline/convergence.csv` 对应实验生成的 C→F 收敛图。 |
| `report/assets/case_study_q_heatmap.svg` | C→F 最终 Q 表的带标注热力图，标出每个状态的贪心动作。 |
| `results/baseline/convergence.csv` | C→F 每回合的完整 Q 表和收敛指标。 |
| `results/ablations/random_seed_summary.csv` | 五个种子的指标均值、样本标准差、最小值和最大值。 |
| `results/ablations/q_star_reference.csv` | 用独立动态规划计算的 C→F 参考 Q*，只用于评估误差，不参与训练。 |
| `results/ablations/training_budget.csv` | 固定种子下五种训练回合数的结果。 |
| `results/ablations/exploration_rates.csv` | 三种探索率策略的结果。 |
| `report/assets/ablation_random_seeds.svg` | 随机种子重复实验的最终 Q 误差图。 |
| `report/assets/ablation_training_budget.svg` | 训练预算与最终 Q 误差图，纵轴为对数尺度。 |
| `report/assets/ablation_exploration.svg` | 探索率策略与最终 Q 误差图，纵轴为对数尺度。 |

