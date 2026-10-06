# IOTA 5201 Mid-Term Assignment: Implementing Q-Learning

本目录包含两组可复现的 Q-learning 实验。运行脚本会重新生成 `results/` 中的 CSV、`report/assets/` 中的 SVG 图，并把结果写入 `report/index.html`。完成报告的其余分析后，将本 README、`code/`、`config/`、`report/`、`results/` 和 `requirements.txt` 打成一个 ZIP 提交。`IOTA_Assignment-1.pdf` 是题目原文，不是要求提交的成果文件。

## 运行方法

需要 Python 3.10 或更新版本；仅使用标准库，`requirements.txt` 无第三方依赖。在本目录执行：

```sh
python3 code/run_experiments.py
```

这条命令依次运行 C→F 和 A→C 两组实验，打印最终 Q 表，并重新生成所有 CSV、两张 SVG 收敛图及 HTML 报告中的结果表格。也可以只运行其中一组：

```sh
python3 code/run_experiments.py --config config/baseline.json
python3 code/run_experiments.py --config config/start_A_goal_C.json
```

核心逻辑自检：

```sh
python3 -m unittest discover -s code -v
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

覆盖运行会写入独立的 `results/baseline_B_to_E/` 和 `report/assets/baseline_B_to_E_convergence.svg`，不会覆盖两组规定实验的结果，也不会改动 HTML 报告。可用 `--output-dir` 和 `--plot` 指定自定义输出位置。直接用浏览器打开 `report/index.html` 即可阅读，无需本地服务器或外部服务。

## 实验设置与输出

两组实验均将所有合法 `Q(s,a)` 初始化为 `0.0`，学习率 `α=0.2`，折扣因子 `γ=0.9`，训练 `10,000` 回合，随机种子 `42`。ε-greedy 的 ε 在前 `12,000` 回合从 `0.4` 线性下降到 `0.1`，所以本次训练最后一个回合约为 `0.15`；继续训练到第 12,000 回合后才达到 `0.1`。每回合至多 `30` 步。探索时在合法动作中均匀随机选择，利用时在最大 Q 值的并列动作中随机选择。进入目标时不再加未来价值；达到步数上限为截断，不视为到达目标。

`convergence.csv` 每行对应一个回合：`Q_状态_动作` 列是该回合结束后的整张 Q 表，`max_abs_q_change` 是相对回合开始时所有合法 Q 值的最大绝对变化，还记录 ε、总奖励、步数和是否到达目标。SVG 上半部分画出所有合法 Q 值，突出起点的动作；下半部分画出 `max_abs_q_change`。图为便于阅读抽样约 300 个回合点，CSV 保留全部 `10,000` 回合。Q 表 CSV 的 `visit_count` 可帮助判断较少探索的动作是否可信；曲线稳定不等于已证明最优。

## 文件说明

| 文件 | 用途和后续需要填写的内容 |
| --- | --- |
| `README.md` | 依赖、参数、复现命令和提交文件说明。 |
| `requirements.txt` | 声明无第三方依赖及 Python 最低版本。 |
| `code/environment.py` | 校验配置并定义状态、合法动作、确定性转移、奖励和终止规则。 |
| `code/q_learning.py` | Q 表初始化、ε-greedy 选动作、Q-learning 更新及逐回合记录。 |
| `code/run_experiments.py` | 运行配置、打印最终 Q 表、输出 CSV 和 SVG，并更新 HTML 中的结果表格。 |
| `code/run_ablations.py` | 运行三组 C→F 消融实验，计算 Q* 误差、成功率、回报、步数和动作覆盖率，输出 CSV/SVG，并更新 HTML 报告。 |
| `code/test_q_learning.py` | 检查终止更新、自定义起点/目标和固定种子的可复现性。 |
| `config/baseline.json` | C→F 实验的图结构、奖励、终止规则与训练参数。 |
| `config/start_A_goal_C.json` | A→C 实验的图结构、奖励、终止规则与训练参数。 |
| `config/ablations.json` | 三组消融实验的随机种子、训练预算和探索率策略。 |
| `report/index.html` | 离线 HTML 报告，包含环境、方法、设置、两组结果表格及收敛图；比较分析等内容尚待完成。 |
| `report/assets/baseline_convergence.svg` | 从 `results/baseline/convergence.csv` 对应实验生成的 C→F 收敛图。 |
| `report/assets/start_A_goal_C_convergence.svg` | 从 `results/start_A_goal_C/convergence.csv` 对应实验生成的 A→C 收敛图。 |
| `results/baseline/q_initial.csv` | C→F 训练前的合法状态动作 Q 值。 |
| `results/baseline/q_final.csv` | C→F 训练后的 Q 值与动作访问次数。 |
| `results/baseline/convergence.csv` | C→F 每回合的完整 Q 表和收敛指标。 |
| `results/start_A_goal_C/q_initial.csv` | A→C 训练前的合法状态动作 Q 值。 |
| `results/start_A_goal_C/q_final.csv` | A→C 训练后的 Q 值与动作访问次数。 |
| `results/start_A_goal_C/convergence.csv` | A→C 每回合的完整 Q 表和收敛指标。 |
| `results/ablations/random_seeds.csv` | 五个随机种子的逐次结果。 |
| `results/ablations/random_seed_summary.csv` | 五个种子的指标均值、样本标准差、最小值和最大值。 |
| `results/ablations/q_star_reference.csv` | 用独立动态规划计算的 C→F 参考 Q*，只用于评估误差，不参与训练。 |
| `results/ablations/training_budget.csv` | 固定种子下五种训练回合数的结果。 |
| `results/ablations/exploration_rates.csv` | 三种探索率策略的结果。 |
| `report/assets/ablation_random_seeds.svg` | 随机种子重复实验的最终 Q 误差图。 |
| `report/assets/ablation_training_budget.svg` | 训练预算与最终 Q 误差图，纵轴为对数尺度。 |
| `report/assets/ablation_exploration.svg` | 探索率策略与最终 Q 误差图，纵轴为对数尺度。 |

仓库中的 `.gitignore` 只用于版本控制；题目 PDF 供参考。这两个文件不属于上述提交成果。

## 提交前补齐

- 补全报告中的跨实验比较、局限、结论、参考资料、姓名、学号和 AI 使用声明。
- 如需声称训练结果对随机性稳健，应在多个随机种子上重复运行，保存并分析各次结果。
- 最后检查报告的图表和文字与 CSV 一致，再将成果文件打包为一个 ZIP。
