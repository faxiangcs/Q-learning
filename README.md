# IOTA 5201 Mid-Term Assignment: Implementing Q-Learning

本目录是作业提交包的文件骨架。目前除本 README、题目 PDF 和 `.gitignore` 外，提交成果文件均为空占位文件，尚不能运行程序或打开报告。完成作业后，将本 README、`code/`、`config/`、`report/`、`results/` 和 `requirements.txt` 打成一个 ZIP 提交。`IOTA_Assignment-1.pdf` 是题目原文，不是要求提交的成果文件。

## 文件说明

| 文件 | 用途和后续需要填写的内容 |
| --- | --- |
| `README.md` | 提交包说明。完成后补充依赖安装方式、两组实验的准确运行命令、图表再生成命令，以及结果文件与报告章节的对应关系。 |
| `requirements.txt` | 记录运行代码和生成报告图表所需的 Python 依赖及具体版本。若改用其他语言，也应在 README 中说明对应依赖。 |
| `code/environment.py` | 定义图 1 的 A-F 状态、可选动作、合法转移、奖励和回合终止条件。 |
| `code/q_learning.py` | 实现 Q 表初始化、动作选择、Q-learning 更新、终止状态处理和训练过程。原有的空 `code.py` 占位文件已整理到这里。 |
| `code/run_experiments.py` | 运行两组配置，固定并记录随机种子，保存 Q 表和收敛数值，并生成报告使用的图。 |
| `config/baseline.json` | 原始起点/目标配置，以及学习率、折扣因子、探索参数、训练轮数、随机种子等设置。图 1 似乎表示机器人从 C 前往 F；题目正文未明示，实施时应核实并在报告中说明采用的解释。 |
| `config/start_A_goal_C.json` | 修改后的配置：起点 A、目标 C；记录其余实验参数，便于与原始配置比较。 |
| `report/index.html` | 必交的 HTML 技术报告。应包含方法、环境和实验设置、两组初始与最终 Q 表、收敛图及指标解释、比较分析、结论、参考资料、姓名、学号和 AI 使用声明。 |
| `report/assets/baseline_convergence.svg` | 原始配置的 Q 值收敛图；完成后由实际实验数据生成，并在报告中引用。 |
| `report/assets/start_A_goal_C_convergence.svg` | A→C 配置的 Q 值收敛图；完成后由实际实验数据生成，并在报告中引用。 |
| `results/baseline/q_initial.csv` | 原始配置训练前的 Q 表，机器可读。 |
| `results/baseline/q_final.csv` | 原始配置训练后的 Q 表，机器可读。 |
| `results/baseline/convergence.csv` | 原始配置收敛图对应的逐轮或定期采样数值，机器可读。 |
| `results/start_A_goal_C/q_initial.csv` | A→C 配置训练前的 Q 表，机器可读。 |
| `results/start_A_goal_C/q_final.csv` | A→C 配置训练后的 Q 表，机器可读。 |
| `results/start_A_goal_C/convergence.csv` | A→C 配置收敛图对应的数值，机器可读。 |

仓库中的 `.gitignore` 只用于版本控制；题目 PDF 供参考。这两个文件不属于上述提交成果。

## 提交前补齐

- 在 `requirements.txt` 填写实际使用的版本，并在两份配置中填写完整参数和随机种子。
- 在此处写明依赖安装、运行原始配置、运行 A→C 配置以及重新生成两幅图的准确命令。
- 运行并核验两组实验，填充所有 Q 表、收敛数据和图；报告中的数值应与这些文件一致。若讨论结果的稳定性或可靠性，按需要增加重复运行数据。
- 确认 `report/index.html` 无需账号、付费服务、AI 工具或实时训练即可阅读；若需要本地服务器，在此处给出启动命令。如使用站点生成器或 Web 框架，附上可编辑源码和构建说明。
- 检查图表坐标轴、图注、重要视觉内容的文字说明、参考资料和 AI 使用声明；最后将上述成果文件打包为一个 ZIP。
