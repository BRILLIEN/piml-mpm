# PIML-MPM：基于问题无关机器学习的物质点法高效计算（研究代码）

Python 3 + numpy/scipy（只使用 MATLAB 有直接对应的运算：向量化数组运算、`bincount`≈`accumarray`、稀疏矩阵、`splu`/`spsolve`≈反斜杠、`eigsh`≈`eigs`、`nnls`/批量 `solve`）。
所有计时必须单线程：通过 `tools/guard.sh` 运行（同时提供截止时间守卫）。

## 目录
| 路径 | 内容 |
|---|---|
| `docs/task_original.docx/.md` | 任务文档 |
| `docs/PLAN.md` | 执行计划 |
| `docs/REPORT.md` | **研究报告（结论、数据、复现命令）** |
| `PROGRESS.md` | 进度检查点与交付记录 |
| `common/` | 形函数（线性/二次B样条）、材料（Neo-Hookean）、非均质生成、计时器、全局组装、多尺度基（`msbasis.py` MsFEM 型；`msbasis_os.py` 边自适应） |
| `explicit/` | 显式：基准 `mpm_explicit.py`、主程序 `main_explicit.py`；多尺度约化 `mpm_explicit_ms.py`、对比主程序 `main_explicit_ms.py`；比较 `compare_explicit.py`；cell-crossing `cellcross_test.py`、`analyze_cc.py` |
| `implicit/` | 隐式：基准 `mpm_implicit.py`、预条件 `precond.py`、约化 Newton `mpm_implicit_red.py`、主程序 `main_implicit.py` |
| `ml/` | PIML：`piml_basis.py`（学习 Φ，无净收益的对照）、`train_basis.py`、`train_edge.py`（学习边条带，有净收益）、`models/`（已训练模型与训练日志） |
| `results/` | 结果 json/png 与运行日志（大体积 npz 不入库） |
| `tools/guard.sh` | 截止守卫 + 单线程 |

## 快速开始
```bash
tools/guard.sh python3 -m common.test_shape          # 形函数检验
tools/guard.sh python3 -m common.test_msbasis        # 多尺度基检验（PU/刚体/一次场）
tools/guard.sh python3 -m explicit.main_explicit --h 0.01 --tend 0.4
GUARD_MAX=3600 tools/guard.sh python3 -m explicit.main_explicit_ms --h 0.01 --tend 0.4 --n 8 --K 50 --agg 1 --basis os os_piml --theta 0.1 --edge_suffix _c100 --tag demo
python3 -m explicit.compare_explicit demo --plot
GUARD_MAX=3000 tools/guard.sh python3 -m implicit.main_implicit --h 0.01 --nload 40 --n 8 --solvers direct red_os_agg red_os_piml_agg --quiet
```
