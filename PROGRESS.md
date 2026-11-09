# 进度与检查点（每完成一个子步骤即更新并本地 git commit）

## 调度
- 开始：2026-11-09 19:45 (UTC+8)；额度恢复续跑：2026-11-10 00:45 起，并每小时整 45 分重试；硬截止：2026-11-10 09:30。
- 守卫：所有长计算经 `tools/guard.sh` 启动（截止前 3 分钟拒绝新操作，强制单线程）。
- 恢复流程：`cat PROGRESS.md` → `git log --oneline | head` → `git status` → 从"下一步"继续；若 `python3 -c "import scipy"` 失败则 `pip install scipy matplotlib`。

## 状态（最近更新 2026-11-09 21:25 UTC+8）
- [x] 准备：任务文档、环境（无 MATLAB → Python3.13+numpy+scipy，单线程）、计划、定时任务。
- [x] 阶段0：common/shape.py（线性/二次B样条，PU/一次再现/梯度跨格连续性检验 PASS）、timer、material(Neo-Hookean)、hetero。
- [x] E1 显式基准：explicit/mpm_explicit.py（USL+FLIP0.99），算例 非均质悬臂梁(L=1,H=0.2,夹杂30%,刚度比10,g=10)，
      周期≈0.79s，自由端最大挠度≈0.43L；细尺度能量漂移 −1.4%/半周期(h=0.01)。
      计时分解（h=0.005~0.00125, Np=32k~512k）：shape≈30%、P2G≈27%、G2P≈31%、本构≈5%、网格≈5%
      ⇒ Amdahl：仅减少网格自由度最多 ~5% 收益；必须减少粒子运算/时间步数。
- [~] E3 显式非学习加速（进行中）：common/msbasis.py 多尺度数值形函数（粗单元粒子材料图像 + ersatz 正则 + Gauss 积分 + 局部凝聚，
      双线骨架使局部问题仅依赖本单元粒子；PU/刚体转动 1e-13 精确，均质区一次场精确；C1 粗二次B样条骨架值），
      explicit/mpm_explicit_ms.py（粗运动学 + 区间冻结映射 TL 子步 + 区间末重建基；一致粗质量）。
      已发现并定位：(1) 行和集中质量在基含负值时失效 → 改一致质量；(2) FLIP 在区间交界投影注入能量 → 区间末 PIC；
      (3) 均质时粗空间频率误差<0.15%，非均质(夹杂尺寸≈粗单元)时 10–15%（MsFEM 边界层/共振误差，尺度不分离）；
      (4) 小切割粗自由度（表面附近质量极小）限制临界步 → 待实现扩展 B 样条式外推约束。
- [x] E2 cell-crossing 机理：explicit/cellcross_test.py（均匀应力块体）：规则格点平移两者均精确(1e-16)；
      7% 非均匀间距：linear 0.56 vs qbspline 0.035（16×）；随机扰动 ppc=2：0.85 vs 0.24，ppc=4：0.37 vs 0.056
      ⇒ 二次B样条显著抑制梯度跳变误差，但不消除粒子积分误差。USL+linear 在 h=0.01 发散（小质量节点），
      动力学批量（USL/MUSL × linear/qbspline × 3 网格偏移）后台运行：results/logs/cc_{MUSL,USL}_*。
- [x] E3a 结果（t5）：ms 基 K=100：n=4 3.0× tip误差17%；n=8 6.1× 18%；bilinear_n8 7.4× 39%。
- [x] 精度瓶颈诊断（全局 Guyan 上界实验）：骨架整体强制为粗函数 → 11–16% 频率误差；仅强制角块 → n=4 2.5%、n=8 0.2–1.4%。
- [x] E3b 边自适应多尺度基 common/msbasis_os.py（两单元块求骨架条带 + 单元内部凝聚；协调；PU/刚体 1e-11）：
      无外推：频率误差 n=4 5.0%、n=8 2.8–4.2%（原 ms 12%/10%）；θ=0.3 外推：dt 放大 2.7×/5.8×，误差 +1–3%。
      显式动力学对比 t6（os，θ=0.1，K=50/100）后台运行中：results/logs/t6.log。
- [x] I1/I3：见上；对比度扫描（h=0.01，n=4）：对比度100：bil 3623 / ms 3340 次 PCG（−8%）；1000：bil 23069 / ms 18298（−21%，68s→52s），
      直接法 18.1s/28.7s 仍更快（2D 小规模）。规模扫描 h=0.005/0.0025（n=8）后台运行：results/logs/isc_*。
- [x] 模板法局部刚度（与 Gauss 伪粒子 1e-14 一致，快 4–10×）——ML 的公平非学习对照。
- [x] PIML-Φ（ml/piml_basis.py, ml/train_basis.py）：合成问题无关数据训练；PU/刚体由投影精确保持；
      n=4 测试 ΔΦ 相对误差 21%，n=8 29%；用于隐式两级预条件：PCG 次数 2368 vs 精确 2339（+1.2%）；
      但基构造时间无收益（3.83s vs 3.35s）：二维局部问题很小，精确凝聚 ~0.04/0.5 MFlop/单元 < 网络 0.28/1.38 MFlop
      ⇒ 结论：直接学习形函数在 2D 无净收益（与任务文档警示一致）。
- [x] PIML-edge（ml/train_edge.py）：t7（h=0.01,n=8,K=50,同 dt）：精确边 34.0s(2.37×,误差5.45%) → PIML 16.6s(4.86×,误差7.4%)，
      ML 增量 2.05×（基重建 21.7→4.0s）；非学习 K=100 仅 4.16×/11.8% ⇒ PIML 在精度-效率上占优。离线 ≈7min。
      c100 类别模型 t7b：5.98×、误差6.8%。跨工况 beam2（x2，不重训）：精确边 3.32×/10.7%，PIML 6.11×/13.3%。
      隐式边自适应粗空间：对比度1000 PCG 13887 次（双线性 23069，MsFEM 18355），仍慢于直接法。
      后台：t8（h=0.005 显式）、isc h=0.0025 pcg。
- [ ] 报告 docs/REPORT.md

## 下一步
1) 查看 t6、isc、cc 结果；2) PIML-edge：数据生成/训练/接入 build_basis_os（ML 预测条带，回退精确）并计时对比；
3) 报告。

## 已设置的定时消息（发送至本会话，UTC+8）
19:45 开始 trig_01HwdM5xUoKAUYGFF6nY5Du7；20:45 trig_01ASWo8Qfz4UEV3JMABuprEB；21:45 trig_01WukxVy8jCuRT1NfZBMntME；
22:45 trig_01XCSPM9bxB4mFTQBQLxoa9u；23:45 trig_0179espSdSFHwmvQm53gHfoV；00:45 额度恢复 trig_015rbqdoMZB5fPS9prVFim8e；
01:45 trig_01TEjvgqs8etpftSK1U1Wd4R；02:45 trig_01EK4afc8CFjo8yuEPxzM8Wi；03:45 trig_01E8FtvtzqZ881gRvPHyUDNi；
04:45 trig_01Nqja6ccpS6DWYTAh6ATQCD；05:45 trig_0128AYAFBSWj1r2yGYERb4Em；06:45 trig_015yRX7uoqDuEt62jsES7rcm；
07:45 trig_01U275AcbLmnmbSTQsi3gfEA；08:45 收尾 trig_01UF9sV9wB7kYFrncR2dMqvg；09:30 硬截止 trig_01DXpPdPHCRph4K3HUqRWVpK
（如需取消：delete_trigger 对应 ID）
