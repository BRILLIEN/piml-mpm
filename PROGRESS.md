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
- [ ] E2 cell-crossing（批量结果在 results/logs/cc_*，待整理分析）
- [x] E3a 结果（results/explicit/t5*，h=0.01，半周期）：小切割外推后临界步放大 2.45×(n=4)/6.2×(n=8)；
      K=100：ms_n4 3.0× 加速 tip误差17%；ms_n8 6.1× 误差18%；bilinear_n8 7.4× 误差39%；K=25 满步长不稳定。
      结论：非均质（夹杂≈粗单元，无尺度分离）下 MsFEM 型光滑骨架约束误差过大（均质<0.15%）——需 LOD/过采样，列为后续。
- [x] I1 隐式基准：implicit/mpm_implicit.py（UL 准静态 Newton，一致切线=材料+几何，载荷步内冻结映射；
      细尺度小支撑节点外推约束 θ=0.05 解决重映射后 Newton 振荡；线搜索有害，默认关闭）。
      h=0.01（8k 粒子，~4.3k 自由度，10 载荷步，g=20 → 自由端挠度 0.43）：direct 9.6s，45 次 Newton；
      计时：pattern 27%、tangent 25%、solve 45%。切线 SPD，条件数 2.5e6。
- [~] I3：PCG+ILU（默认/对称模式、低填充）3000 步不收敛；两级 PCG（对称ILU平滑+Galerkin 粗校正）收敛到与直接法相同解（差 1e-15），
      n=4 ≈52 次/求解、n=8 ≈73 次；对比度 10 时 ms/qb/bil 粗空间迭代数几乎相同；小规模下直接法更快。
- [ ] 待办：对比度扫描(10/100/1000)与规模扫描；pattern 用结构化模板去 unique；基构造改模板化高效实现（ML 公平对照）；PIML 基网络；报告。

## 下一步
1) 后台：隐式对比度扫描；2) msbasis 模板化局部刚度（高效非学习基准）；3) PIML 基网络（PU/刚体约束精确投影）；4) E2 整理；5) 报告。

## 已设置的定时消息（发送至本会话，UTC+8）
19:45 开始 trig_01HwdM5xUoKAUYGFF6nY5Du7；20:45 trig_01ASWo8Qfz4UEV3JMABuprEB；21:45 trig_01WukxVy8jCuRT1NfZBMntME；
22:45 trig_01XCSPM9bxB4mFTQBQLxoa9u；23:45 trig_0179espSdSFHwmvQm53gHfoV；00:45 额度恢复 trig_015rbqdoMZB5fPS9prVFim8e；
01:45 trig_01TEjvgqs8etpftSK1U1Wd4R；02:45 trig_01EK4afc8CFjo8yuEPxzM8Wi；03:45 trig_01E8FtvtzqZ881gRvPHyUDNi；
04:45 trig_01Nqja6ccpS6DWYTAh6ATQCD；05:45 trig_0128AYAFBSWj1r2yGYERb4Em；06:45 trig_015yRX7uoqDuEt62jsES7rcm；
07:45 trig_01U275AcbLmnmbSTQsi3gfEA；08:45 收尾 trig_01UF9sV9wB7kYFrncR2dMqvg；09:30 硬截止 trig_01DXpPdPHCRph4K3HUqRWVpK
（如需取消：delete_trigger 对应 ID）
