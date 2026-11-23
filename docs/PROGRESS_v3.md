# 进度 v3（2026-11-23 起；隐式优先，借鉴用户三维子结构/一致切线方案）

## 定时
- 整点续跑检查：trig_012hUhK7z6RA75vQGBBJVprn（19:00–02:00 每整点附近），02:10 停用提醒 trig_01W7JSNG7c61W25ohfSSQRq8。

## 已完成
- [x] 读用户资料（docs：隐式物质点法子结构加速研究内容（修订稿）；MATLAB：case1_TOP3D_Heterogeneous、implicit_step_heterogeneous、
      beam_basis/transfer/neo/tangent/condense_current、BasicMat_trace 等；结果：0.025 s 步长 1M 自由度 子结构总加速 11.8–12.1×，
      但位移误差 4.5%、应力误差 52%）。
- [x] Python 移植（implicit3d/，函数与变量名与 MATLAB 一致）：setup_heterogeneous_beam、beam_basis（按 27 节点支撑分组）、
      beam_transfer（p2g/force/increment/g2p，APIC）、beam_neo、beam_tangent（按支撑单元积分 81×81，结构化 5³ 模板直接累加 CSR）、
      BasicMat_trace（三线性粗界面、两层共享细界面）、implicit_step_heterogeneous（平均加速度 Newmark、逐轮一致切线、能量线搜索）、
      主脚本 case1_TOP3D_Heterogeneous。
- [x] 一致切线有限差分校验：|K d − FD|/|FD| = 4.7e-6（ε=1e-6）/4.7e-7（ε=1e-7），对称 1e-16。
- [x] 冒烟（16×3×3 块、h=0.125、7.5 万自由度、Δt=0.025 s、1 步）：
      | 方案 | 本步 s | Newton | PCG | 位移误差 | 应力误差 |
      |---|---|---|---|---|---|
      | fine_pcg（细解 Jacobi-PCG 3e-10） | 63.9 | 3 | 526 | — | — |
      | sub_current（用户子结构） | 51.3 | 3 | — | 3.66% | 55.9%（与用户 1M 结果 2.45%/53% 同型） |
      | subpc（子结构预条件细空间 PCG，新） | 74.3 | 3 | 406 | 8e-15 | 1.2e-13 |
      ⇒ subpc 精度与细解一致，但界面 Jacobi 太弱（406 次）；Python 版切线组装占主导（≈14 s/次）。
## 进行中 / 下一步
1. subpc2（错位 2 个细节点的第二分区，加性组合）——首次测试 30 分钟未完成，需剖析耗时（疑为 Condense 中逐块稀疏切片过慢）。
2. 切线组装提速（向量化位置计算/复用每步不变的位置映射）。
3. 规模：16×3×3 → 32×3×3（15 万）→ 用户 20 万配置；记录目标模块（切线+局部+凝聚+界面+迭代+恢复）三类加速比。
