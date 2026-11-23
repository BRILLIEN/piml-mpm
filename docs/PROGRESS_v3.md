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
- [x] 预条件器演进（8×3×3，3.6 万自由度，1 步，PCG 3e-10）：Jacobi 439 次；子结构+界面 Jacobi 407；两错位分区 339；
      **子结构凝聚 + 重叠 3³ 节点块（subpc3）187**；三维中界面节点占 78%（(3/5)³=22% 为块内部），是子结构预条件弱、也是用户应力误差大的同一原因。
- [x] 非精确 Newton（线性容差 1e-3）：subpc3 3 次 Newton、58 次 PCG，误差 1e-9 量级。
- [x] 规模（Δt=0.025，2 步，Python 单线程）：
      | 15.3 万自由度 | 隐式两步 s | PCG | 位移误差 | 应力误差 |
      |---|---|---|---|---|
      | 细解 Jacobi-PCG 3e-10 | 242 | 1165 | — | — |
      | 细解 非精确（1e-3） | 186 | 453 | 1e-9 | 1e-8 |
      | 子结构（用户方案） | 166 | — | 2.7–3.6% | 57–64% |
      | subpc3 非精确（1e-3） | 291 | 154 | 6e-10 | 1e-8 |
      subpc3 的 PCG 次数随规模有界（每步 ≈62，7.5 万→15 万），Jacobi 增长（526→577）；但单次预条件应用 ≈6× Jacobi，Python 下总时不占优。
      Python 版细切线组装占隐式 60–70%（与用户纯 MATLAB 细解 282/361 s 一致）。
- [x] 精度受控（_loose，完整细空间 Newton 1e-3、线性 1e-1）：8×3×3 上 subpc3 仅 22 次 PCG，位移 1.6e-5、应力 2.8e-4（子结构 56%）。
      公平对照（细解同样放宽）15.3 万自由度运行中：results/logs/loose_32.log。
## 进行中 / 下一步
1. 等 loose_32 结果；按目标模块（切线 + 局部 + 凝聚 + 界面 + 迭代 + 恢复）给出三类加速比。
2. 降低 subpc3 构造开销（重叠块提取/求逆、Condense 的逐块切片循环）。
3. 依据用户 MATLAB 实测单位成本（MEX 细切线 2.07 s/次、PCG 0.173 s/次、凝聚 ≈2.1 s/次）给出百万级的预估（标注为估算）。
