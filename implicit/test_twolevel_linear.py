# 线性弹性检验：粗空间解 + 重叠局部（以粗节点为中心的 2×2 粗单元块）限制加性 Schwarz 校正，测精度随扫描次数的变化
import numpy as np, sys, time, scipy.sparse as sp
from scipy.sparse.linalg import splu
sys.path.insert(0,'/home/user/piml-mpm')
from implicit.main_implicit import BASE, case_beam_static
from common.assemble import elastic_K, lumped_mass
from common.shape import shape_2d
from common.msbasis_os import build_basis_os
from common.msbasis import extend_small_cut, coarse_global
h=float(sys.argv[1]); n=int(sys.argv[2]); nsweep=int(sys.argv[3])
P,g,par=case_beam_static(dict(BASE,h=h)); X=P['x']; w=P['V0']; lam=P['lam']; mu=P['mu']; ph=P['phase']; nn=g['nn']; nx=g['nx']
K=elastic_K(X,w,lam,mu,g).tocsr(); M=lumped_mass(X,P['m'],g)
fx=np.zeros(2*nn,bool); fx[par['fixed_nodes']]=True; fx[par['fixed_nodes']+nn]=True
act=(~fx)&(M>0); fd=np.where(act)[0]; Kf=K[fd][:,fd].tocsc()
f=np.zeros(2*nn); f[nn:]=-M[nn:]*1.0; ff=f[fd]
t=time.perf_counter(); us=splu(Kf,permc_spec='MMD_AT_PLUS_A').solve(ff); t_dir=time.perf_counter()-t
idx,N,dNx,dNy=shape_2d(X,g)
def stress(uf):
    u=np.zeros(2*nn); u[fd]=uf; ux=u[:nn][idx]; uy=u[nn:][idx]
    exx=np.sum(ux*dNx,1); eyy=np.sum(uy*dNy,1); exy=0.5*(np.sum(ux*dNy,1)+np.sum(uy*dNx,1))
    tr=exx+eyy; return np.stack([lam*tr+2*mu*exx,2*mu*exy,lam*tr+2*mu*eyy],1), u
key=(np.floor(X[:,0]/h).astype(np.int64)*100003+np.floor(X[:,1]/h).astype(np.int64))*4+ph
_,inv=np.unique(key,return_inverse=True); ws=np.bincount(inv,w)
avg=lambda S: np.stack([np.bincount(inv,w*S[:,k])/ws for k in range(3)],1)
Sref,uref=stress(us); Aref=avg(Sref); tip=par['monitor'][0]
iN,NN,_,_=shape_2d(X[tip:tip+1],g)
def tipy(u): return np.sum(NN*u[nn:][iN])
def metrics(uf):
    S,u=stress(uf); A=avg(S); d=A-Aref
    e=np.sqrt(np.sum(ws*(d[:,0]**2+2*d[:,1]**2+d[:,2]**2))/np.sum(ws*(Aref[:,0]**2+2*Aref[:,1]**2+Aref[:,2]**2)))
    return abs(tipy(u)/tipy(uref)-1), e
# 粗空间：边自适应基 + 外推 + 有限转动缺陷增强
B=build_basis_os(X,w,lam,mu,g,n,fixed_nodes=par['fixed_nodes'])
Phi,_=extend_small_cut(B['Phi'],B['cg'],M,0.1)
xc=B['cg']['xc']; xn=g['xn']; ev=Phi@np.concatenate([xc[:,0],xc[:,1]]); ex=ev[:nn]-xn[:,0]; ey=ev[nn:]-xn[:,1]
live=np.asarray(abs(Phi).sum(1)).ravel(); ex[(live[:nn]==0)&(live[nn:]==0)]=0; ey[(live[:nn]==0)&(live[nn:]==0)]=0
Cq=coarse_global(g,B['cg'],np.arange(nn),'qbspline'); Phi=sp.hstack([Phi,sp.vstack([sp.diags(ex)@Cq,sp.diags(ey)@Cq])]).tocsr()
Pf=Phi[fd].tocsc(); cn=np.sqrt(np.asarray(Pf.multiply(Pf).sum(0)).ravel()); Pf=Pf[:,cn>1e-8*cn.max()].tocsr()
Kc=(Pf.T@Kf@Pf).tocsc(); clu=splu(Kc)
uc=Pf@clu.solve(Pf.T@ff)
print(f'h={h} n={n} ndof={len(fd)} coarse={Pf.shape[1]}  direct {t_dir:.2f}s')
print('  粗空间解         : tip err %.4f  span-phase stress err %.4f'%metrics(uc))
# 重叠块：以每个粗节点为中心、覆盖其相邻 2×2 粗单元的细节点（全局细节点 i,j 落在 [nA-n+?, nA+n+1] 内）
node_of=fd%nn; ix=node_of%nx; iy=node_of//nx; cg=B['cg']
A_of=np.floor((ix-0.5)/n).astype(int); B_of=np.floor((iy-0.5)/n).astype(int)
patches=[]; 
for A in range(A_of.min(),A_of.max()+2):
    for Bq in range(B_of.min(),B_of.max()+2):
        sel=np.where((A_of>=A-1)&(A_of<=A)&(B_of>=Bq-1)&(B_of<=Bq))[0]
        if len(sel)>0: patches.append(sel)
# 单位分解权（限制加性 Schwarz：每个自由度按覆盖次数平均）
cnt=np.zeros(len(fd)); 
for s_ in patches: cnt[s_]+=1
t=time.perf_counter(); lus=[(s_,splu(Kf[s_][:,s_].tocsc())) for s_ in patches]; t_fac=time.perf_counter()-t
u=uc.copy()
for k in range(nsweep):
    r=ff-Kf@u; du=np.zeros_like(u)
    for s_,lu in lus: du[s_]+=lu.solve(r[s_])/cnt[s_]
    u=u+du
    r=ff-Kf@u; u=u+Pf@clu.solve(Pf.T@r)          # 粗校正（两级乘性）
    print('  第 %d 次局部扫描+粗校正: tip err %.4f  span-phase stress err %.4f'%((k+1,)+metrics(u)))
print(f'  块数 {len(patches)}，局部分解 {t_fac:.2f}s')
# ---- PCG + 对称两级加性 Schwarz（局部块精确解 + 粗校正），从粗空间初值出发 ----
from scipy.sparse.linalg import cg, LinearOperator
PT=Pf.T.tocsr()
def M2(r):
    z=np.zeros_like(r)
    for s_,lu in lus: z[s_]+=lu.solve(r[s_])
    return z+Pf@clu.solve(PT@r)
it=[0]; hist=[]
def cb(xk):
    it[0]+=1; hist.append((it[0],)+metrics(xk))
t=time.perf_counter()
x,info=cg(Kf,ff,x0=uc,rtol=1e-8,maxiter=200,M=LinearOperator(Kf.shape,M2),callback=cb)
t_cg=time.perf_counter()-t
for k,a,b in hist[:6]: print('  PCG 第 %d 次: tip err %.4f  span-phase stress err %.4f'%(k,a,b))
print(f'  PCG 收敛到 1e-8：{it[0]} 次，{t_cg:.2f}s（每次 {t_cg/max(it[0],1)*1e3:.1f} ms），info={info}')
# 单次预条件应用中各部分耗时
r=ff-Kf@uc
t=time.perf_counter(); [lu.solve(r[s_]) for s_,lu in lus]; t_loc=time.perf_counter()-t
t=time.perf_counter(); Kf@r; t_mv=time.perf_counter()-t
t=time.perf_counter(); Pf@clu.solve(PT@r); t_c=time.perf_counter()-t
print(f'  一次：局部回代 {t_loc*1e3:.1f} ms，矩阵向量积 {t_mv*1e3:.2f} ms，粗校正 {t_c*1e3:.2f} ms')
