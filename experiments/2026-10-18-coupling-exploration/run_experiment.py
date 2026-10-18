"""Independent experiment based on HTW's log2inter.py. Original repo is untouched.
Run: python run_experiment.py (NumPy, SciPy, Matplotlib).
All time averages use natural logarithms and one map iteration as time unit.
"""
from pathlib import Path
import csv, json, platform
import numpy as np
import scipy
from scipy.optimize import root
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).parent/'results'; OUT.mkdir(exist_ok=True)
RX,RY=3.8318741,3.7389149
DEFAULT=np.array([.12924,.2304])
def advance(z,k):
    a=RX*z[...,0]*(1-z[...,0]); b=RY*z[...,1]*(1-z[...,1])
    # Preserve the arithmetic ordering in the original program.
    return np.stack((a+k*(b-a),b+k*(a-b)),axis=-1)
def jac(z,k):
    a=RX*(1-2*z[...,0]); b=RY*(1-2*z[...,1])
    return np.stack((np.stack(((1-k)*a,k*b),-1),np.stack((k*a,(1-k)*b),-1)),-2)
def period(tail,tol=1e-8,maxp=64):
    # Every retained pair must recur; use smallest accepted lag.
    out=np.zeros(tail.shape[1:],dtype=int)
    for p in range(1,maxp+1):
        good=np.max(np.abs(tail[p:]-tail[:-p]),axis=0)<tol
        out[(out==0)&good]=p
    return out

def batch(ks,seeds,burn=12000,measure=8192,tailn=1024):
    ks=np.asarray(ks); z=np.broadcast_to(seeds,(len(ks),len(seeds),2)).copy(); k=ks[:,None]
    for _ in range(1,250): z=advance(z,0.)
    v=np.broadcast_to([1.,.37],z.shape).copy(); v/=np.linalg.norm(v,axis=-1)[...,None]
    sums=np.zeros(z.shape[:-1]); blocks=[]; buf=[]; tail=[]; dsum=np.zeros_like(sums); dmax=np.zeros_like(sums)
    for i in range(burn+measure):
        w=np.einsum('...ij,...j->...i',jac(z,k),v); norm=np.linalg.norm(w,axis=-1)
        v=w/np.maximum(norm,1e-300)[...,None]
        z=advance(z,k)
        if i>=burn:
            ln=np.log(np.maximum(norm,1e-300)); sums+=ln; buf.append(ln)
            d=np.abs(z[...,0]-z[...,1]); dsum+=d; dmax=np.maximum(dmax,d)
            if len(buf)==1024: blocks.append(np.mean(buf,axis=0));buf=[]
        if i>=burn+measure-tailn: tail.append(z.copy())
    tail=np.array(tail); ps=period(tail)
    rows=[]
    for a,kv in enumerate(ks):
        for b,seed in enumerate(seeds):
            rows.append(dict(k=float(kv),seed=b,x0=float(seed[0]),y0=float(seed[1]),px=int(ps[a,b,0]),py=int(ps[a,b,1]),lambda_max=float(sums[a,b]/measure),block_min=float(np.min(blocks,axis=0)[a,b]),block_max=float(np.max(blocks,axis=0)[a,b]),mean_abs_diff=float(dsum[a,b]/measure),max_abs_diff=float(dmax[a,b])))
    return rows,tail

def writecsv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def trajectory(k,n=400,seed=DEFAULT):
    z=np.empty((n,2));z[0]=seed
    for m in range(1,n):z[m]=advance(z[m-1],0 if m<250 else k)
    return z

def main():
    rng=np.random.default_rng(20261018)
    seeds=np.vstack((DEFAULT,rng.uniform(.001,.999,(15,2))))
    ks=np.r_[np.arange(401)*.0005,np.arange(.202,.50001,.002)]
    rows,tail=batch(ks,seeds);writecsv('scan.csv',rows)
    np.savez_compressed(OUT/'scan_tail.npz',ks=ks,seeds=seeds,tail=tail[-128:,:,0,:])
    selected=[0,.05,.08,.1,.12,.15,.3,.5]
    long,lt=batch(selected,seeds,burn=50000,measure=65536);writecsv('long_checks.csv',long)
    np.savez_compressed(OUT/'long_tail.npz',ks=selected,seeds=seeds,tail=lt)
    short=[]
    for k in selected:
        z=trajectory(k); p=period(z[300:,None,:],maxp=20)[0]
        short.append(dict(k=k,px=int(p[0]),py=int(p[1]),lag4_residual=float(np.max(np.abs(z[378:]-z[374:396])))))
    writecsv('original_400_steps.csv',short)
    # Local stability of the orbit: eigenvalues of D F^p, independently of tangent estimate.
    floquet=[]
    for a,k in enumerate(selected):
        r=long[a*len(seeds)];p=int(np.lcm(r['px'],r['py'])) if r['px']*r['py'] else 0
        if not p:continue
        def fp(z):
            for _ in range(p):z=advance(z,k)
            return z
        sol=root(lambda z:fp(z)-z,lt[-1,a,0]); z=sol.x.copy();M=np.eye(2)
        for _ in range(p):M=jac(z,k)@M;z=advance(z,k)
        eig=np.linalg.eigvals(M)
        floquet.append(dict(k=k,p=p,residual=float(np.max(np.abs(fp(sol.x)-sol.x))),rho=float(max(abs(eig))),eig1=str(eig[0]),eig2=str(eig[1])))
    writecsv('floquet.csv',floquet)
    # Sensitive dependence: perturb an already settled state; coupling-onset states happen to be near a critical point.
    fig,axs=plt.subplots(3,1,figsize=(11,10),constrained_layout=True)
    for col,label in [(0,'x'),(1,'y')]:
        axs[0].plot(np.repeat(ks,128),tail[-128:,:,0,col].T.ravel(),'.',ms=.45,alpha=.6,label=label)
    axs[0].set(ylabel='Late state',title='Output-coupled logistic maps | original initial condition');axs[0].legend()
    rr=np.array([r['lambda_max'] for r in rows]).reshape(len(ks),len(seeds))
    axs[1].plot(ks,rr[:,0],lw=.9,label='original seed');axs[1].fill_between(ks,rr.min(1),rr.max(1),alpha=.25,label='range over 16 initial conditions');axs[1].axhline(0,c='k',lw=.7);axs[1].set(ylabel='Largest Lyapunov exponent');axs[1].legend()
    for target,marker in [(4,'s'),(2,'o')]:
        pp=np.array([r['px']==target and r['py']==target for r in rows]).reshape(len(ks),len(seeds)).mean(1)
        axs[2].plot(ks,pp,marker,ms=2,label=f'both period {target}')
    axs[2].set(xlabel='Symmetric coupling k',ylabel='Fraction of 16 seeds',ylim=(-.05,1.05));axs[2].legend()
    fig.savefig(OUT/'overview.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    sensitivity=[]
    for k in [.05,.08,.1,.12,.15,.5]:
        z=trajectory(k,20251)[-1]; zp=z+np.array([1e-10,0.]); ds=[]
        for t in range(400):
            ds.append(float(np.linalg.norm(z-zp)));z=advance(z,k);zp=advance(zp,k)
        axs[1,1].semilogy(np.maximum(ds,1e-17),label=str(k))
        sensitivity.extend(dict(k=k,t=t,distance=d) for t,d in enumerate(ds))
    writecsv('sensitivity.csv',sensitivity)
    for a,k in enumerate([.08,.1,.12]):
        z=trajectory(k,10000)[-24:];axs[0,0].plot(z[:,0],'.-',label=f'x k={k}');axs[0,1].plot(z[:,1],'.-',label=f'y k={k}')
        axs[1,0].plot(z[:,0],z[:,1],'o',label=f'k={k}')
    axs[0,0].set(title='Late x: period 4 / 4 / 2',xlabel='Late iteration');axs[0,1].set(title='Late y',xlabel='Late iteration')
    axs[1,0].plot([0,1],[0,1],'k--',lw=.8);axs[1,0].set(xlabel='x',ylabel='y',title='Equal period does not imply x = y')
    axs[1,1].set(xlabel='Steps after perturbation at m=20250',ylabel='Euclidean separation',title='Finite perturbation (roundoff floor shown)')
    for ax in axs.flat:ax.legend(fontsize=7)
    fig.savefig(OUT/'examples.png',dpi=160);plt.close(fig)
    metadata=dict(source_commit='3366589902cf4a312f73dd8d7a0a1df2643149c0',source_blob='392986eca08c5b4ee25574ba2b30f65eb82f503c',rx=RX,ry=RY,seed=20261018,burn=12000,measure=8192,tail=1024,period_max=64,period_tol=1e-8,long_burn=50000,long_measure=65536,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__)
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print('DEFAULT LONG CHECKS');print(json.dumps(long[::len(seeds)],indent=2));print('FLOQUET',floquet)
    print('SCAN DEFAULT PERIOD RUNS')
    p=np.array([(r['px'],r['py']) for r in rows[::len(seeds)]])
    start=0
    for i in range(1,len(ks)+1):
        if i==len(ks) or np.any(p[i]!=p[start]):
            print(ks[start],ks[i-1],p[start].tolist());start=i
if __name__=='__main__': main()
