"""Independent right-endpoint study. Run: python research.py. No network writes."""
from pathlib import Path
import json, csv, platform
import numpy as np
from scipy.optimize import root
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).parent/'results'
OUT.mkdir(exist_ok=True)
RX,RY=3.8318741,3.7389149
DEFAULT=np.array([.12924,.2304])

def F(z,k):
    z=np.asarray(z)
    a=RX*z[...,0]*(1-z[...,0]); b=RY*z[...,1]*(1-z[...,1])
    return np.stack((a+k*(b-a),b+k*(a-b)),axis=-1)

def J(z,k):
    a=RX*(1-2*z[0]);b=RY*(1-2*z[1])
    return np.array([[(1-k)*a,k*b],[k*a,(1-k)*b]])

def FK(z):
    a=RX*z[0]*(1-z[0]);b=RY*z[1]*(1-z[1])
    return np.array([b-a,a-b])

def G(z,k): return F(F(z,k),k)-z

def DG(z,k):
    b=F(z,k)
    return np.column_stack((J(b,k)@J(z,k)-np.eye(2), J(b,k)@FK(z)+FK(b)))

def info(q):
    z=q[:2]; k=q[2]; b=F(z,k)
    eig=np.linalg.eigvals(J(b,k)@J(z,k))
    eig=eig[np.argsort(eig.real)]
    return dict(k=float(k),ax=float(z[0]),ay=float(z[1]),bx=float(b[0]),by=float(b[1]),
      gap=float(np.linalg.norm(z-b)),residual=float(np.max(np.abs(G(z,k)))),
      mu1_real=float(eig[0].real),mu1_imag=float(eig[0].imag),
      mu2_real=float(eig[1].real),mu2_imag=float(eig[1].imag),rho=float(max(abs(eig))))

def savecsv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def tangent(q,prev=None):
    _,_,vt=np.linalg.svd(DG(q[:2],q[2]));t=vt[-1]
    if prev is not None and t@prev<0:t=-t
    return t

def trace(q,direction=1,ds=.0002,n=16000):
    t=tangent(q)
    if t[2]*direction<0:t=-t
    out=[q.copy()]; stop='max_steps'
    for i in range(n):
        pred=q+ds*t
        sol=root(lambda u: np.r_[G(u[:2],u[2]),(u-pred)@t],pred,
           jac=lambda u: np.vstack((DG(u[:2],u[2]),t)),tol=1e-11)
        if np.max(abs(sol.fun))>1e-10:raise RuntimeError(('arc corrector',i,sol.message))
        q=sol.x;t=tangent(q,t);out.append(q.copy())
        if not (0<q[2]<.5):stop='parameter_boundary';break
        if np.linalg.norm(q[:2]-F(q[:2],q[2]))<1e-6:stop='near_period1';break
        if np.max(abs(q[:2]))>3:stop='state_boundary';break
        if i>100 and np.linalg.norm(q-out[0])<ds*.8:stop='closed_loop';break
    return np.array(out),stop

def main():
    z=DEFAULT.copy()
    for _ in range(249):z=F(z,0)
    for _ in range(20000):z=F(z,.12)
    checks=[]
    for k in [.12,.145,.1452]:
        s=root(lambda u:G(u,k),z,jac=lambda u:DG(u,k)[:,:2],tol=1e-12)
        assert max(abs(s.fun))<1e-11
        z=s.x;checks.append(info(np.r_[z,k]))
    print('CHECKS',json.dumps(checks),flush=True)
    fold=root(lambda q:np.r_[G(q[:2],q[2]),np.linalg.det(DG(q[:2],q[2])[:,:2])],np.r_[z,.1453],tol=1e-12)
    assert max(abs(fold.fun))<1e-10
    print('FOLD',json.dumps(info(fold.x)),flush=True)
    q0=np.r_[z,.1452]
    branches=[]
    for d in [-1,1]:
        qs,stop=trace(q0,d)
        print('TRACE',d,stop,len(qs),'k range',qs[:,2].min(),qs[:,2].max(),'end',info(qs[-1]),flush=True)
        branches.append(qs)
    qs=np.vstack((branches[0][::-1],branches[1][1:]))
    np.save(OUT/'branch.npy',qs)
    savecsv('period2_branch.csv',[dict(index=i,**info(q)) for i,q in enumerate(qs)])
    savecsv('preliminary_checks.csv',checks)
    (OUT/'summary.json').write_text(json.dumps(dict(checks=checks,fold=info(fold.x),fold_q=fold.x.tolist(),
       python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),indent=2))

if __name__=='__main__':main()
