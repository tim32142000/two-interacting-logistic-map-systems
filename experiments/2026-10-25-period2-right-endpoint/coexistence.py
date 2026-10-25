"""Follow up the seed-dependent window discovered left of the endpoint."""
from simulate import *
from scipy.optimize import root

def main():
    seeds=np.vstack((DEFAULT,np.random.default_rng(20261018).uniform(.001,.999,(15,2))))
    ks=np.array([.1432,.143305,.1434,.1435,.1436])
    t,p,ly,bl,z=batch(ks,seeds,burn=500000,measure=131072,tailn=1024)
    np.savez_compressed(OUT/'coexistence_long.npz',ks=ks,seeds=seeds,tail=t,period=p,lyapunov=ly,blocks=bl)
    checks=[]
    for i,k in enumerate(ks):
        for j in range(16):
            pp=int(p[i,j]);r=dict(k=float(k),seed=j,period=pp,lyapunov=float(ly[i,j]))
            if pp:
                def advancep(u):
                    for _ in range(pp):u=F(u,k)
                    return u
                s=root(lambda u:advancep(u)-u,t[-1,i,j],tol=1e-12)
                u=s.x.copy();M=np.eye(2);orb=[]
                for _ in range(pp):orb.append(u.copy());M=J(u,k)@M;u=F(u,k)
                eig=np.linalg.eigvals(M)
                r.update(residual=float(max(abs(advancep(s.x)-s.x))),rho=float(max(abs(eig))),
                   multiplier1=str(eig[0]),multiplier2=str(eig[1]),root_minimum_period=pp)
                for div in range(1,pp):
                    if pp%div==0 and np.max(abs(np.array(orb)-np.roll(orb,div,axis=0)))<1e-8:
                        r['root_minimum_period']=div;break
            checks.append(r)
    (OUT/'coexistence_checks.json').write_text(json.dumps(checks,indent=2))
    # Direct-onset basin sample at a single parameter with several stable cycles.
    grid=np.linspace(.005,.995,51);xx,yy=np.meshgrid(grid,grid);direct=np.column_stack((xx.ravel(),yy.ravel()))
    bt,bp,bl,bb,bz=batch([.143305],direct,burn=200000,measure=32768,prehistory=0,tailn=1024)
    np.savez_compressed(OUT/'coexistence_basin.npz',seeds=direct,period=bp,lyapunov=bl,final=bz)
    summary=dict(k=.143305,direct_counts={str(q):int(sum(bp[0]==q)) for q in np.unique(bp)},
       long_counts=[dict(k=float(k),counts={str(q):int(sum(p[i]==q)) for q in np.unique(p[i])}) for i,k in enumerate(ks)])
    (OUT/'coexistence_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
