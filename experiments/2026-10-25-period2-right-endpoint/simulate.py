"""Independent-seed bifurcation scan plus direct basin and sweep diagnostics."""
from research import *
from algebra_check import p2cycles

def batch(ks,seeds,burn=30000,measure=4096,prehistory=249,tailn=128):
    ks=np.asarray(ks);k=ks[:,None]
    z=np.broadcast_to(seeds,(len(ks),len(seeds),2)).copy()
    for _ in range(prehistory):z=F(z,0.)
    v=np.ones_like(z);v[...,1]=.37;v/=np.linalg.norm(v,axis=-1)[...,None]
    total=np.zeros(z.shape[:-1]);tail=[];blocks=[];sblock=np.zeros_like(total)
    for i in range(burn+measure):
        # Align the tangent direction during the final 4096 transient steps.
        if i>=max(0,burn-4096):
            a=RX*(1-2*z[...,0]);b=RY*(1-2*z[...,1])
            w0=(1-k)*a*v[...,0]+k*b*v[...,1]
            w1=k*a*v[...,0]+(1-k)*b*v[...,1]
            norm=np.hypot(w0,w1);norm=np.maximum(norm,1e-300)
            v[...,0]=w0/norm;v[...,1]=w1/norm
            if i>=burn:
                logs=np.log(norm);total+=logs;sblock+=logs
                if (i-burn+1)%1024==0:blocks.append(sblock/1024);sblock=np.zeros_like(total)
        z=F(z,k)
        if i>=burn+measure-tailn:tail.append(z.copy())
        if (i+1)%10000==0:print('iterations',i+1,'batch',len(ks),len(seeds),flush=True)
    t=np.array(tail);p=np.zeros(z.shape[:-1],int)
    for lag in range(1,65):
        good=np.max(np.abs(t[lag:]-t[:-lag]),axis=(0,3))<1e-8
        p[(p==0)&good]=lag
    return t,p,total/measure,np.array(blocks),z

def sweep(ks,seed,burn=12000,tailn=128):
    z=seed.copy();tails=[]
    for k in ks:
        for _ in range(burn):z=F(z,k)
        t=[]
        for _ in range(tailn):z=F(z,k);t.append(z.copy())
        tails.append(t)
    return np.array(tails)

def main():
    seeds=np.vstack((DEFAULT,np.random.default_rng(20261018).uniform(.001,.999,(15,2))))
    ks=np.linspace(.143,.147,2401)
    t,p,ly,blocks,z=batch(ks,seeds)
    np.savez_compressed(OUT/'scan.npz',ks=ks,seeds=seeds,tail=t,period=p,lyapunov=ly,blocks=blocks)
    savecsv('scan_summary.csv',[dict(k=float(k),period2_count=int(sum(p[i]==2)),
       period1_count=int(sum(p[i]==1)),unclassified_count=int(sum(p[i]==0)),
       original_period=int(p[i,0]),original_lyapunov=float(ly[i,0]),
       lyapunov_min=float(ly[i].min()),lyapunov_max=float(ly[i].max())) for i,k in enumerate(ks)])
    print('MAIN SCAN DONE',flush=True)
    # Direct coupling-onset seeds sample the interior, without uncoupled prehistory.
    grid=np.linspace(.005,.995,51);xx,yy=np.meshgrid(grid,grid)
    direct=np.column_stack((xx.ravel(),yy.ravel()))
    keys=np.array([.145,.1453,.1453239,.145324,.14533,.1455,.146,.15])
    bt,bp,bl,bb,bz=batch(keys,direct,burn=100000,measure=16384,prehistory=0)
    np.savez_compressed(OUT/'basin_checks.npz',ks=keys,seeds=direct,period=bp,lyapunov=bl,blocks=bb,final=bz)
    rows=[]
    for i,k in enumerate(keys):
        counts={str(p):int(sum(bp[i]==p)) for p in np.unique(bp[i])}
        rows.append(dict(k=float(k),seeds=len(direct),period_counts=json.dumps(counts),
           lyapunov_min=float(bl[i].min()),lyapunov_max=float(bl[i].max()),positive_count=int(sum(bl[i]>0))))
    savecsv('basin_summary.csv',rows);print('BASIN',rows,flush=True)
    # Long independent checks bracket the fold; 80k measurement after 200k transient.
    keys2=np.array([.1453,.1453239,.145324,.14533,.1455,.146,.15])
    lt,lp,ll,lb,lz=batch(keys2,seeds,burn=200000,measure=131072)
    np.savez_compressed(OUT/'long_checks.npz',ks=keys2,period=lp,lyapunov=ll,blocks=lb,tail=lt)
    savecsv('long_summary.csv',[dict(k=float(k),periods=json.dumps(np.unique(lp[i]).tolist()),
      lyapunov_min=float(ll[i].min()),lyapunov_max=float(ll[i].max()),
      block_min=float(lb[:,i].min()),block_max=float(lb[:,i].max())) for i,k in enumerate(keys2)])
    # Sequential quasi-static scans, with no straight lines drawn between attractors.
    sk=np.linspace(.1451,.1456,151)
    up=sweep(sk,np.array([.8863,.56]),burn=6000)
    down=sweep(sk[::-1],up[-1,-1],burn=6000)
    np.savez_compressed(OUT/'sweeps.npz',ks=sk,up=up,down=down[::-1])
    print('ALL SIMULATIONS DONE',flush=True)

if __name__=='__main__':main()
