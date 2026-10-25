"""Independent algebraic enumeration, validation, and endpoint diagnostics."""
from algebra_check import *
import sympy as sp

def exact_slope_poly(ks):
    k=sp.Rational(ks);rx=sp.Rational('3.8318741');ry=sp.Rational('3.7389149')
    a=(1-k)*rx;b=k*ry;c=k*rx;e=(1-k)*ry;D=a*e-b*c;t=sp.Symbol('t')
    mx=(1+(e-b*t)/D)/2;my=(1+(a-c/t)/D)/2
    sx=mx-mx*mx-(e*mx-b*my)/D;sy=my-my*my-(-c*mx+a*my)/D
    return sp.Poly(sp.cancel(t*t*(t*t*sx-sy)),t)

def main():
    # Source-equivalence verification without invoking the source GUI.
    ns={'__name__':'__check__'}
    src=(Path(__file__).parent/'source/source/log2inter.py').read_text().split('plt.figure()')[0]
    exec(compile(src,'source/log2inter.py','exec'),ns)
    original=np.column_stack((ns['x'],ns['y']));ours=[DEFAULT.copy()]
    for m in range(1,400):ours.append(F(ours[-1],0 if m<250 else .1))
    assert np.array_equal(original,np.array(ours))
    z=np.array([.321,.745]);k=.145;h=1e-6
    fd=np.column_stack([(F(z+np.eye(2)[i]*h,k)-F(z-np.eye(2)[i]*h,k))/(2*h) for i in range(2)])
    jd_error=float(np.max(abs(fd-J(z,k))));assert jd_error<1e-8
    q=np.r_[z,k];fdg=np.column_stack([(G((q+np.eye(3)[i]*h)[:2],(q+np.eye(3)[i]*h)[2])-G((q-np.eye(3)[i]*h)[:2],(q-np.eye(3)[i]*h)[2]))/(2*h) for i in range(3)])
    d_error=float(np.max(abs(fdg-DG(z,k))));assert d_error<1e-8
    cycles=[];fixed=[];counts=[]
    for i,k in enumerate(np.linspace(.14,.15,2001)):
        cs,rs,co=p2cycles(k)
        for j,r in enumerate(cs):cycles.append(dict(sample=i,cycle=j,**r))
        fs=fixedpoints(k);fixed.extend(fs)
        counts.append(dict(k=float(k),physical_p2=sum(r['physical'] for r in cs),stable_p2=sum(r['physical'] and r['rho']<1 for r in cs),
            physical_p1=sum(r['physical'] for r in fs),stable_p1=sum(r['physical'] and r['rho']<1 for r in fs)))
    savecsv('all_period2.csv',cycles);savecsv('all_fixedpoints.csv',fixed);savecsv('root_counts.csv',counts)
    # Exact-rational polynomial real-root isolation: independent of root initial guesses.
    isolated=[]
    for ks in ['.14','.145','.1452','.1453','.1453239','.145324','.14533','.146','.15']:
        pol=exact_slope_poly(ks)
        intervals=pol.intervals(eps=sp.Rational(1,10**35))
        roots=[(lo+hi)/2 for (lo,hi),mult in intervals for _ in range(mult)]
        k=float(ks);cs,rs,co=p2cycles(k)
        isolated.append(dict(k=ks,degree=pol.degree(),real_roots=len(roots),
            intervals=[dict(lower=str(lo),upper=str(hi),multiplicity=m) for (lo,hi),m in intervals],
            real_cycles=len(cs),physical_cycles=sum(r['physical'] for r in cs)))
    (OUT/'exact_isolation.json').write_text(json.dumps(isolated,indent=2))
    qs=np.load(OUT/'branch.npy');maxres=max(info(q)['residual'] for q in qs)
    endpoints=[]
    for end in [qs[0],qs[-1]]:
        sol=root(lambda u:G(u,0),end[:2],tol=1e-12)
        endpoints.append(info(np.r_[sol.x,0]))
    (OUT/'component_endpoints.json').write_text(json.dumps(endpoints,indent=2))
    # Refine the left period-doubling of the same two-cycle.
    q0=qs[np.argmin(abs(qs[:,2]-.10207))]
    pd=root(lambda q:np.r_[G(q[:2],q[2]),np.linalg.det(J(F(q[:2],q[2]),q[2])@J(q[:2],q[2])+np.eye(2))],q0,tol=1e-12)
    assert max(abs(pd.fun))<1e-10
    # Directly compare a half-step arclength run through the right fold.
    q0=qs[np.argmin(np.linalg.norm(qs-np.array([.88633941075,.56098433223,.1452]),axis=1))]
    qh,stop=trace(q0,1,ds=.0001,n=300)
    # Re-refine from the largest-k half-step point.
    fh=root(lambda q:np.r_[G(q[:2],q[2]),np.linalg.det(DG(q[:2],q[2])[:,:2])],qh[qh[:,2].argmax()],tol=1e-12)
    summary=json.loads((OUT/'summary.json').read_text())
    validation=dict(source_trace_bitwise_equal=True,jacobian_error=jd_error,DG_error=d_error,
       max_branch_residual=maxres,branch_points=len(qs),half_step_fold_k=float(fh.x[2]),
       half_step_fold_difference=float(abs(fh.x[2]-summary['fold']['k'])),left_flip=info(pd.x),
       physical_p1_counts=sorted(set(c['physical_p1'] for c in counts)),stable_p1_counts=sorted(set(c['stable_p1'] for c in counts)))
    # First exit from a 0.02-radius neighborhood of the former orbit, starting at the fold.
    q=np.array(summary['fold_q']);A=q[:2];B=F(A,q[2]);exitrows=[]
    for delta in [1e-5,1e-6,1e-7,1e-8]:
        z=A.copy();kk=q[2]+delta
        for n in range(1,1000000):
            z=F(z,kk)
            if min(np.linalg.norm(z-A),np.linalg.norm(z-B))>.02:break
        exitrows.append(dict(delta_k=delta,k=float(kk),first_exit_steps=n,radius=.02,steps_times_sqrt_delta=float(n*np.sqrt(delta))))
    savecsv('ghost_exit_times.csv',exitrows)
    validation['ghost_exit_times']=exitrows
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2));print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
