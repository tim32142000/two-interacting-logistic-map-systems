"""Targeted follow-up and independent checks; run after run_experiment.py."""
from run_experiment import *
from scipy.optimize import root

def main():
    # Numerical equivalence with original code, without invoking its GUI.
    ns={'__name__':'__check__'}
    src=(Path(__file__).parent/'source/log2inter.py').read_text().split('plt.figure()')[0]
    exec(compile(src,'source/log2inter.py','exec'),ns)
    original=np.column_stack((ns['x'],ns['y']))
    assert np.array_equal(original,trajectory(.1)), 'Original trace mismatch'
    # Analytic Jacobian versus central finite difference.
    z=np.array([.321,.745]);h=1e-6
    fd=np.column_stack([(advance(z+np.eye(2)[i]*h,.1)-advance(z-np.eye(2)[i]*h,.1))/(2*h) for i in range(2)])
    assert np.max(abs(fd-jac(z,.1)))<1e-8
    a=np.load(OUT/'long_tail.npz');t=a['tail'];seeds=a['seeds']
    check=[]
    for i,k in enumerate(a['ks']):
        ps={str(tol):period(t[:,i],tol=tol).tolist() for tol in [1e-6,1e-8,1e-10]}
        p=int(np.lcm(*ps['1e-08'][0])) if all(ps['1e-08'][0]) else 0
        dist=None
        if p:
            base=t[-p:,i,0];dist=max(float(np.max(np.min(np.linalg.norm(base[:,None,:]-t[-p:,i,s][None,:,:],axis=-1),axis=1))) for s in range(len(seeds)))
        check.append(dict(k=float(k),periods_by_tolerance=ps,orbit_distance_across_seeds=dist))
    (OUT/'validation.json').write_text(json.dumps(dict(original_trace_bitwise_equal=True,jacobian_max_error=float(np.max(abs(fd-jac(z,.1)))),checks=check),indent=2))
    # Continue the period-2 branch through its loss of stability as k decreases.
    z=t[-1,4,0];branch=[]
    for k in np.arange(.12,.09999,-.00005):
        sol=root(lambda u:advance(advance(u,k),k)-u,z,tol=1e-11);z=sol.x
        M=jac(advance(z,k),k)@jac(z,k);eig=np.linalg.eigvals(M)
        branch.append(dict(k=float(k),residual=float(np.max(abs(advance(advance(z,k),k)-z))),eig_min_real=float(np.min(eig.real)),eig_max_real=float(np.max(eig.real)),rho=float(max(abs(eig)))))
    writecsv('period2_branch.csv',branch)
    print('period2 -1 crossings',[(branch[i-1]['k'],r['k']) for i,r in enumerate(branch) if i and (r['eig_min_real']+1)*(branch[i-1]['eig_min_real']+1)<0])
    print('orbit distance',[(x['k'],x['orbit_distance_across_seeds']) for x in check])
    # Near-zero coupling is unsampled by the coarse grid. Include an explicit 5x3 phase grid.
    tr=trajectory(0,300);phase=np.array([[tr[249+i,0],tr[249+j,1]] for i in range(3) for j in range(5)])
    # These are treated as prehistory initial conditions; the same 249 uncoupled steps follow.
    ks=[0,1e-7,1e-6,1e-5,1e-4,.0002,.0003,.0004,.0735,.074,.0745,.075,.0795,.08,.102,.1025,.145,.1455]
    rr,_=batch(ks,np.vstack((DEFAULT,phase)),burn=20000,measure=16384)
    writecsv('phase_and_boundary_checks.csv',rr)
    for k in ks:
        b=[r for r in rr if r['k']==k];print(k,sorted(set((r['px'],r['py']) for r in b)),min(r['lambda_max'] for r in b),max(r['lambda_max'] for r in b))
if __name__=='__main__':main()
