"""80-decimal fold verification and local saddle-node nondegeneracy checks."""
from research import *
import mpmath as mp

def main():
    mp.mp.dps=80
    rx=mp.mpf('3.8318741');ry=mp.mpf('3.7389149')
    def f(z,k):
        a=rx*z[0]*(1-z[0]);b=ry*z[1]*(1-z[1])
        return mp.matrix([a+k*(b-a),b+k*(a-b)])
    def j(z,k):
        a=rx*(1-2*z[0]);b=ry*(1-2*z[1])
        return mp.matrix([[(1-k)*a,k*b],[k*a,(1-k)*b]])
    def equations(x,y,k):
        z=mp.matrix([x,y]);b=f(z,k);g=f(b,k)-z;M=j(b,k)*j(z,k)
        return g[0],g[1],mp.det(M-mp.eye(2))
    q=mp.findroot(equations,(mp.mpf('.88683'),mp.mpf('.56642'),mp.mpf('.145324')),tol=mp.mpf('1e-70'))
    z=q[:2];k=q[2];b=f(z,k);M=j(b,k)*j(z,k)
    eig=mp.eig(M,left=False,right=False)
    # Compute normalized double-precision normal-form coefficients for G=F^2-I.
    qn=np.array([float(x) for x in q]);zz=qn[:2];kk=qn[2];bb=F(zz,kk)
    Mn=J(bb,kk)@J(zz,kk)
    ev,V=np.linalg.eig(Mn);v=V[:,np.argmin(abs(ev-1))].real;v/=np.linalg.norm(v)
    if v[1]<0:v=-v
    ew,W=np.linalg.eig(Mn.T);w=W[:,np.argmin(abs(ew-1))].real;w/=w@v
    L=np.array([[(1-kk)*RX,kk*RY],[kk*RX,(1-kk)*RY]])
    H=lambda u:L@(-2*u*u)
    H2=H(J(zz,kk)@v)+J(bb,kk)@H(v)
    alpha=w@DG(zz,kk)[:,2];beta=.5*w@H2
    fp=mp.findroot(lambda x,y:tuple(f(mp.matrix([x,y]),k)-mp.matrix([x,y])),(.7386,.7330))
    fpeig=mp.eig(j(fp,k),left=False,right=False)
    result=dict(dps=80,fold=dict(k=str(k),A=[str(x) for x in z],B=[str(x) for x in b],
      gap=str(mp.norm(mp.matrix(z)-b)),multipliers=[str(x) for x in eig],
      max_augmented_residual=str(max(abs(x) for x in equations(*q)))),
      fixed_at_fold=dict(z=[str(x) for x in fp],eigenvalues=[str(x) for x in fpeig]),
      normal_form=dict(right_vector=v.tolist(),left_vector=w.tolist(),alpha=float(alpha),beta=float(beta),
         dk_per_s_squared=float(-beta/alpha),other_multiplier=float(ev[np.argmax(abs(ev-1))].real)))
    (OUT/'precision.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
