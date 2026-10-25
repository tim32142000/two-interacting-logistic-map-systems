"""Enumerate all real exact-period-two cycles via a degree-six slope polynomial.

For A=m+d, B=m-d and L=[[a,b],[c,e]], F(z)=L(z-z*z):
L diag(1-2m)d=-d. Set t=d_y/d_x, D=det(L).
mx=(1+(e-b*t)/D)/2, my=(1+(a-c/t)/D)/2.
Sx=mx-mx**2-(e*mx-b*my)/D = dx**2.
Sy=my-my**2-(-c*mx+a*my)/D = dy**2.
t**2*Sx-Sy=0. Multiplication by t**2 gives degree six.
For 0<k<.5, an exact two-cycle cannot have dx=0 or dy=0.
All real roots t with Sx>0 give a real cycle, up to phase exchange.
"""
from research import *
from numpy.polynomial import Polynomial as P

def p2cycles(k):
    a=(1-k)*RX;b=k*RY;c=k*RX;e=(1-k)*RY;D=a*e-b*c
    M0=(1+e/D)/2;M1=-b/(2*D);N0=(1+a/D)/2;Nm=-c/(2*D)
    sx={-1:b*Nm/D,0:(1-e/D)*M0-M0*M0+b*N0/D,
        1:(1-e/D)*M1-2*M0*M1,2:-M1*M1}
    sy={-2:-Nm*Nm,-1:(1-a/D)*Nm-2*N0*Nm,
        0:(1-a/D)*N0-N0*N0+c*M0/D,1:c*M1/D}
    co=np.zeros(7)
    for i,v in sx.items():co[i+4]+=v
    for i,v in sy.items():co[i+2]-=v
    roots=P(co).roots();cycles=[]
    for t in roots:
        if abs(t.imag)>1e-7:continue
        t=t.real
        if abs(t)<1e-12:continue
        m=np.array([M0+M1*t,N0+Nm/t]);S=sum(v*t**i for i,v in sx.items())
        if S<=1e-12:continue
        d=np.sqrt(S)*np.array([1,t]);A=m+d
        s=root(lambda z:G(z,k),A,jac=lambda z:DG(z,k)[:,:2],tol=1e-12)
        if max(abs(s.fun))>1e-9:raise ValueError('bad root')
        r=info(np.r_[s.x,k]);r['slope']=float(t)
        r['physical']=bool(min(r['ax'],r['ay'],r['bx'],r['by'])>=0 and max(r['ax'],r['ay'],r['bx'],r['by'])<=1)
        cycles.append(r)
    return cycles,roots,co

def fixedpoints(k):
    # y = u*x+v*x^2 follows directly from L^-1 z = z-z*z.
    Dk=1-2*k
    u=((1-k)-Dk*RX)/k;v=Dk*RX/k
    X=P([0,1]);Y=P([0,u,v]);pol=RY*(Y-Y*Y)-((1-k)*Y-k*X)/Dk
    out=[]
    for xx in pol.roots():
        if abs(xx.imag)>1e-8:continue
        x=float(xx.real);y=float(Y(x));z=np.array([x,y])
        eig=np.linalg.eigvals(J(z,k));eig=eig[np.argsort(eig.real)]
        out.append(dict(k=float(k),x=x,y=y,physical=bool(min(x,y)>=-1e-12 and max(x,y)<=1+1e-12),
           eig1_real=float(eig[0].real),eig1_imag=float(eig[0].imag),eig2_real=float(eig[1].real),eig2_imag=float(eig[1].imag),
           rho=float(max(abs(eig))),residual=float(max(abs(F(z,k)-z)))))
    return out

if __name__=='__main__':
    for k in [.14,.145,.1452,.1453,.1453239,.145324,.14533,.146,.15]:
        cs,rs,co=p2cycles(k)
        print('CYCLES',k,[(r['ax'],r['ay'],r['gap'],r['rho'],r['physical']) for r in cs])
        print('FIXED',fixedpoints(k))
