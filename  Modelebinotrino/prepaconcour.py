import numpy as np
import matplotlib.pyplot as plt
'''n=500
y=np.random.rand(n)
lam=2
x=-np.log(1-y)*1/lam
print(np.mean(x), np.mean(x**2))

plt.hist(x, bins=50,density=True)
x_vals = np.linspace(0, np.max(x), 1000)
plt.plot(x_vals, lam* np.exp(-lam* x_vals), 'r-', lw=2)
plt.show()
n=1000
x=np.random.rand(n)
y=np.random.rand(n)
nombrepoint=np.sum(x**2+y**2<=1)
pi=4*nombrepoint/n
print(pi)
#REDUCTION DE LA ARIANCE
x=np.random.rand(n//2,2)
x2=1-x
z=np.vstack([x,x2])
fraction=np.sum(z[:,0]**2+z[:,1]**2 <=1)/n
pi=4*fraction
print(pi)
import numpy as np
from scipy import stats
#TARIFICATION D'UNE OPTION
s0,sigma,t,r,k=100,0.15,1,0.05,100
#SOLUTION ANALYIQUE
def function1(s0,sigma,t,r,k):
    d1=(np.log(s0/k)+(r+sigma**2/2)*t)/(sigma*np.sqrt(t))
    d2=d1-sigma*np.sqrt(t)
    call=s0*stats.norm.cdf(d1)-k*np.exp(-r*t)*stats.norm.cdf(d2)
    put = k * np.exp(-r * t) * stats.norm.cdf(-d2)-s0 * stats.norm.cdf(- d1)
    print(f"le prix du call est :{call:.3} et le prix du put est :{put:.3}")
    #return call,put
#MONTE CARLO
n=1000
def fonction2(s0,sigma,t,r,k,n) :
    #EDS: ds=st*mu*dt+sigma*dbt*st
    dt=t/252
    N=252*t
    z=np.random.normal(0,1,(n,N-1))
    s=np.empty((n,N))
    s[:,0]=s0
    for i in range(1,N):
        s[:,i]=s[:,i-1]*np.exp((r-1/2*sigma**2)*dt+sigma*np.sqrt(dt)*z[:,i-1])
    payoff=np.maximum(0,s[:,-1]-k)
    call=np.exp(-r*t)*np.mean(payoff)
    for i in range(1,N):
        s[:,i]=s[:,i-1]*np.exp((r-1/2*sigma**2)*dt+sigma*np.sqrt(dt)*z[:,i-1])
    payoff=np.maximum(0,k-s[:,-1])
    put=np.exp(-r*t)*np.mean(payoff)
    print(f"le prix du call est :{call:.2f}")
    print(f"le prix du call est :{put:.2f}")
fonction2(s0,sigma,t,r,k,n)
x=[10 ,6 ,13,23,0,7]
y=np.sort(x)
print(y[-1])
'''