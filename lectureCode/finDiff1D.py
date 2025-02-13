import numpy as np
import matplotlib.pyplot as plt

N=10 # intervals
h=1/N
n=N-1 # internal nodes
x = np.linspace(0,1,N+1)[1:-1]
print(x)

xx = np.linspace(0,1,1000)[1:-1]
f = np.pi**2*np.sin(np.pi*x)
m = 2/h**2*np.ones(n)
l = -1/h**2*np.ones(n-1)
A = np.diag(m,0) + np.diag(l,1) + np.diag(l,-1)
u = np.linalg.solve(A,f)
uex = np.sin(np.pi*x)
uexx=np.sin(np.pi*xx)
plt.plot(x,u,'ro')
plt.plot(xx,uexx,'b')
plt.show()
err = np.linalg.norm(uex-u,np.inf)
print(err)

allerr=np.array([[10, 0.008265416966228845],[20, 0.002058706764534346], [40,
0.0005142004781495402],[80, 0.00012852038354038697],
[160, 3.212823780796015e-05]])
print(allerr[:,1])
print(allerr.shape)
for i in range(allerr.shape[0]-1):
    print(allerr[i,1]/allerr[i+1,1])
