"""
ADR equations in 1D with FD

@author: Alessandro
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg
import scipy.linalg


def f(x):
#    return 2.+0.*x
    return 0+0*x

def u_ex(x,mu,beta,gamma):
#    return (1-x)*x
    return (np.exp(beta*x/mu)-1)/(np.exp(beta/mu)-1)

def solver_centred(x,f,mu,beta,uL,uR,h,N):
    # Assembly
    ff = f(x)
    Ad =-(mu/h**2)*sp.diags([1., -2., 1.], [-1, 0, 1], shape=[N+1, N+1], format = 'csr')
    Ac =beta/(2*h)*sp.diags([-1., 0., 1.], [-1, 0, 1], shape=[N+1, N+1], format = 'csr')
    A = Ad + Ac
    # boundary conditions
    A[0, 1] = 0.
    A[-1, -2] = 0.
    A[0, 0] = 1./h**2
    A[-1, -1] = 1./h**2
    ff[0] = uL/h**2
    ff[-1] = uR/h**2
    # Solving
    u = sp.linalg.spsolve(A, ff)
    return u

def solver_upwind(x,f,mu,beta,uL,uR,h,N):
    # Assembly
    ff = f(x)
    Ad =-(mu/h**2)*sp.diags([1., -2., 1.], [-1, 0, 1], shape=[N+1, N+1], format = 'csr')
    Ac = beta/(h)*sp.diags([0, -1., 1.], [-1, 0, 1], shape=[N+1, N+1], format = 'csr')
    A = Ad + Ac
    # boundary conditions
    A[0, 1] = 0.
    A[-1, -2] = 0. 
    A[0, 0] = 1./h**2
    A[-1, -1] = 1./h**2
    ff[0] = uL/h**2
    ff[-1] = uR/h**2
    print(A)  
    # Solving    
    u = sp.linalg.spsolve(A, ff)
    return u


# Problem definition
a = 0.
b = 1.
mu = 1. 
beta = -500.0
gamma = 0.0
uL = u_ex(a,mu,beta,gamma)
uR = u_ex(b,mu,beta,gamma)

# Numerical discretization
h = 0.1
N = int((b-a)/h)
x = np.linspace(a,b,N+1)
# 
# For verification
uu = u_ex(x, mu, beta, gamma)

Pe = np.abs(beta)*h/(2*mu)
print(Pe)

#u = solver_centred(x,f,mu*(1+Pe),beta,uL,uR,h,N)
u = solver_upwind(x,f,mu,beta,uL,uR,h,N)

"""
#
# Assembly
ff = f(x)
Ad =-(mu/h**2)*sp.diags([1., -2., 1.], [1, 0, -1], shape=[N+1, N+1], format = 'csr')
Ac =beta/(2*h)*sp.diags([1., 0., -1.], [1, 0, -1], shape=[N+1, N+1], format = 'csr')
A = Ad + Ac
# boundary conditions
A[0, 1] = 0.
A[-1, -2] = 0.
A[0, 0] = 1./h**2
A[-1, -1] = 1./h**2
ff[0] = uL/h**2
ff[-1] = uR/h**2

#print(A)

# Linear system solving
u = sp.linalg.spsolve(A, ff)
"""

# Postprocessing
err = uu-u
err_inf = np.max(np.abs(err))
err_2 = np.sqrt(err.T.dot(err))
err_l2 = np.sqrt(h*err.T.dot(err))
plt.plot(x,u,'x')
xx=np.linspace(0,1,10000)
uu = u_ex(xx,mu,beta,gamma)
plt.plot(xx,uu,'r')
plt.show()

print(err_inf)
print(err_2)
print(err_l2)
plt.plot(err)
plt.show()

