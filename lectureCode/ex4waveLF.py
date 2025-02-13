"""
MATH572 Wave Leap Frog
@author: Alessandro
"""
import sys, time
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg
import scipy.linalg

"""
===== ==========================================================
Name  Description
===== ==========================================================
Nx    The total number of mesh cells; mesh points are numbered
      from 0 to Nx.
T     The stop time for the simulation.
I     Initial condition (Python function of x).
a     Variable coefficient (constant).
L     Length of the domain ([0,L]).
x     Mesh points in space.
t     Mesh points in time.
n     Index counter in time.
u     Unknown at current/new time level.
u_n   u at the previous time level.
dx    Constant mesh spacing in x.
dt    Constant mesh spacing in t.
===== ==========================================================
"""

def visualize(x, t, u):
    plt.plot(x, u, 'r')
    plt.plot(x,u_ex(x, t),'bx')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Solution at time t=%g' % t)
#    umin = 1.2*u.min()
#    umax = -umin
    plt.axis([x[0], x[-1], -1.1, 1.1])
    plt.show()


def error_loc(u,u_ex):
    diff = np.abs(u - u_ex)
    diffnobc = diff[1:-2]
    ei = np.max(diff)
    print(ei)
    el2 = np.sqrt(h*diffnobc.T.dot(diffnobc)  + h/2*(diff[0]**2+diff[-1]**2)) 
    print(el2)
    return ei, el2 


def IC(x):
# initial conditions
#    return 0.5*np.where(x<=0.3,0.,1.)+0.5*np.where(x>=0.7,0.,1.)
    return np.sin(np.pi*x)

def ICP(x):
# initial conditions on the velocity
#    return 0.5*np.where(x<=0.3,0.,1.)+0.5*np.where(x>=0.7,0.,1.)
    return 0.0*x



def BC(xl,xr,t):
    bcl = 0.0
    bcr = 0.0
    return bcl, bcr

def f(x,t):
    return 0.0


def u_ex(x,t): 
    return np.sin(np.pi*x)*np.cos(np.pi*t)


# DEFINITION OF THE PROBLEM
T = 2. # final time
xl = 0. # leftmost point
xr = 1. # righmost point

gamma = 1
gamma2 = gamma**2 # stiffness of the string


# discretization parameters
dt = 0.05
h = 0.1
#r = dt/h**2

Nx = int(round(np.abs(xr-xl)/h))
Nt = int(round(T/dt))

# Space mesh & time mesh
x = np.linspace(xl,xr,Nx+1)
t = np.linspace(0,T,Nt+1)

# Error computing data structures
eil = np.zeros(Nt)
el2l = np.zeros(Nt)

cfl2 = gamma2*(dt/h)**2

# MATRIX ASSEMBLY (for time independent coefficients)
u   = np.zeros(Nx+1)
u_n = np.zeros(Nx+1)
u_nm1 = np.zeros(Nx+1)

# Data structures for the linear system
A = np.zeros((Nx+1, Nx+1))
b = np.zeros(Nx+1)

A =cfl2*sp.diags([1., -2., 1.], [-1, 0, 1], shape=[Nx+1, Nx+1], format = 'csr')
#Ar = sigma*sp.identity(Nx+1, format = 'csr')

#LeftM = sp.identity(Nx+1, format = 'csr') + dt*th*A
RightM = A + 2*sp.identity(Nx+1, format = 'csr')

# boundary conditions
aux_bc = 1.

# Initial conditions
tc = 0 # current time
u_nm1 = IC(x)
visualize(x,tc,u_nm1)

tc += dt
u_n = 0.5*RightM*u_nm1 + dt*ICP(x)
u_n[0], u_n[-1] = BC(xl, xr,t)
visualize(x,tc,u_n)



# TIME LOOP
for n in range(1,Nt):
    tc += dt
    print("Computing at time", tc)
    # right hand side
    b = dt**2*f(x,tc-dt) + RightM*u_n - u_nm1
    b[0], b[-1] = BC(xl, xr,tc)
    u = b 
#
    visualize(x,tc,u)
    u_nm1 = u_n
    u_n = u
    # error computing   
#    eil[n], el2l[n] = error_loc(u,u_ex(x,tc))    

"""    
ei_tot = eil.max()
eL2 = np.sqrt(dt*(el2l[1:-2].T.dot(el2l[1:-2])) + dt/2*(el2l[0]**2+el2l[-1]**2))
print("Error infinity", ei_tot)
print("Error L2(L2)", eL2)
"""