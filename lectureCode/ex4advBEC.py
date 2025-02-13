"""
MATH572 Hyperbolic Problems

@author: Alessandro
"""
import sys, time
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg
import scipy.linalg

"""
INHERITED FROM THE PARABOLIC FILE
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
    plt.plot(x,u_ex(x, t),'b')
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

def BC(inflow,t):
    bc = -np.sin(np.pi*t)
    return bc

def f(x,t):
    return 0.0

def fcl(beta,bp,dt):
# finds the foot of the characteristic line at the boundary
    return bp-dt*beta

def extract_interval(x,fcl):
# finds the interval the fcl belongs to
    interval = np.zeros(3)
    who = np.where(x<=fcl)
    aux = int(who[-1][-1])
    print(fcl,aux)
    if aux == 0:
        interval[0], interval[1], interval[2] = int(0),int(1),int(2)
    else:
        interval[0], interval[1], interval[2] = int(aux-1), int(aux), int(aux+1)   
    return interval

def interpol(x,interval,u,fcl):
    a = int(interval[0])
    b = int(interval[1])
    c = int(interval[2])
    print(a,b,c)
    value  = u[a]*(fcl-x[b])*(fcl-x[c])/((x[a]-x[b])*(x[a]-x[c]))
    value += u[b]*(fcl-x[a])*(fcl-x[c])/((x[b]-x[a])*(x[b]-x[c]))
    value += u[c]*(fcl-x[a])*(fcl-x[b])/((x[c]-x[a])*(x[c]-x[b]))
    return value

def u_ex(x,t): 
    beta = 1.
    xinflow = 0.
    tau = t - (x-xinflow)/beta
    return np.where(x-beta*t>xinflow,np.sin(np.pi*(x-beta*t)),-np.sin(np.pi*tau))


# DEFINITION OF THE PROBLEM
T = 2. # final time
xl = 0. # leftmost point
xr = 1. # righmost point

beta = 1 # convection
sigma = 0 # reaction


# discretization parameters
dt = 0.05
h = 0.1
th = 1. # theta
#r = dt/h**2

Nx = int(round(np.abs(xr-xl)/h))
Nt = int(round(T/dt))

# Space mesh & time mesh
x = np.linspace(xl,xr,Nx+1)
t = np.linspace(0,T,Nt+1)

# Error computing data structures
eil = np.zeros(Nt)
el2l = np.zeros(Nt)


# MATRIX ASSEMBLY (for time independent coefficients)
u   = np.zeros(Nx+1)
u_n = np.zeros(Nx+1)


# Data structures for the linear system
A = np.zeros((Nx+1, Nx+1))
b = np.zeros(Nx+1)

#Ad =-(mu/h**2)*sp.diags([1., -2., 1.], [-1, 0, 1], shape=[Nx+1, Nx+1], format = 'csr')
Ac =beta/(2*h)*sp.diags([-1., 0., 1.], [-1, 0, 1], shape=[Nx+1, Nx+1], format = 'csr')
#Ar = sigma*sp.identity(Nx+1, format = 'csr')
A = Ac # + Ac + Ar

LeftM = sp.identity(Nx+1, format = 'csr') + dt*th*A
RightM = sp.identity(Nx+1, format = 'csr') - dt*(1-th)*A

# boundary conditions
aux_bc = 1.
LeftM[0, 1] = 0.
LeftM[-1, -2] = 0.
LeftM[0, 0] = aux_bc
LeftM[-1, -1] = aux_bc


# Initial conditions
tc = 0 # current time
u_n = IC(x)
visualize(x,tc,u_n)

if beta >0:
    inflow = xl 
    outflow = xr
    flin = 0 
    flout = -1
else:
    inflow = xr
    outflow = xl
    flin = -1 
    flout = 0     

foot = fcl(beta, outflow, dt)
interv = extract_interval(x, foot)

# TIME LOOP
for n in range(0,Nt):
    tc += dt
    print("Computing at time", tc)
    # right hand side
    b = dt*(th*f(x,tc)+(1-th)*f(x,tc-dt)) + RightM*u_n
    u_inflow = BC(inflow,tc)
    u_outflow = interpol(x, interv, u_n, foot)
    b[flin] = aux_bc*u_inflow
# extrapolation along the characteristics
    b[flout] = aux_bc*u_outflow    
    u = sp.linalg.spsolve(LeftM, b)
    visualize(x,tc,u)
    u_n = u
    # error computing   
#    eil[n], el2l[n] = error_loc(u,u_ex(x,tc))    

"""    
ei_tot = eil.max()
eL2 = np.sqrt(dt*(el2l[1:-2].T.dot(el2l[1:-2])) + dt/2*(el2l[0]**2+el2l[-1]**2))
print("Error infinity", ei_tot)
print("Error L2(L2)", eL2)
"""