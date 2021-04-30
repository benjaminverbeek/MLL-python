##### IMPORTS #####
# Imports necessary modules
from math import pi as PI       # for pi, comes with Python
import time                     # for timing, comes with Python
import numpy as np              # requires download, e.g. "$ pip3 install numpy". For scipy and efficient funcs.
from numpy import sin, cos, arccos
from scipy import optimize      # requires download, e.g. "$ pip3 install scipy". For optimization of LL.
import numba                    # requires download, e.g. "$ pip3 install numba". For efficient execution.
from numba import jit           # numba
##### END IMPORTS #####

######### ALT THEORY: ##########
# here alpha = eta = alpha_psi
@jit(nopython=True)
def C00(alpha, dPhi, th): return  2*(1 + alpha * cos(th)**2)

@jit(nopython=True)
def C02(alpha, dPhi, th): return  2*(1-alpha**2)**0.5 * sin(th)*cos(th)*sin(dPhi)

@jit(nopython=True)
def C11(alpha, dPhi, th): return  2*sin(th)**2

@jit(nopython=True)
def C13(alpha, dPhi, th): return  2*(1-alpha**2)**0.5 * sin(th)*cos(th)*cos(dPhi)

@jit(nopython=True)
def C20(alpha, dPhi, th): return  -1*C02(alpha,dPhi,th)

@jit(nopython=True)
def C22(alpha, dPhi, th): return  alpha*C11(alpha,dPhi,th)

@jit(nopython=True)
def C31(alpha, dPhi, th): return  -1*C13(alpha,dPhi,th)

@jit(nopython=True)
def C33(alpha, dPhi, th): return -2*(alpha + cos(th)**2)
##########
# a-funcs should have alpha1/2, th1/2, ph1/2 depending on lambda or lambda-bar
@jit(nopython=True)
def a00(alpha, th, ph): return 1

@jit(nopython=True)
def a10(alpha, th, ph): return alpha * cos(ph) * sin(th)

@jit(nopython=True)            # NOTE: här blir det fel.
def a20(alpha, th, ph): 
    print(alpha)
    return alpha * sin(th) * sin(ph)

@jit(nopython=True)
def a30(alpha, th, ph): return alpha * cos(th)
##########
@jit(nopython=True)
def altWDoubleTag(alpha, dPhi, alpha1, alpha2, th, th1, ph1, th2, ph2):
    return C00(alpha, dPhi, th)/2 * a00(alpha1, th1, ph1) * a00(alpha2, th2, ph2) + \
    C02(alpha, dPhi, th)/2 * a00(alpha1, th1, ph1) * a20(alpha2, th2, ph2) + \
    C11(alpha, dPhi, th)/2 * a10(alpha1, th1, ph1) * a10(alpha2, th2, ph2) + \
    C13(alpha, dPhi, th)/2 * a10(alpha1, th1, ph1) * a30(alpha2, th2, ph2) + \
    C20(alpha, dPhi, th)/2 * a20(alpha1, th1, ph1) * a00(alpha2, th2, ph2) + \
    C22(alpha, dPhi, th)/2 * a20(alpha1, th1, ph1) * a20(alpha2, th2, ph2) + \
    C31(alpha, dPhi, th)/2 * a30(alpha1, th1, ph1) * a10(alpha2, th2, ph2) + \
    C33(alpha, dPhi, th)/2 * a30(alpha1, th1, ph1) * a30(alpha2, th2, ph2)

'''
##### END THEORY #####
 # still dont get same values...
pars = [0.46, 0.785398, 0.75, -0.75]
#pars = [0,0,0,0]
#angs = [0,0,0,0,0]
angs = [0.0321418,        2.31065,        2.80985,        2.02057,       -1.68605]
#angs = [ 1.94843,       0.916242,       -2.08242,        1.53144,      -0.153751]
r1 = altWDoubleTag(*pars, *angs)
print(f"W-func evaluated at {pars}, {angs}, yields:")
print(f"W(...):    {r1}")
'''