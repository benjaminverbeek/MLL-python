# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Program to make 5D fit (Double-tag BESIII)            #
# Benjamin Verbeek, 2021-04-30                          #
# Fully functional. Theory definitions specified in     #
# appropriate places.                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

print("--- RUNNING MAX LOG LIKELIHOOD FIT ---")
dataFrom = 0
dataTo = 100000 + 1
totalTime = []
##### IMPORTS #####
# Imports necessary modules
from math import pi as PI       # for pi, built-in Python
import time                     # for timing, built-in Python
import numpy as np              # requires download, e.g. "$ pip3 install numpy". For scipy and efficient funcs.
from numpy import sin, cos, arccos
from scipy import optimize      # requires download, e.g. "$ pip3 install scipy". For optimization of LL.
import numba                    # requires download, e.g. "$ pip3 install numba". For efficient execution.
from numba import jit
##### END IMPORTS #####

# Set some parameters for the fit.
angleDistributionData_filename = "mcsig100k_JPsi_LLbar.dat"  # specify path if not in same folder.
normalizationData_filename = "mcphsp1000k_JPsi_LLbar.dat"
numberedInput = True        # Is one column of the input data just numbering? Specify that here.

numberedInput = int(numberedInput)  # True - 1, False - 0. This is how many colums to skip in indata files. Can be specified manually further down.
###### THEORY ######    (can be swapped out for whatever )
# Theory from http://uu.diva-portal.org/smash/get/diva2:1306373/FULLTEXT01.pdf , same as ROOT-implementation.
# here alpha = eta = alpha_psi
# C_n,m take theta (no subindex), delta-phi and alpha as input. Only using nonzero ones.
# C defined as per equation 23.
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
# a-funcs should have alpha1/2, th1/2, ph1/2 index depending on lambda (1) or lambda-bar (2). 
# Only defining used ones. Defined as per equation 50.
@jit(nopython=True)
def a00(alpha, th, ph): return 1

@jit(nopython=True)
def a10(alpha, th, ph): return alpha * cos(ph) * sin(th)

@jit(nopython=True)
def a20(alpha, th, ph): return alpha * sin(th) * sin(ph)

@jit(nopython=True)
def a30(alpha, th, ph): return alpha * cos(th)
##########
# Defined as per equation 54.
@jit(nopython=True)
def WDoubleTag(alpha, dPhi, alpha1, alpha2, th, th1, ph1, th2, ph2):
    return C00(alpha, dPhi, th)/2 * a00(alpha1, th1, ph1) * a00(alpha2, th2, ph2) + \
    C02(alpha, dPhi, th)/2 * a00(alpha1, th1, ph1) * a20(alpha2, th2, ph2) + \
    C11(alpha, dPhi, th)/2 * a10(alpha1, th1, ph1) * a10(alpha2, th2, ph2) + \
    C13(alpha, dPhi, th)/2 * a10(alpha1, th1, ph1) * a30(alpha2, th2, ph2) + \
    C20(alpha, dPhi, th)/2 * a20(alpha1, th1, ph1) * a00(alpha2, th2, ph2) + \
    C22(alpha, dPhi, th)/2 * a20(alpha1, th1, ph1) * a20(alpha2, th2, ph2) + \
    C31(alpha, dPhi, th)/2 * a30(alpha1, th1, ph1) * a10(alpha2, th2, ph2) + \
    C33(alpha, dPhi, th)/2 * a30(alpha1, th1, ph1) * a30(alpha2, th2, ph2)

##### END THEORY #####


'''###
pars = [0.46, PI/4, 0.75, -0.75]
alpha, dPhi, alpha1, alpha2 = [0.46, PI/4, 0.75, -0.75]
angs = [0.0321418,        2.31065,        2.80985,        2.02057,       -1.68605]
th,th1,ph1,th2,ph2 = angs
r1 = WDoubleTag(*pars, th,th1,ph1,th2,ph2) # corrected order.
print(f"W-func evaluated at {pars}, {angs}, yielding:")
print(f"v1:    {r1}    ")
print(a20(alpha2, th2, ph2))
'''###

##### MC INTEGRATOR #####
# MC-integrator for normalization factors
@jit(nopython=True) # numba decorator. Significantly improves performance (~factor 100)
def MCintegral(alpha,dPhi,alpha1,alpha2, uniformAngles, distributionFunc):
    """Monte Carlo integration for normalization, for given parameters, a set of normalization angles and a distributionFunc."""
    s = 0.0   # sum
    n = 0.0   # number of points
    for xi in uniformAngles: # xi is a 5D list here
        th,th1,ph1,th2,ph2 = xi
        s += distributionFunc(alpha,dPhi,alpha1,alpha2 , th,th1,ph1,th2,ph2) # evaluate W at a bunch of random points and sum.
        n += 1  # count number of points. Could also use len(uniformAngles)
    return 1/n * s  #* (2*PI)**5    # MC-integral: average value of function, technically multiplied by area (2**3 * (2*PI)**2)
                    # this does not affect results however, since it just becomes adding a constant to the LL-function.
##### END MC INTEGRATOR #####

##### NEG LOG LIKELIHOOD FUNCTION #####
# Help function, split out so numba can optimize it.
@jit(nopython=True)
def iterativeLL(par, var, pdf):
    s = 0  # sum
    alpha,dPhi,alpha1,alpha2 = par
    for v in var: # iterate over samples
        th,th1,ph1,th2,ph2 = v  # unpack angles. Cannot use *v for numba compatibility.
        s -= np.log(pdf(alpha,dPhi,alpha1,alpha2 , th,th1,ph1,th2,ph2)) # log-sum of pdf gives LL. Negative so we minimize.
    return s

# Generalized LL-func.: send in a pdf too, and let par be n-dim, dataset var X be m-dim.
# Note that this is unjitted: It would barely improve performance and would reduce readability/generalizability by a lot.
def negLL(par, var, pdf, normSep=False, normAngs=[]):
    '''Minimize this function for decay parameters to find max of Log-Likelihood for distribution. \n
    par : decay parameters to maximize [list], N-dim \n
    var : dataset of variables (xi) [list of lists] M-dim (NOTE: the inner lists represent observed points, i.e. 
    every variable is not a separate list, but rather part of a set of variables (i.e. a point)). 
    E.g.:
    >>> var = [ [a0, b0], [a1, b1], [a2, b2], ... ] and not \n
    >>> var = [ [a0, a1, a2, ...], [b0, b1, b2, ...] ] 
    where a, b are different variables for the pdf. \n
    pdf : must take arguments pdf(p1,p2, ..., pN, v1, v2, ..., vM)
    Lists (and lists of lists) should be of type typed_list from numba List() to run. \n
    Optional parameters: \n
    normSep : Set to true if separate normalization should be done. Then enter a distribution instead of a PDF. Normalized by MC-integration.
    Must then also enter normalization angles normAngs (a lsit that can be entered into distribution function).'''

    t1 = time.time()
    print("--------")
    if normSep==True:
        normalization = MCintegral(*par, normAngs, pdf)
        t2 = time.time()
        print(f"One normalization done... took {t2 - t1:.5f} seconds. \t   Norm:  {normalization}")
    else:
        normalization = 1   # nothing happens. log(1) = 0.
    # Calculate LL-sum and add normalization
    r = iterativeLL(par,var, pdf) + len(var)*np.log(normalization) # normalize after; -log(W_i/norm) = -log(W_i) + log(norm) 
    t3 = time.time()
    print(f"One LL-sum done. Took {t3 - t2:.5f} seconds. \t\t\t neg LL: {r}")    # takes a long time but not AS long as normalization.
    print(f"Total time for one iteration was {t3 - t1:.5f} seconds.")
    totalTime.append(t3 - t1) 
    return r
##### END NEG LOG LIKELIHOOD FUNCTION #####
############# END HELP FUNCTIONS #############

########## MAIN: ########## 
def main():
    start_time = time.time()
    print("Reading input data... \t (this might take a minute)")
    ########## READ DATA: ##########
    # Read angle distribution data. Becomes python list of numba-lists
    xi_set = [ list(map(float,i.split()))[numberedInput:] for i in open(angleDistributionData_filename).readlines() ]    # list (of lists)
    # Iterate thru lines of datafile, for each line, split it into list of number contents, map the content of that list from
    # str -> float, convert map object -> list, skip first if it is numbered input, all in list comprehension.
    xi_set = np.asarray(xi_set) # converts to numpy.array. Much faster than numba typed list.
    print(f"First row: {xi_set[0]}")
    print(f"Number of measurement points: {len(xi_set)}")
    print("Finished reading signal data.")
    t2 = time.time()
    print(f"--- {(t2 - start_time):.3f} seconds ---")

    # Read normalization data
    print("Reading normalization data...")
    normAngs = [ list(map(float,i.split()))[numberedInput:] for i in open(normalizationData_filename).readlines() ]    # list (of lists) 
    normAngs = np.asarray(normAngs) # needed for numba. Fixed datatype.
    print(f"First row: {normAngs[0]}")
    print(f"Number of points for normalization: {len(normAngs)}")
    print(f"--- {(time.time() - t2):.3f} seconds for normalization data ---")
    print(f"------ {(time.time() - start_time):.3f} seconds total for all input data ------ \n")
    ########## END READ DATA ##########

    ########## OPTIMIZE: ##########
    # input parameter values 1, 2, 3, 4 = 0.461, 0.740, 0.754, -0.754 (from Patrik Adlarson)
    # Parameters to optimize for: alpha, dPhi, alpha1, alpha2
    #initial_guess = [0.3, PI/8, 0.55, -0.55]
    initial_guess = [0.40, 0.5, 0.70, -0.70]
    print(f"Initial guess: {initial_guess}")
    bnds = ((-1,1),(-PI,PI),(-1,1),(-1,1))   # bounds on variables.
    # Options for the optimizer. Can also fix method. Read more on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    #ops = {'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None}
    tolerance = 10**-6
    print("OPTIMIZING...")

    from matplotlib import pyplot as plt
    vals = []
    ranges = np.arange(0.45, 0.47, 0.001)
    for i in ranges:
        vals.append(negLL(np.array([i, 0.74, 0.75, -0.75]), xi_set, WDoubleTag, True, normAngs))
    print(vals)
    plt.title("Likelihood vs alpha for dPhi = 0.74 and 0.75") 
    plt.xlabel("alpha") 
    plt.ylabel("negLikelihood")
    plt.plot(ranges, vals, label = "dPhi = 0.74")

    vals2 = []
    ranges2 = np.arange(0.45, 0.47, 0.001)
    for i in ranges2:
        vals2.append(negLL(np.array([i, 0.75, 0.75, -0.75]), xi_set, WDoubleTag, True, normAngs))
    print(vals)
    plt.plot(ranges2, vals2, label = "dPhi = 0.75")
    plt.legend()
    plt.show()




########## END MAIN ##########

if __name__ == "__main__":  # doesn't run if imported.
    main()