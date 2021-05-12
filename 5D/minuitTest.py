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
dispIterInfo = False    # if set to True, prints info about each iteration (LL, time, Norm). False: just a loading bar.
nIter=0 # counts iterations
##### IMPORTS #####
# Imports necessary modules
from math import pi as PI       # for pi, built-in Python
import time                     # for timing, built-in Python
import numpy as np              # requires download, e.g. "$ pip3 install numpy". For scipy and efficient funcs.
from numpy import sin, cos, arccos
from scipy import optimize      # requires download, e.g. "$ pip3 install scipy". For optimization of LL.
import numba                    # requires download, e.g. "$ pip3 install numba". For efficient execution.
from numba import jit
import iminuit
from iminuit import Minuit
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
    return 1/n * s #* (2**3 * (2*PI)**2)    # MC-integral: average value of function, technically multiplied by area (2**3 * (2*PI)**2)
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
    if dispIterInfo:
        print("--------")
    else:
        global nIter
        nIter = nIter + 1
        print("-"*nIter + f" {nIter}", end="\r")
    if normSep==True:
        normalization = MCintegral(*par, normAngs, pdf)
        t2 = time.time()
        if dispIterInfo: print(f"One normalization done... took {t2 - t1:.5f} seconds. \t   Norm:  {normalization}")
    else:
        normalization = 1   # nothing happens. log(1) = 0.
    # Calculate LL-sum and add normalization
    r = iterativeLL(par,var, pdf) + len(var)*np.log(normalization) # normalize after; -log(W_i/norm) = -log(W_i) + log(norm) 
    t3 = time.time()
    if dispIterInfo:
        print(f"One LL-sum done. Took {t3 - t2:.5f} seconds. \t\t\t neg  LL: {r}")    # takes a long time but not AS long as normalization.
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
    global xi_set
    xi_set = [ list(map(float,i.split()))[numberedInput:] for i in open(angleDistributionData_filename).readlines() ]    # list (of lists)
    # Iterate thru lines of datafile, for each line, split it into list of number contents, map the content of that list from
    # str -> float, convert map object -> list, skip first if it is numbered input, all in list comprehension.
    xi_set = np.asarray(xi_set) # converts to numpy.array. Much faster than numba typed list.
    xi_set = xi_set[dataFrom:dataTo]
    print(f"First row: {xi_set[0]}")
    print(f"Number of measurement points: {len(xi_set)}")
    print("Finished reading signal data.")
    t2 = time.time()
    print(f"--- {(t2 - start_time):.3f} seconds ---")

    # Read normalization data
    print("Reading normalization data...")
    global normAngs
    normAngs = [ list(map(float,i.split()))[numberedInput:] for i in open(normalizationData_filename).readlines() ]    # list (of lists) 
    normAngs = np.asarray(normAngs) # needed for numba. Fixed datatype.
    print(f"First row: {normAngs[0]}")
    print(f"Number of points for normalization: {len(normAngs)}")
    print(f"--- {(time.time() - t2):.3f} seconds for normalization data ---")
    print(f"------ {(time.time() - start_time):.3f} seconds total for all input data ------ \n")
    ########## END READ DATA ##########
    '''
    ########## OPTIMIZE: ##########
    # input parameter values 1, 2, 3, 4 = 0.461, 0.740, 0.754, -0.754 (from Patrik Adlarson)
    # Parameters to optimize for: alpha, dPhi, alpha1, alpha2
    #initial_guess = [0.3, PI/8, 0.55, -0.55]
    initial_guess = [0.46, 0.75, 0.75, -0.75]
    print(f"Initial guess: {initial_guess}")
    bnds = ((-1,1),(-PI,PI),(-1,1),(-1,1))   # bounds on variables.
    # Options for the optimizer. Can also fix method. Read more on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    ftol = 10**-10
    #ops = {'disp': None, 'maxcor': 10, 'ftol': ftol, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None}
    
    print("OPTIMIZING...")
    # scipy existing minimizing function. 
    res = optimize.minimize(negLL, initial_guess, (xi_set[dataFrom:dataTo], WDoubleTag, True, normAngs[0:]), bounds=bnds, tol=ftol)#, method='L-BFGS-B')#, options=ops)
    ########## END OPTIMIZE ##########

    ########## PRESENT RESULTS: ##########
    print(f"\n ------ FINISHED OPTIMIZATION. SCIPY.OPTIMIZE.MINIMIZE RESULTS: ------")
    solvedFor = ('alpha','dPhi','alpha1','alpha2') # what did you solve for?
    print(res)  # scipy default result structure
    if res['success'] == True:
        print(f"CONVERGED SUCCESSFULLY, using tolerance {ftol}")
    else:
        print("!!! OPTIMIZATION WAS NOT SUCCESSFUL !!!")
    print(f"\n------ TOOK A TOTAL OF {time.time() - start_time:.3f} SECONDS ------")
    print(f"{'Solved for:':>20}\t{solvedFor}\n{'Bounded by:':>20}\t{bnds}")
    print(f"{'Initial guess:':>20}\t{initial_guess}")
    print(f"{'Expected result:':>20}\t{(0.460, 0.785398, 0.75, -0.75)}") # input to generate data, according to Patrik
    print(f"{'Actual result:':>20}\t{res['x']}")
    for i in range(len(solvedFor)):
        print(f"{f'Result for {solvedFor[i]}:':>20}{res['x'][i]:>+20.8f}")
    print("")
    print(res['hess_inv'])
    print(res.x)
    #print(np.diag(res['hess_inv']))
    hess = (res['hess_inv']).todense()
    print(np.diag(hess))
    print("Inverse Hessian:")
    print(hess)
    for i in range(len(solvedFor)):
        print(f"{f'Variance {solvedFor[i]}:':>20}{hess[i][i]:>20.8f}")
    print((hess[0][0],hess[1][1],hess[2][2],hess[3][3]))
    ########## END PRESENT RESULTS ##########
    print(sum(totalTime)/len(totalTime))
    print(f'Using {dataTo - dataFrom} points.')

    tmp_i = np.zeros(len(res.x))
    for i in range(len(res.x)):
        tmp_i[i] = 1.0
        hess_inv_i = res.hess_inv(tmp_i)[i]
        uncertainty_i = np.sqrt(max(1, abs(res.fun)) * ftol * hess_inv_i)       # NOTE: weird, it depends on res.fun?
        tmp_i[i] = 0.0
        print('x^{0} = {1:12.4e} ± {2:.1e}'.format(i, res.x[i], uncertainty_i))
'''
########## END MAIN ##########


def negLLMinuit(par):
    normSep = True
    global xi_set
    global normAngs
    var = xi_set
    pdf = WDoubleTag
    normAngs = normAngs
    t1 = time.time()
    if dispIterInfo:
        print("--------")
    else:
        global nIter
        nIter = nIter + 1
        print("-"*nIter + f" {nIter}", end="\r")
    if normSep==True:
        normalization = MCintegral(*par, normAngs, pdf)
        t2 = time.time()
        if dispIterInfo: print(f"One normalization done... took {t2 - t1:.5f} seconds. \t   Norm:  {normalization}")
    else:
        normalization = 1   # nothing happens. log(1) = 0.
    # Calculate LL-sum and add normalization
    r = iterativeLL(par,var, pdf) + len(var)*np.log(normalization) # normalize after; -log(W_i/norm) = -log(W_i) + log(norm) 
    t3 = time.time()
    if dispIterInfo:
        print(f"One LL-sum done. Took {t3 - t2:.5f} seconds. \t\t\t neg LL: {r}")    # takes a long time but not AS long as normalization.
        print(f"Total time for one iteration was {t3 - t1:.5f} seconds.")
    totalTime.append(t3 - t1) 
    return r

if __name__ == "__main__":  # doesn't run if imported.
    t0 = time.time()
    main()
    global xi_set
    print(xi_set[1])
    m = Minuit(negLLMinuit, (0.46, 0.74, 0.75, -0.75))
    m.errordef = Minuit.LIKELIHOOD  # important
    m.migrad()
    print(m.valid)
    print(m.values)
    m.hesse()   # run covariance estimator
    print(m.errors)
    print(m.covariance)
    inp = ''
    print(time.time() - t0)
    while inp != 'q':   # won't get checked here.
        inp = input("command: ")
        if inp == 'q':
            break
        eval(inp)