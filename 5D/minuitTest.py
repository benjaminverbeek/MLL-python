# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Program to make 5D fit (Double-tag BESIII)            #
# Benjamin Verbeek, 2021-04-30                          #
# Now using iminuit for fit, gives proper variance.     #
# Theory definitions specified in appropriate places.   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# NOTE: Seems to require good starting guess. Use scipy for that?

print("--- RUNNING MAX LOG LIKELIHOOD FIT ---")
##### FIT PARAMETERS FOR ANALYSIS #####
dataFrom = 0
dataTo = 100_000 + 1
normFrom = 0
normTo = 1000_000 + 1
totalTime = []
dispIterInfo = False    # if set to True, prints info about each iteration (LL, time, Norm). False: just a loading bar.

##### IMPORTS #####
# Imports necessary modules
from math import pi as PI       # for pi, built-in Python
import time                     # for timing, built-in Python
import numpy as np              # requires download, e.g. "$ pip3 install numpy". For scipy and efficient funcs.
from numpy import sin, cos, arccos
from numba import jit           # requires download, e.g. "$ pip3 install numba". For efficient execution.
from iminuit import Minuit      # requires download, e.g. "$ pip3 install iminuit". For optimization of LL.
##### END IMPORTS #####

# Set some parameters for the fit.
angleDistributionData_filename = "mcsig100k_JPsi_LLbar.dat"  # specify path if not in same folder.
normalizationData_filename = "mcphsp1000k_JPsi_LLbar.dat"
numberedInput = True        # Is one column of the input data just numbering? Specify that here.
###
numberedInput = int(numberedInput)  # True - 1, False - 0. This is how many colums to skip in indata files. Can be specified manually further down.
nIter=0 # counts iterations
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

# Note that this is unjitted: It would barely improve performance and would reduce readability/generalizability by a lot.
def negLLMinuit(par): 
    """A minuit modified negLL function. Only takes in parameters to optimize.
    It is vital that global variables xi_set and normAngs are defined containing data on the correct format."""
    normSep = True
    global xi_set       # bit ugly. Requiered for Minuit fit (function to optimize only dependent on parameters to optimize for)
    global normAngs     # WARNING: Global vairable
    var = xi_set
    pdf = WDoubleTag
    normAngs = normAngs
    
    t1 = time.time()    # time iterations
    
    if dispIterInfo:
        print("--------")
    else:
        global nIter    # WARNING: Global variable
        nIter = nIter + 1
        if nIter < 100:
            print("-"*nIter + f"> {nIter}", end="\r")   # Loading bar.
        else:
            print("*" + f"> {nIter}", end="\r")   # Loading bar.

    if normSep==True:   # Structured this way, a finished PDF can be used too. Just set normSep=False
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
##### END NEG LOG LIKELIHOOD FUNCTION #####
############# END HELP FUNCTIONS #############

########## MAIN: ########## 
def main():
    start_time = time.time()
    print("Reading input data... \t (this might take a minute)")
    ########## READ DATA: ##########
    # Read angle distribution data, save in numpy-array (needed for numba compatibility & speed)
    global xi_set   # Global variable for access in negLL
    xi_set = [ list(map(float,i.split()))[numberedInput:] for i in open(angleDistributionData_filename).readlines() ]    # list (of lists)
    # Iterate thru lines of datafile, for each line, split it into list of number contents, map the content of that list from
    # str -> float, convert map object -> list, skip first if it is numbered input, all in list comprehension.
    xi_set = np.asarray(xi_set) # converts to numpy.array. Much faster than numba typed list.
    xi_set = xi_set[dataFrom:dataTo]    # for analysis
    print(f"First row: {xi_set[0]}")    # sanity-check data
    print(f"Size of signal set: {len(xi_set)}")
    print("Finished reading signal data.")
    t2 = time.time()
    print(f"--- {(t2 - start_time):.3f} seconds ---")

    # Read normalization data
    print("Reading normalization data...")
    global normAngs     # Global variable for access in negLL
    normAngs = [ list(map(float,i.split()))[numberedInput:] for i in open(normalizationData_filename).readlines() ]    # list (of lists) 
    normAngs = np.asarray(normAngs) # needed for numba. Fixed datatype.
    normAngs = normAngs[normFrom:normTo]    # for analysis
    print(f"First row: {normAngs[0]}")      # sanity-check data
    print(f"Number of points for normalization: {len(normAngs)}")
    print(f'{f" {(time.time() - t2):.3f} seconds for normalization data ":-^60}')
    print(f'{f" {(time.time() - start_time):.3f} seconds total for all input data ":-^60}')
    ########## END READ DATA ##########

    ########## OPTIMIZE WITH MINUIT ##########
    #initGuess = (0.461, 0.740, 0.754, -0.754)  # expected results
    initGuess = (0.4, 0.6, 0.6, -0.6)
    m = Minuit(negLLMinuit, initGuess)  # define minuit function and initial guess
    m.errordef = Minuit.LIKELIHOOD      # important
    print(f"OPTIMIZING...          initial guess: {initGuess}")
    m.migrad()  # run minuit optimziation
    #print()    # to see how much is needed for optimizer vs error estimator
    m.hesse()   # run covariance estimator
    print() # offset /r from loading bar.
    varNames = ('alpha', 'dPhi', 'alpha_1', 'alpha_2')
    print(f"Valid optimization: {m.valid}")  # was the optimization successful?
    print("Parameter estimation:")
    for var, val, err in zip(varNames, m.values, m.errors):
        print(f"{var:>10}: {val:>15.10f} ± {err:.10f}")
    print("Covariance matrix:")
    print(m.covariance)
    print(f'Covariance matrix accurate: {m.accurate}')

    print(f"\nFor analysis:          Data: {dataFrom}-{dataTo-1}       Norm: {normFrom}-{normTo-1}")
    print(f"Initial guess: {initGuess}")
    resVals = []
    resErrs = []
    for val, err in zip(m.values, m.errors):
        resVals.append(str(val))
        resErrs.append(str(err))
    print(', '.join(resVals))   # easy to copy-paste to excel.
    print(', '.join(resErrs))
    ########## END OPTIMIZE WITH MINUIT ##########
########## END MAIN ##########

if __name__ == "__main__":  # doesn't run if imported.
    t0 = time.time()
    main()
    print(f"------ TOOK A TOTAL OF {time.time() - t0:.3f} SECONDS ------")