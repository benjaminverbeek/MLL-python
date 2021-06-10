# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Program to make 5D fit (Double-tag BESIII)            #
# Benjamin Verbeek                                      #
# Now using iminuit for fit, gives proper variance.     #
# Theory definitions specified in appropriate places.   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# NOTE: For modification of formalism, simply change the function
# WDoubleTag to any desired probability distribution. Note, if 
# the dimensions of the input data is to be made, according changes
# must also be made when the input is unpacked in "itrativeLL" and
# "MCintegral". "Note"'s mark these positions.
# Of course, also the input-data must be changed.

# NOTE: Be careful with possible negative inputs for log. No such check exists.
# The program will fail.

##### IMPORTS #####
# Imports necessary modules
from math import pi as PI       # for pi, built-in Python
import time                     # for timing, built-in Python
import numpy as np              # requires download, e.g. "$ pip3 install numpy". For scipy and efficient funcs.
from numpy import sin, cos
from scipy import optimize      # requires download, e.g. "$ pip3 install scipy". For optimization of LL.
from numba import jit           # requires download, e.g. "$ pip3 install numba". For efficient execution.
from iminuit import Minuit      # requires download, e.g. "$ pip3 install iminuit". For optimization of LL.
import os
##### END IMPORTS #####


##### FIT PARAMETERS FOR ANALYSIS #####
# Set some parameters for the fit.
dataFrom, dataTo = 0, 0     # ranges of data used. "From" is inclusive, "To" is exclusive (standard)
normFrom, normTo = 0, 0     # Set all to 0 to use all data  (note, this does not work by default in Python). Can also set to None.
dispIterInfo = False    # if set to True, prints info about each iteration (LL, time, Norm). False: just a loading bar.
use_scipy_for_initial_guess = False  # set to True if the initial guess is bad.
signalData_filename = "mcsig100k_JPsi_LLbar.dat"    # specify path if not in same folder. (must use "/")
normData_filename = "mcphsp1000k_JPsi_LLbar.dat"    # Ensure these files can be accessed.
directoryPathForSearch = "C:/Users/Admin/"      # Will search from here if files not found.
numberedInput = True        # Is one column of the input data just numbering? Specify that here.
#initGuess = (0.461, 0.740, 0.754, -0.754)  # expected results for LLbar-data
initGuess = (0.4, 0.7, 0.7, -0.7)      # Initial guess
bnds = ((-1,1),(-PI,PI),(-1,1),(-1,1))  # bounds on parameters (needed for scipy)
ftol = 10**-3                           # tolerance for scipy
parNames = ('alpha', 'dPhi', 'alpha_1', 'alpha_2')  # Parameter names for Minuit

# Do not change:
numberedInput = int(numberedInput)  # True - 1, False - 0. This is how many colums to skip in indata files. Can be specified manually further down.
nIter=0 # counts iterations
totalTime = [] # for timing per iteration. For analysis.

# Searches for file.
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

# Checks if files can be found, else searches for them.  
print()
print(f'{" LOCATING DATA-FILES ":-^60}')
try:
    o = open(signalData_filename)
    o.close()
    o = open(normData_filename)
    o.close()
except FileNotFoundError:
    print("Could not find files from current directory.")
    tSearch = time.time()
    print(f"Searching for in-data files in specified directory:\t '{directoryPathForSearch}'")
    signalData_filename = find(signalData_filename.split('/')[-1], directoryPathForSearch)   # search only for name
    print("Found:\t", signalData_filename)
    # First try search in same dir as first file.
    try:
        print("Looking in same directory:\t",'/'.join(signalData_filename.split('\\')[:-1]))    # NOTE: Could be '/' if using Linux or other OS
        normData_filename_TRY = find(normData_filename.split('/')[-1], '/'.join(signalData_filename.split('\\')[:-1]) )
    except:
        print("Error searching in same dir. Falling back on normal search.")
        normData_filename_TRY = None
    if normData_filename_TRY is None:
        normData_filename = find(normData_filename.split('/')[-1], directoryPathForSearch)
    else:
        print("Found in same directory.")
        normData_filename = normData_filename_TRY

    print("Found:\t", normData_filename)
    print(f"Took {time.time()-tSearch:.3f} s to find files.")
    if time.time()-tSearch > 10:
        print("Try specifying the path more to reduce search-time.")
########## END PREAMBLE ##########

###### THEORY ######    (can be swapped out for whatever)
# Theory from http://uu.diva-portal.org/smash/get/diva2:1306373/FULLTEXT01.pdf , same as ROOT-implementation.
# here alpha = eta = alpha_psi
# C_n,m take theta (no subindex), delta-phi and alpha as input. Only using nonzero ones.
# C defined as per equation 23.
@jit(nopython=True)
def C00(alpha, dPhi, th): return  2*(1 + alpha * cos(th)**2)

@jit(nopython=True)
def C02(alpha, dPhi, th): return 2*(1-alpha**2)**0.5 * sin(th)*cos(th)*sin(dPhi)

@jit(nopython=True)
def C11(alpha, dPhi, th): return 2*sin(th)**2

@jit(nopython=True)
def C13(alpha, dPhi, th): return 2*(1-alpha**2)**0.5 * sin(th)*cos(th)*cos(dPhi)

@jit(nopython=True)
def C20(alpha, dPhi, th): return -1*C02(alpha,dPhi,th)

@jit(nopython=True)
def C22(alpha, dPhi, th): return alpha*C11(alpha,dPhi,th)

@jit(nopython=True)
def C31(alpha, dPhi, th): return -1*C13(alpha,dPhi,th)

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
        th,th1,ph1,th2,ph2 = xi     # NOTE: Change if changing dimension of input
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
    alpha,dPhi,alpha1,alpha2 = par  # NOTE: Change if changing dimension of input
    for v in var: # iterate over samples
        th,th1,ph1,th2,ph2 = v  # unpack angles. Cannot use *v for Numba compatibility. # NOTE: Change if changing dimension of input
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
    pdf = WDoubleTag    # NOTE: Rename these appropriately if changing names.
    normAngs = normAngs
    
    t1 = time.time()    # time iterations
    
    if dispIterInfo:
        print("--------")
    else:
        global nIter    # WARNING: Global variable
        shapes = ['-', '~', '=', '+', '*', ':']
        maxWidth = 50 + 1
        nIter = nIter + 1
        print(shapes[(nIter//maxWidth)%len(shapes)]*(nIter%maxWidth) + f"> {nIter} ", end="\r")   # Loading bar.

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
    print()
    print(f'{" RUNNING MAX LOG LIKELIHOOD FIT ":-^60}')
    start_time = time.time()
    print("Reading input data... \t (this might take a minute)")
    ########## READ DATA: ##########
    # Read angle distribution data, save in numpy-array (needed for numba compatibility & speed)
    global xi_set   # Global variable for access in negLL
    sigData = open(signalData_filename) # open file
    xi_set = [ list(map(float,i.split()))[numberedInput:] for i in sigData.readlines() ]    # list (of lists)
    sigData.close()     # close open file
    # Iterate thru lines of datafile, for each line, split it into list of number contents, map the content of that list from
    # str -> float, convert map object -> list, skip first if it is numbered input, all in list comprehension.
    xi_set = np.asarray(xi_set) # converts to numpy.array. Much faster than numba typed list.
    xi_set = xi_set[dataFrom:dataTo or None]    # for analysis
    #print(f"First row: {xi_set[0]}")    # sanity-check data
    print(f"Size of signal set: {len(xi_set)}")
    print("Finished reading signal data.")
    t2 = time.time()
    print(f'{f" {(t2 - start_time):.3f} seconds ":-^60}')

    # Read normalization data
    print("Reading normalization data...")
    global normAngs     # Global variable for access in negLL
    normData = open(normData_filename)  # open file
    normAngs = [ list(map(float,i.split()))[numberedInput:] for i in normData.readlines() ]    # list (of lists) 
    normData.close()    # close open file
    normAngs = np.asarray(normAngs) # needed for numba. Fixed datatype.
    normAngs = normAngs[normFrom:normTo or None]    # for analysis
    #print(f"First row: {normAngs[0]}")      # sanity-check data
    print(f"Number of points for normalization: {len(normAngs)}")
    print(f'{f" {(time.time() - t2):.3f} seconds ":-^60}')
    print(f'{f" {(time.time() - start_time):.3f} seconds total for all input data ":-^60}')
    print()
    ########## END READ DATA ##########

    ########## OPTIMIZE WITH MINUIT (and maybe scipy) ##########
    if use_scipy_for_initial_guess == True:
        print(f"\nOptimizing for starting guess with scipy optimize.minimize\ninitial guess: {initGuess}")
        # scipy existing minimizing function. 
        res = optimize.minimize(negLLMinuit, initGuess, bounds=bnds, tol=ftol)  # Scipy optimization for starting guess for minuit
        print()
        print(f"Optimizing with minuit\ninitial guess: {res.x}")
        m = Minuit(negLLMinuit, res.x, name=parNames)  # define minuit function and initial guess
    else:
        print(f"Optimizing with Minuit.migrad\ninitial guess: {initGuess}")
        m = Minuit(negLLMinuit, initGuess, name=parNames) # define minuit function and initial guess
    
    m.errordef = Minuit.LIKELIHOOD      # important
    m.migrad()  # run minuit optimziation

    print()
    print(f"Estimating errors with Minuit.hesse")
    m.hesse()   # run covariance estimator
    #m.minos() # alternative error estimator. Gave same results but slower during testing.
    print() # offset /r from loading bar.
    print() # blank line
    print(f'{" RESULTS ":-^60}')
    print(f"Valid optimization: {m.valid}")  # was the optimization successful?
    print("Parameter estimation ± standard deviation:")
    for var, val, err in zip(parNames, m.values, m.errors):
        print(f"{var:>10}: {val:>15.10f} ± {err:.10f}")
    print()
    print("Covariance matrix:")
    print(m.covariance)
    print(f'Covariance matrix accurate: {m.accurate}')

    # For easy analysis. Prints values compactly for export to e.g. excel.
    '''
    print(f"\nFor analysis:          Data: {dataFrom}-{dataTo-1}       Norm: {normFrom}-{normTo-1}")
    print(f"Initial guess: {initGuess}")
    if use_scipy_for_initial_guess:
        print(f"Minuit initial guess (from scipy): {res.x}")
    resVals = []
    resErrs = []
    for val, err in zip(m.values, m.errors):
        resVals.append(str(val))
        resErrs.append(str(err))
    #print(', '.join(resVals))   # easy to copy-paste to excel.
    #print(', '.join(resErrs))
    print(', '.join(resVals + resErrs))
    print(dataFrom-dataTo)
    print(sum(totalTime)/len(totalTime))
    print("Done.")
    '''
    ########## END OPTIMIZE WITH MINUIT ##########
########## END MAIN ##########

if __name__ == "__main__":  # doesn't run if imported.
    t0 = time.time()
    main()
    print(f'{f" TOOK A TOTAL OF {time.time() - t0:.3f} SECONDS ":-^60}', end='\n')