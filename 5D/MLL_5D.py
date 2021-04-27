# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Program to make 5D fit                                #
# Benjamin Verbeek, 2021-04-27                          #
# Updated functions to work with numba, now executes    #
# very fast. Slow part is reading input and converting  #
# to numbda lists.                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

##### IMPORTS #####
# Imports necessary modules
from math import pi as PI       # for pi, comes with Python
import time                     # for timing, comes with Python
import numpy as np              # requires download, e.g. "$ pip3 install numpy". For scipy and efficient funcs.
from scipy import optimize      # requires download, e.g. "$ pip3 install scipy". For optimization of LL.
import numba                    # requires download, e.g. "$ pip3 install numba". For efficient execution.
from numba import jit           # numba
##### END IMPORTS #####

# Set some parameters for the fit.
angleDistributionData_filename = "mcsig100k_JPsi_LLbar.dat"  # specify path if not in same folder.
normalizationData_filename = "mcphsp1000k_JPsi_LLbar.dat"
numberedInput = True        # Is one column of the input data just numbering? Specify that here.

numberedInput = int(numberedInput)  # True - 1, False - 0. This is how many colums to skip in indata files. Can be specified manually further down.

###### THEORY ######
@jit(nopython=True)
def F0(th,th1,th2,ph1,ph2): return 1

@jit(nopython=True)
def F1(th,th1,th2,ph1,ph2): return (np.sin(th))**2 * np.sin(th1) * np.sin(th2) * \
            np.cos(ph1) * np.cos(ph2) + (np.cos(th))**2 * np.cos(th1) * np.cos(th2)

@jit(nopython=True)
def F2(th,th1,th2,ph1,ph2): return np.sin(th) * np.cos(th) * (np.sin(th1) * np.cos(th2) * \
            np.cos(ph1) + np.cos(th1) * np.sin(th2) * np.cos(ph2))

@jit(nopython=True)
def F3(th,th1,th2,ph1,ph2): return np.sin(th) * np.cos(th) * np.sin(th1) * np.sin(ph1)

@jit(nopython=True)
def F4(th,th1,th2,ph1,ph2): return np.sin(th) * np.cos(th) * np.sin(th2) * np.sin(ph2)

@jit(nopython=True)
def F5(th,th1,th2,ph1,ph2): return (np.cos(th))**2

@jit(nopython=True)
def F6(th,th1,th2,ph1,ph2): return np.cos(th1) * np.cos(th2) - (np.sin(th))**2 * \
            np.sin(th1) * np.sin(th2) * np.sin(ph1) * np.sin(ph2)

@jit(nopython=True) # Applies numba magic. nopython=True doesn't appear to make a difference but is apparently recommended.
def WDoubleTag(alpha,dPhi,alpha1,alpha2 , th,th1,th2,ph1,ph2):
    xi = (th,th1,th2,ph1,ph2)
    '''Normalize this to get the PDF to optimize. W is the function from theory (Fäldt, Kupsc)'''
    # https://arxiv.org/pdf/1702.07288.pdf
    return F0(*xi) + alpha*F5(*xi) \
        + alpha1*alpha2 * (F1(*xi) + ((1-alpha**2)**0.5) * np.cos(dPhi) * F2(*xi) + alpha*F6(*xi)) \
        + ((1-alpha**2)**0.5) * np.sin(dPhi) * (alpha1*F3(*xi) + alpha2*F4(*xi))    # W function

##### END THEORY #####

##### MC INTEGRATOR #####
# MC-integrator for normalization factors
@jit(nopython=True) # numba decorator. Significantly improves performance (~factor 100)
def MCintegral(alpha,dPhi,alpha1,alpha2, uniformAngles):
    """Monte Carlo integration for normalization, for given parameters and a set of normalization angles."""
    s = 0.0   # sum
    n = 0.0   # number of points
    for xi in uniformAngles: # xi is a 5D list
        th,th1,th2,ph1,ph2 = xi
        s += WDoubleTag(alpha,dPhi,alpha1,alpha2 , th,th1,th2,ph1,ph2) # evaluate W at a bunch of random points and sum.
        n += 1  # count number of points. Could also use len(uniformAngles)
    return 1/n * s * 2**5    # MC-integral: average value of function * area # NOTE: area is wrong for 5D but results should be same.
                            # (2*2, since cos has range [-1,1]). This area-constant does not affect results.
##### END MC INTEGRATOR #####

## HERE 

@jit(nopython=True)
def iterativeLL(par, var):  # a separate function so numba can optimize it.
    s = 0  # sum
    alpha,dPhi,alpha1,alpha2 = par    # TODO
    for v in var: # iterate over samples of xi
        th,th1,th2,ph1,ph2 = v # TODO
        s -= np.log(WDoubleTag(alpha,dPhi,alpha1,alpha2 , th,th1,th2,ph1,ph2)) # log-sum of pdf gives LL. Negative so we minimize.
    return s

# Generalized LL-func.: send in a pdf too, and let par be n-dim, dataset var X be m-dim.
def negLL(par, var, pdf, normalizeSeparately=False, normalizationAngles=[]):
    '''Minimize this function for decay parameters to find max of Log-Likelihood for distribution. \n
    par : decay parameters to maximize [list], N-dim \n
    var : dataset of variables (xi) [list of lists] M-dim (NOTE: the inner lists represent observed points, i.e. 
    every variable is not a separate list, but rather part of a set of variables (i.e. a point)). 
    E.g.:
    >>> var = [ [a0, b0], [a1, b1], [a2, b2], ... ] and not \n
    >>> var = [ [a0, a1, a2, ...], [b0, b1, b2, ...] ] 
    where a, b are different variables for the pdf. \n
    pdf : must take arguments pdf(p1,p2, ..., pN, v1, v2, ..., vM)
    Lists (and lists of lists) should be of type typed_list from numba List() to run.'''
    
    t1 = time.time()

    print("--------")
    if normalizeSeparately==True:
        normalization = MCintegral(*par, normalizationAngles)
        print(normalization)
        t2 = time.time()
        print(f"One normalization done... took {t2 - t1:.5f} seconds.")
    else:
        normalization = 1
    
    r = iterativeLL(par,var) + len(var)*np.log(normalization) # normalize after; -log(W_i/norm) = -log(W_i) + log(norm) 
    
    t3 = time.time()
    print(f"One set of parameters done. Took {t3 - t2:.5f} seconds.")    # takes a long time but not AS long as normalization.
    print(f"Total time for one run was {t3 - t1:.5f} seconds.")
    return r

############# END HELP FUNCTIONS #############

########## MAIN: ########## 
def main():

    start_time = time.time()
    print("Reading input data... \t (this might take a minute)")

    ########## READ DATA: ##########
    # Read angle distribution data. Becomes python list of numba-lists
    xi_set = [ list(map(float,i.split()))[numberedInput:] for i in open(angleDistributionData_filename).readlines() ]    # list (of lists)
    # Iterate thru lines of datafile, for each line, split it into list of number contents,
    # map the content of that list from str -> float, convert map object -> list, 
    # skip first if it is numbered input, all in list comprehension.
    xi_set = np.asarray(xi_set) # converts to numpy.array. Much faster than numba typed list.

    print("Finished reading.")
    print(xi_set[0])
    print(f"Number of measurement points: {len(xi_set)}")
    print("DONE")
    t2 = time.time()
    print(f"--- {(t2 - start_time):.3f} seconds ---")

    # Read normalization data
    normalizationAngles = [ list(map(float,i.split()))[numberedInput:] for i in open(normalizationData_filename).readlines() ]    # list (of lists) 
    normalizationAngles = np.asarray(normalizationAngles) # needed for numba. Fix datatype.

    print(normalizationAngles[0])
    print(f"Number of random angles for normalization: {len(normalizationAngles)}")
    # NOTE: The normalization angles are not angles but rather cos(angles).
    print(f"--- {(time.time() - t2):.3f} seconds for normalization data ---")
    print(f"--- {(time.time() - start_time):.3f} seconds total for all input data ---")
    ########## END READ DATA ##########

    ########## OPTIMIZE: ##########
    #  input parametervärden 1, 2, 3, 4 = 0.460, 0.740, 0.754, -0.754
    # Variables alpha, dPhi, alpha1, alpha2
    initial_guess = [0.4, 0.7, 0.6, -0.6]
    print(f"Initial guess: {initial_guess}")
    bnds = ((-1,1),(-7,7),(-7,7),(-7,7))   # bounds on variables. NOTE: The (-7,7) bound on last 3 is pretty arbitrary.
    #q = 2.396 # GeV, reaction energy (momentum transfer)
    #mLambda = 1.115683 # GeV, mass of lambda baryon (from PDG-live)
    #tau = q**2/(4*mLambda**2)   # form factor #tau = 1.15442015725 # Viktors värde? #tau = 1.1530071615814588 # mitt beräknade 
    tolerance = 10**-6

    print("Optimizing...")
    # scipy existing minimizing function. 
    res = optimize.minimize(negLL, initial_guess, (xi_set[0:], WDoubleTag, True, normalizationAngles[0:]), tol=tolerance, bounds=bnds)
    ########## END OPTIMIZE ##########

    ########## PRESENT RESULTS: ##########
    print(res)  # scipy default result structure
    print(f"------ TOOK A TOTAL OF {time.time() - start_time:.3f} SECONDS ------")
    print(f"Initial guess: \t\t {initial_guess}")
    print(f"Expected result: \t {(0.460, 0.740, 0.754, -0.754)}") # input to generate data, according to Viktor
    alpha_res = res['x'][0]
    dphi_res = res['x'][1]
    alpha1_res = res['x'][2]
    alpha2_res = res['x'][3]
    print(f"Result for alpha: \t {alpha_res}")
    #R = tau**(0.5) * ((1-eta_res)/(1+eta_res))**(0.5)   # according to formalism
    #print(f"Yielding R = {R}")
    print(f"delta-phi = {dphi_res} rad, or delta-phi = {dphi_res*180/PI} deg")
    print(f"Result for alpha1: \t {alpha1_res}")
    print(f"Result for alpha2: \t {alpha2_res}")
    
    print("")
    hess = (res['hess_inv']).todense()
    print("Inverse Hessian:")
    print(hess)
    print(f'Variance alpha: \t\t {hess[0][0]} \nVariance delta-phi: \t {hess[1][1]} (rad)')
    ########## END PRESENT RESULTS ##########

########## END MAIN ##########

if __name__ == "__main__":  # doesn't run if imported.
    main()