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
from scipy import integrate     # requires download
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
def F1(th,th1,th2,ph1,ph2): return ( (np.sin(th))**2 * np.sin(th1) * np.sin(th2) * \
            np.cos(ph1) * np.cos(ph2) + (np.cos(th))**2 * np.cos(th1) * np.cos(th2) )

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

@jit(nopython=True) # Applies numba magic. nopython=True doesn't appear to make a difference but is apparently recommended.
def WDoubleTag2(alpha,dPhi,alpha1,alpha2 , th,th1,th2,ph1,ph2):
    xi = (th,th1,th2,ph1,ph2)
    '''Normalize this to get the PDF to optimize. W is the function from theory (Fäldt, Kupsc)'''
    # https://arxiv.org/pdf/1702.07288.pdf
    return alpha1*alpha2 * ((np.sin(th))**2 * np.sin(th1)*np.sin(th2)*np.cos(ph1)*np.cos(ph2) + (np.cos(th))**2 * np.cos(th1)*np.cos(th2) + (1-alpha**2)**0.5 * np.cos(dPhi) * np.sin(th) * np.cos(th) * (np.sin(th1)*np.cos(th2)*np.cos(ph1) + np.cos(th1)*np.sin(th2)*np.cos(ph2)) + alpha*(np.cos(th1)*np.cos(th2) - (np.sin(th))**2 * np.sin(th1)*np.sin(th2)*np.sin(ph1)*np.sin(ph2))) + alpha1 * (1-alpha**2)**0.5 * np.sin(dPhi)*np.sin(th)*np.cos(th)*np.sin(th1)*np.sin(ph1) + alpha2 * (1-alpha**2)**0.5 * np.sin(dPhi) * np.sin(th)*np.cos(th)*np.sin(th2)*np.sin(ph2) + alpha*(np.cos(th))**2 + 1                                                                                          

def WDoubleTagIntegrable(alpha,dPhi,alpha1,alpha2):
    '''Normalize this to get the PDF to optimize. W is the function from theory (Fäldt, Kupsc)'''
    # https://arxiv.org/pdf/1702.07288.pdf
    return (lambda *xi : F0(*xi) + alpha*F5(*xi) \
        + alpha1*alpha2 * (F1(*xi) + ((1-alpha**2)**0.5) * np.cos(dPhi) * F2(*xi) + alpha*F6(*xi)) \
        + ((1-alpha**2)**0.5) * np.sin(dPhi) * (alpha1*F3(*xi) + alpha2*F4(*xi)) )    # W function
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
    return 1/n * s * (2*PI)**5    # MC-integral: average value of function * area # NOTE: area is wrong for 5D but results should be same?
                            # (2pi)**5, since angles from -pi, pi. This area-constant does not affect results.
##### END MC INTEGRATOR #####

@jit(nopython=True)
def iterativeLL(par, var):  # a separate function so numba can optimize it.
    s = 0  # sum
    alpha,dPhi,alpha1,alpha2 = par
    for v in var: # iterate over samples of xi
        th,th1,th2,ph1,ph2 = v 
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
    initial_guess = [0.4405018 ,  0.8273168 , -0.0947856 ,  0.11611481]
    initial_guess = np.asarray(initial_guess)
    print(f"Initial guess: {initial_guess}")
    bnds = ((-1,1),(-0,7),(-1,1),(-1,1))   # bounds on variables. NOTE: What should they be?

    res = negLL(initial_guess, xi_set, WDoubleTag, normalizeSeparately=True, normalizationAngles=normalizationAngles)
    print(f"""
    Negative log likeligood at point: {initial_guess}
    neg LL:                           {res}
    """)

    """ # INTEGRAL TESTING
    print(MCintegral(*[0.0, 0.0, 0.4, -0.4], normalizationAngles[:]))
    print(integrate.nquad(WDoubleTagIntegrable(*[0.0, 0.0, 0.4, -0.4]), [[-PI,PI],[-PI,PI],[-PI,PI],[-PI,PI],[-PI,PI]]))
    """
    #print("Optimizing...")
    # scipy existing minimizing function. 
    #res = optimize.minimize(negLL, initial_guess, (xi_set[0:], WDoubleTag, True, normalizationAngles[0:]), tol=tolerance, method='L-BFGS-B',  bounds=bnds)#, options=ops)
    ########## END OPTIMIZE ##########

    ########## END PRESENT RESULTS ##########

########## END MAIN ##########

if __name__ == "__main__":  # doesn't run if imported.
    main()