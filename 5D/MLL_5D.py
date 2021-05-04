# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Program to make 5D fit (Double-tag BESIII)            #
# Benjamin Verbeek, 2021-04-30                          #
# Fully functional. Theory definitions specified in     #
# appropriate places.                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

print("--- RUNNING MAX LOG LIKELIHOOD FIT ---")

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
    # input parameter values 1, 2, 3, 4 = 0.460, 0.740, 0.754, -0.754 (from Patrik Adlarson)
    # Parameters to optimize for: alpha, dPhi, alpha1, alpha2
    initial_guess = [0.3, PI/8, 0.55, -0.55]
    print(f"Initial guess: {initial_guess}")
    bnds = ((-1,1),(-PI,PI),(-1,1),(-1,1))   # bounds on variables.
    # Options for the optimizer. Can also fix method. Read more on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    #ops = {'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None}
    tolerance = 10**-6
    print("OPTIMIZING...")
    # scipy existing minimizing function. 
    res = optimize.minimize(negLL, initial_guess, (xi_set[0:], WDoubleTag, True, normAngs[0:]), bounds=bnds, tol=tolerance)#, method='L-BFGS-B')#, options=ops)
    ########## END OPTIMIZE ##########

    ########## PRESENT RESULTS: ##########
    print("\n------ FINISHED OPTIMIZATION, SCIPY.OPTIMIZE.MINIMIZE RESULTS: ------")
    print(res)  # scipy default result structure
    if res['success'] == True:
        print(f"CONVERGED SUCCESSFULLY, using tolerance {tolerance}")
    else:
        print("!!! OPTIMIZATION WAS NOT SUCCESSFUL !!!")
    print(f"\n------ TOOK A TOTAL OF {time.time() - start_time:.3f} SECONDS ------")
    print(f"          Solved for: \t alpha, dPhi, alpha1, alpha2 \n          Bounded by: \t {bnds}")
    print(f"       Initial guess: \t {initial_guess}")
    print(f"     Expected result: \t {(0.460, 0.785398, 0.75, -0.75)}") # input to generate data, according to Patrik
    print(f"       Actual result: \t {res['x']}")
    alpha_res = res['x'][0]
    dphi_res = res['x'][1]
    alpha1_res = res['x'][2]
    alpha2_res = res['x'][3]
    print(f"    Result for alpha: \t {alpha_res}")
    print(f"Result for delta-phi: \t {dphi_res} rad  =  {dphi_res*180/PI} deg") 
    print(f"   Result for alpha1: \t {alpha1_res}")
    print(f"   Result for alpha2: \t {alpha2_res}")
    
    print("")
    hess = (res['hess_inv']).todense()
    print("Inverse Hessian:")
    print(hess)
    print(f'    Variance alpha: \t {hess[0][0]} \nVariance delta-phi: \t {hess[1][1]} (rad) \n   Variance alpha1: \t {hess[2][2]} \n   Variance alpha2: \t {hess[3][3]} \n')
    ########## END PRESENT RESULTS ##########

########## END MAIN ##########

if __name__ == "__main__":  # doesn't run if imported.
    main()

# SAMPLE RUN (PERSONAL LAPTOP, as per 2021-04-30 16:30)
#
# --- RUNNING MAX LOG LIKELIHOOD FIT ---
# Reading input data...    (this might take a minute)
# First row: [ 0.0321418  2.31065    2.80985    2.02057   -1.68605  ]
# Number of measurement points: 100000
# Finished reading signal data.       
# --- 0.986 seconds ---
# Reading normalization data...       
# First row: [ 0.0321418  2.31065    2.80985    2.02057   -1.68605  ]
# Number of angles for normalization: 1000000
# --- 9.620 seconds for normalization data ---
# --- 10.606 seconds total for all input data ---       
# Initial guess: [0.3, 0.39269908169872414, 0.55, -0.55]
# Optimizing...
# --------
# One normalization done... took 2.46460 seconds. Norm = 1.1002344428646174
# neg LL:          -2615.8326718302633
# One set of parameters done. Took 0.34604 seconds.
# Total time for one iteration was 2.81064 seconds.
# --------
# One normalization done... took 0.39079 seconds. Norm = 1.100234446201749
# neg LL:          -2615.832684453
# One set of parameters done. Took 0.05532 seconds.
# Total time for one iteration was 0.44612 seconds.
# --------
# One normalization done... took 0.38181 seconds. Norm = 1.1002344428658377
# neg LL:          -2615.832680226911
# One set of parameters done. Took 0.04844 seconds.
# Total time for one iteration was 0.43025 seconds.
# --------
# One normalization done... took 0.37050 seconds. Norm = 1.1002344428657294
# neg LL:          -2615.832693160619
# One set of parameters done. Took 0.05915 seconds.
# Total time for one iteration was 0.42965 seconds.
# --------
# One normalization done... took 0.38322 seconds. Norm = 1.1002344428614075
# neg LL:          -2615.8326507381626
# One set of parameters done. Took 0.05796 seconds.
# Total time for one iteration was 0.44118 seconds.
# --------
# One normalization done... took 0.37721 seconds. Norm = 1.3341855074942957
# neg LL:          4017.2292090307637
# One set of parameters done. Took 0.05590 seconds.
# Total time for one iteration was 0.43312 seconds.
# --------
# One normalization done... took 0.39803 seconds. Norm = 1.334185510954694
# neg LL:          4017.3655647193773
# One set of parameters done. Took 0.05129 seconds.
# Total time for one iteration was 0.44933 seconds.
# --------
# One normalization done... took 0.38389 seconds. Norm = 1.3341855074942957
# neg LL:          4017.2292090307637
# One set of parameters done. Took 0.04978 seconds.
# Total time for one iteration was 0.43367 seconds.
# --------
# One normalization done... took 0.36802 seconds. Norm = 1.3341855074888582
# neg LL:          4017.2288167848455
# One set of parameters done. Took 0.06504 seconds.
# Total time for one iteration was 0.43306 seconds.
# --------
# One normalization done... took 0.38120 seconds. Norm = 1.3341855074888584
# neg LL:          4017.22881678486
# One set of parameters done. Took 0.05199 seconds.
# Total time for one iteration was 0.43319 seconds.
# --------
# One normalization done... took 0.38174 seconds. Norm = 1.100297059304985
# neg LL:          -2617.1963044183485
# One set of parameters done. Took 0.05235 seconds.
# Total time for one iteration was 0.43409 seconds.
# --------
# One normalization done... took 0.37008 seconds. Norm = 1.1002970626420259
# neg LL:          -2617.1963170345334
# One set of parameters done. Took 0.06097 seconds.
# Total time for one iteration was 0.43105 seconds.
# --------
# One normalization done... took 0.38049 seconds. Norm = 1.1002970593061259
# neg LL:          -2617.1963128074694
# One set of parameters done. Took 0.05240 seconds.
# Total time for one iteration was 0.43288 seconds.
# --------
# One normalization done... took 0.38053 seconds. Norm = 1.1002970593060535
# neg LL:          -2617.1963257512543
# One set of parameters done. Took 0.04828 seconds.
# Total time for one iteration was 0.42881 seconds.
# --------
# One normalization done... took 0.36606 seconds. Norm = 1.100297059301727
# neg LL:          -2617.196283332283
# One set of parameters done. Took 0.06234 seconds.
# Total time for one iteration was 0.42840 seconds.
# --------
# One normalization done... took 0.38365 seconds. Norm = 1.2173095776006382
# neg LL:          -2951.791833706695
# One set of parameters done. Took 0.04811 seconds.
# Total time for one iteration was 0.43177 seconds.
# --------
# One normalization done... took 0.37254 seconds. Norm = 1.2173095809372234
# neg LL:          -2951.791824531487
# One set of parameters done. Took 0.06096 seconds.
# Total time for one iteration was 0.43350 seconds.
# --------
# One normalization done... took 0.37589 seconds. Norm = 1.2173095776006326
# neg LL:          -2951.7918301389473
# One set of parameters done. Took 0.05746 seconds.
# Total time for one iteration was 0.43335 seconds.
# --------
# One normalization done... took 0.38100 seconds. Norm = 1.2173095776030727
# neg LL:          -2951.7918167755524
# One set of parameters done. Took 0.05279 seconds.
# Total time for one iteration was 0.43380 seconds.
# --------
# One normalization done... took 0.38268 seconds. Norm = 1.2173095775941924
# neg LL:          -2951.791850978603
# One set of parameters done. Took 0.04986 seconds.
# Total time for one iteration was 0.43254 seconds.
# --------
# One normalization done... took 0.37870 seconds. Norm = 1.1756232244274951
# neg LL:          -3232.679610991994
# One set of parameters done. Took 0.05186 seconds.
# Total time for one iteration was 0.43056 seconds.
# --------
# One normalization done... took 0.37998 seconds. Norm = 1.1756232277642102
# neg LL:          -3232.679612085136
# One set of parameters done. Took 0.05107 seconds.
# Total time for one iteration was 0.43105 seconds.
# --------
# One normalization done... took 0.39558 seconds. Norm = 1.1756232244279865
# neg LL:          -3232.6796078455936
# One set of parameters done. Took 0.04889 seconds.
# Total time for one iteration was 0.44447 seconds.
# --------
# One normalization done... took 0.37992 seconds. Norm = 1.1756232244288845
# neg LL:          -3232.679617352742
# One set of parameters done. Took 0.05704 seconds.
# Total time for one iteration was 0.43695 seconds.
# --------
# One normalization done... took 0.40498 seconds. Norm = 1.1756232244214808
# neg LL:          -3232.679604935405
# One set of parameters done. Took 0.04908 seconds.
# Total time for one iteration was 0.45406 seconds.
# --------
# One normalization done... took 0.37164 seconds. Norm = 1.1644895586653647
# neg LL:          -3294.6761320833466
# One set of parameters done. Took 0.06294 seconds.
# Total time for one iteration was 0.43458 seconds.
# --------
# One normalization done... took 0.40564 seconds. Norm = 1.1644895620022344
# neg LL:          -3294.676134338253
# One set of parameters done. Took 0.04870 seconds.
# Total time for one iteration was 0.45434 seconds.
# --------
# One normalization done... took 0.37570 seconds. Norm = 1.1644895586660844
# neg LL:          -3294.676129034773
# One set of parameters done. Took 0.06315 seconds.
# Total time for one iteration was 0.43884 seconds.
# --------
# One normalization done... took 0.38401 seconds. Norm = 1.1644895586667583
# neg LL:          -3294.676138701361
# One set of parameters done. Took 0.04989 seconds.
# Total time for one iteration was 0.43390 seconds.
# --------
# One normalization done... took 0.38868 seconds. Norm = 1.1644895586595352
# neg LL:          -3294.676125703183
# One set of parameters done. Took 0.04932 seconds.
# Total time for one iteration was 0.43799 seconds.
# --------
# One normalization done... took 0.38267 seconds. Norm = 1.1199074158156315
# neg LL:          -3251.5852108744493
# One set of parameters done. Took 0.05857 seconds.
# Total time for one iteration was 0.44125 seconds.
# --------
# One normalization done... took 0.37685 seconds. Norm = 1.1199074191532181
# neg LL:          -3251.585218858436
# One set of parameters done. Took 0.06114 seconds.
# Total time for one iteration was 0.43799 seconds.
# --------
# One normalization done... took 0.38202 seconds. Norm = 1.1199074158171543
# neg LL:          -3251.5852184157957
# One set of parameters done. Took 0.06169 seconds.
# Total time for one iteration was 0.44371 seconds.
# --------
# One normalization done... took 0.38825 seconds. Norm = 1.1199074158173414
# neg LL:          -3251.5852183994957
# One set of parameters done. Took 0.05652 seconds.
# Total time for one iteration was 0.44478 seconds.
# --------
# One normalization done... took 0.38945 seconds. Norm = 1.1199074158116962
# neg LL:          -3251.5852034206055
# One set of parameters done. Took 0.05476 seconds.
# Total time for one iteration was 0.44421 seconds.
# --------
# One normalization done... took 0.39382 seconds. Norm = 1.14332831351073
# neg LL:          -3371.932385009295
# One set of parameters done. Took 0.05743 seconds.
# Total time for one iteration was 0.45125 seconds.
# --------
# One normalization done... took 0.39157 seconds. Norm = 1.1433283168480106
# neg LL:          -3371.932389098245
# One set of parameters done. Took 0.05485 seconds.
# Total time for one iteration was 0.44642 seconds.
# --------
# One normalization done... took 0.38967 seconds. Norm = 1.1433283135119015
# neg LL:          -3371.932384442569
# One set of parameters done. Took 0.04987 seconds.
# Total time for one iteration was 0.43953 seconds.
# --------
# One normalization done... took 0.38698 seconds. Norm = 1.1433283135121293
# neg LL:          -3371.9323927368405
# One set of parameters done. Took 0.05552 seconds.
# Total time for one iteration was 0.44250 seconds.
# --------
# One normalization done... took 0.37781 seconds. Norm = 1.1433283135055947
# neg LL:          -3371.9323774214554
# One set of parameters done. Took 0.05463 seconds.
# Total time for one iteration was 0.43244 seconds.
# --------
# One normalization done... took 0.38156 seconds. Norm = 1.162840008316981
# neg LL:          -3378.4844358027003
# One set of parameters done. Took 0.05488 seconds.
# Total time for one iteration was 0.43644 seconds.
# --------
# One normalization done... took 0.37393 seconds. Norm = 1.1628400116545061
# neg LL:          -3378.4844321541223
# One set of parameters done. Took 0.06313 seconds.
# Total time for one iteration was 0.43706 seconds.
# --------
# One normalization done... took 0.36739 seconds. Norm = 1.1628400083183603
# neg LL:          -3378.4844399021695
# One set of parameters done. Took 0.06239 seconds.
# Total time for one iteration was 0.42978 seconds.
# --------
# One normalization done... took 0.37691 seconds. Norm = 1.1628400083189396
# neg LL:          -3378.4844390272574
# One set of parameters done. Took 0.05977 seconds.
# Total time for one iteration was 0.43668 seconds.
# --------
# One normalization done... took 0.37806 seconds. Norm = 1.1628400083123855
# neg LL:          -3378.4844326359926
# One set of parameters done. Took 0.05421 seconds.
# Total time for one iteration was 0.43226 seconds.
# --------
# One normalization done... took 0.37971 seconds. Norm = 1.1531851824491002
# neg LL:          -3398.6174645204082
# One set of parameters done. Took 0.04987 seconds.
# Total time for one iteration was 0.42958 seconds.
# --------
# One normalization done... took 0.36587 seconds. Norm = 1.1531851857865065
# neg LL:          -3398.6174647021217
# One set of parameters done. Took 0.06260 seconds.
# Total time for one iteration was 0.42847 seconds.
# --------
# One normalization done... took 0.37707 seconds. Norm = 1.1531851824503658
# neg LL:          -3398.6174660739416
# One set of parameters done. Took 0.05682 seconds.
# Total time for one iteration was 0.43390 seconds.
# --------
# One normalization done... took 0.37155 seconds. Norm = 1.1531851824507733
# neg LL:          -3398.617470337207
# One set of parameters done. Took 0.06140 seconds.
# Total time for one iteration was 0.43295 seconds.
# --------
# One normalization done... took 0.37642 seconds. Norm = 1.1531851824442076
# neg LL:          -3398.6174587570877
# One set of parameters done. Took 0.05946 seconds.
# Total time for one iteration was 0.43588 seconds.
# --------
# One normalization done... took 0.37808 seconds. Norm = 1.159793069247651
# neg LL:          -3415.943065063475
# One set of parameters done. Took 0.05151 seconds.
# Total time for one iteration was 0.42959 seconds.
# --------
# One normalization done... took 0.37886 seconds. Norm = 1.1597930725851453
# neg LL:          -3415.943063249586
# One set of parameters done. Took 0.05044 seconds.
# Total time for one iteration was 0.42929 seconds.
# --------
# One normalization done... took 0.36562 seconds. Norm = 1.1597930692489682
# neg LL:          -3415.9430642020598
# One set of parameters done. Took 0.06284 seconds.
# Total time for one iteration was 0.42846 seconds.
# --------
# One normalization done... took 0.36939 seconds. Norm = 1.1597930692494909
# neg LL:          -3415.943062489134
# One set of parameters done. Took 0.06298 seconds.
# Total time for one iteration was 0.43237 seconds.
# --------
# One normalization done... took 0.37460 seconds. Norm = 1.1597930692423015
# neg LL:          -3415.9430677550154
# One set of parameters done. Took 0.05782 seconds.
# Total time for one iteration was 0.43242 seconds.
# --------
# One normalization done... took 0.38156 seconds. Norm = 1.150590649355402
# neg LL:          -3422.289601866223
# One set of parameters done. Took 0.05170 seconds.
# Total time for one iteration was 0.43326 seconds.
# --------
# One normalization done... took 0.38395 seconds. Norm = 1.1505906526928558
# neg LL:          -3422.289602366567
# One set of parameters done. Took 0.04819 seconds.
# Total time for one iteration was 0.43214 seconds.
# --------
# One normalization done... took 0.36546 seconds. Norm = 1.1505906493566627
# neg LL:          -3422.2896019662076
# One set of parameters done. Took 0.06377 seconds.
# Total time for one iteration was 0.42923 seconds.
# --------
# One normalization done... took 0.36806 seconds. Norm = 1.1505906493571414
# neg LL:          -3422.2896019600485
# One set of parameters done. Took 0.06263 seconds.
# Total time for one iteration was 0.43069 seconds.
# --------
# One normalization done... took 0.37517 seconds. Norm = 1.1505906493502178
# neg LL:          -3422.2896018704632
# One set of parameters done. Took 0.05787 seconds.
# Total time for one iteration was 0.43304 seconds.
# --------
# One normalization done... took 0.39505 seconds. Norm = 1.1532809217685722
# neg LL:          -3422.4591996447907
# One set of parameters done. Took 0.06451 seconds.
# Total time for one iteration was 0.45957 seconds.
# --------
# One normalization done... took 0.37387 seconds. Norm = 1.1532809251059812
# neg LL:          -3422.4591995450155
# One set of parameters done. Took 0.06398 seconds.
# Total time for one iteration was 0.43785 seconds.
# --------
# One normalization done... took 0.36955 seconds. Norm = 1.153280921769857
# neg LL:          -3422.4591996247764
# One set of parameters done. Took 0.06345 seconds.
# Total time for one iteration was 0.43300 seconds.
# --------
# One normalization done... took 0.36712 seconds. Norm = 1.153280921770333
# neg LL:          -3422.459199829793
# One set of parameters done. Took 0.06596 seconds.
# Total time for one iteration was 0.43308 seconds.
# --------
# One normalization done... took 0.38940 seconds. Norm = 1.1532809217633617
# neg LL:          -3422.459199540297
# One set of parameters done. Took 0.05214 seconds.
# Total time for one iteration was 0.44154 seconds.
# --------
# One normalization done... took 0.37594 seconds. Norm = 1.1527628893016437
# neg LL:          -3422.4918541384086
# One set of parameters done. Took 0.05937 seconds.
# Total time for one iteration was 0.43532 seconds.
# --------
# One normalization done... took 0.37768 seconds. Norm = 1.152762892639126
# neg LL:          -3422.4918541194293
# One set of parameters done. Took 0.05200 seconds.
# Total time for one iteration was 0.42968 seconds.
# --------
# One normalization done... took 0.36986 seconds. Norm = 1.1527628893028936
# neg LL:          -3422.491854136915
# One set of parameters done. Took 0.06082 seconds.
# Total time for one iteration was 0.43068 seconds.
# --------
# One normalization done... took 0.36814 seconds. Norm = 1.1527628893034365
# neg LL:          -3422.491854138303
# One set of parameters done. Took 0.06362 seconds.
# Total time for one iteration was 0.43176 seconds.
# --------
# One normalization done... took 0.36906 seconds. Norm = 1.1527628892964898
# neg LL:          -3422.4918542007163
# One set of parameters done. Took 0.06244 seconds.
# Total time for one iteration was 0.43151 seconds.
# --------
# One normalization done... took 0.36972 seconds. Norm = 1.1526802117137671
# neg LL:          -3422.494414439552
# One set of parameters done. Took 0.06116 seconds.
# Total time for one iteration was 0.43088 seconds.
# --------
# One normalization done... took 0.38270 seconds. Norm = 1.1526802150511797
# neg LL:          -3422.494414450961
# One set of parameters done. Took 0.05387 seconds.
# Total time for one iteration was 0.43656 seconds.
# --------
# One normalization done... took 0.37657 seconds. Norm = 1.1526802117149564
# neg LL:          -3422.4944144454385
# One set of parameters done. Took 0.05768 seconds.
# Total time for one iteration was 0.43426 seconds.
# --------
# One normalization done... took 0.38057 seconds. Norm = 1.1526802117154447
# neg LL:          -3422.4944144665933
# One set of parameters done. Took 0.04993 seconds.
# Total time for one iteration was 0.43050 seconds.
# --------
# One normalization done... took 0.37846 seconds. Norm = 1.152680211708497
# neg LL:          -3422.494414486373
# One set of parameters done. Took 0.04950 seconds.
# Total time for one iteration was 0.42796 seconds.

# ------ FINISHED OPTIMIZATION, SCIPY.OPTIMIZE.MINIMIZE RESULTS: ------
#       fun: -3422.494414439552
#  hess_inv: <4x4 LbfgsInvHessProduct with dtype=float64>
#       jac: array([-1.14087015, -0.58862497, -2.70410963, -4.6820787 ])
#   message: 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
#      nfev: 75
#       nit: 9
#      njev: 15
#    status: 0
#   success: True
#         x: array([ 0.45668555,  0.74505663,  0.76339588, -0.74182001])
# CONVERGED SUCCESSFULLY, using tolerance 1e-06

# ------ TOOK A TOTAL OF 45.801 SECONDS ------
# Solved for:              alpha, dPhi, alpha1, alpha2
# Bounded by:              ((-1, 1), (-3.141592653589793, 3.141592653589793), (-1, 1), (-1, 1))
# Initial guess:           [0.3, 0.39269908169872414, 0.55, -0.55]
# Expected result:         (0.46, 0.785398, 0.75, -0.75)
# Actual result:           [ 0.45668555  0.74505663  0.76339588 -0.74182001]
# Result for alpha:        0.4566855519105043
# Result for delta-phi:    0.7450566254248492 rad,         or delta-phi = 42.68860013510332 deg
# Result for alpha1:       0.7633958823362114
# Result for alpha2:       -0.7418200140733076

# Inverse Hessian:
# [[0.01789666 0.06019965 0.0168174  0.00340302]
#  [0.06019965 0.20414961 0.05949282 0.01435339]
#  [0.0168174  0.05949282 0.438382   0.47162549]
#  [0.00340302 0.01435339 0.47162549 0.52017877]]
# Variance alpha:          0.017896660505261067
# Variance delta-phi:      0.2041496074042821 (rad)
# Variance alpha1:         0.43838200382715514
# Variance alpha2:         0.52017876873727