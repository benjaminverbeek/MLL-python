# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Program to make 3D fit (Actually simplifies to 2D)    #
# Benjamin Verbeek, 2021-04-23                          #
# Updated functions to work with numba, now executes    #
# very fast. Slow part is reading input and converting  #
# to numbda lists.                                      #
# NOTE: Multithread input? Could work.                  #
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

alpha = 0.754   # assumed.  Had 0.753 before, which stood in formalism_viktor.pdf
angleDistributionData_filename = "lAngles.txt"  # specify path if not in same folder.
normalizationData_filename = "lPHSP_4Pi.txt"

###### THEORY ######
@jit(nopython=True) # Applies numba magic. nopython=True doesn't appear to make a difference but is apparently recommended.
def WSingleTagNum(eta, delta_phi, cos_th, cos_thP):
    '''Normalize this to get the PDF to optimize. W is the function from theory (Fäldt, Kupsc)'''
    # https://arxiv.org/pdf/1702.07288.pdf
    return 1 + eta * (cos_th)**2 + alpha * (1 - eta**2)**(0.5) * np.sin(delta_phi) \
            * np.sin(np.arccos(cos_th)) * cos_th * cos_thP  # W function
##### END THEORY #####
    
##### MC INTEGRATOR #####
# MC-integrator for normalization factors
@jit(nopython=True) # numba decorator. Significantly improves performance (~factor 100)
def MCintegralNum(eta, delta_phi, uniformAngles):
    """Monte Carlo integration for normalization, for given parameters and a set of normalization angles."""
    s = 0.0   # sum
    n = 0.0   # number of points
    for xi in uniformAngles: # xi is a 2D list
        cos_th, cos_thP = xi
        s += WSingleTagNum(eta, delta_phi, cos_th, cos_thP) # evaluate W at a bunch of random points and sum.
        n += 1  # count number of points. Could also use len(uniformAngles)
    return 1/n * s * 2*2    # MC-integral: average value of function * area 
                            # (2*2, since cos has range [-1,1]). This area-constant does not affect results.
##### END MC INTEGRATOR #####

# Generalized LL-func.: send in a pdf too, and let par be n-dim, dataset var X be m-dim.
def negLogLikelihood(par, var, pdf, normalizeSeparately=False, normalizationAngles=[]):
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
        normalization = MCintegralNum(*par, normalizationAngles)
        print(normalization)
        t2 = time.time()
        print(f"One normalization done... took {t2 - t1:.5f} seconds.")
    else:
        normalization = 1
    
    @jit(nopython=True)
    def iterativeLL(par, var):  # a separate function so numba can optimize it.
        s = 0  # sum
        eta, delta_phi = par
        for v in var: # iterate over samples of xi
            cos_th, cos_thP = v
            s -= np.log(WSingleTagNum(eta, delta_phi, cos_th, cos_thP)) # log-sum of pdf gives LL. Negative so we minimize.
        return s
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
    xi_set = [ list(map(float,i.split())) for i in open(angleDistributionData_filename).readlines() ]    # list (of numba lists)
    # Iterate thru lines of datafile, for each line, split it into list of number contents,
    # map the content of that list from str -> float, convert map object -> list, all in list comprehension.
    xi_set = np.asarray(xi_set) # converts to numpy.array. Much faster than numba typed list.

    print("Finished reading.")
    print(xi_set[0])
    print(f"Number of measurement points: {len(xi_set)}")
    print("DONE")
    t2 = time.time()
    print(f"--- {(time.time() - start_time):.3f} seconds ---")

    # Read normalization data
    normalizationAngles = [ list(map(float,i.split())) for i in open(normalizationData_filename).readlines() ]    # list (of numba lists) 
    normalizationAngles = np.asarray(normalizationAngles)

    print(normalizationAngles[0])
    print(f"Number of random angles for normalization: {len(normalizationAngles)}")
    # NOTE: The normalization angles are not angles but rather cos(angles).
    print(f"--- {(time.time() - t2):.3f} seconds for normalization data ---")
    print(f"--- {(time.time() - start_time):.3f} seconds total for all input data ---")
    ########## END READ DATA ##########

    ########## OPTIMIZE: ##########
    # Generated data with R=0.91 and delta_phi = 42 deg (0.733 rad)
    # Variables eta, delta-phi
    initial_guess = [0.4, 60*PI/180]
    print(f"Initial guess: {initial_guess}")
    bnds = ((-1,1),(-7,7))   # bounds on variables. NOTE: The (-7,7) bound on dPhi is pretty arbitrary.
    q = 2.396 # GeV, reaction energy (momentum transfer)
    mLambda = 1.115683 # GeV, mass of lambda baryon (from PDG-live)
    tau = q**2/(4*mLambda**2)   # form factor #tau = 1.15442015725 # Viktors värde? #tau = 1.1530071615814588 # mitt beräknade 
    tolerance = 10**-6

    print("Optimizing...")
    # scipy existing minimizing function. 
    res = optimize.minimize(negLogLikelihood, initial_guess, (xi_set[0:-1], WSingleTagNum, True, normalizationAngles[0:-1]), tol=tolerance, bounds=bnds)
    ########## END OPTIMIZE ##########

    ########## PRESENT RESULTS: ##########
    print(res)  # scipy default result structure
    print(f"------ TOOK A TOTAL OF {time.time() - start_time:.3f} SECONDS ------")
    print(f"Initial guess: \t\t {initial_guess}")
    print(f"Expected result: \t {(0.217, 42*PI/180)}") # input to generate data, according to Viktor
    eta_res = res['x'][0]
    dphi_res = res['x'][1]
    print(f"Result for eta: \t {eta_res}")
    R = tau**(0.5) * ((1-eta_res)/(1+eta_res))**(0.5)   # according to formalism
    print(f"Yielding R = {R}")
    print(f"delta-phi = {dphi_res} rad, or delta-phi = {dphi_res*180/PI} deg")
    print("")
    hess = (res['hess_inv']).todense()
    print("Inverse Hessian:")
    print(hess)
    print(f'Variance eta: \t\t {hess[0][0]} \nVariance delta-phi: \t {hess[1][1]} (rad)')
    ########## END PRESENT RESULTS ##########

########## END MAIN ##########

if __name__ == "__main__":  # doesn't run if imported.
    main()



# EXAMPLE OF RUN:   (ran on office Linux/ubuntu machine)
#
# Reading input data...    (this might take a minute)
# Finished reading.
# [0.999483, -0.67418]
# Number of measurement points: 1000000
# DONE
# --- 21.951 seconds ---
# [0.350705, 0.492364]
# Number of random angles for normalization: 4950000
# --- 105.084 seconds for normalization data ---
# --- 127.035 seconds total for all input data ---
# Initial guess: [0.4, 1.0471975511965976]
# Optimizing...
# --------
# 4.533010576373457
# One normalization done... took 0.49616 seconds.
# One set of parameters done. Took 0.16491 seconds.
# Total time for one run was 0.66107 seconds.
# --------
# 4.533010589700987
# One normalization done... took 0.21176 seconds.
# One set of parameters done. Took 0.16081 seconds.
# Total time for one run was 0.37257 seconds.
# --------
# 4.533010576372809
# One normalization done... took 0.21229 seconds.
# One set of parameters done. Took 0.15921 seconds.
# Total time for one run was 0.37151 seconds.
# --------
# 2.6672213991197546
# One normalization done... took 0.21179 seconds.
# One set of parameters done. Took 0.16012 seconds.
# Total time for one run was 0.37191 seconds.
# --------
# 2.6672214242546115
# One normalization done... took 0.21145 seconds.
# One set of parameters done. Took 0.16012 seconds.
# Total time for one run was 0.37157 seconds.
# --------
# 2.6672213991197546
# One normalization done... took 0.21229 seconds.
# One set of parameters done. Took 0.23076 seconds.
# Total time for one run was 0.44305 seconds.
# --------
# 4.532967132148787
# One normalization done... took 0.21066 seconds.
# One set of parameters done. Took 0.16186 seconds.
# Total time for one run was 0.37252 seconds.
# --------
# 4.532967145477396
# One normalization done... took 0.21269 seconds.
# One set of parameters done. Took 0.16031 seconds.
# Total time for one run was 0.37300 seconds.
# --------
# 4.532967132148146
# One normalization done... took 0.21376 seconds.
# One set of parameters done. Took 0.16000 seconds.
# Total time for one run was 0.37376 seconds.
# --------
# 3.6001646157557508
# One normalization done... took 0.21110 seconds.
# One set of parameters done. Took 0.16068 seconds.
# Total time for one run was 0.37178 seconds.
# --------
# 3.6001646290842912
# One normalization done... took 0.21213 seconds.
# One set of parameters done. Took 0.15805 seconds.
# Total time for one run was 0.37018 seconds.
# --------
# 3.6001646157574267
# One normalization done... took 0.21251 seconds.
# One set of parameters done. Took 0.15977 seconds.
# Total time for one run was 0.37228 seconds.
# --------
# 4.487810467932925
# One normalization done... took 0.21235 seconds.
# One set of parameters done. Took 0.23552 seconds.
# Total time for one run was 0.44787 seconds.
# --------
# 4.487810481260856
# One normalization done... took 0.21170 seconds.
# One set of parameters done. Took 0.16083 seconds.
# Total time for one run was 0.37253 seconds.
# --------
# 4.487810467932097
# One normalization done... took 0.21087 seconds.
# One set of parameters done. Took 0.16538 seconds.
# Total time for one run was 0.37625 seconds.
# --------
# 2.6672213991197546
# One normalization done... took 0.21800 seconds.
# One set of parameters done. Took 0.15356 seconds.
# Total time for one run was 0.37156 seconds.
# --------
# 2.6672213949722967
# One normalization done... took 0.21206 seconds.
# One set of parameters done. Took 0.15369 seconds.
# Total time for one run was 0.36575 seconds.
# --------
# 2.6672213991197546
# One normalization done... took 0.21061 seconds.
# One set of parameters done. Took 0.15888 seconds.
# Total time for one run was 0.36950 seconds.
# --------
# 3.854114647760378
# One normalization done... took 0.21113 seconds.
# One set of parameters done. Took 0.15667 seconds.
# Total time for one run was 0.36780 seconds.
# --------
# 3.854114661088056
# One normalization done... took 0.21294 seconds.
# One set of parameters done. Took 0.23348 seconds.
# Total time for one run was 0.44642 seconds.
# --------
# 3.854114647760006
# One normalization done... took 0.21156 seconds.
# One set of parameters done. Took 0.15871 seconds.
# Total time for one run was 0.37027 seconds.
# --------
# 4.269613433635415
# One normalization done... took 0.21253 seconds.
# One set of parameters done. Took 0.15792 seconds.
# Total time for one run was 0.37045 seconds.
# --------
# 4.269613446962967
# One normalization done... took 0.20963 seconds.
# One set of parameters done. Took 0.16335 seconds.
# Total time for one run was 0.37297 seconds.
# --------
# 4.269613433634691
# One normalization done... took 0.21048 seconds.
# One set of parameters done. Took 0.15769 seconds.
# Total time for one run was 0.36817 seconds.
# --------
# 4.177855943676114
# One normalization done... took 0.21554 seconds.
# One set of parameters done. Took 0.15878 seconds.
# Total time for one run was 0.37432 seconds.
# --------
# 4.177855957003884
# One normalization done... took 0.21299 seconds.
# One set of parameters done. Took 0.15971 seconds.
# Total time for one run was 0.37269 seconds.
# --------
# 4.1778559436753815
# One normalization done... took 0.21000 seconds.
# One set of parameters done. Took 0.24009 seconds.
# Total time for one run was 0.45009 seconds.
# --------
# 4.228231171894719
# One normalization done... took 0.21253 seconds.
# One set of parameters done. Took 0.15742 seconds.
# Total time for one run was 0.36995 seconds.
# --------
# 4.228231185222605
# One normalization done... took 0.21170 seconds.
# One set of parameters done. Took 0.15957 seconds.
# Total time for one run was 0.37127 seconds.
# --------
# 4.228231171893704
# One normalization done... took 0.21175 seconds.
# One set of parameters done. Took 0.15725 seconds.
# Total time for one run was 0.36900 seconds.
# --------
# 4.2214496157168515
# One normalization done... took 0.21243 seconds.
# One set of parameters done. Took 0.16547 seconds.
# Total time for one run was 0.37790 seconds.
# --------
# 4.221449629044873
# One normalization done... took 0.21001 seconds.
# One set of parameters done. Took 0.15788 seconds.
# Total time for one run was 0.36789 seconds.
# --------
# 4.221449615715762
# One normalization done... took 0.21194 seconds.
# One set of parameters done. Took 0.15767 seconds.
# Total time for one run was 0.36961 seconds.
#       fun: 1380360.0691064522
#  hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>
#       jac: array([-42.14234652, -42.0724971 ])
#   message: 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
#      nfev: 33
#       nit: 4
#      njev: 11
#    status: 0
#   success: True
#         x: array([0.16621864, 0.72430347])
# ------ TOOK A TOTAL OF 140.611 SECONDS ------
# Initial guess:           [0.4, 1.0471975511965976]
# Expected result:         (0.217, 0.7330382858376184)
# Result for eta:          0.16621864050768131
# Yielding R = 0.9079294510333814
# delta-phi = 0.7243034698685151 rad, or delta-phi = 41.499531910146914 deg

# Inverse Hessian:
# [[ 4.81180533e-05 -1.30955508e-05]
#  [-1.30955508e-05  1.35035850e-04]]
# Variance eta:            4.811805325549507e-05 
# Variance delta-phi:      0.00013503585019338521 (rad)


# On my private Windows laptop:

# Reading input data...    (this might take a minute)
# Finished reading.
# [0.999483, -0.67418]
# Number of measurement points: 1000000
# DONE
# --- 78.561 seconds ---
# [0.350705, 0.492364]
# Number of random angles for normalization: 4950000
# --- 353.995 seconds for normalization data ---
# --- 432.557 seconds total for all input data ---
# Initial guess: [0.4, 1.0471975511965976]
# Optimizing...
# --------
# 4.533010576373457
# One normalization done... took 1.15258 seconds.
# One set of parameters done. Took 0.43618 seconds.
# Total time for one run was 1.58875 seconds.
# --------
# 4.533010589700987
# One normalization done... took 0.50864 seconds.
# One set of parameters done. Took 0.41093 seconds.
# Total time for one run was 0.91957 seconds.
# --------
# 4.533010576372809
# One normalization done... took 0.43722 seconds.
# One set of parameters done. Took 0.38102 seconds.
# Total time for one run was 0.81824 seconds.
# --------
# 2.6672213991197546
# One normalization done... took 0.44170 seconds.
# One set of parameters done. Took 0.37404 seconds.
# Total time for one run was 0.81574 seconds.
# --------
# 2.6672214242546115
# One normalization done... took 0.43683 seconds.
# One set of parameters done. Took 0.39597 seconds.
# Total time for one run was 0.83280 seconds.
# --------
# 2.6672213991197546
# One normalization done... took 0.44000 seconds.
# One set of parameters done. Took 0.40891 seconds.
# Total time for one run was 0.84891 seconds.
# --------
# 4.532967132148787
# One normalization done... took 0.45227 seconds.
# One set of parameters done. Took 0.50062 seconds.
# Total time for one run was 0.95289 seconds.
# --------
# 4.532967145477396
# One normalization done... took 0.43706 seconds.
# One set of parameters done. Took 0.40896 seconds.
# Total time for one run was 0.84601 seconds.
# --------
# 4.532967132148146
# One normalization done... took 0.51496 seconds.
# One set of parameters done. Took 0.46280 seconds.
# Total time for one run was 0.97775 seconds.
# --------
# 3.6001646157557508
# One normalization done... took 0.45578 seconds.
# One set of parameters done. Took 0.41990 seconds.
# Total time for one run was 0.87568 seconds.
# --------
# 3.6001646290842912
# One normalization done... took 0.46932 seconds.
# One set of parameters done. Took 0.40792 seconds.
# Total time for one run was 0.87724 seconds.
# --------
# 3.6001646157574267
# One normalization done... took 0.48215 seconds.
# One set of parameters done. Took 0.46579 seconds.
# Total time for one run was 0.94793 seconds.
# --------
# 4.487810467932925
# One normalization done... took 0.46407 seconds.
# One set of parameters done. Took 0.37999 seconds.
# Total time for one run was 0.84406 seconds.
# --------
# 4.487810481260856
# One normalization done... took 0.45977 seconds.
# One set of parameters done. Took 0.55851 seconds.
# Total time for one run was 1.01827 seconds.
# --------
# 4.487810467932097
# One normalization done... took 0.48869 seconds.
# One set of parameters done. Took 0.39793 seconds.
# Total time for one run was 0.88663 seconds.
# --------
# 2.6672213991197546
# One normalization done... took 0.46476 seconds.
# One set of parameters done. Took 0.39597 seconds.
# Total time for one run was 0.86073 seconds.
# --------
# 2.6672213949722967
# One normalization done... took 0.44227 seconds.
# One set of parameters done. Took 0.38398 seconds.
# Total time for one run was 0.82624 seconds.
# --------
# 2.6672213991197546
# One normalization done... took 0.43583 seconds.
# One set of parameters done. Took 0.38899 seconds.
# Total time for one run was 0.82483 seconds.
# --------
# 3.854114647760378
# One normalization done... took 0.43472 seconds.
# One set of parameters done. Took 0.38198 seconds.
# Total time for one run was 0.81670 seconds.
# --------
# 3.854114661088056
# One normalization done... took 0.45059 seconds.
# One set of parameters done. Took 0.40595 seconds.
# Total time for one run was 0.85654 seconds.
# --------
# 3.854114647760006
# One normalization done... took 0.44485 seconds.
# One set of parameters done. Took 0.54154 seconds.
# Total time for one run was 0.98639 seconds.
# --------
# 4.269613433635415
# One normalization done... took 0.44925 seconds.
# One set of parameters done. Took 0.38932 seconds.
# Total time for one run was 0.83857 seconds.
# --------
# 4.269613446962967
# One normalization done... took 0.49979 seconds.
# One set of parameters done. Took 0.41861 seconds.
# Total time for one run was 0.91840 seconds.
# --------
# 4.269613433634691
# One normalization done... took 0.43040 seconds.
# One set of parameters done. Took 0.39010 seconds.
# Total time for one run was 0.82051 seconds.
# --------
# 4.177855943676114
# One normalization done... took 0.46289 seconds.
# One set of parameters done. Took 0.40921 seconds.
# Total time for one run was 0.87210 seconds.
# --------
# 4.177855957003884
# One normalization done... took 0.46696 seconds.
# One set of parameters done. Took 0.38062 seconds.
# Total time for one run was 0.84758 seconds.
# --------
# 4.1778559436753815
# One normalization done... took 0.43446 seconds.
# One set of parameters done. Took 0.41996 seconds.
# Total time for one run was 0.85442 seconds.
# --------
# 4.228231171894719
# One normalization done... took 0.54922 seconds.
# One set of parameters done. Took 0.54227 seconds.
# Total time for one run was 1.09149 seconds.
# --------
# 4.228231185222605
# One normalization done... took 0.47596 seconds.
# One set of parameters done. Took 0.39102 seconds.
# Total time for one run was 0.86699 seconds.
# --------
# 4.228231171893704
# One normalization done... took 0.45576 seconds.
# One set of parameters done. Took 0.45601 seconds.
# Total time for one run was 0.91177 seconds.
# --------
# 4.2214496157168515
# One normalization done... took 0.46433 seconds.
# One set of parameters done. Took 0.38912 seconds.
# Total time for one run was 0.85345 seconds.
# --------
# 4.221449629044873
# One normalization done... took 0.43783 seconds.
# One set of parameters done. Took 0.39932 seconds.
# Total time for one run was 0.83716 seconds.
# --------
# 4.221449615715762
# One normalization done... took 0.47639 seconds.
# One set of parameters done. Took 0.42332 seconds.
# Total time for one run was 0.89971 seconds.
#       fun: 1380360.0691064522
#  hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>
#       jac: array([-42.14234652, -42.0724971 ])
#   message: 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
#      nfev: 33
#       nit: 4
#      njev: 11
#    status: 0
#   success: True
#         x: array([0.16621864, 0.72430347])
# ------ TOOK A TOTAL OF 464.076 SECONDS ------
# Initial guess:           [0.4, 1.0471975511965976]
# Expected result:         (0.217, 0.7330382858376184)
# Result for eta:          0.16621864050768131
# Yielding R = 0.9079294510333814
# delta-phi = 0.7243034698685151 rad, or delta-phi = 41.499531910146914 deg

# Inverse Hessian:
# [[ 4.81180533e-05 -1.30955508e-05]
#  [-1.30955508e-05  1.35035850e-04]]
# Variance eta:            4.811805325549549e-05
# Variance delta-phi:      0.00013503585019338532 (rad)