# Import necessary modules
from HitAndMissGenerator import hit_miss_generator
import numpy as np
from scipy import optimize
import time

########## HELP FUNCTIONS ##########

# probability density function = func / (integral of func w.r.t. variables X), i.e. normalize integral.
# pdf(x,alpha) now yields the probability of seeing x given a distribution with parameter alpha.
# int [-1,1] of 1+ax^2 dx = 2*(x + ax^3/3) | x=1 = (2*3 + 2*a)/3
# pdf = lambda alpha, costheta : (3 /(6+2*alpha)) * (1 + alpha*costheta**2) # normalized to integrate to 1 on [-1,1]
def pdf(alpha, costheta):
    return (3 /(6+2*alpha)) * (1 + alpha*costheta**2) # normalized to integrate to 1 on [-1,1]

# what we want to maximize... (minimize neg.)
def negLogLikelihoodFunc(alpha, X):
    '''Parameter alpha [float], dataset X [list]'''
    s = 0       # sum
    for i in X: # iterate over samples
        s -= np.log(pdf(alpha, i)) # log-sum of pdf gives LL
    return s

# Prototype: (use instead of older version)
# Generalized LL-func.: send in a pdf too, and let vars be n-dim, dataset X be m-dim.
def generalNegLogLikelihoodFunc(var, par, pdf):
    '''Minimize this function for input variables to find max of Log-Likelihood for distribution. \n
    var : variables to maximize [list], N-dim \n
    par : dataset of parameters [list of lists] M-dim (NOTE: the inner lists represent observed points, i.e. 
    every parameter is not a separate list, but rather part of a set of parameters). 
    E.g.:
    >>> par = [ [a0, b0], [a1, b1], [a2, b2], ... ] and not \n
    >>> par = [ [a0, a1, a2, ...], [b0, b1, b2, ...] ] 
    where a, b are different parameters for the pdf. \n
    pdf : must take arguments pdf(v1,v2, ..., vN, p1, p2, ..., pM)'''

    s = 0  # sum
    for p in par: # iterate over samples
        # * unpacks the list of arguments
        s -= np.log(pdf(*var, *p)) # log-sum of pdf gives LL
    return s

######## END HELP FUNCTIONS ########

# Running code:
def main():
    
    start_time = time.time()

    # Set some parameters for the event generator
    N = 10000   # number of events to be generated
    alpha = -0.2 # ratio between G_E and G_M

    print(f"Generating {N} random points...")
    costheta_dist = []
    for i in hit_miss_generator(N, alpha):
        costheta_dist.append([i])

    # Time for generation of distribution
    print('Done generating.')
    cp1 = time.time()
    print(f"--- {cp1 - start_time} seconds since start---")

    print("Minimizing negLogLikelihoodFunc...")
    # NOTE: This can easily be made multivalued. Just pack extra arguments in args-parameter (as many as needed)
    # and pack variables to optimize for aswell (make sure initial guess is same dimension as variables)...
    res = optimize.minimize(generalNegLogLikelihoodFunc, [-0.3], (costheta_dist, pdf), tol=10**-3)

    print(f"Deviation from alpha = {alpha}: {res['x'][0]-(alpha)}") # depends on set of points obviously.
    print(res)
    print(f'Convergence: {res["success"]}')

    cp2 = time.time()
    endMsg = (  f"--- {cp2 - cp1} seconds since start last checkpoint --- \n"
                f"      --- {cp2 - start_time} seconds since start---" )
    print(endMsg)
########## END MAIN ##########

# Runs only if not imported
if __name__ == "__main__":
    main()