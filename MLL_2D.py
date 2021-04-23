# Testing out higher dimensional (2D) implementation of MLL
# 2021-04-15    Benjamin Verbeek

from MLL_1D_HitAndMiss import generalNegLogLikelihoodFunc
from HitAndMissGenerator import hit_miss_generator_2D
import numpy as np
from scipy import optimize
import time

# probability density function = func / (integral of func w.r.t. variables X), i.e. normalize integral.
# pdf(x,alpha) now yields the probability of seeing x given a distribution with parameter alpha.
# int [-1,1] of 1+ax^2 dx = 2*(x + ax^3/3) | x=1 = (2*3 + 2*a)/3
# pdf = lambda alpha, costheta : (3 /(6+2*alpha)) * (1 + alpha*costheta**2) # normalized to integrate to 1 on [-1,1]
def pdf(alpha, beta, costheta):
    return (3 /(6*beta+2*alpha)) * (beta + alpha*costheta**2) # normalized to integrate to 1 on [-1,1]

def main():
    # Set some parameters for the event generator
    N = 100000   # number of events to be generated
    alpha = -0.2 # ratio between G_E and G_M
    beta = 2
    
    print(f"Generating {N} random points...")
    # Create and fill the cos(theta) distribution
    costheta_dist = []
    for i in hit_miss_generator_2D(N, alpha, beta):
        costheta_dist.append([i])   # NOTE: Append as list, just happens to be 1D observation

    print("Minimizing negLogLikelihoodFunc...")
    # NOTE: This can easily be made multivalued. Just pack extra arguments in args-parameter (as many as needed)
    # and pack variables to optimize for aswell (make sure initial guess is same dimension as variables)...
    alphaGuess = -0.3
    betaGuess = 1
    initialGuess = [alphaGuess, betaGuess]
    # this is useless. Dependent variables, inf solutions?
    res = optimize.minimize(generalNegLogLikelihoodFunc, initialGuess, (costheta_dist, pdf), tol=10**-3)

    print(res)
    print(f'Convergence: {res["success"]}')


if __name__ == "__main__":
    main()