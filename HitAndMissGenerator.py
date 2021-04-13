# Import the any necessary modules
import numpy as np
import matplotlib.pyplot as plt


# Set the seed for the random number generator
np.random.seed()


# Set some parameters for the event generator
N = 100000   # number of events to be generated
alpha = -0.2 # ratio between G_E and G_M
bins = 100 # number of bins for the histogram


# Implement the hit or miss generator
# generates distribution following 1 + alpha * x^2
def hit_miss_generator(N, alpha):
  i = 0
  while (i < N):
    costheta = np.random.uniform(-1, 1)
    rand = np.random.uniform(0, 1)
    dsdcostheta = 1 + alpha * costheta**2 # calculate the angular distribution
    upper_limit = 1 + abs(alpha) # and normalise it
    i += 1
    if (rand < dsdcostheta / upper_limit): # makes it follow distribution
      yield costheta

# Create and fill the cos(theta) distribution
costheta_dist = []
for i in hit_miss_generator(N, alpha):
  costheta_dist.append(i)

'''
# Let's look at the result
plt.hist(costheta_dist, bins)
plt.xlabel('$\cos(\\theta)$')
plt.ylabel('entries')
plt.show()
'''
# probability density function = func / (integral of func w.r.t. variables X), i.e. normalize integral.
# pdf(x,alpha) now yields the probability of seeing x given a distribution with parameter alpha.
# int [-1,1] of 1+ax^2 dx = 2*(x + ax^3/3) | x=1 = (2*3 + 2*a)/3
pdf = lambda costheta, alpha : (3 /(6+2*alpha)) * (1 + alpha*costheta**2) # normalized to integrate to 1 on [-1,1]

step = 0.05
a = [-1+step*i for i in range(int(1//step))] # range of alphas
#a = [-0.3, -0.2, -0.1] # guesses for alpha

# what we want to maximize
def logLikelihoodFunc(X, alpha):
  '''Dataset X [list], parameter alpha [float]'''
  s = 0       # sum
  for i in X: # iterate over samples
    s += np.log(pdf(i,alpha)) # log-sum of pdf gives LL
  return s

# this is a brute-force solution. I should check zeroes of derivative (numerically or analytically).
# takes quite a while... 
# NOTE: built in numpy-function maximizing inner for-loop?
# TODO: Replace this with built-in Python which finds max for all alpha, not just a given range.
L = []              # log-likelihood. We want to maximize this
for alph in a:      # for each alpha, check the log-likelihood of observing dataset
  L.append(logLikelihoodFunc(costheta_dist, alph))


#print(a)
print(len(L))
ind = L.index(max(L))
print(f"Max at alpha = {a[ind]:.2f}, yields L = {max(L)}")
# note to self: f"txt {x:totalChars.decimalPrecision f}" formats floats. use totalChars for "tabs"