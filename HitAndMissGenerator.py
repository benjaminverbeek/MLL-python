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
def hit_miss_generator(N, alpha):
  i = 0
  while (i < N):
    costheta = np.random.uniform(-1, 1)
    rand = np.random.uniform(0, 1)
    dsdcostheta = 1 + alpha * costheta**2 # calculate the angular distribution
    upper_limit = 1 + abs(alpha) # and normalise it
    i += 1
    if (rand < dsdcostheta / upper_limit):
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
pdf = lambda costheta, alph : 6* (1 + alph*costheta**2)/(6+alph)
print(pdf(0,-0.5))

step = 0.1
a = [-1+step*i for i in range(10)]

L = []
# integral from 0.8 to 1.2 is 0.376206  
for alp in a:
    p = 1
    for i in costheta_dist:
        p *= pdf(i,alp)
    L.append(p)

print(a)
print(L)