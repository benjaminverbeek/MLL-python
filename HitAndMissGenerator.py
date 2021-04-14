



# Implement the hit or miss generator
# generates distribution following 1 + alpha * x^2
def hit_miss_generator(N, alpha):
  # Import the any necessary modules
  import numpy as np
  # Set the seed for the random number generator
  np.random.seed()

  i = 0
  while (i < N):
    costheta = np.random.uniform(-1, 1)
    rand = np.random.uniform(0, 1)
    dsdcostheta = 1 + alpha * costheta**2 # calculate the angular distribution
    upper_limit = 1 + abs(alpha) # and normalise it
    i += 1
    if (rand < dsdcostheta / upper_limit): # makes it follow distribution
      yield costheta

# Code that won't run on import
if __name__ == "__main__":
  import matplotlib.pyplot as plt
  import time
  start_time = time.time()

  # Set some parameters for the event generator
  N = 1000   # number of events to be generated
  alpha = -0.2 # ratio between G_E and G_M
  bins = 100 # number of bins for the histogram

  print(f"Generating {N} random points...")

  # Create and fill the cos(theta) distribution
  costheta_dist = []
  for i in hit_miss_generator(N, alpha):
    costheta_dist.append(i)

  print('Done generating.')
  cp1 = time.time()
  print(f"--- {cp1 - start_time} seconds since start---")

  # Let's look at the result
  plt.hist(costheta_dist, bins)
  plt.xlabel('$\cos(\\theta)$')
  plt.ylabel('entries')
  plt.show()

  '''
  #print(a)
  print(len(L))
  ind = L.index(max(L))
  print(f"Max at alpha = {a[ind]:.2f}, yields L = {max(L)}")
  # note to self: f"txt {x:totalChars.decimalPrecision f}" formats floats. use totalChars for "tabs"
  '''
