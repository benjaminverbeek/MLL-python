# testing input of 200k values from .dat
# 2021-04-12 Benjamin Verbeek

import os
import time
start_time = time.time()

print(os.getcwd())

datContent = ( i.strip().split() for i in open("test200k.dat").readlines() )    # generator object

s = 0
for i in datContent:
    s += int(i[0])

print(s)
print("--- %s seconds ---" % (time.time() - start_time))

# execuition time 0.0008268356323242188 seconds for 100 lines
# execuition time 0.33054375648498535 seconds for 200k lines (roughly x400) list format
# This should be fine. 
# NOTE: only about half as long (0.14) with a generator (i.e. () instead of [] in comprehension)
# This also circumvents any potential memory issues for larger datasets.
# 0.682 s for 10**6 points, vs 2.07 s for list.
