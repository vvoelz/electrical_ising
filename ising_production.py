import numpy as np
import os, sys

from IsingSampler import *


usage = """Usage: python ising_production.py NSTEPS

    NOTES:    nsteps = 10000000    # 10M    - makes 4 Mb trajectory files
              nsteps = 100000000   # 100M   - makes 40 Mb trajectory files
"""

if len(sys.argv) < 2:
    print usage
    sys.exit(1)

nsteps = int(sys.argv[1])
e = None


my_depsilons = [(24.0 - i*0.5) for i in range(20)] 
my_depsilons += [(14.5 + i*0.5) for i in range(20)]

print 'my_depsilons', my_depsilons
sys.exit(1)

ntrials = 40 
#nsteps = 10000000   # 10M    - makes 4 Mb trajectory files
nsteps = 100000000   # 100M  - makes 40 Mb trajectory files

e = None
for trial in range(ntrials):
    
    # create a sampler and simulate a trajectory
    s = IsingSampler(initial_state=e, depsilon=my_depsilons[trial])
    t = s.sample(nsteps)   # sample() returns a Trajectory object
    t.save('test%d_10M_deps%3.1f'%(trial,my_depsilons[trial]))
    
    # get last state
    e = np.copy(s.e)
    
    # get rid of the old sampler and trajectory
    del s
    del t
