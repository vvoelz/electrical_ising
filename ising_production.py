import numpy as np
import os, sys

from IsingSampler import *


usage = """Usage: python ising_production.py nsteps ramp outname

    INPUTS
    nsteps	the number of ampler steps to simulate.  Examples:
    		10000000    # 10M    - makes 4 Mb trajectory files,  each takes ~7 min on a MacBook Pro, 1 CPU
              	100000000   # 100M   - makes 40 Mb trajectory files, each takes ~70 min on a MacBook Pro, 1 CPU
    ramp        0 (ramp up) or 1 (ramp down) depsilon values
    outname	write trajectory files with name <outname>0_10M_deps24.0   (for example)
"""

if len(sys.argv) < 4:
    print usage
    sys.exit(1)

nsteps  = int(sys.argv[1])
ramp    = int(sys.argv[2])
outname = sys.argv[3]

if nsteps < 1e6:
    tag = '%dk'%(nsteps/1000)
elif nsteps < 1e9:
    tag = '%dk'%(nsteps/1000000)

if ramp:
    my_depsilons = [(24.0 - i*0.5) for i in range(20)] 
else:
    my_depsilons = [(14.5 + i*0.5) for i in range(20)]

print 'my_depsilons', my_depsilons

ntrials = len(my_depsilons) 

e = None
for trial in range(ntrials):
    
    # create a sampler and simulate a trajectory
    s = IsingSampler(initial_state=e, depsilon=my_depsilons[trial])
    t = s.sample(nsteps)   # sample() returns a Trajectory object
    t.save('%s%d_%s_deps%3.1f'%(outname, trial, tag, my_depsilons[trial]))
    
    # get last state
    e = np.copy(s.e)
    
    # get rid of the old sampler and trajectory
    del s
    del t
