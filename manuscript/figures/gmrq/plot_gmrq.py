import os, sys
import numpy as np

import matplotlib.pyplot as plt


data = {}

ntica_values = [6,8,10,15,20]
nstates_values  = [50,100,250,500,750,1000,1500,2000]

for ntica in ntica_values:
    infile = 't1_m1_n%d.dat'%ntica

    data[ntica] = np.loadtxt(infile)

"""
       fold  msm_lag  n_states  n_tica test_score  tica_lag  train_score
0      0        1        50      20    2.874323         1     2.870937
1      1        1        50      20    2.908684         1     2.913711
2      2        1        50      20    2.829298         1     2.863758
"""

# Calculate the means
mean_test, mean_train = {}, {}
for ntica in ntica_values:

    mean_test[ntica] = np.zeros( (len(nstates_values), 3) )  # nstates, mean, std
    mean_train[ntica] = np.zeros( (len(nstates_values), 3) )

    for i in range(len(nstates_values)):
        nstates = nstates_values[i]
        Ind = (data[ntica][:,3] == nstates)
        mean_test[ntica][i,:] = nstates, data[ntica][Ind,5].mean(), data[ntica][Ind,5].std()
        mean_train[ntica][i,:] = nstates, data[ntica][Ind,7].mean(), data[ntica][Ind,7].std()

print 'mean_test', mean_test
print 'mean_train', mean_train

colors = ['b','r','g','m','y','c','k']

plt.figure(figsize=(6,8))
for i in range(len(ntica_values)):
    ntica = ntica_values[i]
    plt.errorbar(mean_test[ntica][:,0], mean_test[ntica][:,1], yerr=mean_test[ntica][:,2], fmt='%so-'%colors[i], label='test %d tICs'%ntica)
    plt.errorbar(mean_train[ntica][:,0], mean_train[ntica][:,1], yerr=mean_train[ntica][:,2], fmt='%so--'%colors[i], label='train %d tICs'%ntica)

#plt.plot(n_tica,mean_test_score,'r-')
#plt.plot(n_tica,mean_train_score,'b-')
#plt.xlim(0,11)
plt.legend(loc='best', fontsize=8)
plt.xlabel('number of states')
plt.ylabel('GMRQ score')

plt.show()

