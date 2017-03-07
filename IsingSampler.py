import numpy as np

from IsingTrajectory import *

class IsingSampler(object):
    """An object for performing kinetic Monte Carlo sampling of the 
    'electrical Ising' model. """
    
    def __init__(self, nx=20, ny=20, voltage=0.0, debug=False, initial_state=None, depsilon=24.0):
        """Initialize the sampler object."""
        
        self.nx = nx  # number of cells in the x dimension
        self.ny = ny  # number of cells in the y dimension
        self.ncells = nx*ny   # total number of cells
        
        self.debug = debug
        
        # set electrical constants
        self.gating_charge = 1.0  # units eu
        self.dq = self.gating_charge/float(self.ncells)  # the activating charge of each cell
        self.voltage = voltage   # units mV  
        self.k_Boltzmann = 0.0861733034 # Boltzmann's constant in meV/K
        self.temperature = 290.0 # units K
        self.kT          = self.k_Boltzmann*self.temperature  # 25 meV at 290 K
        self.depsilon    = depsilon ## SHOULD be 24.0  # meV (this is equal to 2J of the standard Ising model) 
        
        # set kinetic Monte Carlo constants
        self.nu = 2.0e7 # pre_exponential_factor, units  s^{-1}
        self.Bronsted_slope = 0.5    #
               
        # initialize a random configuration of electric dipoles  e
        if initial_state==None:
            self.e = np.random.randint( 2, size=(self.nx, self.ny) )
        else:
            self.e = initial_state
        if self.debug:
            print 'self.e', self.e
        
        # compile a neighbor list for fast lookup
        self.neighborlist = None
        self.compile_neighborlist()
               
        # initialize the number of activated nearest neighbors
        self.n = self.count_all_activated_neighbors(self.e)   
        if self.debug:
            print 'self.n', self.n
            
        # Calculate the total energy from the states and their neighbors
        self.energy = self.calc_total_energy(self.e, self.n)
        if self.debug:
            print 'self.energy', self.energy
            print 'self.energy/self.kT', self.energy/self.kT

        # The 0->1 activation energy depends *only* on the number of activated neighbors (0 through 4),
        # which we can precompute.  
        self.activation_energy = np.array([ (2.0*(2.0-na)*self.depsilon - 1.0*self.dq*self.voltage) for na in range(5)])
        if self.debug:
            print 'self.activation_energy', self.activation_energy

        # The alpha 0->1 rates also depend *only* on the number of activated neighbors (0 through 4),
        # which we can precompute.        
        self.alphas = np.array([self.nu*np.exp(-self.Bronsted_slope*self.activation_energy[na]/self.kT) for na in range(5)])
        if self.debug:
            print 'self.alphas', self.alphas

        # The beta 1->0 rates also depend *only* on the number of activated neighbors (0 through 4),
        # which we can precompute.        
        self.betas = np.array([self.nu*np.exp((1.-self.Bronsted_slope)*self.activation_energy[na]/self.kT) for na in range(5)])
        if self.debug:
            print 'self.betas', self.betas

        # initialize rate constants for flipping a dipole
        self.alpha_beta = self.calculate_all_alpha_beta(self.e, self.n)
        if self.debug:
            print 'self.alpha_beta', self.alpha_beta
            
            
    def count_all_activated_neighbors(self, e):
        """Count the numbers of north/south/east/west neighbors that
        are activated."""
        
        n = np.zeros( e.shape, dtype=int) 
        
        for i in range(self.nx):      
            for j in range(self.ny):
                n[i,j] = self.count_activated_neighbors(e, i, j)
        return n

    def compile_neighborlist(self):
        """Construct a lookup table for periodic-boundary neighbor indices."""
        self.neighborlist = np.zeros( (self.nx, self.ny, 4, 2), dtype=int)
            # [i,j,:,:]  = [[inorth, jnorth],
            #               [isouth, jsouth],
            #               [iwest,  jwest],
            #               [ieast,  jeast]]
        for i in range(self.nx):
            for j in range(self.ny):
                self.neighborlist[i,j,0,:] = [i, (j-1)%(self.ny)]
                self.neighborlist[i,j,1,:] = [i, (j+1)%(self.ny)]
                self.neighborlist[i,j,2,:] = [(i-1)%(self.nx), j]
                self.neighborlist[i,j,3,:] = [(i+1)%(self.nx), j]

    
    def count_activated_neighbors(self, e, i, j):
        """Count the number of activated neighbors for a particular i,j cell."""
        
        count = 0
        
        # ESCHEW FOR-LOOPS!
        # for k in range(4):
        #    count += e[self.neighborlist[i,j,k,0],self.neighborlist[i,j,k,1]]       

        count += e[self.neighborlist[i,j,0:4,0],self.neighborlist[i,j,0:4,1]].sum()       

        return count

    
    def calc_total_energy(self, e, n):
        """Calculate the energy of the input state.
        NOTE:  Expensive 
        
        INPUT
        e     - an nx,ny array of 0 (resting) or 1 (activated)"""
        
        energy = 0.0
        for i in range(self.nx):
            for j in range(self.ny):
                
                energy +=  -1.0*self.dq*float(e[i,j])*self.voltage
                
                if e[i,j] == 0:
                    # NOTE divide by 2 to avoid double-counting
                    energy += n[i,j]*self.depsilon / 2.0 
                else:
                    energy += (4.0-n[i,j])*self.depsilon / 2.0 

        return energy 
 

    def calculate_all_alpha_beta(self, e, n):
        """Calculate activating/deactivating rate constants for each cell"""
        
        alpha_beta = np.zeros( e.shape, dtype=float)
        for i in range(self.nx):
            for j in range(self.ny):
                
                if e[i,j] == 0:
                    alpha_beta[i,j] = self.alphas[n[i,j]]
                else:
                    alpha_beta[i,j] = self.betas[n[i,j]]
                    
        return alpha_beta
                
        
    def sample(self, nsteps, print_every=10000, save_every=1000, energy_check=False):
        """Perform kinetic Monte Carlo """
        
        nframes = nsteps/save_every 
        self.t = IsingTrajectory(nframes, nx=self.nx, ny=self.ny)
                        
        time = 0.0
        tau  = 0.0
        
        for step in range(nsteps):
            
            ###########################
            ##### save a snapshot #####
            
            fwd_rate = self.alpha_beta[(self.e==0)].sum()
            back_rate = self.alpha_beta[(self.e==1)].sum()
                        
            # append a snapshot to the the trajectory
            if step%save_every == 0:
                self.t.tally(self.e, time, tau, self.energy, fwd_rate, back_rate)
            
            if step%print_every == 0:
                print 'step', step, 'of', nsteps, 'q =', self.e.sum()*self.dq, 
                if energy_check:
                    print '| energy (on-the-fly total-recalc)', self.energy, self.calc_total_energy(self.e, self.n)
                else:
                    print

            ###############################
            
            # Sum all the previously assigned rate constants ai together to obtain the total rate constant a
            # of leaving the current configuration. 
            total_rate_a = fwd_rate + back_rate
            # total_rate_a = self.alpha_beta.sum()
                    
            # Then pick a uniform random number r1
            r1 = np.random.rand()
            # ... to draw a dwell time from an exponential distribution
            # using the formula:
            tau = -np.log(r1)/total_rate_a

            # Pick a second random number r2 in order to determine which cell to flip.
            # Each cell has probability ai/a of being chosen.
            if (0):
                # OLD routine (slow)
                p = self.alpha_beta.ravel()/total_rate_a
                r2 = np.random.rand()
                flip_index = -1
                p_cumulative = 0.0
                while r2 > p_cumulative:
                    p_cumulative += p[flip_index+1]
                    flip_index += 1
                if self.debug:
                    print 'r2', r2, 'flip_index', flip_index,
            else:
                # NEW routine (fast, using numpy routines)
                flip_index = np.searchsorted(np.cumsum(self.alpha_beta.ravel()/total_rate_a), [np.random.rand()])[0]
                if self.debug:
                    print 'flip_index', flip_index,

            iflip, jflip = np.unravel_index(flip_index, (self.nx,self.ny))
            if self.debug:
                print 'iflip, jflip', iflip, jflip
                
            # update the energy
            if self.e[iflip,jflip] == 1:  # the move will be 1->0
                self.energy -= self.activation_energy[ self.n[iflip,jflip] ]
            else: # the move will be 0->1
                self.energy += self.activation_energy[ self.n[iflip,jflip] ]

            # flip the selected cell
            self.e[iflip,jflip] = int(self.e[iflip,jflip]==0)
            
            
            ### update neighbors ###
            ineighbors = self.neighborlist[iflip,jflip,0:4,0]
            jneighbors = self.neighborlist[iflip,jflip,0:4,1]
            
            # update the neighbor counts
            if self.e[iflip,jflip] == 1:  # the flip is 0 --> 1; add +1 to neighbor counts
                self.n[ineighbors, jneighbors] += 1      
            else:  # the flip is 1 --> 0; subtract 1 from neighbor counts
                self.n[ineighbors, jneighbors] -= 1 
            
            # update the rates for the flipped cell...   
            if self.e[iflip,jflip] == 0:
                self.alpha_beta[iflip,jflip] = self.alphas[self.n[iflip, jflip]]
            else:
                self.alpha_beta[iflip,jflip] = self.betas[self.n[iflip, jflip]]
            
            # ... and its neighbors
            neighboring_e = self.e[ineighbors, jneighbors]
            neighboring_n = self.n[ineighbors, jneighbors]
            self.alpha_beta[ineighbors, jneighbors] = \
                    np.where(neighboring_e == 0, self.alphas[neighboring_n], self.betas[neighboring_n])
                        
            # update current time
            time += tau

                    
        return self.t
    
    
