import numpy as np
import os, sys

class IsingTrajectory(object):
    """An object to hold kinetic Monte Carlo trajectories (non-discrete time jumps), 
    with methods to convert to discrete-time MSM trajectories."""
    
    def __init__(self, nframes=None, nx=20, ny=20):
        """Initialize the trajectory object.
        PARAMETERS
        nframes    - if None, then do not allocate arrays
                   - if supplied, allocate arrays of this length"""
        
        self.nframes = nframes
        self.nx = nx
        self.ny = ny
        
        self.stats_header = None
        self.stats = None
        
        self.traj_header = None
        self.traj = None
        
        if (nframes != None) and (nx != None) and (ny != None):
            self.allocate_frames(nframes, nx, ny)
        
    def allocate_frames(self, nframes, nx, ny):
        """Allocate a trajectory arrays of the right size"""

        import numpy as np
        
        self.nframes = nframes
        self.nx = nx
        self.ny = ny
        
        self.stats_header = '%d %d %d\ntime(s)\tdwell_time(s)\tenergy(meV)\tfwd_rate(Hz)\tback_rate(Hz)'%(self.nframes, self.nx, self.ny)
        self.stats = np.zeros( (nframes, 5), dtype=float)  
        
        self.traj_header = '%d frames of (%d x %d = %d cells)'%(nframes, nx, ny, nx*ny)
        self.traj = np.zeros( (nframes, nx*ny), dtype=bool)
        
        self.current_frame = 0

    def tally(self, state, time, dwell_time, energy, fwd_rate, back_rate):
        """Write a frame."""
        
        self.stats[self.current_frame,:] = [time, dwell_time, energy, fwd_rate, back_rate]
        self.traj[self.current_frame,:] = state.ravel().astype(bool)
        self.current_frame += 1
        
    
    def save(self, name):
        """Save <name>.stats.npy and <name>.traj.npy to file"""
        
        outfile_stats = name + '.stats.dat'
        np.savetxt(outfile_stats, self.stats, header=self.stats_header, fmt='%10.8e')
        print 'Wrote', outfile_stats
        
        outfile_traj = name + '.traj.npy'
        np.save(outfile_traj, self.traj)
        print 'Wrote', outfile_traj

        
    def load(self, name):
        """Load <name>.stats.npy and <inname>.traj.npy"""
        
        # parse filenames
        infile_stats = name + '.stats.dat'   
        infile_traj = name + '.traj.npy'
        
        # get header info
        fin = open(infile_stats, 'r')
        header_lines = [fin.readline() for i in range(2)]
        #print header_lines
        fin.close()
        self.stats_header = header_lines[0] + header_lines[1]
        #print 'self.stats_header', self.stats_header  
        
        # get array sizes and allocate arrays
        fields = header_lines[0].strip('#').split()
        nframes, nx, ny = [int(fields[i]) for i in [0,1,2]]
        #print 'nframes, nx, ny', nframes, nx, ny
        self.allocate_frames(nframes, nx, ny)
        print 'Read', infile_stats
        
        # read in stats
        self.stats = np.loadtxt(infile_stats)
        
        # read in trajectory data   
        self.traj = np.load(infile_traj).astype(int)
        print 'Read', infile_traj
        
        self.nframes = self.traj.shape[0]
        self.current_frame = self.traj.shape[0]
        

        
        
    def discretize(self, outname, dt=1.0e-7, debug=False):
        """Discretize the kinetic MC trajectory to a discrete-time trajectory
        amenable to Markov State Model analysis.
        
        INPUT
        outname    - save output to <outname>.msmtraj.npy 
        
        PARAMETERS
        dt         - the discrete time interval, in seconds.  (Default: 1.0e-7 s = 100 ns)
        """

        discrete_snaps = []
        discrete_times = []
        
        i = 0
        timestep = 0   # number of steps dt
        while i < (self.nframes-1):
            
            while timestep*dt < self.stats[i,0]:
                if debug:
                    print timestep, timestep*dt, self.stats[i,0], self.traj[i,0:30],'...'
                discrete_snaps.append( self.traj[i,:] )
                discrete_times.append( timestep)
                timestep += 1            
            i += 1

        # write this new trajectory to file
        msm_traj = np.array(discrete_snaps).astype(bool)
        outfile = outname+'.msmtraj.npy'
        np.save(outfile, msm_traj)
        print 'Wrote:', outfile
        


