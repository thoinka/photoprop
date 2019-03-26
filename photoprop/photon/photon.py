import numpy as np
from scipy.interpolate import interp1d
from .scattering import henyey_greenstein, scattering2d, scattering3d


class Photon:
    '''
    Photon. Note: Only works for 2 and 3 dimensions.

    Attributes
    ----------
    x : numpy array, shape=(n_photons, n_steps, n_dim)
    v : numpy array, shape=(n_photons, n_steps, n_dim)
    t : numpy array, shape=(n_photons, n_steps)
    w : numpy array, shape=(n_photons, n_steps)
    '''

    def __init__(self, x0, v0, t0, c_sca, c_abs, c0=1.0,
                 dphi_gen=henyey_greenstein):
        '''
        Parameters
        ----------
        x0 : numpy array, shape=(n_photons, n_dim)
            Initial positions.
        v0 : numpy array, shape=(n_photons, n_dim)
            Initial directions (have to be normalized!).
        t0 : numpy array, shape=(n_photons)
            Initial times.
        c_sca : float > 0
            Scattering coefficient.
        c_abs: float > 0
            Absorption coefficient.
        c0 : float > 0
            Speed of Light.
        d_phi_gen : callable
            Generator of angular displacement at each scattering.

        Notes
        -----
        Speed of light is always 1.
        '''
        self.x = [x0]
        self.v = [v0]
        self.v[0] /= np.linalg.norm(self.v[0], axis=1).reshape(-1,1)
        self.t = [t0]
        self.ndim = x0.shape[1]
        if self.ndim == 2:
            self.scatter = scattering2d
        elif self.ndim == 3:
            self.scatter = scattering3d
        else:
            raise ValueError('Number of dimensions must be either 2 or 3.')
        self.c0 = c0
        self.c_sca = c_sca
        self.c_abs = c_abs
        self.n_photons = len(x0)
        self.n_steps = 0
        self.dphi_gen = dphi_gen


    def propagate(self, n_steps=2):
        '''
        Parameters
        ----------
        n_steps : int
            Number of propagation steps
        '''
        for step in range(n_steps - 1):
            s_ = np.random.exponential(1.0 / self.c_sca, (self.n_photons, 1))
            self.x.append(self.x[-1] + s_ * self.v[-1])
            phi = self.dphi_gen(self.n_photons)
            self.v.append(self.scatter(self.v[-1], phi))
            self.t.append(self.t[-1] + s_.flatten())
        self.v = np.array(self.v).swapaxes(0, 1)
        self.x = np.array(self.x).swapaxes(0, 1)
        self.t = np.array(self.t).swapaxes(0, 1)
        self.w = np.exp(-self.c_abs * (self.t - self.t[:,0].reshape(-1,1)))
        self.n_steps += n_steps

    def __len__(self):
        return len(self.x)    
    
    def absorb(self):
        L = np.random.exponential(1.0 / self.c_abs, self.__len__()) + self.t[:, 0]
        return L, np.array([interp1d(self.t[i,:], self.x[i,:,:], axis=0, bounds_error=False)(L[i])
                            for i in range(self.__len__())])