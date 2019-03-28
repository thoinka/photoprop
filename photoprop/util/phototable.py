import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import RectBivariateSpline as RBS, RegularGridInterpolator as RGI


class PhotoTable:
    '''Interpolation table for photons.

    Attributes
    ----------
    ndim : int
        Effective number of dimensions of the table. Takes assumed symmetries
        into account.
    bins : list(numpy array), len=ndim
        Bin edges for all dimensions.
    signature : str
        Signature of the table, e.g. `f(t,r,phi)`.
    H : numpy array, shape=(nbins) * ndim
        Underlying histogram
    self.t : list(numpy array)
        Underlying points corresponding to H.
    '''
    def __init__(self, photons, assume_symmetry=[], binning='auto'):
        self.ndim = 1 + photons.ndim - len(assume_symmetry)
        if binning == 'auto':
            binning = np.linspace(0.0, 1.0, int((len(photons) * photons.n_steps) ** (1.0 / (self.ndim + 1.0))) + 1)
        if type(binning) == int:
            binning = np.linspace(0.0, 1.0, binning)
        t = photons.t.flatten()
        x = photons.x.reshape(-1, photons.ndim)
        r = np.linalg.norm(x, axis=-1)
        vals = [t, r]
        self.bins = [binning * np.max(t), binning * np.max(r)]
        self.signature = 'f(t,r'

        if photons.ndim == 2:
            if 'phi' not in assume_symmetry:
                phi = np.arccos(x[:, 0] / r)
                vals.append(phi)
                self.bins.append(binning * 2.0 * np.pi)
                self.signature += ',phi'

        elif photons.ndim == 3:
            if 'theta' not in assume_symmetry:
                theta = np.arccos(x[:, 2] / r)
                vals.append(theta)
                self.bins.append(binning * np.pi)
                self.signature += ',theta'

            if 'phi' not in assume_symmetry:
                phi = np.arctan(x[:, 1], x[:, 0])
                vals.append(phi)
                self.bins.append(2.0 * np.pi * binning - np.pi)
                self.signature += ',phi'

        self.signature += ')'
        self.H, _ = np.histogramdd(vals, self.bins)
        self.t = [(b[1:] + b[:-1]) * 0.5 for b in self.bins]
        if self.ndim == 2:
            self.interpolator = RBS(*self.t, np.log(self.H + 1e-8),
                                    bbox=[0.0, np.max(t), 0.0, np.max(r)],
                                    kx=1, ky=1)
        else:
            self.interpolator_ = RGI(self.t, self.H, bounds_error=False,
                                     fill_value=0.0)
            self.interpolator = lambda *x: self.interpolator_(x)
    
    def __call__(self, *xi, **kwargs):
        output = self.interpolator(*xi, **kwargs)
        within_bounds = np.all([(x > b[0]) & (x < b[-1]) for x, b in zip(xi, self.bins)], axis=0)
        output[~within_bounds] = -20.0
        return output