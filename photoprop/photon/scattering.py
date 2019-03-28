import numpy as np


class ScatteringModelBase:
    __gen__ = None
    
    def __init__(self, *args, **kwargs):
        self.gen_args = args
        self.gen_kwargs = kwargs

    def __call__(self, n_samples):
        return self.__gen__(n_samples, *self.gen_args, **self.gen_kwargs)


class SimpleGaussian(ScatteringModelBase):
    def __gen__(self, n_samples, sigma=0.01):
        '''Generates scattering angles according to some super simple Gaussian
        model'''
        return np.random.randn(n_samples) * sigma


class HenyeyGreenstein(ScatteringModelBase):
    def __gen__(self, n_samples, g=0.94):
        '''Generates scattering angles according to the Henyey Greenstein
        approximation. From
        https://www.astro.umd.edu/~jph/HG_note.pdf
        '''
        r = np.random.uniform(-1.0, 1.0, n_samples)
        cosphi = (1 + g ** 2 - ((1.0 - g **2) / (1.0 + g * r)) ** 2) / 2.0 / g
        return np.arccos(cosphi)
        

def rotmat2d(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s],
                     [s,  c]])


def scattering2d(v, phi):
    '''Rotates randomly in two dimensions around scattering angle phi.
    '''
    rot = rotmat2d(phi).T
    return np.sum((rot * v.reshape(-1,2,1)), axis=1)


def rotmat3d(phi, ux, uy, uz):
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([
            [c + ux ** 2 * (1.0 - c),
             ux * uy * (1.0 - c) - uz * s,
             ux * uz * (1.0 - c) + uy * s],
            [ux * uy * (1.0 - c) - uz * s,
             c + uy ** 2 * (1.0 - c),
             uy * uz * (1.0 - c) - ux * s],
            [ux * uz * (1.0 - c) + uy * s,
             uy * uz * (1.0 - c) - ux * s,
             c + uz ** 2 * (1.0 - c)]
        ])


def scattering3d(v, phi):
    out = np.empty_like(v)
    u_inter = np.random.randn(len(phi), 3)
    u1 = np.cross(v, u_inter)
    u2 = np.cross(v, u1)
    r = np.random.rand(len(phi), 1)
    u = r * u1 + (1.0 - r) * u2
    u /= np.linalg.norm(u, axis=-1).reshape(-1,1)
    rot = rotmat3d(phi, *u.T).T
    return  np.sum((rot * v.reshape(-1,3,1)), axis=1)
