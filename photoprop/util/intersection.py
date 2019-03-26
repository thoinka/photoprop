import numpy as np


def check_intersect(paths, times, weights, r, m, idx=0, return_index=False):
    '''Checks whether a path intersects with a sphere at `m` with radius `r`.
    
    Parameters
    ----------
    paths : numpy.array, shape=(n_paths, n_points, n_dim)
        Points of the paths.
    r : float
        Radius of the sphere.
    m : numpy.array, shape=(n_dim)
        Center of the sphere.
    return_index : bool, default=False
        Whether or not to return the indices of the part of the path
        that intersects with the sphere.
    
    Returns
    -------
    result : bool
        Whether or not at least one part of the path intersects with
        the circle. _Note_: Only returned when `return_index` is False!
    indices : numpy.array, shape=n_points, dtype=int
        Indices of the path segments that intersect with the sphere.
    
    '''
    w1_shift = paths[:,:-1] - m.reshape(1, -1)
    w2_shift = paths[:,1:] - m.reshape(1, -1)
    
    v = w2_shift - w1_shift
    L = np.linalg.norm(v, axis=-1)
    v = v / L.reshape(*L.shape, 1)
    d = np.sum(v * w1_shift, axis=-1)
    delta = d ** 2 - np.sum(w1_shift ** 2, axis=-1) + r ** 2
    cond1 = delta > 0
    t = -d + np.sqrt(np.clip(delta, 0, np.inf))
    t2 = -d - np.sqrt(np.clip(delta, 0, np.inf))
    negative = t < 0
    larger_L = t > L
    t[negative | larger_L] = t2[negative | larger_L]
    t[~cond1] = np.max(L) + 1
    if return_index:
        ma, mb = np.meshgrid(np.arange(paths.shape[0]),
                             np.arange(paths.shape[1] - 1),
                             indexing='ij')
        hits = cond1 & (t <= L) & (t > 0)
        m_rep = np.tile(m, (np.sum(hits), 1))
        return np.column_stack((np.full(np.sum(hits), idx), ma[hits],
                                m_rep, times[ma[hits], mb[hits]],
                                times[ma[hits], 0]))
    
    return (cond1 & (t <= L) & (t > 0)).any(axis=-1)