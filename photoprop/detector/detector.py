import numpy as np
import pandas as pd
from ..util import check_intersect, isotropic
from ..photon import Photon
from tqdm import tqdm
from joblib import Parallel, delayed

class Detector:

    def __init__(self, dom_pos, radius):
        self.ndim = dom_pos.shape[-1]
        self.doms = dom_pos
        self.radius = radius

    def detect(self, photons, n_jobs=1):
        hits = Parallel(n_jobs)([delayed(check_intersect)(photons.x,
                                                          photons.t,
                                                          photons.w,
                                                          self.radius,
                                                          self.doms[i],
                                                          i,
                                                          return_index=True)
                                 for i in range(len(self.doms))])
        multi_index = pd.MultiIndex.from_arrays([
            np.vstack(hits)[:,0].astype(int),
            np.vstack(hits)[:,1].astype(int)], names=['dom', 'photon'])
        cols = ['x{}'.format(i + 1) for i in range(self.ndim)] + ['t', 'w']
        df = pd.DataFrame(np.vstack(hits)[:,2:],
                          index=multi_index,
                          columns=cols)
        visible = df.t - df.w < np.random.exponential(1.0 / photons.c_abs,
                                                      len(df))
        return df[visible]


def retro_prop(photons, df_hits, n_photons_per_hit=1, n_steps=100):
    loc_cols = ['x{}'.format(i + 1) for i in range(photons.ndim)]
    x0 = df_hits[loc_cols].values.repeat(n_photons_per_hit, axis=0)
    t0 = -df_hits['t'].values.repeat(n_photons_per_hit)
    v0 = isotropic(len(df_hits) * n_photons_per_hit, photons.ndim)

    pt_retro = Photon(x0, v0, t0, photons.c_abs, photons.c_sca, photons.c0,
                      photons.dphi_gen)
    pt_retro.propagate(n_steps)
    time_retro, loc_retro = pt_retro.absorb()
    return time_retro, loc_retro