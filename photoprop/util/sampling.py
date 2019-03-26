import numpy as np


def isotropic(n_samples, n_dim):
    v = np.random.randn(n_samples, n_dim)
    return v / np.linalg.norm(v, axis=-1).reshape(-1, 1)


def track(x_start, x_end, t_start, n_photons, n_losses=-1, speed=1.0):
    x_start = np.array(x_start).reshape(-1, 1)
    x_end = np.array(x_end).reshape(-1, 1)
    ndim = x_start.shape[0]

    if n_losses < 0:
        t_vec = np.random.uniform(0.0, 1.0, n_photons)
        x0 = x_start + (x_end - x_start) * t_vec
    else:
        R = np.sort(np.random.randint(1, n_photons, size=n_losses - 1))
        losses = np.diff(np.r_[0, R, n_photons]).astype(int)
        t_loss = np.random.uniform(0.0, 1.0, n_losses)
        x0_loss = x_start + (x_end - x_start) * t_loss 
        x0 = np.vstack([x] * l for x, l in zip(x0_loss.T, losses)).T
    t0 = np.linalg.norm(x0 - x_start, axis=0) / speed + t_start
    v0 = isotropic(n_photons, ndim)
    return x0.T, v0, t0


def cascade(x0, t0, n_photons):
    x0 = np.array(x0).reshape(1, -1).repeat(n_photons, axis=0)
    t0 = np.array(t0).repeat(n_photons)
    v0 = isotropic(n_photons, x0.shape[-1])
    return x0, v0, t0