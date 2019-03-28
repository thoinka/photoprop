# Photon Propagation

A simple photon propagation written in Python. 

## Usage

To propagate photons use the `Photon` class:

```python
import photoprop as ph


x0, v0, t0 = ph.util.cascade(x0=[0.0, 0.0], t0=0.0, n_photons=10)
photon = ph.Photon(x0, v0, t0, c_sca=0.05, c_abs=0.01)
photon.propagate(10)
```
This will propagate 10 photons (in 2 dimensions) originating from the origin at the time 0 for 10 steps.

The output is stored in attributed of the `Photon`-object:

```python
from matplotlib import pyplot as plt


plt.figure()
plt.scatter(*photon.x.reshape(-1, 2).T, c=photon.t.reshape(-1), s=photon.w.reshape(-1))
plt.show()
```

These photons are not properly absorbed, but instead memorize a weight corresponding to the probability of absorption. To absorb the photons properly, use the corresponding method `absorb`

```
absorption_loc, absorption_time = photon.absorb()


