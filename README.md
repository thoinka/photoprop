# Photon Propagation

A simple photon propagation written in Python. 

## Usage

### Propagation and Absorption

To propagate photons use the `Photon` class:

```python
import photoprop as ph
import numpy as np


x0, v0, t0 = ph.util.cascade(x0=[0.0, 0.0], t0=0.0, n_photons=10)
photons = ph.Photon(x0, v0, t0, c_sca=0.05, c_abs=0.01)
photons.propagate(10)
```
This will propagate 10 photons (in 2 dimensions) originating from the origin at the time 0 for 10 steps.

The output is stored in attributed of the `Photon` object:

```python
from matplotlib import pyplot as plt


plt.figure()
plt.scatter(*photons.x.reshape(-1, 2).T, c=photons.t.reshape(-1), s=photons.w.reshape(-1))
plt.show()
```

These photons are not properly absorbed, but instead memorize a weight corresponding to the probability of absorption. To absorb the photons properly, use the corresponding method `absorb`

```
absorption_loc, absorption_time = photons.absorb()
```

### Detection

A detector can be simulated using the `Detector` class.

```python
detector = ph.Detector(np.random.uniform(-10, 10, size=(15, 2)), radius=1.0)
```
This creates a `Detector` object that contains 15 modules at random positions in a 10x10 box. Each of the modules has a radius of 1.

To check whether any of the photons generated before have been detected, use the `detect` method:

```python
hits = detector.detect(photons)
```

### Interpolation Tables

The class `PhotoTable` provides means to histogram and interpolate the photons:

```python
ptable = ph.PhotoTable(photons, assume_symmetry=['phi'])
```
