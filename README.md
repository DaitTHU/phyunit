# PhyUnit

PhyUnit is a Python package for physical units and quantities.

## Installation

PhyUnit has been uploaded as a package in [pypi: phyunit](https://pypi.org/project/phyunit/).
You can directly install it using `pip`.

```shell
pip install phyunit
```

## Quickstart: sub-package `phyunit.SI`

Sub-package `phyunit.SI` implements **SI unit** definitions, 
from which you can import many very useful classes: 

|class|content|
|:-|:-|
|`phyunit.SI.SI`|physical constants in SI unit|
|`phyunit.SI.si`|SI units|
|`phyunit.SI.prefix`|SI unit prefixes|
|`phyunit.SI.particle`|constants in particle physics|
|`phyunit.SI.conversion.au`|atomic unit value in SI|
|`phyunit.SI.conversion.nu`|natural unit value in SI|
|`phyunit.SI.conversion.Planck`|Planck unit value in SI|

```python
>>> from phyunit.SI import SI, si
```

### `SI`: SI physical constant class

class `SI` provide physical constants defined by (or derived from) SI unit system.

Like speed of light _c_, Planck constant _h_, electron mass _me_...
directly use them.

```python
>>> from phyunit.SI import SI

>>> SI.c
Constant(299792458, m/s)

>>> print(SI.me)
9.1093837139e-31 kg

>>> print(SI.me * SI.c**2)
8.18710578796845e-14 kg·m²/s²
```

`particle` class contains constants in particle physics, it is separated from `SI` class for symbol and namespace clarity.

```python
>>> from phyunit.SI import particle

>>> print(particle.mtau)
3.16754e-27 kg
```

### `si`: SI unit class

class `si` provides common SI units,
like meter _m_, second _s_,
and units with prefix like centimeter _cm_.

```python
>>> from phyunit.SI import si

>>> print(1 * si.m / si.s)
1 m/s

>>> print((760 * si.mmHg).to(si.Pa))
101325.0 Pa
```

### `prefix`: unit prefix factor

class `prefix` contains prefix from _quetta-_ (_Q-_, = 10^30) to _quecto-_ (_q-_, = 10^-30), and byte prefix like _ki-_ (2^10 = 1024), _Mi-_ (2^20), _Gi-_ (2^30)... It's just a number factor, not Quantity.

```python
>>> from phyunit.SI import prefix

>>> prefix.mega
1000000.0

>>> prefix.Pi  # 2**50
1125899906842624
```

### `conversion` module: `au`, `nu` and `Planck`

Convert values in other unit system to SI unit,
i.e. `au.length` is the atomic unit (a.u.) of length in SI unit.

```python
>>> from phyunit.SI.conversion import au

>>> print(1 * au.length)  # 1 Bohr radius
5.291772105437146e-11 m
```

## Tutorial: Define `phyunit.Quantity`

Import class `phyunit.Quantity` to define a quantity object with a certain value and unit:

```python
>>> from phyunit import Quantity

>>> F = Quantity(1, 'kg.m/s2')
```

where _F_ is a `Quantity` object, and it has properties:

```python
>>> F.value
1

>>> F.unit
Unit('kg·m/s²')

>>> F.dimension
Dimension(T=-2, L=1, M=1, I=0, Theta=0, N=0, J=0)
```

## Using with `numpy`

`phyunit` is compatible with `numpy`.
You can directly operate on `numpy.ndarray` with units.

```python
>>> import numpy as np

>>> length = np.array([1, 2, 3]) * si.m
>>> print(length)
[1 2 3] m

>>> print(length**2)
[1 4 9] m²

>>> print(np.sum(lengths))
6 m
```

### Example 

`phyunit` is also compatible with `matplotlib`.

Note that `phyunit.Quantity` as a parameter in `pyplot.plot()` is actually its `value` property.

```python
import numpy as np
from matplotlib import pyplot as plt
from phyunit.SI import SI, si

lam = np.linspace(0.01, 5, 100)[:, None] * si.um  # wavelength
T = np.array([3000, 4000, 5000])[None, :] * si.K  # temperature

nu = SI.c / lam  # frequency
I = 2 * SI.h * SI.c**2 / (lam**5 * (np.exp(SI.h * nu / (SI.kB * T)) - 1))  # intensity

plt.plot(lam, I)  # same as `plt.plot(lam.value, I.value)`
plt.xlabel('Wavelength [um]')
plt.ylabel('Intensity [W/m^3/sr]')
plt.title('Blackbody Radiation')
plt.show()
```
