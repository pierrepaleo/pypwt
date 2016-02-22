## pypwt

pypwt is a python module for parallel Discrete Wavelet Transform.
This is a wrapper of [PDWT](https://github.com/pierrepaleo/PDWT).


## Features

* Pythonic interface providing the full potential of [PDWT](https://github.com/pierrepaleo/PDWT)
* Compatible with Python >=2.7 and Python >=3.4
* Test suite
* Documentation and examples


## Installation

### Requirements

You need cython and nvcc (the Nvidia CUDA compiler, available in the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)).

For the tests, you need pywavelets. `python-pywt` is packaged for Debian-like distributions, more recent changes are available on [the new repository](https://github.com/PyWavelets/pywt).

### Compiling

Running

```python
python setup.py install --user
```

should build and install the module. For python3, just replace `python` with `python3`.


### Testing

If `pywt` is available, you can check if pypwt gives consistent results :

```bash
cd test
python test_all.py
```

the results are stored in `results.log`.


## Getting started

Computing a Wavelet Transform wity pypwt is simple. In `ipython`:

```python
from pypwt import Wavelets
from scipy.misc import lena
l = lena()
W = Wavelets(l, "db2", 3)
W
------------- Wavelet transform infos ------------
Wavelet name : db2
Number of levels : 3
Stationary WT : no
Cycle spinning : no
Separable transform : yes
Estimated memory footprint : 5.2 MB
Running on device : GeForce GTX TITAN X
--------------------------------------------------
W.forward()
W.soft_threshold(10)
W.inverse()
imshow(W.image)
```


