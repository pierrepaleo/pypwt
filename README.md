## pycudwt

pycudwt is a python module for parallel Discrete Wavelet Transform.
This is a wrapper of [PDWT](https://github.com/pierrepaleo/PDWT).

**Note:** this project was formerly named `pypwt`.
It has been renamed `pycudwt` to have a spot on [pypi](https://pypi.org/project/pycudwt).

## Installation

### Requirements

You need cython and nvcc (the Nvidia CUDA compiler, available in the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)).

For the tests, you need pywavelets. `python-pywt` is packaged for Debian-like distributions, more recent changes are available on [the new repository](https://github.com/PyWavelets/pywt).

### Stable version (from pypi)

```bash
pip install pycudwt
```

### Development version (from github)

```bash
git clone https://github.com/pierrepaleo/pypwt
cd pypwt
pip install .
```

You can specify the compute capability when building the library:  
```bash
PYCUDWT_CC=86 pip install .
```

### Testing

If `pywt` is available, you can check if pycudwt gives consistent results :

```bash
cd test
python test_all.py
```

the results are stored in `results.log`.


## Getting started

Computing a Wavelet Transform wity pycudwt is simple. In `ipython`:

```python
from pycudwt import Wavelets
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


