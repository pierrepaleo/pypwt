import numpy as np
cimport numpy as np
assert sizeof(int) == sizeof(np.int32_t)


cdef extern from "../ppdwt/wt.h":
    cdef cppclass C_Wavelets "Wavelets":

        # C++ attributes should be declared here if we want to access them
        # -----------------------------------------------------------------
        int Nr
        int Nc
        int nlevels
        int do_cycle_spinning
        int do_separable
        int hlen

        # Methods
        # -------
        C_Wavelets()
        # Wavelets(float* img, int Nr, int Nc, const char* wname, int levels, int memisonhost=1, int do_separable = 1, int do_cycle_spinning = 0, int do_swt = 0, int ndim = 2);
        C_Wavelets(float*, int, int, const char*, int, int, int, int, int, int)
#~         ~C_Wavelets()
        void forward()
        void soft_threshold(float, int)
        void hard_threshold(float, int)
        void shrink(float, int)
        void circshift(int, int, int)
        void inverse()
        float norm2sq()
        float norm1()
        int get_image(float*)
        void print_informations()
        int get_coeff(float*, int)
        void set_image(float*, int)

cdef class Wavelets:
    """
    Initializes the Wavelet transform from an image and given parameters.

    img: 2D numpy.ndarray, float32
        Input image
    wname: string
        Name of the wavelet
    levels: int
        Number of decomposition levels
    do_separable: int
        if not 0, perform a separable transform
    do_cycle_spinning: int
        if not 0, perform a random shift on the image
        (useful for iterative algorithms)
    do_swt: int
        if not 0, perform a Stationary (non-decimated) wavelet transform
    """
    # Attributes in cdef classes behave differently from attributes in regular classes:
    #   - All attributes must be pre-declared at compile-time
    #   - Attributes are by default only accessible from Cython (typed access)
    #   - Properties can be declared to expose dynamic attributes to Python-space
    # The attributes may be Python objects (either generic or of a particular extension type),
    # or they may be of any C data type. So you can use extension types to wrap arbitrary
    # C data structures and provide a Python-like interface to them.
    # By default, extension type attributes are only accessible by direct access,
    # not Python access, which means that they are not accessible from Python code.
    # To make them accessible from Python code, you need to declare them as public or readonly.

    cdef C_Wavelets* w # pointer to the C Wavelet object
    cdef readonly int Nr
    cdef readonly int Nc
    cdef readonly char* _wname
    cdef readonly str wname
    cdef readonly int levels
    cdef readonly int do_cycle_spinning
    cdef int hlen
    cdef readonly int do_swt
    cdef readonly int do_separable
    cdef readonly int ndim
    cdef list _coeffs
    cdef tuple shape
    cdef readonly int batched1d


    def __cinit__(self,
                    np.ndarray img,
                    str wname,
                    int levels,
                    int do_separable = 1,
                    int do_cycle_spinning = 0,
                    int do_swt = 0,
                    int ndim = 2
                  ):

        img = self._checkarray(img)

        # We can have ndim != img.ndim, which means batched 1D transform
        self.batched1d = 0
        if img.ndim == 2:
            self.Nr = img.shape[0]
            self.Nc = img.shape[1]
            if (img.ndim != ndim): self.batched1d = 1
        elif img.ndim == 1: # For 1D, make Nr = 1 and Nc = img.shape[0] for a contiguous C array
            self.Nr = 1
            self.Nc = img.shape[0]
        else:
            raise NotImplementedError("Wavelets(): Only 1D and 2D transforms are supported for now")

        # for ND
        shp = []
        for i in range(img.ndim):
            shp.append(img.shape[i])
        self.shape = tuple(shp)
        self.wname = wname

        py_wname = wname.encode("ASCII") # python variable keeps the reference
        cdef char* c_wname = py_wname
        self._wname = c_wname
        self.levels = levels
        self.do_separable = do_separable
        self.do_cycle_spinning = do_cycle_spinning
        self.do_swt = do_swt
        self.ndim = img.ndim

        # Build the C++ Wavelet object
        self.w = new C_Wavelets(<float*> np.PyArray_DATA(img),
                                self.Nr, self.Nc,
                                self._wname, self.levels,
                                1, self.do_separable, self.do_cycle_spinning, self.do_swt, ndim)
        # Retrieve the possibly updated attributes after the C++ initialization
        self.levels = self.w.nlevels
        self.hlen = self.w.hlen
        self.do_separable = self.w.do_separable

        # Initialize the python coefficients
        # for 2D : [A, [H1, V1, D1], [H2, V2, D2], ... ]
        # for 1D : [A, D1, ... Dn]
        self._coeffs = []
        Nr2 = self.Nr
        Nc2 = self.Nc
        _factor = 2**self.levels if (self.do_swt == 0) else 1
        if (self.ndim == 2) and not(self.batched1d): # 2D
            self._coeffs.append(np.zeros((Nr2//_factor, Nc2//_factor), dtype=np.float32))
            for i in range(self.levels):
                ahvd = []
                if self.do_swt == 0:
                    if not(self.batched1d): Nr2 = Nr2//2
                    Nc2 = Nc2//2
                for i in range(3):
                    ahvd.append(np.zeros((Nr2, Nc2), dtype=np.float32))
                self._coeffs.append(ahvd)
        else: # (batched) 1D
            _shp = (Nc2//_factor, ) if self.ndim == 1 else (Nr2, Nc2//_factor)
            self._coeffs.append(np.zeros(_shp, dtype=np.float32))
            for i in range(self.levels):
                if self.do_swt == 0: Nc2 = Nc2//2
                _shp = (Nc2,) if self.ndim == 1 else (Nr2, Nc2)
                self._coeffs.append(np.zeros(_shp, dtype=np.float32))


    def info(self):
        """
        Print some information on the current ``Wavelets`` instance.
        """
        self.w.print_informations()

    def __repr__(self):
        self.info()
        return ""

    def __str__(self):
        self.info()
        return ""


    @staticmethod
    def _checkarray(arr, shp=None):
        res = arr
        if arr.dtype != np.float32 or not(arr.flags["C_CONTIGUOUS"]):
            res = np.ascontiguousarray(arr, dtype=np.float32)
        if shp is not None:
            if arr.ndim != len(shp):
                raise ValueError("Invalid number of dimensions (expected %d, got %d)" % (len(shp), arr.ndim))
            for i in range(arr.ndim):
                if arr.shape[i] != shp[i]:
                    raise ValueError("The image does not have the correct shape (expected %s, got %s)" % (str(shp), str(arr.shape)))
        return res


    def coeff_only(self, int num):
        """
        Get only the coeff "num" from the C++ class instance.
        You should use it if you know that you will access only one coeff,
        since this is faster than coeff[level][k]

        num : int
            Number of the coefficient. The indexing is as follows :
            2D : [0: A, 1: H1, 2: V1, 3: D1,  4: H2, ...]
            1D : [0: A, 1: D1, 2: D2, ...]
        """
        if num == 0:
            coeff_ref = self._coeffs[0]
        else:
            if (self.ndim == 2) and not(self.batched1d):
                curr_level = (num-1)//3 +1
                curr_coeff = (num-1)%3
                coeff_ref = self._coeffs[curr_level][curr_coeff]
            else: # (batched) 1D
                coeff_ref = self._coeffs[num]

        c_dstbuf = <float*> np.PyArray_DATA(coeff_ref)
        numc = self.w.get_coeff(c_dstbuf, num)
        if numc != coeff_ref.size:
            raise RuntimeError("Wavelets.coeff_only(): something went wrong when retrieving coefficients numbef %d, expected %d coeffs, got %d" % (num, coeff_ref.size, numc))
        return coeff_ref


    @property
    def coeffs(self):
        """
        Get all the coefficients from the C++ class instance
        Returns the following list : [A, [H1, V1, D1], [H2, V2, D2], ...]
        Note: using coeffs[0] is slower than coeff_only(0),
        since calling coeff() copies all the coefficients from the device
        """

        self.coeff_only(0)
        if self.ndim == 2 and not(self.batched1d):
            i_end = 3*self.levels
        else: # 1D
            i_end = self.levels
        for cnt in range(1, i_end + 1):
            self.coeff_only(cnt)
        return self._coeffs


    @property
    def image(self):
        res = np.zeros((self.Nr, self.Nc), dtype=np.float32)
        c_dstbuf = <float*> np.PyArray_DATA(res)
        numc = self.w.get_image(c_dstbuf)
        if numc != res.size:
            raise RuntimeError("Wavelets.image(): something went wrong when retrieving image, expected %d coeffs, got %d" % (res.size, numc))
        return res


#~     @image.setter # Not working in cython (?)
    def set_image(self, img):
        """
        Modifies the image of the wavelet class.

        img: numpy.ndarray
            Provided image. The dimensions have to be consistent with the current ``Wavelets`` instance.

        **Note**: it does not update the coefficients. You have to perform ``Wavelets.forward()`` to update the coefficients.

        """
        img = self._checkarray(img, (self.Nr, self.Nc))
        self.w.set_image(<float*> np.PyArray_DATA(img), 0)


    def forward(self, img = None):
        """
        Performs the foward wavelet transform with the current configuration.

        img: numpy.ndarray
            Optionnal. If an image is provided, the transform is performed on this image.
            Otherwise, the transform is performed on ``Wavelets.image``
        """
        if img is not None:
            img = self._checkarray(img, self.shape)
            self.w.set_image(<float*> np.PyArray_DATA(img), 0)
        self.w.forward()


    def inverse(self):
        """
        Invert the wavelet transform with the current configuration.
        It Transforms back the coefficients ``Wavelets.coeffs`` to an image.

        **Note**: The inverse function modifies the coefficients.
        This means that performing ``Wavelets.inverse()`` twice leads to an inconsistent result.
        The underlying library prevents any further usage of ``Wavelets.coeffs`` (including ``Wavelets.inverse()``)
        once ``Wavelets.inverse()`` has been performed once. This mechanism is reset as soon as ``Wavelets.forward()``
        is performed.
        """
        self.w.inverse()



    def soft_threshold(self, float beta, int do_threshold_appcoeffs = 1):
        """
        Soft threshold the wavelets coefficients.
        The soft thresholding is defined by

        .. math::

            \\text{ST}(x, t) = (|x| - t)_+ \\cdot \\text{sign}(x)

        This is the proximal operator of beta * L1 norm.

        beta: float
            threshold factor
        do_threshold_appcoeffs : int
            if not 0, the approximation coefficients will also be thresholded
        """
        cdef float c_beta = beta
        cdef int c_dt = do_threshold_appcoeffs
        self.w.soft_threshold(c_beta, c_dt)


    def hard_threshold(self, float beta, int do_threshold_appcoeffs = 1):
        """
        Hard threshold the wavelets coefficients.
        The hard thresholding is defined by

        .. math::

            \\text{HT}(x, t) = x \\cdot 1_{|x| > t}

        beta: float
            threshold factor
        do_threshold_appcoeffs : int
            if not 0, the approximation coefficients will also be thresholded
        """
        cdef float c_beta = beta
        cdef int c_dt = do_threshold_appcoeffs
        self.w.hard_threshold(c_beta, c_dt)


    def shrink(self, float beta, int do_threshold_appcoeffs = 1):
        """
        Shrink the wavelets coefficients.
        The shrink is defined by

        .. math::

            \\text{shrink}(x, t) = \\frac{x}{1+t}

        This is the proximal operator of beta * L2 norm.

        beta: float
            shrink factor
        do_threshold_appcoeffs : int
            if not 0, the approximation coefficients will also be shrunk.
        """
        cdef float c_beta = beta
        cdef int c_dt = do_threshold_appcoeffs
        self.w.shrink(c_beta, c_dt)


    def norm1(self):
        """
        Returns the L1 norm of the Wavelets coefficients :

        .. math::

            \\left\\| w \\right\\|_1 = \\sum_i |w_i|

        """
        return self.w.norm1()


    def norm2sq(self):
        """
        Returns the squared L2 norm of the Wavelets coefficients :

            .. math::

                \\left\\| w \\right\\|_2^2 = \\sum_i |w_i|^2

        """
        return self.w.norm2sq()

