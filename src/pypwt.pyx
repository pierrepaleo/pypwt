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
        int hlen

        # Methods
        # -------
        C_Wavelets()
        C_Wavelets(float*, int, int, const char*, int, int, int, int, int)
#~         ~C_Wavelets()
        void forward()
        void soft_threshold(float, int)
        void hard_threshold(float, int)
        void circshift(int, int, int)
        void inverse()
        float norm2sq()
        float norm1()
        int get_image(float*)
        void print_informations()
        int get_coeff(float*, int)
        void set_image(float*, int)

cdef class Wavelets:

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
    cdef readonly char* wname
    cdef readonly int levels
    cdef readonly int do_cycle_spinning
    cdef int hlen
    cdef readonly int do_swt
    cdef readonly int do_separable
    cdef list _coeffs


    def __cinit__(self,
                    np.ndarray[ndim=2, dtype=np.float32_t] img,
                    str wname,
                    int levels,
                    int do_separable = 1,
                    int do_cycle_spinning = 0,
                    int do_swt = 0,
                  ):
        """
        Class constructor. Initializes the Wavelet transform
        from an image and given parameters.

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

        if img.ndim != 2: # TODO
            raise NotImplementedError("Wavelets(): Only 2D transform is supported for now")

        self.Nr = img.shape[0]
        self.Nc = img.shape[1]
        py_wname = wname.encode("ASCII") # python variable keeps the reference
        cdef char* c_wname = py_wname
        self.wname = c_wname
        self.levels = levels
        self.do_separable = do_separable
        self.do_cycle_spinning = do_cycle_spinning
        self.do_swt = do_swt

        # Build the C++ Wavelet object
        #~ Wavelets(float* img, int Nr, int Nc, const char* wname, int levels, int memisonhost=1, int do_separable = 1, int do_cycle_spinning = 0, int do_swt = 0);
#~         cdef float[:] c_data = img.ravel()
#~         cfunc(&c_data[0]...)

        self.w = new C_Wavelets(<float*> np.PyArray_DATA(img),
                                self.Nr, self.Nc,
                                self.wname, self.levels,
                                1, self.do_separable, self.do_cycle_spinning, self.do_swt)
        # Retrieve the possibly updated attributes after the C++ initialization
        self.levels = self.w.nlevels
        self.hlen = self.w.hlen

        # Initialize the python coefficients
        # [A, [H1, V1, D1], [H2, V2, D2], ... ]
        self._coeffs = []
        Nr2 = self.Nr
        Nc2 = self.Nc
        _factor = 2**self.levels if (self.do_swt == 0) else 1
        self._coeffs.append(np.zeros((Nr2//_factor, Nc2//_factor), dtype=np.float32))
        for i in range(self.levels):
            ahvd = []
            if self.do_swt == 0:
                Nr2 = Nr2//2
                Nc2 = Nc2//2
            for i in range(3):
                ahvd.append(np.zeros((Nr2, Nc2), dtype=np.float32))
            self._coeffs.append(ahvd)


    def info(self):
        self.w.print_informations()


    def coeff_only(self, int num): # TODO : handle a (level, type) syntax ?
        """
        Get only the coeff "num" from the C++ class instance.
        You should use it if you know that you will access only one coeff,
        since this is faster than coeff[level][k]

        num : int
            Number of the coefficient. The indexing is as follows :
            [0: A, 1: H1, 2: V1, 3: D1,  4: H2, ...]
        """
        if num == 0:
            coeff_ref = self._coeffs[0]
        else:
            curr_level = (num-1)//3 +1
            curr_coeff = (num-1)%3
            coeff_ref = self._coeffs[curr_level][curr_coeff]
        c_dstbuf = <float*> np.PyArray_DATA(coeff_ref)
        numc = self.w.get_coeff(c_dstbuf, num)
        if numc != coeff_ref.size:
            raise RuntimeError("Wavelets.coeff_only(): something went wrong when retrieving coefficients numbef %d, expected %d coeffs, got %d" % (num, coeff_ref.size, numc))
        return coeff_ref


    @property
    def coeffs(self):
        """
        Get all the coefficients from the C++ class instance
        Returns the following list :
            [A, [H1, V1, D1], [H2, V2, D2], ...]
        Note: using coeffs[0] is slower than coeff_only(0),
        since calling coeff() copies all the coefficients from the device
        """

        self.coeff_only(0)
        for cnt in range(1, 3*self.levels+1):
            self.coeff_only(cnt)
        return self._coeffs

    # TODO (?) : @coeffs.setter (self._coeffs = ...), @coeffs.deleter (del self._coeffs)


    def forward(self, np.ndarray[ndim=2, dtype=np.float32_t] img = None):
        if img is not None:
            if img.shape[0] != self.Nr or img.shape[1] != self.Nc:
                raise ValueError("Wavelets.forward(): provided image does not match the geometry (expected %s, got %s)" % (str(tuple(self.Nr, self.Nc)), str(tuple(img.shape[0], img.shape[1]))))
            self.w.set_image(<float*> np.PyArray_DATA(img), 0)
        self.w.forward()


    def soft_threshold(self, float beta):
        cdef float c_beta = beta
        self.w.soft_threshold()



#~         py_dstbuf = np.zeros((Nr//2**levels, Nc//2**levels), dtype=np.float32)
#~         c_dstbuf = <float*> np.PyArray_DATA(py_dstbuf)
#~         print(self.w.get_coeff(c_dstbuf, 0))
#~         import matplotlib.pyplot as plt
#~         plt.figure()
#~         plt.imshow(py_dstbuf, interpolation="nearest"); plt.colorbar()
#~         plt.show()


#~         self.dim1 = len(arr)
#~         self.g = new C_GPUAdder(&arr[0], self.dim1)


#~     def retreive(self):
#~         cdef np.ndarray[ndim=1, dtype=np.int32_t] a = np.zeros(self.dim1, dtype=np.int32)
#~         self.g.retreive_to(&a[0], self.dim1)

#~         return a
