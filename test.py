import numpy as np
from pypwt import Wavelets
from scipy.misc import lena
from time import time
import matplotlib.pyplot as plt

def ims(img):
	plt.figure()
	plt.imshow(img, interpolation="nearest")
	plt.colorbar()
	plt.show()


if __name__ == '__main__':

	l = lena().astype(np.float32)
	#W = Wavelets(lena(), "db3", 2) # 0.27 ms
	W = Wavelets(l, "db3", 2)
	# W.info()
	t0 = time()
	W.forward()
	el = (time()-t0)*1e3
	print("pypwt took %.3f ms" % (el))
	#W.get_all_coeffs()

	import pywt
	t0 = time()
	Wpy = pywt.wavedec2(l, "db3", mode='per', level=2)
	el = (time()-t0)*1e3
        print("pywt took %.3f ms" % (el))

	diff = W.coeffs[0] - Wpy[0]
	print(np.max(np.abs(diff)))
	ims(diff)


	W.set_image(0*l)
	ims(W.image)	
	W.inverse()
	print("Inversion OK")
	ims(W.image)
