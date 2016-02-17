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

	l = lena()#.astype(np.float32)
	wname = "haar"#"db3"
	nlevels = 8#2
	#W = Wavelets(lena(), "db3", 2) # 0.27 ms
	W = Wavelets(l, wname, nlevels)
	print(l.dtype)
	print(W.wname)
	# W.info()
	t0 = time()
	W.forward()
	el = (time()-t0)*1e3
	print("pypwt took %.3f ms" % (el))
	#W.get_all_coeffs()

	import pywt
	t0 = time()
	Wpy = pywt.wavedec2(l, wname, mode='per', level=nlevels)
	el = (time()-t0)*1e3
        print("pywt took %.3f ms" % (el))

	diff = W.coeffs[0] - Wpy[0]
	print(np.max(np.abs(diff)))
	#ims(diff)

	#W.set_image(0*l)
	#ims(W.image)
	
	#W.soft_threshold(90)
	# W.hard_threshold(100.0)
	# W.shrink(90.0)

	t0 = time()	
	W.inverse()
	el = (time()-t0)*1e3
        print("(inversion) pypwt took %.3f ms" % (el))
	t0 = time()
	pywt.waverec2(Wpy, wname, mode='per')
	el = (time()-t0)*1e3
        print("(inversion) pywt took %.3f ms" % (el))

	print(np.max(np.abs(l - W.image)))
	ims(W.image)
