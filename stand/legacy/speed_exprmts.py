from numba import njit
from numba.typed import List
import numpy as np
import timeit

N = 10000

data = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[0,0,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1], #3
	[0,0,0,0,0,0,1,1,1,1,1,0,0,255,0,0,0], #1
	[0,0,0,0,1,0,1,1,1,1,1,0,0,255,0,0,0], #1
	[0,0,1,0,1,0,1,1,1,1,1,0,0,255,0,0,0], #1
	[1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,1], #2
	[0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0], #2
	[1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0], #2
	],np.uint8)
data = np.reshape(data,(-1))

@njit(nogil=True,fastmath=True)
def nb_argwhere(x):
	nl,nr,nn = 0,0,0
	l = np.empty(x.shape,np.uint32)
	r = np.empty(x.shape,np.uint32)
	n = np.empty(x.shape,np.uint32)
	
	for i in range(len(x)):
		x_i = x[i]
		if(x_i == 255):
			n[nn] = i
			nn += 1
		else:
			if(x[i]):
				r[nr] = i
				nr += 1
			else:
				l[nl] = i
				nl += 1

	# return np.array(l), np.array(r), np.array(n)
	return l[:nl], r[:nr], n[:nn]



@njit(nogil=True,fastmath=True)
def argwhere(x):
	l = np.argwhere(~x)[:,0]
	r = np.argwhere(x)[:,0]
	n = np.argwhere(x == 255)[:,0]
	
	return l, r, n



def time_ms(f):
		f() #warm start
		return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))


@njit(nogil=True,fastmath=True)
def go_nb():
	nb_argwhere(data)


@njit(nogil=True,fastmath=True)
def go_np():
	argwhere(data)

print()

print("ARGWHERE:")
print("NB", time_ms(go_nb))
print("NP", time_ms(go_np))
