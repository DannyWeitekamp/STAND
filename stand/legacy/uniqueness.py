import numpy as np
from stand.tree_classifiers import gini
from numba import njit, objmode
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.types import UniTuple, intp,float32, intp
from numba.types.containers import UniTuple

data = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[0,0,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1], #3
	[0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0], #1
	[0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1
	[0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1
	[1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,1], #2
	[0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0], #2
	[1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0], #2
	],np.bool);

def uniqueness1(X):
	p = np.sum(X,axis=0)
	n_p = X.shape[0]-p
	print(p)
	print(n_p)
	# p *= X.shape[0]
	# n_p *= X.shape[0]
	# z = (X * n_p.reshape((1,-1)) + (~X * p.reshape((1,-1))))

	s = np.ones((X.shape[0]))
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			if(X[i,j]):
				print(1.0/p[j])
				if(p[j] > 0.0): s[i] += p[j]
			else:
				if(n_p[j] > 0.0): s[i] += n_p[j]
				
	# s = s
	print("S", s)
	print("sS", np.sum(s))
	# z = (X * p.reshape((1,-1))) + (~X * n_p.reshape((1,-1)))
	# print("Z", 1.0/(X * p.reshape((1,-1)))  )
	# print("Z", 1.0/(~X * n_p.reshape((1,-1))) )
	# z = np.log(np.sum(z/X.shape[1],axis=1))
	# print(z)
	# for d in range(X.shape[1]):
	# 	p = np.sum(X[:,d])/X.shape[0]

# uniqueness(data)


def uniqueness2(X):
	p = np.sum(X,axis=0,dtype=np.uint32)
	n_p = X.shape[0]-p
	z = np.concatenate((p.reshape(-1,1),n_p.reshape(-1,1)),axis=1)
	print(z)
	print(z.shape)
	print(gini(z))

def uniqueness3(X):
	n = np.sum(X,axis=0,dtype=np.uint32)
	p = n/X.shape[0]

	out = np.empty((X.shape[0],))
	for i in range(X.shape[0]):
		u = 0
		for j in range(X.shape[1]):
			# u += (1-p[j])/p[j] if X[i,j] else p[j]/(1-p[j])
			# u += np.log((1-p[j])/p[j]) if X[i,j] else np.log(p[j]/(1-p[j]))
			u += (1-p[j]) if X[i,j] else p[j]
		out[i] = u
	# out /= np.sum(n)
	print(out, 2*sum(out)/X.shape[0])



def argsort_arrays(X):
	flat_type = np.dtype((np.void, X.dtype.itemsize * X.shape[1]))
	return np.argsort(X.view(flat_type).ravel(), axis=0)
# 
@njit(nogil=True,fastmath=True,cache=True)
def _unique_arrays(inp):
	# print(inp.astype(np.int))
	# counts = [];
	uniques = [];
	inds = np.zeros(len(inp),np.uint32);
	ind=0;
	last = 0;
	for i in range(1,len(inp)):
		if((inp[i-1] != inp[i]).any()):
			# counts.append(i-last);
			uniques.append(inp[i-1]);
			last = i;
			ind += 1;
		inds[i] = ind;
	# counts.append((i+1)-last);
	uniques.append(inp[i]);
	u = np.empty((len(uniques),inp.shape[1]),dtype=np.int32)
	for i,x in enumerate(uniques):
		u[i] = x
	# c = np.asarray(counts,dtype=np.uint32)
	# u = np.asarray(uniques,dtype=np.int32)
	return u

# from numba.unsafe.ndarray import to_fixed_tuple
@njit(nogil=True,fastmath=True,cache=True)
def unique_arrays(X):
	inds = np.zeros((X.shape[0],),dtype=np.int64)
	with objmode(inds='i8[:]'):
		inds += argsort_arrays(X)
	print(inds)
	u = _unique_arrays(X[inds])
	return u

@njit(nogil=True,fastmath=True,cache=True)
def uniqueness4(X):
	print("---------")
	print("unique\n",)
	

	X = unique_arrays(X)
	print(X)
	out = np.zeros((X.shape[0],))
	# m = np.zeros((X.shape[0],X.shape[0]))
	for i in range(X.shape[0]):
		for j in range(i+1,X.shape[0]):
			s = np.sum(X[i] != X[j])
			# m[i,j] = s
			out[i] += s
			out[j] += s
	out /= 2
	# print(m)
	# print(np.sum(m),(2**X.shape[1])*((2**X.shape[1])-2) )
	print(out, np.sum(out),"/",(2**X.shape[1])*((2**X.shape[1])-2),)


uniqueness = uniqueness4




data = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[1,0,0,0],
	[1,0,0,0],
	[1,0,0,0],
	[1,0,0,0],
	[1,0,0,0],
	[1,0,0,0],
	
	],np.bool);

# uniqueness(data)
uniqueness(data)

data = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[0,1,0],
	[0,1,1],
	[0,1,1],
	[0,1,1],
	[1,0,0],
	[1,0,0],
	[1,0,0],
	[1,0,0],
	],np.bool);

uniqueness(data)

data = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	# [0,1,1],
	[0,1,1],
	[1,0,0],
	# [1,0,0],
	],np.bool);

uniqueness(data)

data = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[0,0,0],
	[1,0,0],
	[0,1,0],
	[1,1,0],
	[0,0,1],
	[1,0,1],
	[0,1,1],
	[1,1,1],
	],np.bool);

uniqueness(data)
# uniqueness(data)

data = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[0,0],
	[0,1],
	[1,0],
	[1,1],
	],np.bool);
uniqueness(data)

data = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[1,0,0],
	[0,1,0],
	[0,0,1],
	],np.bool);

uniqueness(data)
