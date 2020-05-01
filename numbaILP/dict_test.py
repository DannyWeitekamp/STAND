import numpy as np
import numba
from numba import types, njit,jitclass, guvectorize,vectorize,prange
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.types import ListType, unicode_type
import timeit
from sklearn import tree as SKTree
from compile_template import compile_template
from enum import IntEnum
from numba.pycc import CC
from fnvhash import hasharray



def flatten_state(state):
    tup = Tuplizer()
    flt = Flattener()
    state = flt.transform(tup.transform(state))
    return state

def vectorize(X,ignore=[]):
	X = [{k:{_k:_v for _k,_v in v.items() if _k not in ignore } if isinstance(v,dict) else v \
			 for k,v in x.items()} for x in X]
	X = [flatten_state(x) for x in X]
	d = {}
	n = 0
	for state in X:
		for k,v in state.items(): 
			if(k not in d):
				d[k] = n
				n += 1

	L = len(d.keys())
	out = np.zeros((len(X),L))
	for i,state in enumerate(X):
		for k,v in state.items():
			out[i,d[k]] = v

	return out