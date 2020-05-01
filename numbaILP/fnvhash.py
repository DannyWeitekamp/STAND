#From here: https://github.com/znerol/py-fnvhash/blob/master/fnvhash/__init__.py
from numba import types, njit,jitclass
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import ListType, unicode_type
import numpy as np

FNV_32_PRIME = 0x01000193
FNV_64_PRIME = 0x100000001b3

FNV0_32_INIT = 0
FNV0_64_INIT = 0
FNV1_32_INIT = 0x811c9dc5
FNV1_32A_INIT = FNV1_32_INIT
FNV1_64_INIT = 0xcbf29ce484222325
FNV1_64A_INIT = FNV1_64_INIT


# @jitclass()
# def HashTable():
# 	def __init__(self):
# 		self.bins = List.empty_list(ListType())

@njit(nogil=True,fastmath=True,cache=True)
def fnv(data, hval_init, fnv_prime, fnv_size):
    """
    Core FNV hash algorithm used in FNV0 and FNV1.
    """
    hval = hval_init
    for i in range(len(data)):
        byte = data[i]
        hval = (hval * fnv_prime) % fnv_size
        hval = hval ^ byte
    return hval

@njit(u4(u1[:]),nogil=True,fastmath=True,cache=True)
def fnv0_32(data):
    """
    Returns the 32 bit FNV-0 hash value for the given data.
    """
    return fnv(data, FNV0_32_INIT, FNV_32_PRIME, 2**32)

# @njit(u8(u1[:]),nogil=True,fastmath=True,cache=True)
# def fnv0_64(data):
#     """
#     Returns the 64 bit FNV-0 hash value for the given data.
#     """
#     return fnv(data, FNV0_64_INIT, FNV_64_PRIME, 2**64)

@njit(nogil=True,fastmath=True,cache=True)
def hasharray(array):
	return fnv0_32(array.view(np.uint8))