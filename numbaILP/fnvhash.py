#From here: https://github.com/znerol/py-fnvhash/blob/master/fnvhash/__init__.py
from numba import types, njit
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import ListType, DictType, unicode_type
from numba.experimental import jitclass
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

bytarr_type = u1[:]
lst_bytarr_type = ListType(bytarr_type)
def AKD(typ):
    lst_custom_type = ListType(typ)

    # @jitclass([('key', bytarr_type),
    #            ('value', typ),
    #            ('next', types.de])
    # class BinElem(object):


    @jitclass([('bin_map', DictType(u4,i8)),
               ('bins', ListType(lst_bytarr_type)),
               ('values', ListType(lst_custom_type))])
    class ArrayKeyedDict(object):
        def __init__(self):
            self.bin_map = Dict.empty(u4,i8)
            self.bins = List.empty_list(lst_bytarr_type)
            self.values = List.empty_list(lst_custom_type)
    return ArrayKeyedDict



@njit(nogil=True,fastmath=True,cache=True)
def akd_insert(akd,_arr,item):
    arr = _arr.view(np.uint8)
    h = hasharray(arr)
    if(h in akd.bin_map):
        l = akd.bins[akd.bin_map[h]]
        is_in = False
        for i in range(len(l)):
            if((l[i] == arr).all()): is_in = True
        # if(arr not in l):
        if(not is_in):
            l.append(arr)
            akd.values[akd.bin_map[h]].append(item)
    else:
        akd.bin_map[h] = len(akd.bins)
        l = List.empty_list(bytarr_type)
        l.append(arr)
        akd.bins.append(l)
        l2 = List()
        l2.append(item)
        akd.values.append(l2)

@njit(nogil=True,fastmath=True,cache=True)
def akd_includes(akd,_arr):
    arr = _arr.view(np.uint8)
    h = hasharray(arr)
    if(h in akd.bin_map):
        l = akd.bins[akd.bin_map[h]]
        is_in = False
        for i in range(len(l)):
            if((l[i] == arr).all()): is_in = True
        # if(arr in l):
        if(is_in):
            return True
    return False

@njit(nogil=True,fastmath=True,cache=True)
def akd_get(akd,_arr):
    arr = _arr.view(np.uint8)
    h = hasharray(arr)
    if(h in akd.bin_map):
        l = akd.bins[akd.bin_map[h]]
        is_in = False
        for i in range(len(l)):
            if((l[i] == arr).all()):
                return akd.values[akd.bin_map[h]][i]
        
    return None


import unittest

class TestAKD(unittest.TestCase):

    def test_insert_include(self):
        akd = AKD(i8)()
        a = np.array([1,2,3,4,5],np.float)
        akd_insert(akd,a,1)
        self.assertTrue(akd_includes(akd,a))
        print(akd_get(akd,a))
        self.assertEqual(akd_get(akd,a),1)

    def test_false_pos(self):
        akd = AKD(unicode_type)()
        a = np.array([1,2,3,4,5],np.uint32)
        b = np.array([1,2,3,4],np.uint32)
        c = np.array([5,5,3,4],np.uint32)
        akd_insert(akd,a,"A")
        akd_insert(akd,b,"B")
        self.assertTrue(akd_includes(akd,a))
        self.assertTrue(akd_includes(akd,b))
        self.assertFalse(akd_includes(akd,c))
        self.assertEqual(akd_get(akd,a),"A")
        self.assertEqual(akd_get(akd,b),"B")
        self.assertEqual(akd_get(akd,c),None)
        




if __name__ == '__main__':
    unittest.main()

