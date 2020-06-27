#From here: https://github.com/znerol/py-fnvhash/blob/master/fnvhash/__init__.py
from numba import types, njit, jit
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import ListType, DictType, unicode_type, NamedTuple
from numba.experimental import jitclass
from collections import namedtuple
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
# lst_bytarr_type = ListType(bytarr_type)


def AKD(typ,ret_be=False):
    # lst_custom_type = ListType(typ)

    BE_deffered = deferred_type()
    # BinElem = namedtuple("BinElem",['key','value','next'])
    # BE = NamedTuple([bytarr_type,typ,optional(BE_deffered)],BinElem)
    @jitclass([('key', bytarr_type),
               ('value', typ),
               ('next', optional(BE_deffered))])
    class BinElem(object):
        def __init__(self,key,value):
            self.key = key
            self.value = value
            self.next = None
    # print(BinElem)
    BE = BinElem.class_type.instance_type
    BE_deffered.define(BE)

    @njit(nogil=True,fastmath=True)
    def akd_insert(akd,_arr,item,h=None):
        arr = _arr.view(np.uint8)
        if(h is None): h = hasharray(arr)
        elem = akd.get(h,None)
        is_in = False
        while(elem is not None):
            if(len(elem.key) == len(arr) and
                (elem.key == arr).all()): 
                is_in = True
                break
            if(elem.next is None): break 
            elem = elem.next
        if(not is_in):
            new_elem = BinElem(arr,item)
            new_elem.next = elem
            akd[h] = new_elem
            # print("HERE",elem.value,new_elem.value)

        # else:
        #     new_elem = BinElem(arr,item)
        #     akd[h] = new_elem
            
    @njit(nogil=True,fastmath=True)
    def akd_includes(akd,_arr,h=None):
        arr = _arr.view(np.uint8)
        if(h is None): h = hasharray(arr)
        elem = akd.get(h,None)
        # if(elem is not None):
        is_in = False
        while(elem is not None):
            if(len(elem.key) == len(arr) and
                (elem.key == arr).all()): 
                is_in = True
                break
            if(elem.next is None): break 
            elem = elem.next
            # if(is_in):
            #     return True
        return is_in

    @njit(nogil=True,fastmath=True)
    def akd_get(akd,_arr,h=None):
        arr = _arr.view(np.uint8)
        if(h is None): h = hasharray(arr) 
        elem = akd.get(h,None)
        while(elem is not None):
            # print(":",elem.value)
            if(len(elem.key) == len(arr) and
                (elem.key == arr).all()): 
                return elem.value
            if(elem.next is None): break 
            elem = elem.next
            
        return -1


    # @jitclass([('bin_map', DictType(u4,i8)),
    #            ('bins', ListType(lst_bytarr_type)),
    #            ('values', ListType(lst_custom_type))])
    # class ArrayKeyedDict(object):
    #     def __init__(self):
    #         self.bin_map = Dict.empty(u4,i8)
    #         self.bins = List.empty_list(lst_bytarr_type)
    #         self.values = List.empty_list(lst_custom_type)

    # @jitclass([('bin_map', DictType(u4,BinElem))])
    # class ArrayKeyedDict(object):
    #     def __init__(self):
    #         self.bin_map = Dict.empty(u4,i8)
    #         self.bins = List.empty_list(lst_bytarr_type)
    #         self.values = List.empty_list(lst_custom_type)

    # @jit(fastmath=True)
    # def new_AKD():
    #     d = Dict.empty(u4,BE)
    #     return d
    # new_AKD()
    if(not ret_be):
        return BE,akd_get, akd_includes, akd_insert
    else:
        return BE,akd_get, akd_includes, akd_insert,BinElem




# @njit(nogil=True,fastmath=True,cache=True)
# def akd_insert(akd,_arr,item):
#     arr = _arr.view(np.uint8)
#     h = hasharray(arr)
#     if(h in akd.bin_map):
#         l = akd.bins[akd.bin_map[h]]
#         is_in = False
#         for i in range(len(l)):
#             if((l[i] == arr).all()): is_in = True
#         # if(arr not in l):
#         if(not is_in):
#             l.append(arr)
#             akd.values[akd.bin_map[h]].append(item)
#     else:
#         akd.bin_map[h] = len(akd.bins)
#         l = List.empty_list(bytarr_type)
#         l.append(arr)
#         akd.bins.append(l)
#         l2 = List()
#         l2.append(item)
#         akd.values.append(l2)

# @njit(nogil=True,fastmath=True,cache=True)
# def akd_includes(akd,_arr):
#     arr = _arr.view(np.uint8)
#     h = hasharray(arr)
#     if(h in akd.bin_map):
#         l = akd.bins[akd.bin_map[h]]
#         is_in = False
#         for i in range(len(l)):
#             if((l[i] == arr).all()): is_in = True
#         # if(arr in l):
#         if(is_in):
#             return True
#     return False

# @njit(nogil=True,fastmath=True,cache=True)
# def akd_get(akd,_arr):
#     arr = _arr.view(np.uint8)
#     h = hasharray(arr)
#     if(h in akd.bin_map):
#         l = akd.bins[akd.bin_map[h]]
#         is_in = False
#         for i in range(len(l)):
#             if((l[i] == arr).all()):
#                 return akd.values[akd.bin_map[h]][i]
        
#     return None


import unittest

class TestAKD(unittest.TestCase):

    def test_insert_include(self):
        BE, akd_get, akd_includes, akd_insert = AKD(i8)
        akd = Dict.empty(u4,BE)
        # akd, akd_get, akd_includes, akd_insert = AKD(i8)
        a = np.array([1,2,3,4,5],np.float)
        akd_insert(akd,a,1)
        self.assertTrue(akd_includes(akd,a))
        print(akd_get(akd,a))
        self.assertEqual(akd_get(akd,a),1)

    def test_false_pos(self):
        BE, akd_get, akd_includes, akd_insert = AKD(unicode_type)
        akd = Dict.empty(u4,BE)
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

    def test_hash_conflict(self):
        BE, akd_get, akd_includes, akd_insert = AKD(unicode_type)
        akd = Dict.empty(u4,BE)
        a = np.array([1,2,3,4,5],np.uint32)
        b = np.array([1,2,3,4],np.uint32)
        c = np.array([5,5,3,4],np.uint32)
        akd_insert(akd,a,"C",0)
        akd_insert(akd,b,"D",0)
        self.assertTrue(akd_includes(akd,a,0))
        self.assertEqual(akd_get(akd,a,0),"C")
        self.assertTrue(akd_includes(akd,b,0))
        self.assertEqual(akd_get(akd,b,0),"D")

        self.assertFalse(akd_includes(akd,c,0))
        self.assertEqual(akd_get(akd,c,0),None)

        # BE_deffered = deferred_type()
# @jitclass([('key', u4),
#            ('value', u4),
#            ])
# class BinElem(object):
#     def __init__(self,key,value):
#         self.key = key
#         self.value = value
#         # self.next = None

#         @njit
#         def new_BinElm(key,value):
#             return BinElem(key.view(np.uint8),value)
        # BE, akd_get, akd_includes, akd_insert, BinElem = AKD(unicode_type,True)
        # # print(BE.class_type)
        # @njit(nogil=True,fastmath=True)
        # def add_to(akd,y,x1,v1,x2,v2):
        #     akd = Dict.empty(u4,BE)
        #     h = hasharray(y)
        #     be_a = BinElem(x1.view(np.uint8),v1)
        #     be_b = BinElem(x2.view(np.uint8),v2)
        #     # be_b.next = be_a
        #     akd[h] = be_b
        #     print("MEEP",akd_get(akd,x1))
        #     # return akd

        # # BE = BinElem.class_type.instance_type
        # # BE_deffered.define(BE)        
        
        
        # a = np.array([1,2,3,4,5],np.uint32)
        # b = np.array([1,2,3,4],np.uint32)

        # # be_a = new_BinElm(a,"A")
        # # be_b = new_BinElm(b,"A")
        # # be_b.next = be_a
        # akd = Dict.empty(u4,BE)
        # add_to(akd,a,a,"A",b,"B")
        # add_to(akd,a,be_b)
        # h = hasharray(a)
        # akd[h] = be_b
        # akd_includes(akd,a)
        # self.assertTrue(akd_includes(akd,a))
        # self.assertEqual(akd_get(akd,a),"A")


if __name__ == '__main__':
    # new_AKD, akd_get, akd_includes, akd_insert = AKD(unicode_type)
    # akd = new_AKD()
    unittest.main()

