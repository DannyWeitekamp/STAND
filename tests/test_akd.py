import numpy as np
import operator

from numba import njit, u8, i8
# from numba.core import types
# from numba.types import ListType, DictType, Tuple
# from numba.experimental import structref
# from numba.experimental.structref import new
# from numba.typed.dictobject import _dict_lookup, DKIX, _nonoptional, _cast
# from numba.typed import List, Dict
# from numba.extending import overload, overload_classmethod

from stand.fnvhash import hasharray
from stand.akd import akd_contains, AKD, AKDType

def test_type():
    typ = AKDType(i8[::1], i8)
    print(typ)

def test_getset():
    key = np.ones(5,dtype=np.int64)
    akd = AKD.empty(i8[::1], i8)
    akd[key] = 7
    assert akd[key] == 7
    assert akd.get(key,0) == 7
    assert akd.get(np.zeros(0),0) == 0
    assert key in akd

def test_convert_type():
    key = np.ones(5)
    akd = AKD.empty(i8[::1], i8)
    akd[key] = 7
    assert akd[key] == 7
    assert key in akd


def test_hashconflict():
    a = np.array([2, 3, 4, 4, 0, 4, 0, 4, 1, 4],dtype=np.int64)
    b = np.array([0, 1, 0, 4, 4, 3, 1, 1, 4, 2],dtype=np.int64)
    ha = hasharray(a)
    hb = hasharray(b)
    if(ha == hb):
        akd = AKD.empty(i8[::1], i8)
        akd[a] = 1
        akd[b] = 2
        assert akd[a] == 1
        assert akd[b] == 2
        assert a in akd
        assert b in akd

def test_jitted():
    i8_arr = i8[::1]
    @njit(cache=True)
    def foo():
        # Dict.empty(i8_arr, i8)
        akd = AKDType.empty(i8_arr, i8)
        # akd = akd_empty(i8_arr, i8)
        okay = True
        for i in range(100):
            r = np.random.randint(0,5,size=(10,))
            # print(r)
            akd[r] = i
            if(akd[r] != i):
                okay = False
        return okay

    assert foo() == True


if __name__ == "__main__":
    test_type()
    test_getset()
    test_convert_type()
    test_hashconflict()
