from numba import types, njit, jit
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
import stand.m_patch_array_hash
import numpy as np

f8_lst = f8[:]

def test_array_keyed_dict():

    @njit(cache=True)
    def no_conflicts():
        out = np.ones(10,np.uint8)
        d = Dict.empty(f8_lst,i8)

        a = np.zeros(5, dtype=np.float64)
        a1 = np.zeros(5, dtype=np.float64)
        d[a] = 2

        out[0] = a in d
        out[1] = a1 in d
        out[2] = d[a] == 2
        out[3] = d[a1] == 2
        out[4] = a + 1 not in d

        b = np.zeros(6, dtype=np.float64)

        out[5] = b not in d

        c = np.arange(5, dtype=np.float64)

        out[6] = c not in d        

        d[c] = 4

        out[7] = c in d        
        out[8] = d[c] == 4

        c_rev = c[::-1]
        
        out[8] = c_rev not in d

        out[9] = d.get(c,0) == 4


        return out

    assert all(no_conflicts())
    assert all(no_conflicts.py_func())
