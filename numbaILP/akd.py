import numpy as np
import operator

from numba import njit, u8, i8, generated_jit, objmode
from numba.core import types
from numba.types import ListType, DictType, Tuple, unicode_type
from numba.experimental import structref
from numba.experimental.structref import new
from numba.typed.dictobject import _dict_lookup, DKIX, _nonoptional, _cast
from numba.typed import List, Dict
from numba.extending import overload, overload_classmethod, overload_method

from numbaILP.fnvhash import hasharray


akd_fields = {
    # An underlying i8 -> key_typ dictionary
    "dict" : DictType(u8, ListType(Tuple((types.Any, types.Any))))
}


# -------------------------------------------------------------------------
# : Array Keyed Dictionary (AKD) Type

@structref.register
class AKDType(types.StructRef):
    def __init__(cls, key_typ, val_typ):
        l_typ = ListType(Tuple((key_typ, val_typ)))
        super().__init__([('dict', DictType(u8, l_typ))])
    def __str__(self):
        key_type, val_typ = self._fields[0][1].value_type.item_type
        return f'AKDType[{key_type},{val_typ}]'


class AKD(structref.StructRefProxy):
    @classmethod
    def empty(cls, key_typ, val_typ):
        return new_akd(key_typ, val_typ)

    def __setitem__(self, key, val):
        akd_setitem(self, key, val)

    def __getitem__(self, key):
        return akd_getitem(self, key)

    def get(self, key, default=None):
        return akd_get(self, key, default)

    def __contains__(self, key):
        return akd_contains(self, key)

    def __str__(self):
        return akd_str(self)

structref.define_proxy(AKD, AKDType, ["dict"])
# structref.define_boxing(AKDType, AKD)


@generated_jit(nopython=True)
@overload_classmethod(AKDType, 'empty')
def new_akd(_key_typ, _val_typ):
    key_typ, val_typ = _key_typ.instance_type, _val_typ.instance_type
    l_typ = ListType(Tuple((key_typ, val_typ)))
    akd_typ = AKDType(key_typ, val_typ)
    def impl(_key_typ, _val_typ):
        akd = new(akd_typ)
        akd.dict = Dict.empty(u8, l_typ)
        return akd
    return impl

# -------------------------------------------------------------------------
# : AKD lowlevel lookup functions

@generated_jit(nopython=True)
def _akd_bin_lookup(akd, arr_key):
    def impl(akd, arr_key):
        arr = arr_key.view(np.uint8)
        h = u8(hasharray(arr))
        ix, val_lst = _dict_lookup(akd.dict, h, h)
        return h, ix, val_lst
    return impl

@generated_jit(nopython=True)
def _akd_lookup(akd, arr_key):
    def impl(akd, arr_key):
        h, ix, val_lst = _akd_bin_lookup(akd, arr_key)
        if(ix != DKIX.EMPTY):
            # if(len(val_lst) > 1):
            #     print("::", val_lst)
            #     raise ValueError()
            for key_item, val in val_lst:
                if(len(key_item) == len(arr_key) and
                    (key_item == arr_key).all()): 
                    return h, ix, val
            return h, DKIX.EMPTY, None
        else:
            return h, ix, None 
    return impl

# -------------------------------------------------------------------------
# : AKD access functions

@generated_jit(nopython=True)
@overload(operator.contains)
def akd_contains(akd, _arr_key):
    l_typ = akd._fields[0][1].value_type
    item_type = l_typ.item_type
    key_type, val_typ = l_typ.item_type
    arr_item_type = key_type.dtype

    def impl(akd, _arr_key):
        arr_key = _arr_key.astype(arr_item_type)#_cast(_arr_key, key_type)
        _, ix, _ = _akd_lookup(akd, arr_key)
        return ix > DKIX.EMPTY
    return impl


@generated_jit(nopython=True)
@overload(operator.setitem)
def akd_setitem(akd, _arr_key, val):
    if not isinstance(akd, AKDType):
        return

    l_typ = akd._fields[0][1].value_type
    item_type = l_typ.item_type
    key_type, val_typ = l_typ.item_type
    arr_item_type = key_type.dtype

    def impl(akd, _arr_key, val):
        arr_key = _arr_key.astype(arr_item_type)#_cast(_arr_key, key_type)
        h, ix, val_lst = _akd_bin_lookup(akd, arr_key)
        tup = (arr_key, val)
        if ix == DKIX.EMPTY:
            val_lst = List.empty_list(item_type)
        else:
            for i, (key_item, _) in enumerate(val_lst):
                if(len(key_item) == len(arr_key) and
                    (key_item == arr_key).all()): 
                    val_lst[i] = tup
                    return 

        val_lst.append(tup)
        akd.dict[h] = val_lst

    return impl

        
@generated_jit(nopython=True)
@overload(operator.getitem)
def akd_getitem(akd, _arr_key):
    if not isinstance(akd, AKDType):
        return

    l_typ = akd._fields[0][1].value_type
    item_type = l_typ.item_type
    key_type, val_typ = l_typ.item_type
    arr_item_type = key_type.dtype

    def impl(akd, _arr_key):
        arr_key = _arr_key.astype(arr_item_type)#_cast(_arr_key, key_type)
        _, ix, val = _akd_lookup(akd, arr_key)
        if ix == DKIX.EMPTY:
            raise KeyError()
        elif ix < DKIX.EMPTY:
            raise AssertionError("internal dict error during lookup")
        else:
            return _nonoptional(val)

    return impl


@generated_jit(nopython=True)
@overload_method(AKDType, 'get')
def akd_get(akd, _arr_key, default=None):
    if not isinstance(akd, AKDType):
        return
    l_typ = akd._fields[0][1].value_type
    item_type = l_typ.item_type
    key_type, val_typ = l_typ.item_type
    arr_item_type = key_type.dtype

    def impl(akd, _arr_key, default=None):
        arr_key = _arr_key.astype(arr_item_type)#_cast(_arr_key, key_type)
        _, ix, val = _akd_lookup(akd, arr_key)
        if ix > DKIX.EMPTY:
            return _nonoptional(val)
        return default
    return impl

@generated_jit(nopython=True)
@overload(str)
def akd_str(akd):
    if not isinstance(akd, AKDType):
        return
    def impl(akd):
        s = "{\n"
        for i, (h, val_lst) in enumerate(akd.dict.items()):
            # print(h)
            # print(val_lst)
            for j, (arr_key, val) in enumerate(val_lst):
                # print(arr_key, val)
                with objmode(s='unicode_type'):
                    s += f'  {arr_key} : {val}'
                if(i < len(akd.dict)-1 or j < len(val_lst)-1):
                    s += ",\n"

        return  s + "\n}"
        # return "BOB"
    return impl















    # def __new__(cls, key_typ):
    #     return structref.StructRefProxy.__new__(cls, key_typ)



# @njit(nogil=True,fastmath=True)
# def akd_insert(akd,_arr,item,h=None):
#     '''Inserts an i4 item into the dictionary keyed by an array _arr'''
#     arr = _arr.view(np.uint8)
#     if(h is None): h = hasharray(arr)
#     elems = akd.get(h,List.empty_list(BE))
#     is_in = False
#     for elem in elems:
#         if(len(elem[0]) == len(arr) and
#             (elem[0] == arr).all()): 
#             is_in = True
#             break
#     if(not is_in):
#         elems.append((arr,item))
#         akd[h] = elems

# @njit(nogil=True,fastmath=True)
# def akd_get(akd,_arr,h=None):
#     '''Gets an i4 from a dictionary keyed by an array _arr'''
#     arr = _arr.view(np.uint8)
#     if(h is None): h = hasharray(arr) 
#     if(h in akd):
#         for elem in akd[h]:
#             if(len(elem[0]) == len(arr) and
#                 (elem[0] == arr).all()): 
#                 return elem[1]
#     return -1
