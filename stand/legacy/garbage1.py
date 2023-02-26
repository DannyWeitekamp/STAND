from numba import njit, cfunc
from numba.types import unicode_type, i8
from cre.utils import PrintElapse

@njit(cache=True)
def foo(x,a0,a1,a2):
    return len(x)

@cfunc(i8(unicode_type, i8, i8, i8), nopython=True, cache=True)
def bar(x,a0,a1,a2):
    return len(x)

@njit(i8(unicode_type, i8, i8, i8), nopython=True, cache=True)
def baz(x,a0,a1,a2):
    return len(x)

baz_entry_point = baz.overloads[(unicode_type, i8, i8, i8)].entry_point

print(foo("HELLO WORLD",0,1,2))
print(bar("HELLO WORLD","MOOP",1,2))
print(bar("HELLO WORLD",0,1,2))
print(baz_entry_point("HELLO WORLD",0,1,2))


with PrintElapse("njit"):
    for i in range(1000):
        foo("HELLO WORLD",i,1,2)

with PrintElapse("python"):
    f = foo.py_func
    for i in range(1000):
        f("HELLO WORLD",i,1,2)

with PrintElapse("cfunc"):
    for i in range(1000):
        bar("HELLO WORLD",i,1,2)

with PrintElapse("entry_point"):
    for i in range(1000):
        baz_entry_point("HELLO WORLD",i,1,2)    
