from numba import njit, objmode
import numpy as np
import ctypes
import time

CLOCK_MONOTONIC = 0x1
clock_gettime_proto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_long))
pybind = ctypes.CDLL(None)
clock_gettime_addr = pybind.clock_gettime
clock_gettime_fn_ptr = clock_gettime_proto(clock_gettime_addr)


@njit
def timenow():
    timespec = np.zeros(2, dtype=np.int64)
    clock_gettime_fn_ptr(CLOCK_MONOTONIC, timespec.ctypes)
    ts = timespec[0]
    tns = timespec[1]
    return np.float64(ts) + 1e-9 * np.float64(tns)


@njit
def pointless_delay(seconds):
    with objmode():
        s = time.time()
        e = 0
        while (e < seconds):
            e = time.time() - s


@njit
def do_stuff(n):
    t0 = timenow()
    pointless_delay(n)
    print("Elapsed", timenow() - t0)


do_stuff(1)
do_stuff(2)
do_stuff(3.21)
