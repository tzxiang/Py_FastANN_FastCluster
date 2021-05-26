import fastann
import numpy as np
import numpy.random as npr
import ctypes
from ctypes.util import find_library

A = npr.rand(1000, 128)
B = npr.rand(1000, 128)

dll_path = find_library('fastann')  # ctypes.util.find_library(name), name without pre-'lib' and '.so'
# dll_path = "/usr/local/lib/libfastann.so"
if dll_path:
    try:
        lib = ctypes.CDLL(dll_path)  # ctypes.cdll.LoadLibrary()
        print(dll_path)
        print(lib)
        print("Succeed to load dll.")
    except OSError as e:
        print(e, "Fail to load dll.")

nno = fastann.build_exact(A)
argmins, mins = nno.search_nn(B)
nno_kdt = fastann.build_kdtree(A, 8, 768)
argmins_kdt, mins_kdt = nno_kdt.search_nn(B)

print(float(np.sum(argmins == argmins_kdt))/1000)
print("finished!")
