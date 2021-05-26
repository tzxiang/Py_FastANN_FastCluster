

# Py_FastANN_FastCluster

A modified C++ library used for approximate nearest neighbours and fast distributed clustering, with `Python 3.x` interfaces. The `Py_FastANN_FastCluster` is used for approximate K-Means (AKM).



## Description

It contains two libraries: `FASTANN` and `FASTCLUSTER`.

- `FASTANN`: A library for fast approximate nearest neighbours

- `FASTCLUSTER`: A library for fast, distributed clustering (using MPI) for very large datasets.



Environment:  

- [Fastann and fastcluster](https://www.robots.ox.ac.uk/~vgg/software/fastanncluster/) (released in 2009.11)

- Ubuntu 18.04 x86_64 with GCC 7.5.0
- CMake 3.15.0

- python 3.6



## FastANN Installation

Requirements: Linux, CMake, Yasm (optional). Here I donot use Yasm.

```
$ cd fastann
$ PREFIX=/usr/local/ cmake . && make
$ make test  # all PASSED
$ make perf
$ sudo make install
```

The compiled library is installed in `PREFIX/lib` (libfastann.so) and `PREFIX/include/fastann` folder. 



### 1) Test C++ example

Go into `examples` folder, and run the `example1.cpp` in `fastann/examples` folders. 

```
$ g++ example1.cpp -o example1 -I /usr/local/include/fastann -L /usr/local/lib -lfastann
$ ./example1
```

If it doesnot find lib, write the lib  path into environment variable first, e.g.

```
$ export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
$ sudo ldconfig
```



### 2) Install Python interfaces

Requirements: python and numpy.

First, in `./fastann/interfaces/python/fastann.py`, chang `raise` to

```python
raise TypeError('query type must be the same as the base type')
```

In `fastann.py`: `build_exact()` and  `build_kdtree()` functions, define the types of retured value before calling functions. Specifically, in `build_exact()` function, between `suffix=***` and `ptr=***`, add: 
```python
if suffix == "d":
    lib.fastann_nn_obj_build_exact_d.restype = ctypes.POINTER(ctypes.c_void_p)
elif suffix == "s":
    lib.fastann_nn_obj_build_exact_s.restype = ctypes.POINTER(ctypes.c_void_p)
else:
    lib.fastann_nn_obj_build_exact_c.restype = ctypes.POINTER(ctypes.c_void_p)
```

In `build_kdtree()` function, between `suffix=***` and `ptr=***`, add: 
```python
if suffix == "d":
    lib.fastann_nn_obj_build_kdtree_d.restype = ctypes.POINTER(ctypes.c_void_p)
elif suffix == "s":
    lib.fastann_nn_obj_build_kdtree_s.restype = ctypes.POINTER(ctypes.c_void_p)
else:
    lib.fastann_nn_obj_build_kdtree_c.restype = ctypes.POINTER(ctypes.c_void_p)
```

Otherwise, when running `test.py` in `fastann/examples`, it will cause error `Process finished with exit code 139 (interrupted by signal 11: SIGSEGV). Address: getattr in fastann.py.` This is because the type of the return value is not defined, so the pointer returned by calling functions is wrong and cannot access the correct memory. (Invalid/illegal pointer access)

Then  compile

```
$ cd interfaces/python && python setup.py install
```

Thus we can get the `fastann.py` and `libfastann.so`, copy these two files into python `python**/site-packages/`. Then you can call its functions in your own python files.

**Test**. Go into `fastann/examples` folders and run `python test.py` to test the fastann library in python. Before run, modify the `print` to python 3.x format in `test.py` files. The final library is: `fastann.py`, `libfastann.so` and `PREFIX/include/fastann` (header files). The `fastann` folder includes: `fastann.hpp, randomkit.h, rand_point_gen.hpp, fastann.h`.




## FastCluster Installation

Requirements: PyTables, fastann, and MPI library (OpenMPI is recommended). Here, fastann is installed above.



### 1) OpenMPI Under Ubuntu

Download  `openmpi-1.1.5` (released in 2007) and install. Other versions of [OpenMPI](https://www-lb.open-mpi.org/) may cause some errors.

```
$ wget https://download.open-mpi.org/release/open-mpi/v1.1/openmpi-1.1.5.tar.gz
$ tar -zxvf openmpi-1.1.5.tar.gz
$ cd openmpi-1.1.5
$ ./configure --prefix="/usr/local/openmpi"
$ make
$ sudo make install # -j8
```

Then configure environment variables:

```
# Method 1
# vim ~/.bashrc and then add
export PATH=$PATH:/usr/local/openmpi/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/openmpi/lib/
# save .bashrc and then 
$ sudo ldconfig

# Method 2
$ sudo gedit /etc/profile # and add
export PATH=/usr/local/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH
# open new terminal
```

Test if the installation was successful.

```
$ mpirun
# or
$ cd examples
$ make
$ mpirun -np 8 hello_c
```



### 2) PyTables

PyTables is used to read and write hdf5 files.

```
conda install -c conda-forge pytables
```



### 3) Fastcluster

Modify the files:

- In `kmeans.cpp`, add `#include <stdio.h>` and `#include <iostream>`

- In `test_mpi_queue.cpp`, add `#include <iostream>`

Then compile:

```
$ PREFIX=/usr/local/ cmake . && make
$ sudo make install
```

The compiled library is installed in `PREFIX/lib` (libfastcluster.so) and `PREFIX/include/fastcluster` folder. 



**a) Test**. Run `./test_mpi_queue`. "PASSED" means successful compilation.



**b) Python 3.x interfaces**

Modify `fastcluster.py`.

1)Modify the functions in pyTables 2.x (python 2.x)  to pyTables 3.x version. For pyTables 2.x -> 3.x, you can see [Pytables Migrating](http://www.pytables.org/MIGRATING_TO_3.x.html). Specifically,

- In main function of `fastcluster.py`, modify
  
  ```python
  # Change pyTables 2.x
  pnts_fobj = tables.openFile('pnts.h5','w')
  pnts_fobj.createArray(pnts_fobj.root, 'pnts', pnts)
  # To pyTables 3.x
  pnts_fobj = tables.open_file('pnts.h5','w')
  pnts_fobj.create_array(pnts_fobj.root, 'pnts', pnts)
  ```
  
- In `kmeans()` function of `fastcluster.py`, about line 180+, modify

  ```python
  pnts_fobj = tables.openFile(pnts_fn, 'r')     -> pnts_fobj = tables.open_file(pnts_fn, 'r')
  pnts_fobj.walkNodes('/', classname = 'Array') -> pnts_fobj.walk_nodes('/', classname = 'Array')
  ctypes.c_char_p(chkpnt_fn)                    -> ctypes.c_char_p(chkpnt_fn.encode('utf-8'))
  clst_fobj = tables.openFile(clst_fn, 'w')     -> clst_fobj = tables.open_file(clst_fn, 'w')
  createCArray()                                -> create_carray()
  ```

2)Modify the code in python 3.x format, specifically,  

- change `except OSError, e:` to `OSError as e:`; 

- change `raise ***` to `raise TypeError('****')`.

- In `build_nn_obj()` function of `nn_obj_exact_builder` class, define the type of returned value. Specifically, before `ptr=getattr**`, add 
  
  ```python
  if self.suffix == "d":
      libfastann.fastann_nn_obj_build_exact_d.restype = ctypes.c_void_p
  elif self.suffix == "s":
      libfastann.fastann_nn_obj_build_exact_s.restype = ctypes.c_void_p
  else:
    libfastann.fastann_nn_obj_build_exact_c.restype = ctypes.c_void_p
  ```

- In `build_nn_obj()` function of `nn_obj_approx_builder` class, define the type of returned value. Specifically, before `ptr=getattr**`, add 

  

  ```python
  if self.suffix == "d":
      libfastann.fastann_nn_obj_build_kdtree_d.restype = ctypes.c_void_p
  elif self.suffix == "s":
      libfastann.fastann_nn_obj_build_kdtree_s.restype = ctypes.c_void_p
  else:
      libfastann.fastann_nn_obj_build_kdtree_c.restype = ctypes.c_void_p
  ```

  


Test and compile:

- Run `fastcluster.py` to test the `fastcluster` lib. 

- Run `python setup.py install` to compile the library in python 3.x.

The final library is `fastcluster.py`, `libfastcluster.so` and `fastcluster` (header files) folders. The `fastcluster` includes: `randomkit.h, mpi_queue.hpp, kmeans.h`. 



**c) Other issues**

- Running error: `mca_base_component_repository_open: unable to open mca_op_avx: /usr/local/openmpi/lib/openmpi/` `mca_op_avx.so: undefined symbol: ompi_op_base_module_t_class (ignored)`

  It seems to not influence the program running. It is the version issue of OpenMPI. You should use `openmpi-1.1.5` (released in 2007). Other versions will cause this error.

- Error: Permission denied, cannot create folders or write files.

  Use `sudo` command, or modify the folder permission by `$chmod 777 /***/***`.


## Appendix

Other similar library used for AKM (Approximate K-Means): 

- [Faiss](https://github.com/facebookresearch/faiss): a library for efficient similarity search and clustering of dense vectors, released by FAIR.
