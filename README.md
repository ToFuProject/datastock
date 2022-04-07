[![Conda]( https://anaconda.org/conda-forge/datastock/badges/version.svg)](https://anaconda.org/conda-forge/datastock)
[![](https://anaconda.org/conda-forge/datastock/badges/downloads.svg)](https://anaconda.org/conda-forge/datastock)
[![](https://anaconda.org/conda-forge/datastock/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/datastock)
[![](https://anaconda.org/conda-forge/datastock/badges/platforms.svg)](https://anaconda.org/conda-forge/datastock)
[![](https://anaconda.org/conda-forge/datastock/badges/license.svg)](https://github.com/conda-forge/datastock/blob/master/LICENSE.txt)
[![](https://anaconda.org/conda-forge/datastock/badges/installer/conda.svg)](https://anaconda.org/conda-forge/datastock)
[![](https://codecov.io/gh/ToFuProject/datastock/branch/master/graph/badge.svg)](https://codecov.io/gh/ToFuProject/datastock)
[![](https://badge.fury.io/py/datastock.svg)](https://badge.fury.io/py/datastock)


# datacollection
Provides a generic DataStock class, useful for containing classes and multiple data arrays, with interactive plots


datastock
=========

Provides a generic class for storing multiple heterogeneous numpy arrays with non-uniform shapes and built-in interactive visualization routines.
Also stores the relationships between arrays (e.g.: matching dimensions...)
Also provides an elegant way of storing objects of various categories depending on the storeed arrays


The full power of datastock is unveiled when using the DataStock class and sub-classing it for your own use.

But a simpler and more straightforward use is possible if you are just looking for a ready-to-use interactive visualization tool of 1d, 2d and 3d numpy arrays by using a shortcut


Installation:
-------------

datastock is available on Pypi and anaconda.org

``
pip install datastock
``

``
conda install -c conda-forge datastock
``

Examples:
=========
 

Straightforward array visualization:
------------------------------------

```
import datastock as ds

# any 1d, 2d or 3d array
aa = np.np.random.random((100, 100, 100))

# plot interactive figure using shortcut to method
dax = ds.plot_as_array(aa)
```

Now do **shift + left clic** on any axes, the rest of the interactive commands are automatically printed in your python console


<p align="center">
<img align="middle" src="https://github.com/ToFuProject/datastock/blob/Issue020_README/README_figures/DirectVisualization_3d.png" width="600" alt="Direct 3d array visualization"/>
</p>


The DataStock class:
--------------------

You will want to instanciate the DataStock class (which is the cor of datastock) if:
* You have many numpy arrays, not just one, especially if they do not have the same shape
* You want to define a variety of objects from these data arrays (DataStock can be seen as a class storing many sub-classes)


DataStock has 3 main dict attributes:
* dref: to store the size of each dimension, each under a unique key
* ddata: to store all numpy arrays, each under a unique key
* dobj: to store any number of arbitrary sub-dict, each containing a category of object

Thanks to dref, the class knows the relationaships between all numpy arrays.
In particular it knows which arrays share the same references / dimensions


```
import numpy as np
import datastock as ds

# -----------
# Define data
# Here: time-varying profiles representing velocity measurement across the radius of a tube

nt, nx = 100, 80
t = np.linspace(0, 10, nt)
x = np.linspace(1, 2, nx)
prof = (1 + np.cos(t)[:, None]) * x[None, :]

# ------------------
# Populate DataStock

# instanciate 
st = ds.DataStock()

# add references (i.e.: store size of each dimension under a unique key)
st.add_ref(key='nt', size=nt)
st.add_ref(key='nx', size=nx)

# add data dependening on these references
# you can, optionally, specify units, physical dimensionality (ex: distance, time...), quantity (ex: radius, height, ...) and name (to your liking)
st.add_data(key='t', data=t, dimension='time', units='s')
st.add_data(key='x', data=x, dimension='distance', quant='radius', units='m')
st.add_data(key='prof', data=prof, dimension='velocity', units='m/s')

# print in the console the content of st
st

# plot any array interactively
dax = st.plot_as_array('t')
dax = st.plot_as_array('x')
dax = st.plot_as_array('prof')
dax = st.plot_as_array('prof', keyX='t', keyY='x')
```

DataStock can then be used to store any object category

```
# add arbitrary object category as sub-dict of self.dobj
st.add_obj(which='')

```





