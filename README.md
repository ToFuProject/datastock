[![Conda]( https://anaconda.org/conda-forge/datastock/badges/version.svg)](https://anaconda.org/conda-forge/datastock)
[![](https://anaconda.org/conda-forge/datastock/badges/downloads.svg)](https://anaconda.org/conda-forge/datastock)
[![](https://anaconda.org/conda-forge/datastock/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/datastock)
[![](https://anaconda.org/conda-forge/datastock/badges/platforms.svg)](https://anaconda.org/conda-forge/datastock)
[![](https://anaconda.org/conda-forge/datastock/badges/license.svg)](https://github.com/conda-forge/datastock/blob/master/LICENSE.txt)
[![](https://anaconda.org/conda-forge/datastock/badges/installer/conda.svg)](https://anaconda.org/conda-forge/datastock)
[![](https://badge.fury.io/py/datastock.svg)](https://badge.fury.io/py/datastock)



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
<img align="middle" src="https://github.com/ToFuProject/datastock/blob/devel/README_figures/DirectVisualization_3d.png" width="600" alt="Direct 3d array visualization"/>
</p>


The DataStock class:
--------------------

You will want to instanciate the DataStock class (which is the cor of datastock) if:
* You have many numpy arrays, not just one, especially if they do not have the same shape
* You want to define a variety of objects from these data arrays (DataStock can be seen as a class storing many sub-classes)


DataStock has 3 main dict attributes:
* `dref`: to store the size of each dimension, each under a unique key
* `ddata`: to store all numpy arrays, each under a unique key
* `dobj`: to store any number of arbitrary sub-dict, each containing a category of object

Thanks to dref, the class knows the relationaships between all numpy arrays.
In particular it knows which arrays share the same references / dimensions


```
import numpy as np
import datastock as ds

# -----------
# Define data
# Here: time-varying profiles representing velocity measurement across the radius of a tube
# we assume 5 measurement campaigns were conducted, each yielding a different number of measurements, all sampled on 80 radial points

nc = 5
nx = 80
lnt = [100, 90, 80, 120, 110]

x = np.linspace(1, 2, nx)
lt = [np.linspace(0, 10, nt) for nt in lnt]
lprof = [(1 + np.cos(t)[:, None]) * x[None, :] for t in lt]

# ------------------
# Populate DataStock

# instanciate 
st = ds.DataStock()

# add references (i.e.: store size of each dimension under a unique key)
st.add_ref(key='nc', size=nc)
st.add_ref(key='nx', size=nx)
for ii, nt in enumerate(lnt):
    st.add_ref(key=f'nt{ii}', size=nt)

# add data dependening on these references
# you can, optionally, specify units, physical dimensionality (ex: distance, time...), quantity (ex: radius, height, ...) and name (to your liking)

st.add_data(key='x', data=x, dimension='distance', quant='radius', units='m', ref='nx')
for ii, nt in enumerate(lnt):
    st.add_data(key=f't{ii}', data=lt[ii], dimension='time', units='s', ref=f'nt{ii}')
    st.add_data(key=f'prof{ii}', data=lprof[ii], dimension='velocity', units='m/s', ref=(f'nt{ii}', 'x'))

# print in the console the content of st
st
```

<p align="center">
<img align="middle" src="https://github.com/ToFuProject/datastock/blob/devel/README_figures/DataStock_refdata.png" width="600" alt="Direct 3d array visualization"/>
</p>

You can see that DataStock stores the relationships between each array and each reference
Specifying explicitly the references is only necessary if there is an ambiguity (i.e.: several references have the same size, like nx and nt2 in our case)


```
# plot any array interactively
dax = st.plot_as_array('x')
dax = st.plot_as_array('t0')
dax = st.plot_as_array('prof0')
dax = st.plot_as_array('prof0', keyX='t0', keyY='x', aspect='auto')
```

You can then decide to store any object category
Let's create a 'campaign' category to store the characteristics of each measurements campaign
and let's add a 'campaign' parameter to each profile data

```
# add arbitrary object category as sub-dict of self.dobj
for ii in range(nc):
    st.add_obj(
        which='campaign',
	    key=f'c{ii}',
        start_date=f'{ii}.04.2022',
        end_date=f'{ii+5}.05.2022',
        operator='Barnaby' if ii > 2 else 'Jack Sparrow',
        comment='leak on tube' if ii == 1 else 'none',
        index=ii,
    )

# create new 'campaign' parameter for data arrays
st.add_param('campaign', which='data')

# tag each data with its campaign
for ii in range(nc):
    st.set_param(which='data', key=f't{ii}', param='campaign', value=f'c{ii}')	
    st.set_param(which='data', key=f'prof{ii}', param='campaign', value=f'c{ii}')	

# print in the console the content of st
st
```

<p align="center">
<img align="middle" src="https://github.com/ToFuProject/datastock/blob/devel/README_figures/DataStock_Obj.png" width="600" alt="Direct 3d array visualization"/>
</p>

DataStock also provides built-in object selection method to allow return all
objects matching a criterion, as lits of int indices, bool indices or keys.

```
In [9]: st.select(which='campaign', index=2, returnas=int)
Out[9]: array([2])

# list of 2 => return all matches inside the interval
In [10]: st.select(which='campaign', index=[2, 4], returnas=int)
Out[10]: array([2, 3, 4])

# tuple of 2 => return all matches outside the interval
In [11]: st.select(which='campaign', index=(2, 4), returnas=int)
Out[11]: array([0, 1])

# return as keys
In [12]: st.select(which='campaign', index=(2, 4), returnas=str)
Out[12]: array(['c0', 'c1'], dtype='<U2')

# return as bool indices
In [13]: st.select(which='campaign', index=(2, 4), returnas=bool)
Out[13]: array([ True,  True, False, False, False])

# You can combine as many constraints as needed
In [17]: st.select(which='campaign', index=[2, 4], operator='Barnaby', returnas=str)
Out[17]: array(['c3', 'c4'], dtype='<U2')

```

You can also decide to sub-class DataStock to implement methods and visualizations specific to your needs


Other useful built-in methods:
-----------------------------

DataStock provides built-in methods like:
* `get_nbytes()`: return a tuple (size, dsize) where:
    - size is the total size of all data stored in the instance in bytes
    - dsize is a dict with the detail (size for each item in each sub-dict of the instance)
* `save()`: will save the instance
* `ds.load()`: will load a saved instance


