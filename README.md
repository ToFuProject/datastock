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

The interactive commands are automatically printed in your python console


<p align="center">
<img align="middle" src="https://github.com/ToFuProject/datastock/blob/main/README_figures/DirectVisualization_3d.png" width="600" alt="Direct 3d array visualization"/>
</p>







