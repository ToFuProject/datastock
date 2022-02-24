""" A tomography library for fusion devices

See:
https://github.com/ToFuProject/datastock
"""

# Built-in
import os
import subprocess
from codecs import open
# ... setup tools
from setuptools import setup, find_packages


# ... local script
import _updateversion as up


# == Getting version =====================================================
_HERE = os.path.abspath(os.path.dirname(__file__))

version = up.updateversion()

print("")
print("Version for setup.py : ", version)
print("")


# =============================================================================
# Get the long description from the README file
# Get the readme file whatever its extension (md vs rst)

_README = [
    ff
    for ff in os.listdir(_HERE)
    if len(ff) <= 10 and ff[:7] == "README."
]
assert len(_README) == 1
_README = _README[0]
with open(os.path.join(_HERE, _README), encoding="utf-8") as f:
    long_description = f.read()
if _README.endswith(".md"):
    long_description_content_type = "text/markdown"
else:
    long_description_content_type = "text/x-rst"


# =============================================================================


# =============================================================================
#  Compiling files

setup(
    name="datastock",
    version=f"{version}",
    # Use scm to get code version from git tags
    # cf. https://pypi.python.org/pypi/setuptools_scm
    # Versions should comply with PEP440. For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    # The version is stored only in the setup.py file and read from it (option
    # 1 in https://packaging.python.org/en/latest/single_source_version.html)
    use_scm_version=False,

    # Description of what library does
    description="A python library for generic class and data handling",
    long_description=long_description,
    long_description_content_type=long_description_content_type,

    # The project's main homepage.
    url="https://github.com/ToFuProject/datastock",
    # Author details
    author="Didier VEZINET",
    author_email="didier.vezinet@gmail.com",

    # Choose your license
    license="MIT",

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        # In which language most of the code is written ?
        "Natural Language :: English",
    ],

    # What does your project relate to?
    keywords="data analysis class container generic interactive plot",

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(
        exclude=[
            "doc",
        ]
    ),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    # py_modules=["my_module"],
    # List run-time dependencies here. These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    python_requires=">=3.6",

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        "dev": [
            "check-manifest",
            "coverage",
            "pytest",
            "sphinx",
            "sphinx-gallery",
            "sphinx_bootstrap_theme",
        ]
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here. If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # package_data={
    #    # If any package contains *.txt, *.rst or *.npz files, include them:
    #    '': ['*.txt', '*.rst', '*.npz'],
    #    # And include any *.csv files found in the 'ITER' package, too:
    #    'ITER': ['*.csv'],
    # },
    # package_data={},
    # include_package_data=True,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html
    # installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # executable scripts can be declared here
    # They can be python or non-python scripts
    # scripts=[
    # ],

    # entry_points point to functions in the package
    # Theye are generally preferable over scripts because they provide
    # cross-platform support and allow pip to create the appropriate form
    # of executable for the target platform.
    # entry_points={},
    # include_dirs=[np.get_include()],

    py_modules=['_updateversion'],
)
