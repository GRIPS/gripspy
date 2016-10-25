.. gripspy documentation master file, created by
   sphinx-quickstart on Wed Feb 10 13:27:14 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gripspy's documentation!
===================================

Introduction
------------
This Python package is intended for analysis of data from the `GRIPS <http://grips.ssl.berkeley.edu>`_
long-duration flight of 2016 January 19–30.
It can also be used to analyze the pre-flight calibration data that was already being recorded
using flight-like packet formats.
The code is still very much a work in progress, so not only are many elements still missing,
even the parts that have been coded are likely to go through changes in their interfaces and
organization.

Prerequisites
-------------
This package depends on:

* Python 2.7 (untested on Python 3.x)
* NumPy
* SciPy
* matplotlib
* Cython, and corresponding build environment for compiling C code
* scikit-image
* Astropy

However, rather than installing these packages individually, it is highly recommended that one
instead use a scientific Python distribution (e.g., `Anaconda <https://www.continuum.io/downloads>`_),
which will include useful packages such as Jupyter Notebook (formerly IPython Notebook).

The build environment for C code is a bit trickier to set up on Windows than on the other platforms.
Assuming that you are using Python 2.7, here are the steps:

* Install `Microsoft Visual C++ Compiler for Python 2.7 <http://www.microsoft.com/en-us/download/details.aspx?id=44266>`_
* Patch `C:\\yourpythoninstall\\Lib\\distutils\\msvc9compiler.py` by adding the following highlighted lines at the top
  of the `find_vcvarsall()` function:

  .. code-block:: python
     :emphasize-lines: 8-13

     def find_vcvarsall(version):
         """Find the vcvarsall.bat file

         At first it tries to find the productdir of VS 2008 in the registry. If
         that fails it falls back to the VS90COMNTOOLS env var.
         """
         vsbase = VS_BASE % version
         vcpath = os.environ['ProgramFiles']
         vcpath = os.path.join(vcpath, 'Common Files', 'Microsoft',
             'Visual C++ for Python', '9.0', 'vcvarsall.bat')
         if os.path.isfile(vcpath): return vcpath
         vcpath = os.path.join(os.environ['LOCALAPPDATA'], 'Programs', 'Common', 'Microsoft', 'Visual C++ for Python', '9.0', 'vcvarsall.bat')
         if os.path.isfile(vcpath): return vcpath
         ...

* Create a file `distutils.cfg` in `C:\\yourpythoninstall\\Lib\\distutils\\` with the following:

  .. code-block:: none

     [build]
     compiler=msvc

You should now be able to build extensions.
If these steps don't work for you, or you are using a different version of Python,
`this page <https://github.com/cython/cython/wiki/CythonExtensionsOnWindows>`_ or
`this page <https://wiki.python.org/moin/WindowsCompilers>`_ may be helpful.

Installation
------------
You can download `gripspy` either as a Git repository or just the code itself.
Installing `grispy` can be done through the basic ways (e.g., `python setup.py`),
although I prefer using `pip` so that tracking installations is easier.
I recommend installing `gripspy` in "editable" mode so that it is more convenient
for active development, e.g.::

   pip install -e .

Use
---
Simply import `gripspy` to start.

.. code-block:: python

   import gripspy

The classes for handling science data are under `gripspy.science`.
The classes for handling housekeeping data are under `gripspy.housekeeping`.
Here are some examples of what has been implemented so far:

.. code-block:: python

   data = gripspy.science.GeData(detector_number, filename)
   data = gripspy.science.BGOCounterData(filename)
   data = gripspy.science.BGOEventData(filename)
   data = gripspy.science.PYSequence(filelist)
   data = gripspy.housekeeping.GPSData(filename)

Code Reference
--------------

.. toctree::
   :maxdepth: 2

   code_ref/index.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
