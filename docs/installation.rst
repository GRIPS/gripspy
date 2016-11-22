Installation
============

Prerequisites
-------------
This package depends on:

* Python 3.5 or 2.7 (untested on other versions)
* NumPy
* SciPy
* matplotlib
* Cython, and corresponding build environment for compiling C code
* scikit-image
* Astropy

However, rather than installing these packages individually, it is highly recommended that one
instead use a scientific Python distribution (e.g., `Anaconda <https://www.continuum.io/downloads>`_),
which will include useful packages such as Jupyter Notebook (formerly IPython Notebook).

The build environment for C code is a bit trickier to set up on Windows than on the other platforms:

* Python 3.5:

  * Install `Microsoft Visual C++ Build Tools 2015 <http://landinghub.visualstudio.com/visual-cpp-build-tools>`_ (check both Windows 8.1 SDK and Windows 10 SDK)

* Python 2.7:

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
`this other page <https://wiki.python.org/moin/WindowsCompilers>`_ may be helpful.

Installing `gripspy`
--------------------
You can download `gripspy` either as a Git repository or just the code itself.
Installing `gripspy` can be done through the basic ways (e.g., `python setup.py`),
although I prefer using `pip` so that tracking installations is easier.
I recommend installing `gripspy` in "editable" mode so that it is more convenient
for active development, e.g.::

   pip install -e .
