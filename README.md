Introduction
============
This Python package is intended for analysis of data from the [GRIPS](http://grips.ssl.berkeley.edu) long-duration flight of 2016 January 19–30.  It can also be used to analyze the pre-flight calibration data that was already being recorded using flight-like packet formats.  The code is still very much a work in progress, so not only are many elements still missing, even the parts that have been coded are likely to go through changes in their interfaces and organization.

Prerequisites
=============
This package depends on:

* Python 2.7
* NumPy
* SciPy
* matplotlib
* Cython, and corresponding build environment for compiling C code

However, rather than installing these packages individually, it is highly recommended that one instead use a scientific Python distribution (e.g., [Anaconda](https://www.continuum.io/downloads)), which will include useful packages such as Jupyter Notebook (fomerly IPython Notebook).

The build environment for C code is a bit trickier to set up on Windows than on the other platforms.  I will write out the steps at a later time.

Installation
============
You can download `gripspy` either as a Git repository or just the code itself.  Installing `grispy` can be done through the basic ways (e.g., `python setup.py`), although I prefer using `pip` so that tracking installations is easier.  I recommend installing `gripspy` in "editable" mode so that it is more convenient for active development, e.g.:
```
pip install -e .
```

Use
===
Simply import `gripspy` to start.
```python
import gripspy
```
The class for handling science data are under `gripspy.science`.  Here are some examples of what has been implemented so far:
```python
data = gripspy.science.GeData(detector_number, filename)
data = gripspy.science.BGOCounterData(filename)
data = gripspy.science.BGOEventData(filename)
data = gripspy.science.PYSequence(filelist)
```

