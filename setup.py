from distribute_setup import use_setuptools
use_setuptools()

from setuptools import setup, Extension

from Cython.Build import cythonize

ext_modules = [Extension("gripspy.util.checksum", ["gripspy/util/checksum.pyx"])]

setup(
    name='gripspy',
    version='0.1',
    author='',
    author_email='',
    packages=['gripspy'],
    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'cython'],
    ext_modules = cythonize(ext_modules, annotate=True),
    url='',
    license='See LICENSE.txt',
    description='',
    long_description=open('README.md').read(),
)
