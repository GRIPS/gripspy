from setuptools import setup, Extension, find_packages

from Cython.Build import cythonize

ext_modules = [Extension("gripspy.util.checksum", ["gripspy/util/checksum.pyx"]),
               Extension("gripspy.util.coincidence", ["gripspy/util/coincidence.pyx"])]

setup(
    name='gripspy',
    version='0.1',
    author='Albert Y. Shih',
    author_email='ayshih@gmail.com',
    packages=find_packages(),
    package_data={'gripspy.science': ['asicmap/*.txt', 'cms/*.npz']},
    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'cython',
                      'scikit-image',
                      'astropy'],
    ext_modules = cythonize(ext_modules, annotate=True, compiler_directives={'embedsignature': True}),
    url='',
    license='See LICENSE.txt',
    description='',
    long_description=open('README.md').read(),
)
