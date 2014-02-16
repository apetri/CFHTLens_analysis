from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

pyxfile = 'KSI'

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension(pyxfile, ["%s.pyx"%(pyxfile)], include_dirs=[np.get_include()])]
)
