from distutils.core import setup,Extension
import numpy.distutils.misc_util

setup(
	name="WLInterpolate",
	version="0.1",
	author = "Andrea Petri",
	author_email = "apetri@phys.columbia.edu",
	url = "https://github.com/apetri/WL_Emulator",
	description = "Python wrapper of Weak Lensing Emulator",
	install_requires = ["numpy"],
	ext_modules=[Extension("_WLInterpolate",["_WLInterpolate.c","function.c"])],
	include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)