import os,sys,re

try:
	from setuptools import setup
	setup
except ImportError:
	from distutils.core import setup
	setup

def rd(filename):
	
	f = file(filename,"r")
	r = f.read()
	f.close()

	return r


vre = re.compile("__version__ = \"(.*?)\"")
m = rd(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "GaussianProcess", "__init__.py"))
version = vre.findall(m)[0]


setup(
	name="GaussianProcess",
	version=version,
	author="Andrea Petri, Jia Liu, Jan M. Kratochvil",
	author_email="apetri@phys.columbia.edu",
	packages=["GaussianProcess"],
	url="?",
	license="?",
	description="Toolkit for Gaussian Process Regression",
	long_description=rd("README.md"),
	install_requires=["numpy","scipy"],
	classifiers=[
		"Development Status :: 2 - Pre-Alpha",
		"Intended Audience :: Science/Research",
		"Operating System :: OS Independent",
		"Programming Language :: Python",
		"Programming Language :: C",
		"License :: Public Domain"
	],
)
