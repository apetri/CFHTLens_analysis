/*Python wrapper module for JMK interpolation code;
the method used is the same as in 
http://dan.iel.fm/posts/python-c-extensions/ 

The module is called _WLInterpolate and it defines a method called
WLInterpolate()
*/

#include <Python.h>
#include <numpy/arrayobject.h>

#include "function.h"

//Python module docstrings 
static char module_docstring[] = "This module provides a python interface for the Weak Lensing Statistics interpolator";
static char WLInterpolate_docstring[] = "Calculate the binned statistic in question given a set of cosmological parameters";

//WL_Interpolate declaration
static PyObject *_WLInterpolate_WLInterpolate(PyObject *self, PyObject *args); 

//_WLInterpolate method definitions
static PyMethodDef module_methods[] = {
	{"WLInterpolate",_WLInterpolate_WLInterpolate,METH_VARARGS,WLInterpolate_docstring},
	{NULL,NULL,0,NULL}
};

//_WLInterpolate constructor
PyMODINIT_FUNC init_WLInterpolate(void){

	PyObject *m = Py_InitModule3("_WLInterpolate",module_methods,module_docstring);
	if(m==NULL) return;

	/*Load numpy functionality*/
	import_array();

}

//WL_Interpolate implementation
static PyObject *_WLInterpolate_WLInterpolate(PyObject *self, PyObject *args){

	PyObject *cosmo_params_obj;
	int Nbins;

	/*Parse the input tuple*/
	if(!PyArg_ParseTuple(args,"Oi",&cosmo_params_obj,&Nbins)){ 
		return NULL;
	}

	/*Interpret the input object as numpy array*/
	PyObject *cosmo_params_array = PyArray_FROM_OTF(cosmo_params_obj,NPY_DOUBLE,NPY_IN_ARRAY);

	/*If that didn't work, throw an exception*/
	if(cosmo_params_array==NULL){
		return NULL;
	}

	/*How many cosmological parameters are there?*/
	int Nparams = (int)PyArray_DIM(cosmo_params_array,0);

	/*Get pointer to the cosmological parameters as C types*/
	double *cosmo_params = (double *)PyArray_DATA(cosmo_params_array);

	/*Create a new array object, which will be the output of the method*/
	npy_intp dims[] = {(npy_intp) Nbins};
	PyObject *descriptor_array = PyArray_SimpleNew(1,dims,NPY_DOUBLE);

	/*Throw exception if this failed*/
	if(descriptor_array==NULL){
		Py_DECREF(cosmo_params_array);
		return NULL;
	}

	/*Call the interpolation function, which will write to the array memory*/
	function(cosmo_params,Nparams,(double *)PyArray_DATA(descriptor_array),Nbins);

	/*Clean up and decrease reference count for cosmo_params_array since we don't need it anymore*/
	Py_DECREF(cosmo_params_array); 

	/*Return pointer to the numpy array which contains the interpolated descriptor*/

	return descriptor_array;
	
}

