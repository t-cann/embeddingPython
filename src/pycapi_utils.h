//pycapi_utils.h
#ifndef PYCAPI_UTILS_H
#define PYCAPI_UTILS_H

#include <Python.h>
#include <numpy/arrayobject.h>

PyObject* array_double_to_pyobj(double* v_c, long int NUMEL); //Convert from array to Python list (double)

#endif //PYCAPI_UTILS_H
