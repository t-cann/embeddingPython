//pycapi_utils.cpp
// https://stackoverflow.com/questions/32899621/numpy-capi-error-with-import-array-when-compiling-multiple-modules
#include "pycapi_utils.h"


PyObject* array_double_to_pyobj(double* v_c, long int NUMEL){
    //Convert a double array to a Numpy array
    PyObject* out_array = PyArray_SimpleNew(1, &NUMEL, NPY_DOUBLE);
    double* v_b = (double*) ((PyArrayObject*) out_array)->data;
    for (int i=0;i<NUMEL;i++) v_b[i] = v_c[i];
    free(v_c);
    return out_array;
}