#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <gtest/gtest.h>

TEST(Numpy_API_Tests, init)
{
    wchar_t* program = Py_DecodeLocale("init", NULL);
    Py_SetProgramName(program);
    Py_Initialize();
    if(_import_array()<0){
        ASSERT_TRUE(false);
    }
        

    // Build the 2D array in C++
    const int SIZE = 10;
    npy_intp dims[2]{SIZE, SIZE};
    const int ND = 2;
    long double(*c_arr)[SIZE]{ new long double[SIZE][SIZE] };
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            c_arr[i][j] = i * SIZE + j;

    ASSERT_FALSE(c_arr == NULL);
    // std::cout << "Finish Creating 2D Array in C++" << std::endl;

    // Outputing to Console
    // for (int i = SIZE - 1; i >= 0; i--){
    //     for(int j = SIZE -1; j>=0; j--){
    //         std::cout << c_arr[i][0] << " ";
    //     }
    //     //std::cout << std::endl;
    // }   
    

    // Convert it to a NumPy array.
    PyObject *pArray = PyArray_SimpleNewFromData(ND, dims, NPY_LONGDOUBLE, reinterpret_cast<void*>(c_arr));
    // std::cerr << &pArray;
    ASSERT_NE(pArray, nullptr) ;
    // PyArrayObject *np_arr = reinterpret_cast<PyArrayObject*>(pArray); //NOt Used ??

    // std::cout << "Converted to Numpy Array" << std::endl;

    // // import mymodule.array_tutorial
    const char *module_name = "mymodule";
    PyObject *pName = PyUnicode_FromString(module_name);
    PyObject* pModule = PyImport_Import(pName);
    const char *func_name = "array_tutorial";
    PyObject *pFunc = PyObject_GetAttrString( pModule , func_name);
    if(pFunc == NULL){
        std::cout << "Error";
        // return -1;
    }
    // np_ret = mymodule.array_tutorial(np_arr)
    PyObject *pReturn = PyObject_CallFunctionObjArgs(pFunc, pArray, NULL);
    if(pReturn == NULL){
        std::cerr << "Error";
    }    
    PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(pReturn);
    
    // Convert back to C++ array and print.
    int len = PyArray_SHAPE(np_ret)[0];
    long double* c_out;
    c_out = reinterpret_cast<long double*>(PyArray_DATA(np_ret));
    std::cout << "Printing output array" << std::endl;
    for (int i = 0; i < len; i++)
        std::cout << c_out[i] << ' ';
    std::cout << std::endl;
    
    // result = EXIT_SUCCESS;

    if(PyErr_Occurred()){
        PyErr_Print();  
    }
    Py_FinalizeEx();
    
    // return result;
}