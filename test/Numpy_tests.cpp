#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

/* Fixture for lots of similar test */
struct NumpyAPITests : public ::testing::Test
{
    wchar_t *program;

    virtual void SetUp() override
    {
        // printf("Starting Up!\n");
        program = Py_DecodeLocale(::testing::UnitTest::GetInstance()->current_test_info()->name(), NULL);
        Py_SetProgramName(program); /* optional but recommended */
        Py_Initialize();
        if(PyArray_API == NULL)
        {
            _import_array(); //https://stackoverflow.com/questions/32899621/numpy-capi-error-with-import-array-when-compiling-multiple-modules
        }
    }

    virtual void TearDown() override
    {
        // printf("Tearing Down!\n");
        if(PyErr_Occurred()){
            PyErr_Print();  
        }
        if (Py_FinalizeEx() < 0)
        {
            exit(120);
        }
        Py_Finalize();
        PyMem_RawFree(program);
    }
};

TEST_F(NumpyAPITests, init)
{
    
    // Build the 2D array in C++
    const int SIZE = 10;
    npy_intp dims[2]{SIZE, SIZE};
    const int ND = 2;
    long double(*c_arr)[SIZE]{ new long double[SIZE][SIZE] };
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            c_arr[i][j] = i * SIZE + j;

    ASSERT_TRUE(c_arr != NULL);
    // std::cout << "Finish Creating 2D Array in C++" << std::endl;

    // Outputing to Console
    // for (int i = SIZE - 1; i >= 0; i--){
    //     for(int j = SIZE -1; j>=0; j--){
    //         std::cout << c_arr[i][0] << " ";
    //     }
    //     std::cout << std::endl;
    // }   
    

    // Convert it to a NumPy array.
    PyObject *pArray = PyArray_SimpleNewFromData(ND, dims, NPY_LONGDOUBLE, reinterpret_cast<void*>(c_arr));
    // std::cerr << &pArray;
    ASSERT_NE(pArray, nullptr) ;
    // PyArrayObject *np_arr = reinterpret_cast<PyArrayObject*>(pArray); //Not Used. only recasting varible into PyArrayObject type which might enable extra functions.

    // std::cout << "Converted to Numpy Array" << std::endl;

    // // import mymodule.array_tutorial
    const char *module_name = "mymodule";
    PyObject *pName = PyUnicode_FromString(module_name);
    PyObject* pModule = PyImport_Import(pName);
    ASSERT_TRUE(pModule != NULL) << "Error importing module -- Check Module is compiled and in PYTHONPATH";
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
    // std::cout << "Printing output array" << std::endl;
    // for (int i = 0; i < len; i++)
    //     std::cout << c_out[i] << ' ';
    // std::cout << std::endl;

}

TEST_F(NumpyAPITests, PyArray_NewFromData_Test)
{

    // array dimensions
    npy_intp dim[] = {5, 5};

    // array data
    std::vector<double> buffer(25, 1.0);

    // create a new array using 'buffer'
    PyObject* array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, &buffer[0]);
    PyArrayObject *np_arr = reinterpret_cast<PyArrayObject*>(array_2d);

    // https://gist.github.com/maartenbreddels/82a3778c9a79b7ef048e inspiration for tests.

    ASSERT_TRUE(array_2d != NULL);                      //Checks Object is not NULL e.g. Created/Defined.
    ASSERT_TRUE( PyArray_Check(array_2d));              //Checks Object is PyArray Type
    ASSERT_TRUE((int)PyArray_NDIM(np_arr) == 2 );       //Checks Object is 2 Dimensional as defined. 
    ASSERT_TRUE(PyArray_TYPE(np_arr) == NPY_DOUBLE );   //Checks the type of the object is correct. 


    //Don't understand Strides. 

}


TEST_F(NumpyAPITests, PyArray_SimpleNewFromData_Test)
{

    const int ND = 2;
    const int SIZE = 10;
    npy_intp dims[2]{SIZE, SIZE};
    int typeint = NPY_DOUBLE; // https://numpy.org/doc/stable/reference/c-api/dtype.html#c.NPY_FLOAT
    void* data;

    PyObject *pArray = PyArray_SimpleNewFromData(ND, dims, NPY_LONGDOUBLE, reinterpret_cast<void*>(data));
    PyArrayObject *np_arr = reinterpret_cast<PyArrayObject*>(pArray);
    ASSERT_TRUE(pArray != NULL);                      //Checks Object is not NULL e.g. Created/Defined.
    ASSERT_TRUE( PyArray_Check(pArray));              //Checks Object is PyArray Type
    ASSERT_TRUE((int)PyArray_NDIM(np_arr) == 2 );       //Checks Object is 2 Dimensional as defined. 
    ASSERT_TRUE(PyArray_TYPE(np_arr) == NPY_LONGDOUBLE );   //Checks the type of the object is correct. 

}

TEST_F(NumpyAPITests, Types_Test)
{
    bool x = true;
}

TEST_F(NumpyAPITests, PyArray_Dims_Test)
{
    // PyArray_Dims

    // This structure is very useful when shape and/or strides information is supposed to be interpreted. The structure is:

    // typedef struct {
    //     npy_intp *ptr;
    //     int len;
    // } PyArray_Dims;

    // The members of this structure are

    // npy_intp *PyArray_Dims.ptr

    //     A pointer to a list of (npy_intp) integers which usually represent array shape or array strides.

    // int PyArray_Dims.len

    //     The length of the list of integers. It is assumed safe to access ptr [0] to ptr [len-1].

}


// Testing Advance GoogleTest Features 

// int main(int argc, char** argv) {
//   // Disables elapsed time by default.
//   ::testing::GTEST_FLAG(print_time) = false;

//   // This allows the user to override the flag on the command line.
//   ::testing::InitGoogleTest(&argc, argv);

//   return RUN_ALL_TESTS();
// }