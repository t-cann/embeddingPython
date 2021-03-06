#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION //https://numpy.org/devdocs/release/1.19.2-notes.html

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

TEST_F(NumpyAPITests, PyArray_NewFromData_withvector_Test)
{

    // array dimensions
    npy_intp dim[] = {5, 5};

    // array data
    // double a[] = {25,1.0};
    // std::vector<double> *buffer = new std::vector<double>;
    // buffer->assign(a,a+2);
    std::vector<double> buffer(25,1.0);

    // create a new array using 'buffer'
    PyObject* array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, &buffer[0]);
    PyArrayObject *np_arr = reinterpret_cast<PyArrayObject*>(array_2d);

    // https://gist.github.com/maartenbreddels/82a3778c9a79b7ef048e inspiration for tests.

    ASSERT_TRUE(array_2d != NULL);                      //Checks Object is not NULL e.g. Created/Defined.
    ASSERT_TRUE( PyArray_Check(array_2d));              //Checks Object is PyArray Type
    ASSERT_TRUE((int)PyArray_NDIM(np_arr) == 2 );       //Checks Object is 2 Dimensional as defined. 
    ASSERT_TRUE(PyArray_TYPE(np_arr) == NPY_DOUBLE );   //Checks the type of the object is correct. 

    //Don't understand Strides. 

    double* c_out = reinterpret_cast<double*>(PyArray_DATA(np_arr));
    EXPECT_TRUE(c_out[0]=25)<< c_out[0];
    EXPECT_TRUE(c_out[1]==1.0) << c_out[1]; //Issue with understand of Buffers and Memory Allocation
    buffer.clear();
    // std::cout << "After buffer/data is deleted Numpy Array double go to " << c_out[0] << std::endl;
    // EXPECT_FALSE(c_out[0] == 25)<< c_out[0]; // TODO does not work
    // EXPECT_FALSE(c_out[1] == 1.0) << c_out[1];


}

TEST_F(NumpyAPITests, PyArray_NewFromData_witharray_Test)
{
    // array dimensions
    npy_intp dim[] = {5, 5};

    // array data
    double* buffer = new double[1];
    buffer[0] = 25;
    buffer[1] = 1.0;

    // create a new array using 'buffer'
    PyObject* array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, &buffer[0]);
    PyArrayObject *np_arr = reinterpret_cast<PyArrayObject*>(array_2d);

    // https://gist.github.com/maartenbreddels/82a3778c9a79b7ef048e inspiration for tests.

    ASSERT_TRUE(array_2d != NULL);                      //Checks Object is not NULL e.g. Created/Defined.
    ASSERT_TRUE( PyArray_Check(array_2d));              //Checks Object is PyArray Type
    ASSERT_TRUE((int)PyArray_NDIM(np_arr) == 2 );       //Checks Object is 2 Dimensional as defined. 
    ASSERT_TRUE(PyArray_TYPE(np_arr) == NPY_DOUBLE );   //Checks the type of the object is correct. 

    //Don't understand Strides. 

    double* c_out = reinterpret_cast<double*>(PyArray_DATA(np_arr));
    // EXPECT_TRUE(c_out[0]==buffer[0]);
    // EXPECT_TRUE(c_out[1]==buffer[1]);
    EXPECT_TRUE(c_out[0] == 25)<< c_out[0];
    EXPECT_TRUE(c_out[1] == 1.0) << c_out[1]; 

    delete(buffer);
    std::cout << "After buffer/data is deleted Numpy Array double go to " << c_out[0] << std::endl;
    EXPECT_FALSE(c_out[0] == 25)<< c_out[0]; // 
    EXPECT_FALSE(c_out[1] == 1.0) << c_out[1];
}

TEST_F(NumpyAPITests, PyArray_SimpleNewFromData_Test)
{
    //Inputs
    const int ND = 2;
    const int SIZE = 10;
    npy_intp dims[2]{SIZE, SIZE};
    int typeint = NPY_DOUBLE; // https://numpy.org/doc/stable/reference/c-api/dtype.html#c.NPY_FLOAT
    void* data;

    //Manipulation
    PyObject *pArray = PyArray_SimpleNewFromData(ND, dims, NPY_LONGDOUBLE, reinterpret_cast<void*>(data));
    PyArrayObject *np_arr = reinterpret_cast<PyArrayObject*>(pArray);

    //Tests
    ASSERT_TRUE(pArray != NULL);                      //Checks Object is not NULL e.g. Created/Defined.
    ASSERT_TRUE( PyArray_Check(pArray));              //Checks Object is PyArray Type
    ASSERT_TRUE((int)PyArray_NDIM(np_arr) == 2 );       //Checks Object is 2 Dimensional as defined. 
    ASSERT_TRUE(PyArray_TYPE(np_arr) == NPY_LONGDOUBLE );   //Checks the type of the object is correct. 

}

TEST_F(NumpyAPITests, PyArray_Copy_Test)
{
    // array dimensions
    npy_intp dim[] = {2};

    // array data
    double* buffer = new double[1];
    buffer[0] = 25;
    buffer[1] = 1.0;

    // create a new array using 'buffer'
    PyObject* array_2d = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, &buffer[0]);
    PyArrayObject *np_arr = reinterpret_cast<PyArrayObject*>(array_2d);
    PyObject* copy_arr = PyArray_Copy(np_arr);
    PyArrayObject* npcopy_arr = reinterpret_cast<PyArrayObject*>(copy_arr);

    double* c_out = reinterpret_cast<double*>(PyArray_DATA(npcopy_arr));
    // EXPECT_TRUE(c_out[0]==buffer[0]);
    // EXPECT_TRUE(c_out[1]==buffer[1]);
    EXPECT_TRUE(c_out[0] == 25)<< c_out[0];
    EXPECT_TRUE(c_out[1] == 1.0) << c_out[1]; 

    delete(buffer);
    // std::cout << "After buffer/data is deleted Numpy Array double go to " << c_out[0] << std::endl;
    EXPECT_TRUE(c_out[0] == 25)<< c_out[0]; // 
    EXPECT_TRUE(c_out[1] == 1.0) << c_out[1];


}

TEST_F(NumpyAPITests, Arrays_to_Python_Test)
{
    PyObject *m;
    m = PyImport_AddModule("__main__");
    
    // array
    npy_intp dim[] = {2};
    double* buffer = new double[1];
    buffer[0] = 25;
    buffer[1] = 1.0;

    // create a new array using 'buffer'
    PyObject* array = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, &buffer[0]);
    PyArrayObject *np_arr_tocopy = reinterpret_cast<PyArrayObject*>(array);
    PyObject* py_copy= PyArray_Copy(np_arr_tocopy);

    PyObject_SetAttrString(m, "move", array);
    PyObject_SetAttrString(m, "copy", py_copy);

    PyObject *move = PyObject_GetAttrString(m, "move");
    PyObject *copy = PyObject_GetAttrString(m, "copy");
    
    PyArrayObject *np_arr = reinterpret_cast<PyArrayObject*>(move);

    double* c_out_move = reinterpret_cast<double*>(PyArray_DATA(np_arr));
    EXPECT_TRUE(c_out_move[0] == 25)<< c_out_move[0];
    EXPECT_TRUE(c_out_move[1] == 1.0) << c_out_move[1]; 

    delete(buffer);
    // std::cout << "After buffer/data is deleted Numpy Array double go to " << c_out[0] << std::endl;
    EXPECT_FALSE(c_out_move[0] == 25)<< c_out_move[0]; // 
    EXPECT_FALSE(c_out_move[1] == 1.0) << c_out_move[1];

    PyArrayObject *np_arr_copy = reinterpret_cast<PyArrayObject*>(copy);

    double* c_out = reinterpret_cast<double*>(PyArray_DATA(np_arr_copy));
    EXPECT_TRUE(c_out[0] == 25)<< c_out[0];
    EXPECT_TRUE(c_out[1] == 1.0) << c_out[1]; 
}



// Work in progress

TEST_F(NumpyAPITests, PyArray_NewFromDescr_Test)
{
    //Inputs
    PyTypeObject* subtype;
    PyArray_Descr* descr;
    int nd;
    npy_intp const* dims;
    npy_intp const* strides;
    void* data;                     // If data is provided, it must stay alive for the life of the array.
    int flags;
    PyObject* obj; 

    // Max number of DIMs in TIO 


    // descr = PyArray_DescrFromType();

    // //Manipulation
    //  PyObject *pArray = PyArray_NewFromDescr(subtype, descr, nd, dims , strides, data, flags, obj);
    // PyArrayObject *np_arr = reinterpret_cast<PyArrayObject*>(pArray);

    // //Tests
    // ASSERT_TRUE(pArray != NULL);                      //Checks Object is not NULL e.g. Created/Defined.
    // ASSERT_TRUE( PyArray_Check(pArray));              //Checks Object is PyArray Type
    // ASSERT_TRUE((int)PyArray_NDIM(np_arr) == 2 );       //Checks Object is 2 Dimensional as defined. 
    // ASSERT_TRUE(PyArray_TYPE(np_arr) == NPY_LONGDOUBLE );   //Checks the type of the object is correct. 

}

TEST_F(NumpyAPITests, Types_Test)
{
    int expected_array[3]= {1,2,3};

    NPY_TYPES type;
    //  PyArray_TypeNumFromName();
    // switch (type)
    // {
    // case :
    //     /* code */
    //     break;
    
    // default:
    //     break;
    // }

    //if NUMPY Enabled  in headers  add to TIOBROWSE ??

    // Static Array of max length rather than pointer / 

    // const Max_Dims =7;

    // if def ## 
    //ifndef

    //ARRAYQUANTS only onsite

    // 1D
    // 2D
    // 3D
    // Varibles -> Dim7  -- Come up with an example?
    // Vector type -> as invidual components

    // NPY_BOOL=0,
    // NPY_BYTE
    // NPY_UBYTE
    // NPY_SHORT
    // NPY_USHORT
    // NPY_INT
    // NPY_UINT
    // NPY_LONG
    // NPY_ULONG
    // NPY_LONGLONG
    // NPY_ULONGLONG
    // NPY_FLOAT
    // NPY_DOUBLE
    // NPY_LONGDOUBLE
    // NPY_CFLOAT
    // NPY_CDOUBLE
    // NPY_CLONGDOUBLE
    // NPY_OBJECT
    // NPY_STRING
    // NPY_UNICODE
    // NPY_VOID
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