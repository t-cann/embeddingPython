#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <string>   // Require for
#include <iostream> // Require for
#include <fstream>  // Require for
#include <gtest/gtest.h>

/* Fixture for lots of similar test */
struct Python_C_API_Tests : public ::testing::Test
{
    wchar_t *program;
   
    virtual void SetUp() override
    {
        // printf("Starting Up!\n");
        program = Py_DecodeLocale(::testing::UnitTest::GetInstance()->current_test_info()->name(), NULL);
        Py_SetProgramName(program); /* optional but recommended */
        Py_Initialize();
    }

    virtual void TearDown() override
    {
        // printf("Tearing Down!\n");
        if(PyErr_Occurred()){
            PyErr_Print();  
        }
        // if (Py_FinalizeEx() < 0)
        // {
        //     exit(120);
        // }
        Py_Finalize();
        PyMem_RawFree(program);
    }
};

TEST_F(Python_C_API_Tests, Py_Initialise_Test)
{
    // int argc = 1;
    // const char *argv[] = {"Py_Initialise_Test", NULL};
    // wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    // ASSERT_TRUE(program != NULL);
    // Py_SetProgramName(program);
    // Py_Initialize();

    PyRun_SimpleString("from time import time,ctime\n"
                       "print('Today is', ctime(time()))\n");

    // if (Py_FinalizeEx() < 0)
    // {
    //     exit(120);
    // }
    // PyMem_RawFree(program);

}

TEST_F(Python_C_API_Tests, PyRun_SimpleString_Test)
{

    const char *inputs[] = {"int = 10", "double = 3.14", "string = 'Thomas'"};

    PyRun_SimpleString(inputs[0]);
    PyRun_SimpleString(inputs[1]);
    PyRun_SimpleString(inputs[2]);

    PyObject *m, *i, *d, *s;
    m = PyImport_AddModule("__main__");

    i = PyObject_GetAttrString(m, "int");
    d = PyObject_GetAttrString(m, "double");
    s = PyObject_GetAttrString(m, "string");

    Py_XDECREF(m);

    //Int
    EXPECT_EQ(10, PyLong_AsLong(i));
    
    //Doubles
    EXPECT_EQ(3.14, PyFloat_AsDouble(d));

    //Strings
    EXPECT_TRUE(PyUnicode_Check(s)); 
    const char* str = PyUnicode_AsUTF8(s);
    const char* name = "Thomas";
    EXPECT_STREQ(str, name);

    // Release reference to object
    Py_XDECREF(i);
    Py_XDECREF(d);
    Py_XDECREF(s);


}

TEST_F(Python_C_API_Tests, PyRun_SimpleFile_Test)
{

    std::ofstream myfile;
    myfile.open("test.py");
    myfile << "#!/usr/bin/python\n\n";
    myfile << "int = 10\n";
    myfile << "double = 3.14\n";
    myfile << "string = 'Thomas'";
    myfile.close();


    const char *inputs[] = {"int = 10", "double = 3.14", "string = 'Thomas'"};

    FILE* pFile;
    
    pFile = fopen ("test.py","r");
    if(pFile!=NULL)
        PyRun_AnyFile(pFile, "test");

    PyObject *m, *i, *d, *s;
    m = PyImport_AddModule("__main__");

    i = PyObject_GetAttrString(m, "int");
    d = PyObject_GetAttrString(m, "double");
    s = PyObject_GetAttrString(m, "string");

    Py_XDECREF(m);

    //Int
    EXPECT_EQ(10, PyLong_AsLong(i));
    
    //Doubles
    EXPECT_EQ(3.14, PyFloat_AsDouble(d));

    //Strings
    EXPECT_TRUE(PyUnicode_Check(s)); 
    const char* str = PyUnicode_AsUTF8(s);
    const char* name = "Thomas";
    EXPECT_STREQ(str, name);

    // Release reference to object
    Py_XDECREF(i);
    Py_XDECREF(d);
    Py_XDECREF(s);

    remove("test.py");
}

//Work in Progress Below

TEST_F(Python_C_API_Tests, Python_C_API_Memory_Test)
{
    //TODO
    EXPECT_TRUE(true);
}

TEST_F(Python_C_API_Tests, PyBuildValue_Test)
{

    PyObject *m;
    m = PyImport_AddModule("__main__");

    /* Python call: f() */
    // Py_BuildValue("")                                       // None
    // Py_BuildValue("i", 123)                                 // 123
    // Py_BuildValue("iii", 123, 456, 789)                     // (123, 456, 789)
    // Py_BuildValue("s", "hello")                             // 'hello'
    // Py_BuildValue("y", "hello")                             // b'hello'
    // Py_BuildValue("ss", "hello", "world")                   // ('hello', 'world')
    // Py_BuildValue("s#", "hello", 4)                         // 'hell'
    // Py_BuildValue("y#", "hello", 4)                         // b'hell'
    // Py_BuildValue("()")                                     // ()
    // Py_BuildValue("(i)", 123)                               // (123,)
    // Py_BuildValue("(ii)", 123, 456)                         // (123, 456)
    // Py_BuildValue("(i,i)", 123, 456)                        // (123, 456)
    // Py_BuildValue("[i,i]", 123, 456)                        // [123, 456]
    // Py_BuildValue("{s:i,s:i}", "abc", 123, "def", 456)      // {'abc': 123, 'def': 456}
    // Py_BuildValue("((ii)(ii)) (ii)", 1, 2, 3, 4, 5, 6)      // (((1, 2), (3, 4)), (5, 6))

    PyObject_SetAttrString(m, "x",Py_BuildValue(""));                                  // None
    //EXPECT_EQ(NULL, PyUnicode_AsUTF8(PyObject_GetAttrString(m, "x")));
    PyObject_SetAttrString(m, "x",Py_BuildValue("i", 123));                               // 123
    PyObject_SetAttrString(m, "x",Py_BuildValue("iii", 123, 456, 789));                   // (123, 456, 789)
    PyObject_SetAttrString(m, "x",Py_BuildValue("s", "hello") );                          // 'hello'
    PyObject_SetAttrString(m, "x",Py_BuildValue("y", "hello"));                           // b'hello'
    PyObject_SetAttrString(m, "x",Py_BuildValue("ss", "hello", "world"));                 // ('hello', 'world')
    PyObject_SetAttrString(m, "x",Py_BuildValue("s#", "hello", 4));                       // 'hell'
    PyObject_SetAttrString(m, "x",Py_BuildValue("y#", "hello", 4));                       // b'hell'
    PyObject_SetAttrString(m, "x",Py_BuildValue("()"));                                   // ()
    PyObject_SetAttrString(m, "x",Py_BuildValue("(i)", 123));                             // (123,)
    PyObject_SetAttrString(m, "x",Py_BuildValue("(ii)", 123, 456) );                      // (123, 456)
    PyObject_SetAttrString(m, "x",Py_BuildValue("(i,i)", 123, 456));                      // (123, 456)
    PyObject_SetAttrString(m, "x",Py_BuildValue("[i,i]", 123, 456));                      // [123, 456]
    PyObject_SetAttrString(m, "x",Py_BuildValue("{s:i,s:i}", "abc", 123, "def", 456));    // {'abc': 123, 'def': 456}
    PyObject_SetAttrString(m, "x",Py_BuildValue("((ii)(ii)) (ii)", 1, 2, 3, 4, 5, 6));    // (((1, 2), (3, 4)), (5, 6))
    
    // v = Py_BuildValue("s","Thomas");
    // if(v==NULL)
    //     return -1;
    // if(PyObject_SetAttrString(m, "name", v) == -1){
    //     return -1;
    // }

    // v = Py_BuildValue("[i,i]", 123, 456);
    // if(v==NULL)
    //     return -1;
    // if(PyObject_SetAttrString(m, "array", v) == -1){
    //     return -1;
    // }


    Py_XDECREF(m);
}

// TEST_F(Python_C_API_Tests, Arrays_to_Python_Test)
// {
//     EXPECT_TRUE(true);
// }
