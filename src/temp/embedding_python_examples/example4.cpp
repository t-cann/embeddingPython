#define PY_SSIZE_T_CLEAN
#include <Python.h>

/**
 * @brief Example of using Py_BuildValue and PyObject_SetAttrString to make variable that are accessible in the Global/__main__ namespace. 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char* argv[])
{   
    PyObject *m , *v;

    wchar_t** _argv = (wchar_t **)PyMem_Malloc(sizeof(wchar_t*) * argc);
    for (size_t i = 0; i < argc; i++)
    {
        wchar_t* arg = Py_DecodeLocale(argv[i], NULL);
        _argv[i] = arg;
    }

    Py_Initialize();
    m = PyImport_AddModule("__main__");
    if(m==NULL)
        return -1;

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

    v = Py_BuildValue("s","Thomas");
    if(v==NULL)
        return -1;
    if(PyObject_SetAttrString(m, "name", v) == -1){
        return -1;
    }

    v = Py_BuildValue("[i,i]", 123, 456);
    if(v==NULL)
        return -1;
    if(PyObject_SetAttrString(m, "array", v) == -1){
        return -1;
    }
    
    Py_Main(argc,_argv);

    if (Py_FinalizeEx() < 0) {
        exit(120);
    }
    return 0;
}


