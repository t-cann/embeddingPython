#define PY_SSIZE_T_CLEAN
#include <Python.h>

static int numargs=0;

/* Return the number of arguments of the application command line */
static PyObject*
emb_numargs(PyObject *self, PyObject *args)
{
    if(!PyArg_ParseTuple(args, ":numargs"))
        return NULL;
    return PyLong_FromLong(numargs);
}

static PyMethodDef EmbMethods[] = {
    {"numargs", emb_numargs, METH_VARARGS,
     "Return the number of arguments received by the process."},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef EmbModule = {
    PyModuleDef_HEAD_INIT, "emb", NULL, -1, EmbMethods,
    NULL, NULL, NULL, NULL
};

static PyObject*
PyInit_emb(void)
{
    return PyModule_Create(&EmbModule);
}

int
main(int argc, char *argv[])
{
    numargs = argc;
    PyImport_AppendInittab("emb", &PyInit_emb);
    
    wchar_t** _argv = (wchar_t **)PyMem_Malloc(sizeof(wchar_t*) * argc);
    for (size_t i = 0; i < argc; i++)
    {
        wchar_t* arg = Py_DecodeLocale(argv[i], NULL);
        _argv[i] = arg;
    }
    
    Py_Initialize();
    Py_Main(argc, _argv);
    Py_Finalize();
}

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
