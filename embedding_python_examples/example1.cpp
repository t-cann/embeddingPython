#define PY_SSIZE_T_CLEAN
#include <Python.h>

static int numargs=0;

/* Return the number of arguments of the application command line */
/* Beginning of Python Module Definition Code */
static PyObject* emb_numargs(PyObject *self, PyObject *args)
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

static PyObject* PyInit_emb(void)
{
    return PyModule_Create(&EmbModule);
}
/* End of Python Module Definition  Code */

/**
 * @brief  Import module before, Initialising embedded Python Interpeter.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char *argv[])
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


