#define PY_SSIZE_T_CLEAN
#include <Python.h>

int main(int argc, char* argv[])
{
    wchar_t** _argv = (wchar_t **)PyMem_Malloc(sizeof(wchar_t*) * argc);
    for (size_t i = 0; i < argc; i++)
    {
        wchar_t* arg = Py_DecodeLocale(argv[i], NULL);
        _argv[i] = arg;
    }

    Py_Initialize();
    PyObject* main_module = PyImport_AddModule("__main__");

    PyObject* tuple = PyTuple_New(Py_ssize_t);
    tuple = PyTuple_SetItem();
    Py_Main(argc,_argv);

    if (Py_FinalizeEx() < 0) {
        exit(120);
    }
    return 0;
}


