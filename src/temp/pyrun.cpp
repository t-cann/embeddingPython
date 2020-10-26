#define PY_SSIZE_T_CLEAN

#include <Python.h> //Must be included before standard headers
#include <cstdio>
#include "pyrun.hpp"

int Simplestring(int argc, char *argv[])
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }
    Py_SetProgramName(program);  /* optional but recommended */
    Py_Initialize();
    PyRun_SimpleString("from time import time,ctime\n"
                       "print('Today is', ctime(time()))\n");
 


    if (Py_FinalizeEx() < 0) {
        exit(120);
    }
    PyMem_RawFree(program);
    return 0;
}

int Anyfile(int argc, char *argv[])
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    wchar_t** _argv = (wchar_t **)PyMem_Malloc(sizeof(wchar_t*) * argc);
    for (size_t i = 0; i < argc; i++)
    {
        wchar_t* arg = Py_DecodeLocale(argv[i], NULL);
        _argv[i] = arg;
    }

    Py_SetProgramName(program);  /* optional but recommended */
    Py_Initialize();

    PyRun_SimpleString(
            "import sys, os\n"
            "sys.path.append(os.getcwd())\n"
    );

    FILE* pFile;
    
    pFile = fopen ("helloworld.py","r");
    if(pFile!=NULL)
        PyRun_AnyFile(pFile, "helloworld");
    
    Py_Main(argc,_argv);

    if (Py_FinalizeEx() < 0)
        exit(120);
    
    PyMem_RawFree(program);
    return 0;
}