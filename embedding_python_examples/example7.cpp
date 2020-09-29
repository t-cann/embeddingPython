#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#include <thread>

int main(int argc, char *argv[])
{
    std::cout << "Begining";

    wchar_t** _argv = (wchar_t **)PyMem_Malloc(sizeof(wchar_t*) * argc);
    for (int i = 0; i < argc; i++)
    {
        wchar_t* arg = Py_DecodeLocale(argv[i], NULL);
        _argv[i] = arg;
    }
    FILE * stderr, *stdout,  *stdin;
     
    Py_Initialize();
    
    std::thread first (Py_Main,argc, _argv);
    std::cerr  << "End"; 
    first.join();

    Py_FinalizeEx();

    
}
