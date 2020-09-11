#define PY_SSIZE_T_CLEAN
#include <Python.h> //Must be included before standard headers
#include <cstdio>

int
main(int argc, char *argv[])
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
    PyRun_SimpleString(
        "a_foo = None\n"
        "\n"
        "def setup(a_foo_from_cxx):\n"
        "    print 'setup called with', a_foo_from_cxx\n"
        "    global a_foo\n"
        "    a_foo = a_foo_from_cxx\n"
        "\n"
        "def run():\n"
        "    a_foo.doSomething()\n"
        "\n"
        "print 'main module loaded'\n"
    );

    if (Py_FinalizeEx() < 0) {
        exit(120);
    }
    PyMem_RawFree(program);
    return 0;
}