#define PY_SSIZE_T_CLEAN  /* Make "s#" use Py_ssize_t rather than int. */
#include <Python.h>

/**
 * @brief Not Finshed - Program that converts C-Type Array into Python Variable. 
 * 
 */

int ok;
int i, j;
long k, l;
const char *s;
Py_ssize_t size;

//PyObject* tuple = PyTuple_New();
//tuple = PyTuple_SetItem();
    
ok = PyArg_ParseTuple(args, ""); /* No arguments */


    // PyRun_SimpleString(
    //     "a_foo = None\n"
    //     "\n"
    //     "def setup(a_foo_from_cxx):\n"
    //     "    print 'setup called with', a_foo_from_cxx\n"
    //     "    global a_foo\n"
    //     "    a_foo = a_foo_from_cxx\n"
    //     "\n"
    //     "def run():\n"
    //     "    a_foo.doSomething()\n"
    //     "\n"
    //     "print 'main module loaded'\n"
    // );