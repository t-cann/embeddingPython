#define PY_SSIZE_T_CLEAN  /* Make "s#" use Py_ssize_t rather than int. */
#include <Python.h>

int ok;
int i, j;
long k, l;
const char *s;
Py_ssize_t size;

ok = PyArg_ParseTuple(args, ""); /* No arguments */
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