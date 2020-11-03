#define PY_SSIZE_T_CLEAN  /* Make "s#" use Py_ssize_t rather than int. */
#include <Python.h>
// https://docs.python.org/2.0/ext/parseTuple.html

/**
 * @brief Program examples of using parseTuple . 
 * 
 */

int main(int argc, const char* argv[])
{  
PyObject* args;

int ok;
int i, j;
long k, l;
char *s;
int size;

ok = PyArg_ParseTuple(args, ""); /* No arguments */ 
/* Python call: f() */
    

ok = PyArg_ParseTuple(args, "s", &s); /* A string */
    /* Possible Python call: f('whoops!') */

ok = PyArg_ParseTuple(args, "lls", &k, &l, &s); /* Two longs and a string */
    /* Possible Python call: f(1, 2, 'three') */

ok = PyArg_ParseTuple(args, "(ii)s#", &i, &j, &s, &size);
    /* A pair of ints and a string, whose size is also returned */
    /* Possible Python call: f((1, 2), 'three') */


char *file;
const char *mode = "r";
int bufsize = 0;
ok = PyArg_ParseTuple(args, "s|si", &file, &mode, &bufsize);
/* A string, and optionally another string and an integer */
/* Possible Python calls:
    f('spam')
    f('spam', 'w')
    f('spam', 'wb', 100000) */


int left, top, right, bottom, h, v;
ok = PyArg_ParseTuple(args, "((ii)(ii))(ii)",
            &left, &top, &right, &bottom, &h, &v);
/* A rectangle and a point */
/* Possible Python call:
    f(((0, 0), (400, 300)), (10, 10)) */

Py_complex c;
ok = PyArg_ParseTuple(args, "D:myfunction", &c);
/* a complex, also providing a function name for errors */
/* Possible Python call: myfunction(1+2j) */

//PyObject* tuple = PyTuple_New();
//tuple = PyTuple_SetItem();
return ok;
}

//Example of Better formated Running of Simple String

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