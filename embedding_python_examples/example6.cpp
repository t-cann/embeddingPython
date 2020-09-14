#define PY_SSIZE_T_CLEAN  /* Make "s#" use Py_ssize_t rather than int. */
#include <Python.h> //Must be included before standard headers

// https://docs.python.org/3/extending/extending.html#calling-python-functions-from-c
// NOT Finished
// TODO How to register this function with the interpreter using the METH_VARARGS flag;

static PyObject *my_callback = NULL;

static PyObject *
my_set_callback(PyObject *dummy, PyObject *args)
{
    PyObject *result = NULL;
    PyObject *temp;

    if (PyArg_ParseTuple(args, "O:set_callback", &temp)) {
        if (!PyCallable_Check(temp)) {
            PyErr_SetString(PyExc_TypeError, "parameter must be callable");
            return NULL;
        }
        Py_XINCREF(temp);         /* Add a reference to new callback */
        Py_XDECREF(my_callback);  /* Dispose of previous callback */
        my_callback = temp;       /* Remember new callback */
        /* Boilerplate to return "None" */
        Py_INCREF(Py_None);
        result = Py_None;
    }
    return result;
}
/**
 * @brief  Calling Python FUnctions from C Example
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char* argv[])
{  

int arg;
PyObject *arglist;
PyObject *result;
arg = 123;

/* Time to call the callback */
arglist = Py_BuildValue("(i)", arg);
result = PyObject_CallObject(my_callback, arglist);
Py_DECREF(arglist);

}