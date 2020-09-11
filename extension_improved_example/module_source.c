#include <Python.h>

#if PY_MAJOR_VERSION < 3
#error "Requires Python 3"
#include "stopcompilation"
#endif

//https://ishantheperson.github.io/posts/python-c-ext-2/

static PyObject* hello(PyObject* Py_UNUSED(self), PyObject* Py_UNUSED(args)) {
  printf("Hello world\n");

  Py_RETURN_NONE;
}

static PyObject* my_function(PyObject* Py_UNUSED(self), PyObject* args) {
  const char* a;
  int b;
  double c;

  if (!PyArg_ParseTuple(args, "sid", &a, &b, &c))
    return NULL;

  if (b < 0) {
    PyErr_SetString(PyExc_ValueError, "Cannot be negative");
    return NULL;
  }

  for (int i = 0; i < b; i++)
    printf("%f %s\n", c, a);

  return PyFloat_FromDouble((double)b * c);
}

static PyObject* process_list(PyObject* Py_UNUSED(self), PyObject* args) {
  PyObject* list;

  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list))
    return NULL;

  Py_ssize_t list_size = PyList_Size(list);

  double sum = 0;

  for (Py_ssize_t i = 0; i < list_size; i++) {
    PyObject* sublist = PyList_GetItem(list, i);

    if (!PyList_Check(sublist)) {
      PyErr_SetString(PyExc_TypeError, "List must contain lists");
      return NULL;
    }

    Py_ssize_t sublist_size = PyList_Size(sublist);

    for (Py_ssize_t j = 0; j < sublist_size; j++) {
      sum += PyFloat_AsDouble(PyList_GetItem(sublist, j));

      if (PyErr_Occurred()) return NULL;
    }
  }

  return PyFloat_FromDouble(sum);
}

static PyMethodDef methods[] = {
  { "hello", &hello, METH_VARARGS, "Hello world function" },
  { "my_function", &my_function, METH_VARARGS, "Takes string, int, float"},
  { "process_list", &process_list, METH_VARARGS, "Adds up list of lists of doubles" },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef module_def = {
  PyModuleDef_HEAD_INIT, // always required
  "test_module",         // module name
  "Testing module",      // description
  -1,                    // module size (-1 indicates we don't use this feature)
  methods,               // method table
};

PyMODINIT_FUNC PyInit_test_module() {
  printf("Initialization\n");
  return PyModule_Create(&module_def);
}
