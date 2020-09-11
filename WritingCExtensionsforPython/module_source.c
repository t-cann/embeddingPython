#include <Python.h>

#if PY_MAJOR_VERSION < 3
#error "Requires Python 3"
#include "stopcompilation"
#endif

static PyObject* hello(PyObject* Py_UNUSED(self), PyObject* Py_UNUSED(args)) {
  printf("Hello world\n");

  Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
  { "hello", &hello, METH_VARARGS, "Hello world function" },
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
