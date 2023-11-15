#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API

#include <numpy/arrayobject.h>
#include "p5_matrixA.hpp"

static PyObject * wrap_p5_matrixA( PyObject * self, PyObject * args ) {
  PyObject *XYZW_;

  if( !PyArg_ParseTuple( args, "O", &XYZW_ ) )
    return NULL;

  PyArrayObject * pXYZW = reinterpret_cast<PyArrayObject *>( XYZW_ );

  if( pXYZW == 0  )
    return NULL;

  double * xyzw = reinterpret_cast< double * >( PyArray_DATA( pXYZW ) );

  npy_intp * dims = PyArray_DIMS( pXYZW );
  npy_intp  dimsa[] = { 20, 10 };

  if( dims[0] != 4 || dims[1] != 9 )
    return NULL;

  PyArrayObject * A_ = ( PyArrayObject * )PyArray_SimpleNew(
    2, dimsa, NPY_DOUBLE );

  double *a = reinterpret_cast< double * >( PyArray_DATA( A_ ) );

  double *x = xyzw, *y = xyzw + 9, *z = xyzw + 18, *w = xyzw + 27;

  p5_matrixA( x, y, z, w, a );

  PyObject * out = ( PyObject* )A_;

  return out;
}

static PyMethodDef Methods[] =
  {
   { "p5_matrixA", wrap_p5_matrixA, METH_VARARGS,
     "Five-point relative pose problem: the matrix 'A'."
   },
   { NULL, NULL, 0, NULL }
  };

static struct PyModuleDef p5module =
  {
   PyModuleDef_HEAD_INIT,
   "ext", // name of module
   NULL, // module documentation or NULL
   -1,   // size of per-interpreter state of the module,
         //  or -1 if the module keeps state in global variables.
   Methods
  };

PyMODINIT_FUNC PyInit_ext( void ) {
  import_array();
  return PyModule_Create( &p5module );
}
