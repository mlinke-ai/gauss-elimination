#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef __AVX2__
#include <immintrin.h>
#define STRIDE 8
#endif

static PyObject *cgauss_solve(PyObject *self, PyObject *args) {
  PyObject *py_coefficients_obj;
  PyObject *py_constants_obj;

  // parse the Python arguments to get the matrix and vector objects
  if (!PyArg_ParseTuple(args, "OO", &py_coefficients_obj, &py_constants_obj))
    return NULL;

  // convert the objects to arrays and check if the conversion succeeded
  PyArrayObject *np_coefficients_obj = (PyArrayObject *)PyArray_FROM_OTF(
      py_coefficients_obj, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
  if (np_coefficients_obj == NULL) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Coefficients matrix is either not of type FLOAT, "
                    "C-contiguous, or properly aligned.");
    return NULL;
  }
  PyArrayObject *np_constants_obj = (PyArrayObject *)PyArray_FROM_OTF(
      py_constants_obj, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
  if (np_constants_obj == NULL) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Constants vector is either not of type FLOAT, "
                    "C-contiguous, or properly aligned.");
    return NULL;
  }

  // get size of system and create vector for solution
  npy_intp size = PyArray_SIZE(np_constants_obj);
  PyArrayObject *np_solutions_obj = (PyArrayObject *)PyArray_NewLikeArray(
      np_constants_obj, NPY_ANYORDER, NULL, 1);

  // solve the system using the avaliable methods
#ifdef __AVX2__
  int i, j, k, s = 0, t = 0;
  float v = 0.0;
  __m256 e;
  __m128 h, l;
  float *m = (float *)malloc(size * size * sizeof(float));
  if (m == NULL) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for matrix copy.");
    return NULL;
  }
  memcpy(m, PyArray_DATA(np_coefficients_obj), size * size * sizeof(float));
  float *b = (float *)PyArray_DATA(np_constants_obj);
  int *p = (int *)malloc(size * sizeof(int));
  if (p == NULL) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for pivot lookup table.");
    return NULL;
  }
  for (i = 0; i < size; i++) {
    p[i] = i;
  }
  float *y = (float *)malloc(size * sizeof(float));
  if (y == NULL) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Unable to allocate temporary memory for forward substitution.");
    return NULL;
  }
  float *x = (float *)PyArray_DATA(np_solutions_obj);
  for (i = 0; i < size; i++) {
    s = i;
    for (j = i; j < size; j++) {
      if (fabsf(m[j * size + i]) > fabsf(m[s * size + i])) {
        s = j;
      }
    }
    t = p[i];
    p[i] = p[s];
    p[s] = t;
    if (m[p[i] * size + i] == 0.0) {
      PyErr_SetString(PyExc_ValueError, "Pivoting failed");
      return NULL;
    } else {
      for (j = i + 1; j < size; j++) {
        m[p[j] * size + i] /= m[p[i] * size + i];
        for (k = i + 1; k + STRIDE - 1 < size; k += STRIDE) {
          e = _mm256_mul_ps(_mm256_set1_ps(m[p[j] * size + i]),
                            _mm256_loadu_ps(m + (p[i] * size + k)));
          e = _mm256_sub_ps(_mm256_loadu_ps(m + (p[j] * size + k)), e);
          _mm256_storeu_ps(m + (p[j] * size + k), e);
        }
        for (; k < size; k++) {
          m[p[j] * size + k] -= m[p[j] * size + i] * m[p[i] * size + k];
        }
      }
    }
  }
  for (i = 0; i < size; i++) {
    for (j = 0; j + STRIDE - 1 < i; j += STRIDE) {
      e = _mm256_mul_ps(_mm256_loadu_ps(m + (p[i] * size + j)), _mm256_loadu_ps(y + j));
      l = _mm256_extractf128_ps(e, 0);
      h = _mm256_extractf128_ps(e, 1);
      l = _mm_hadd_ps(h, l);
      l = _mm_hadd_ps(l, l);
      v += _mm_cvtss_f32(l);
      l = _mm_permute_ps(l, 0b00000001);
      v += _mm_cvtss_f32(l);
    }
    for (; j < i; j++) {
      v += m[p[i] * size + j] * y[j];
    }
    y[i] = b[p[i]] - v;
    v = 0.0;
  }
  for (i = size - 1; i >= 0; i--) {
    for (j = i + 1; j + STRIDE - 1 < size; j += STRIDE) {
      e = _mm256_mul_ps(_mm256_loadu_ps(m + (p[i] * size + j)), _mm256_loadu_ps(x + j));
      l = _mm256_extractf128_ps(e, 0);
      h = _mm256_extractf128_ps(e, 1);
      l = _mm_hadd_ps(h, l);
      l = _mm_hadd_ps(l, l);
      v += _mm_cvtss_f32(l);
      l = _mm_permute_ps(l, 0b00000001);
      v += _mm_cvtss_f32(l);
    }
    for (; j < size; j++) {
      v += m[p[i] * size + j] * x[j];
    }
    x[i] = (y[i] - v) / m[p[i] * size + i];
    v = 0.0;
  }
  free(y);
  free(p);
  free(m);
#else
  int i, j, k, s = 0, t = 0;
  float v = 0.0;
  float *m = (float *)malloc(size * size * sizeof(float));
  if (m == NULL) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for matrix copy.");
    return NULL;
  }
  memcpy(m, PyArray_DATA(np_coefficients_obj), size * size * sizeof(float));
  float *b = (float *)PyArray_DATA(np_constants_obj);
  int *p = (int *)malloc(size * sizeof(int));
  if (p == NULL) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for pivot lookup table.");
    return NULL;
  }
  for (i = 0; i < size; i++) {
    p[i] = i;
  }
  float *y = (float *)malloc(size * sizeof(float));
  if (y == NULL) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Unable to allocate temporary memory for forward substitution");
    return NULL;
  }
  float *x = (float *)PyArray_DATA(np_solutions_obj);
  for (i = 0; i < size; i++) {
    s = i;
    for (j = i; j < size; j++) {
      if (fabsf(m[j * size + i]) > fabsf(m[s * size + i])) {
        s = j;
      }
    }
    t = p[i];
    p[i] = p[s];
    p[s] = t;
    if (m[p[i] * size + i] == 0.0) {
      PyErr_SetString(PyExc_ValueError, "Pivoting failed");
      return NULL;
    } else {
      for (j = i + 1; j < size; j++) {
        m[p[j] * size + i] /= m[p[i] * size + i];
        for (k = i + 1; k < size; k++) {
          m[p[j] * size + k] -= m[p[j] * size + i] * m[p[i] * size + k];
        }
      }
    }
  }
  for (i = 0; i < size; i++) {
    for (j = 0; j < i; j++) {
      v += m[p[i] * size + j] * y[j];
    }
    y[i] = b[p[i]] - v;
    v = 0.0;
  }
  for (i = size - 1; i >= 0; i--) {
    for (j = i + 1; j < size; j++) {
      v += m[p[i] * size + j] * x[j];
    }
    x[i] = (y[i] - v) / m[p[i] * size + i];
    v = 0.0;
  }
  free(y);
  free(p);
  free(m);
#endif

  // decrease the reference count of the array, as created by PyArray_FROM_OTF
  Py_DECREF(np_coefficients_obj);
  Py_DECREF(np_constants_obj);

  // return the solutions vector which we newly created
  return (PyObject *)np_solutions_obj;
}

static PyMethodDef cgauss_methods[] = {
    {"solve", cgauss_solve, METH_VARARGS,
     "Solve the LGS using Gauss Elimination."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef cgauss_module = {PyModuleDef_HEAD_INIT, "cgauss",
                                           "module documentation", -1,
                                           cgauss_methods};

PyMODINIT_FUNC PyInit_cgauss(void) {
  PyObject *m;
  m = PyModule_Create(&cgauss_module);
  if (m == NULL)
    return NULL;
  import_array(); // necessary for NumPy initialization
  return m;
}
