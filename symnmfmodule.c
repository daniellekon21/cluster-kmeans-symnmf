#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <math.h>
#include "symnmf.h"

/* Function Declarations */
double **fromPyObjectToCMatrix(PyObject *py_mat, int rows, int cols);
PyObject *fromCMatrixToPyObject(double **c_mat, int rows, int cols);
void freeMatrix(double **c_mat, int rows);
static PyObject *sym_wrapper(PyObject *self, PyObject *args);
static PyObject *ddg_wrapper(PyObject *self, PyObject *args);
static PyObject *norm_wrapper(PyObject *self, PyObject *args);
static PyObject *symnmf_wrapper(PyObject *self, PyObject *args);

/* Convert Python object to C matrix */
double **fromPyObjectToCMatrix(PyObject *py_mat, int rows, int cols)
{
    int i, j;
    double **c_mat;
    PyObject *py_row, *py_val;
    c_mat = (double **)malloc(rows * sizeof(double *));
    /* Memory alloc check */
    if (!c_mat){
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for C matrix rows");
        return NULL;
    }

    for (i = 0; i < rows; i++){
        c_mat[i] = (double *)malloc(cols * sizeof(double));
        /* Memory alloc check */
        if (!c_mat[i]){
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for C matrix columns");
            /* Free allocated memory for previous rows */
            for (j = 0; j < i; j++){
                free(c_mat[j]);
            }
            free(c_mat);
            return NULL;
        }
        py_row = PyObject_GetItem(py_mat, PyLong_FromSsize_t(i));
        for (j = 0; j < cols; j++){
            py_val = PyObject_GetItem(py_row, PyLong_FromSsize_t(j));
            c_mat[i][j] = PyFloat_AsDouble(py_val);
        }
    }
    return c_mat;
}

/* Convert C matrix to Python object */
PyObject *fromCMatrixToPyObject(double **c_mat, int rows, int cols)
{
    int i, j;
    PyObject *py_mat, *py_row, *py_val;
    py_mat = PyList_New(rows);
    if (!py_mat){
        PyErr_SetString(PyExc_RuntimeError, "Failed to create Python list");
        return NULL;
    }
    /* Fill the list with the results row by row */
    for (i = 0; i < rows; i++){
        py_row = PyList_New(cols);
        if (!py_row){
            PyErr_SetString(PyExc_RuntimeError, "Failed to create Python sublist");
            Py_XDECREF(py_mat); /* Handle to prevent memory leak */
            return NULL;
        }
        for (j = 0; j < cols; j++){
            py_val = PyFloat_FromDouble(c_mat[i][j]);
            if (!py_val){
                PyErr_SetString(PyExc_RuntimeError, "Failed to convert double to PyFloat");
                Py_XDECREF(py_row); /* Handle to prevent memory leak */
                Py_XDECREF(py_mat); /* Handle to prevent memory leak */
                return NULL;
            }
            PyList_SetItem(py_row, j, py_val);
        }
        PyList_SetItem(py_mat, i, py_row);
    }
    return py_mat;
}

/* Free C matrix */
void freeMatrix(double **c_mat, int rows)
{
    int i;
    for (i = 0; i < rows; i++){
        free(c_mat[i]);
    }
    free(c_mat);
}

/* Similarity matrix wrapper function */
static PyObject *sym_wrapper(PyObject *self, PyObject *args)
{
    int N, d;
    double **X, **A;
    PyObject *row_obj, *py_X, *py_A;

    if (!PyArg_ParseTuple(args, "O", &py_X)){
        return NULL; /* Parsing failed */
    }
    N = (int)PyList_Size(py_X); /* Calculate N */
    row_obj = PyObject_GetItem(py_X, PyLong_FromSsize_t(0));
    d = (int)PyList_Size(row_obj); /* Calculate d */
    X = fromPyObjectToCMatrix(py_X, N, d);
    if (!X){
        printf("108 %d\n", d);
        return NULL;
    }
    A = compute_similarity_matrix(X, N, d); /* Compute similarity matrix */
    if (!A){
        freeMatrix(X, N);
        PyErr_SetString(PyExc_RuntimeError, "sym function failed");
        return NULL;
    }
    py_A = fromCMatrixToPyObject(A, N, N);
    /* Free allocated memory */
    freeMatrix(X, N);
    freeMatrix(A, N);
    return py_A;
}

/* Diagonal degree matrix wrapper */
static PyObject *ddg_wrapper(PyObject *self, PyObject *args)
{
    int N, d;
    double **X, **D;
    PyObject *row_obj, *py_X, *py_D;
    if (!PyArg_ParseTuple(args, "O", &py_X)){
        return NULL; /* Parsing failed*/
    }
    N = (int)PyList_Size(py_X); /* Calculate N */
    row_obj = PyObject_GetItem(py_X, PyLong_FromSsize_t(0));
    d = (int)PyList_Size(row_obj); /* Calculate d */
    X = fromPyObjectToCMatrix(py_X, N, d);
    if (!X){
        return NULL;
    }
    D = compute_ddg_matrix(X, N, d); /* Compute diagonal degree matrix */
    if (!D){
        freeMatrix(X, N);
        PyErr_SetString(PyExc_RuntimeError, "ddg function failed");
        return NULL;
    }
    py_D = fromCMatrixToPyObject(D, N, N);
    /* Free allocated memory */
    freeMatrix(X, N);
    freeMatrix(D, N);
    return py_D;
}

/* Normalized similarity matrix wrapper */
static PyObject *norm_wrapper(PyObject *self, PyObject *args)
{
    int N, d;
    double **X, **W;
    PyObject *row_obj, *py_X, *py_W;
    if (!PyArg_ParseTuple(args, "O", &py_X)){
        return NULL; /* Parsing failed*/
    }
    N = (int)PyList_Size(py_X); /* Calculate N */
    row_obj = PyObject_GetItem(py_X, PyLong_FromSsize_t(0));
    d = (int)PyList_Size(row_obj); /* Calculate d */
    X = fromPyObjectToCMatrix(py_X, N, d);
    if (!X){
        return NULL;
    }
    W = compute_norm_matrix(X, N, d);
    if (!W){
        freeMatrix(X, N);
        PyErr_SetString(PyExc_RuntimeError, "norm function failed");
        return NULL;
    }
    py_W = fromCMatrixToPyObject(W, N, N);
    /* Free allocated memory */
    freeMatrix(X, N);
    freeMatrix(W, N);
    return py_W;
}

/* Final H matrix wrapper */
static PyObject *symnmf_wrapper(PyObject *self, PyObject *args)
{
    int N, k;
    PyObject *py_W, *py_H_init, *py_H;
    double **W, **H_init, **H;
    if (!PyArg_ParseTuple(args, "OOi", &py_W, &py_H_init, &k)){
        return NULL; /* Parsing failed*/
    }
    N = (int)PyList_Size(py_H_init); /* Calculate N */
    W = fromPyObjectToCMatrix(py_W, N, N);
    H_init = fromPyObjectToCMatrix(py_H_init, N, k);
    if ((!W) || (!H_init)){
        return NULL;
    }
    H = symnmf(W, H_init, N, k);
    if (!H){
        freeMatrix(W, N);
        freeMatrix(H_init, N);
        PyErr_SetString(PyExc_RuntimeError, "symnmf function failed");
        return NULL;
    }
    py_H = fromCMatrixToPyObject(H, N, k);
    /* Free allocated memory */
    freeMatrix(W, N);
    freeMatrix(H_init, N);
    freeMatrix(H, N);
    return py_H;
}

/* Define the methods of the module */
static PyMethodDef SymNMFMethods[] = {
    {"sym", (PyCFunction)sym_wrapper, METH_VARARGS, PyDoc_STR("Compute similarity matrix")},
    {"ddg", (PyCFunction)ddg_wrapper, METH_VARARGS, PyDoc_STR("Compute degree matrix")},
    {"norm", (PyCFunction)norm_wrapper, METH_VARARGS, PyDoc_STR("Compute normalized matrix")},
    {"symnmf", (PyCFunction)symnmf_wrapper, METH_VARARGS, PyDoc_STR("Run SymNMF algorithm")},
    {NULL, NULL, 0, NULL}};

/* Define Module */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "mysymnmf",
    "Python interface for the SymNMF C library",
    -1,
    SymNMFMethods};

/* Module Initialization */
PyMODINIT_FUNC PyInit_mysymnmf(void)
{
    PyObject *m = PyModule_Create(&symnmfmodule);
    if (!m){
        return NULL;
    }

    return m;
}