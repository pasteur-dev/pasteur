#define PY_SSIZE_T_CLEAN /* Make "s#" use Py_ssize_t rather than int. */
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>

extern void sum(
    int dom, int l, uint32_t *out,
    int n_u8, int *mul_u8, uint8_t **arr_u8,
    int n_u16, int *mul_u16, uint16_t **arr_u16,
    int n_u32, int *mul_u32, uint32_t **arr_u32);

extern void sum_non_simd(
    uint64_t dom, uint64_t l, uint32_t *out,
    int n_u8, int *mul_u8, uint8_t **arr_u8,
    int n_u16, int *mul_u16, uint16_t **arr_u16,
    int n_u32, int *mul_u32, uint32_t **arr_u32);

static PyObject *sum_wrapper(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyArrayObject *out;
    PyObject *ops;
    int simd = 1;

    static char *kwlist[] = {"out", "ops", "simd", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|p", kwlist, &out, &ops, &simd))
        return NULL;

    if (!PyList_Check(ops))
    {
        PyErr_SetString(PyExc_TypeError, "Ops (2nd arg) is not a list.");
        return NULL;
    }

    uint64_t dom = PyArray_DIM(out, 0);

    int n_u8 = 0;
    int mul_u8[100];
    uint8_t *arr_u8[100];
    int n_u16 = 0;
    int mul_u16[100];
    uint16_t *arr_u16[100];
    int n_u32 = 0;
    int mul_u32[100];
    uint32_t *arr_u32[100];

    uint64_t l = 0;
    for (int i = 0; i < PyList_Size(ops); i++)
    {
        PyObject *t = PyList_GetItem(ops, i);

        int mul = PyLong_AsLong(PyTuple_GetItem(t, 0));
        PyArrayObject *arr = (PyArrayObject *)PyTuple_GetItem(t, 1);
        l = PyArray_DIM(arr, 0);

        switch (PyArray_TYPE(arr))
        {
        case NPY_UINT8:
            mul_u8[n_u8] = mul;
            arr_u8[n_u8] = (uint8_t *)PyArray_DATA(arr);
            n_u8 += 1;
            break;
        case NPY_UINT16:
            mul_u16[n_u16] = mul;
            arr_u16[n_u16] = (uint16_t *)PyArray_DATA(arr);
            n_u16 += 1;
            break;
        case NPY_UINT32:
            mul_u32[n_u32] = mul;
            arr_u32[n_u32] = (uint32_t *)PyArray_DATA(arr);
            n_u8 += 1;
            break;
        default:
            PyErr_SetString(PyExc_TypeError, "Uknown Type passed in ops, only uint 8, 16, 32 supported");
            return NULL;
        }
    }

    uint32_t *data_out = (uint32_t *)PyArray_DATA(out);

    if (simd)
    {
        sum(dom, l, data_out, n_u8, mul_u8, arr_u8, n_u16, mul_u16, arr_u16, n_u32, mul_u32, arr_u32);
    }
    else
    {
        sum_non_simd(dom, l, data_out, n_u8, mul_u8, arr_u8, n_u16, mul_u16, arr_u16, n_u32, mul_u32, arr_u32);
    }

    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    /* The cast of the function is necessary since PyCFunction values
     * only take two PyObject* parameters, and native_parrot() takes
     * three.
     */
    {"marginal", (PyCFunction)(void (*)(void))sum_wrapper, METH_VARARGS | METH_KEYWORDS,
     "Calculates the marginal of the provided columns"},
    {NULL, NULL, 0, NULL} /* sentinel */
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "native",
    NULL,
    -1,
    methods};

PyMODINIT_FUNC
PyInit_native(void)
{
    return PyModule_Create(&module);
}