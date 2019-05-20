//#define EIGEN_USE_MKL_ALL        //uncomment if available
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include "numpy/arrayobject.h"
#include "fmm.h"

#include <future>
#include <thread>

#include <iostream>
# include <iomanip>

static PyObject *GmmcError;

static PyObject *new_gmmc(PyObject *self, PyObject *args)
{
  unsigned K=100;
  unsigned iters = 2000;
  unsigned n_save = 0; // don't save intermediate and final results to files
  PyArrayObject *in_array_mean;
  PyArrayObject *in_array_var;

  if (!PyArg_ParseTuple(args,"IIO!O!:new_gmmc", &K, &iters, &PyArray_Type, &in_array_mean, &PyArray_Type, &in_array_var))
    return NULL;

  npy_intp numPoints = PyArray_DIM(in_array_mean, 0);
  npy_intp numDims = PyArray_DIM(in_array_mean, 1);

  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  npy_intp idx[2] = {0, 0};
  double * fnp_mean = reinterpret_cast< double * >( PyArray_GetPtr(in_array_mean, idx) );
  Eigen::Map<Eigen::MatrixXd> meanMatrix(fnp_mean, numDims, numPoints);

  std::cout<< PyArray_NDIM(in_array_var) <<std::endl;
  if (PyArray_NDIM(in_array_var)==2)
  {
    numPoints = PyArray_DIM(in_array_var, 0);
    if(numDims != PyArray_DIM(in_array_var, 1))
      throw std::runtime_error("Mean and variance dimensions do not match!");
  }
  else if (PyArray_NDIM(in_array_var)==1)
  {
    numPoints = 1;
    if(numDims != PyArray_DIM(in_array_var, 0))
      throw std::runtime_error("Mean and variance dimensions do not match!");
  }
  else
      throw std::runtime_error("Invalid variance!"); 
  double * fnp_var = reinterpret_cast< double * >( PyArray_GetPtr(in_array_var, idx) );
  Eigen::Map<Eigen::MatrixXd> varMatrix(fnp_var, numDims, numPoints);

  utils::ParsedArgs params(K, iters, "simple", n_save);
  model* canopy = model::init(params, meanMatrix, varMatrix, 0);
  size_t int_ptr = reinterpret_cast< size_t >(canopy);

  return Py_BuildValue("n", int_ptr);
}

static PyObject *delete_gmmc(PyObject *self, PyObject *args)
{
  model *obj;
  size_t int_ptr;

  if (!PyArg_ParseTuple(args,"n:delete_gmmc", &int_ptr))
    return NULL;

  obj = reinterpret_cast< model * >(int_ptr);
  delete obj;

  return Py_BuildValue("n", int_ptr);
}


static PyObject *gmmc_fit(PyObject *self, PyObject *args)
{
  model *obj;
  size_t int_ptr;
  PyArrayObject *in_array_trng;
  PyArrayObject *in_array_test;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!O!:kmeansc_fit", &int_ptr, &PyArray_Type, &in_array_trng, &PyArray_Type, &in_array_test))
    return NULL;

  npy_intp numPoints_trng = PyArray_DIM(in_array_trng, 0);
  npy_intp numDims_trng = PyArray_DIM(in_array_trng, 1);
  npy_intp numPoints_test = PyArray_DIM(in_array_test, 0);
  npy_intp numDims_test = PyArray_DIM(in_array_test, 1);
  if (numDims_trng != numDims_test)
  {
      throw std::runtime_error("Train and test dimensions do not match!");
  }
  //std::cout<<numPoints_trng<<", "<<numDims_trng<<std::endl;
  //std::cout<<numPoints_test<<", "<<numDims_test<<std::endl;
  
  npy_intp idx[2] = {0, 0};
  double * fnp_trng = reinterpret_cast< double * >( PyArray_GetPtr(in_array_trng, idx) );
  Eigen::Map<Eigen::MatrixXd> trngMatrix(fnp_trng, numDims_trng, numPoints_trng);
  double * fnp_test = reinterpret_cast< double * >( PyArray_GetPtr(in_array_test, idx) );
  Eigen::Map<Eigen::MatrixXd> testMatrix(fnp_test, numDims_test, numPoints_test);

  obj = reinterpret_cast< model * >(int_ptr);
  double score = obj->fit(trngMatrix, testMatrix);

  return Py_BuildValue("d", score);
}

static PyObject *gmmc_evaluate(PyObject *self, PyObject *args) {

  model *obj;
  size_t int_ptr;
  PyArrayObject *in_array;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!:kmeansc_evaluate", &int_ptr, &PyArray_Type, &in_array))
    return NULL;

  npy_intp numPoints = PyArray_DIM(in_array, 0);
  npy_intp numDims = PyArray_DIM(in_array, 1);
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  npy_intp idx[2] = {0, 0};
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<Eigen::MatrixXd> pointMatrix(fnp, numDims, numPoints);

  obj = reinterpret_cast< model * >(int_ptr);
  double score = obj->evaluate(pointMatrix);

  return Py_BuildValue("d", score);
}

static PyObject *gmmc_predict(PyObject *self, PyObject *args) {

  model *obj;
  size_t int_ptr;
  PyArrayObject *in_array;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!:kmeansc_predict", &int_ptr, &PyArray_Type, &in_array))
    return NULL;

  npy_intp idx[2] = {0,0};
  npy_intp numPoints = PyArray_DIM(in_array, 0);
  npy_intp numDims = PyArray_DIM(in_array, 1);
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<Eigen::MatrixXd> queryPts(fnp, numDims, numPoints);

  obj = reinterpret_cast< model * >(int_ptr);
  std::vector<unsigned> results = obj->predict(queryPts);
  unsigned* new_ptr = new unsigned[numPoints];
  for(int i =0; i < numPoints; i++)
    new_ptr[i] = results[i];

  npy_intp dims[1] = {numPoints};
  PyObject *out_array = PyArray_SimpleNewFromData(1, dims, NPY_UINT, new_ptr);
  PyArray_ENABLEFLAGS((PyArrayObject *)out_array, NPY_ARRAY_OWNDATA);

  //Py_INCREF(out_array);
  return out_array;
}

static PyObject *gmmc_centers(PyObject *self, PyObject *args) {

  model *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:kmeansc_centers", &int_ptr))
    return NULL;

  obj = reinterpret_cast< model * >(int_ptr);
  std::pair<Eigen::Map<Eigen::MatrixXd>, Eigen::Map<Eigen::MatrixXd>> results = obj->get_centers();

  npy_intp numPoints = results.first.cols();
  npy_intp numDims = results.first.rows();
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  npy_intp dims[2] = {numDims, numPoints};

  // means
  PyObject *out_mean = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, results.first.data());
  PyArray_ENABLEFLAGS((PyArrayObject *)out_mean, NPY_ARRAY_OWNDATA);

  // variance
  PyObject *out_var = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, results.second.data());
  PyArray_ENABLEFLAGS((PyArrayObject *)out_var, NPY_ARRAY_OWNDATA);

  return Py_BuildValue("NN", out_mean, out_var);
}

static PyObject *gmmc_serialize(PyObject *self, PyObject *args)
{
  model *obj;
  size_t int_ptr;

  if (!PyArg_ParseTuple(args,"n:kmeansc_serialize", &int_ptr))
    return NULL;
  
  obj = reinterpret_cast< model * >(int_ptr);
  char* buff = obj->serialize();
  size_t len = obj->msg_size();
  
  return Py_BuildValue("y#", buff, len);
}

static PyObject *gmmc_deserialize(PyObject *self, PyObject *args)
{
  char* buff;
  size_t len;
  if (!PyArg_ParseTuple(args,"y#:kmeansc_deserialize", &buff, &len))
    return NULL;
      
  model* canopy = new model();
  canopy->deserialize(buff);
  size_t int_ptr = reinterpret_cast< size_t >(canopy);

  return Py_BuildValue("n", int_ptr);
}

PyMODINIT_FUNC PyInit_gmmc(void)
{
  PyObject *m;
  static PyMethodDef GmmcMethods[] = {
    {"new", new_gmmc, METH_VARARGS, "Initialize a new kmeans."},
    {"delete", delete_gmmc, METH_VARARGS, "Delete the kmeans."},
    {"fit", gmmc_fit, METH_VARARGS, "Predict cluster assignments."},
    {"evaluate", gmmc_evaluate, METH_VARARGS, "Get objective function value."},
    {"predict", gmmc_predict, METH_VARARGS, "Get labels."},
    {"centers", gmmc_centers, METH_VARARGS, "Get the learnt cluster centers."},
    {"serialize", gmmc_serialize, METH_VARARGS, "Serialize the current kmeans."},
    {"deserialize", gmmc_deserialize, METH_VARARGS, "Construct a kmeans from deserializing."},
    {NULL, NULL, 0, NULL}
  };
  static struct PyModuleDef mdef = {PyModuleDef_HEAD_INIT,
                                 "gmmc",
                                 "Example module that creates an extension type.",
                                 -1,
                                 GmmcMethods};
  m = PyModule_Create(&mdef);
  if ( m == NULL )
    return NULL;

  /* IMPORTANT: this must be called */
  import_array();

  GmmcError = PyErr_NewException("gmmc.error", NULL, NULL);
  Py_INCREF(GmmcError);
  PyModule_AddObject(m, "error", GmmcError);
  
  return m;
}

int main(int argc, char *argv[])
{
  /* Convert to wchar */
  wchar_t *program = Py_DecodeLocale(argv[0], NULL);
  if (program == NULL) {
     std::cerr << "Fatal error: cannot decode argv[0]" << std::endl;
     return 1;
  }
  
  /* Add a built-in module, before Py_Initialize */
  //PyImport_AppendInittab("gmmc", PyInit_covertreec);
    
  /* Pass argv[0] to the Python interpreter */
  Py_SetProgramName(program);

  /* Initialize the Python interpreter.  Required. */
  Py_Initialize();

  /* Add a static module */
  //PyInit_kmeansc();
}

