#define PY_SSIZE_T_CLEAN
#include <Python.h>

static inline int
imax(int a, int b) {
  if (a > b)
    return a;
  return b;
}

/**** Conversion from Python to C ******/

long* PySequenceToLongArray(PyObject* pylist) {
  if (!(pylist && PySequence_Check(pylist)))
    return NULL;
  Py_ssize_t len = PySequence_Size(pylist);
  long* result = calloc(len, sizeof(long));
  if (!result) return NULL;
  for(Py_ssize_t i = 0; i < len; ++i) {
    PyObject* item = PySequence_GetItem(pylist, i);
    result[i] = PyLong_AsLong(item);
    if (result[i] == -1 && PyErr_Occurred()) {
      free(result);
      return NULL;
    }
    Py_DECREF(item);
  }
  return result;
}

long** PySequenceToLongArray2D(PyObject* pylist) {
  if (!(pylist && PySequence_Check(pylist)))
    return NULL;
  Py_ssize_t len = PySequence_Size(pylist);
  long** result = calloc(len, sizeof(long*));
  if (!result) return NULL;
  for(Py_ssize_t i = 0; i < len; ++i) {
    PyObject* item = PySequence_GetItem(pylist, i);
    result[i] = PySequenceToLongArray(item);
    Py_DECREF(item);
    if (!result[i]) {
      for(Py_ssize_t j = 0; j < i; ++j) {
	free(result[j]);
      }
      return NULL;
    }
  }
  return result;
}

double* PySequenceToDoubleArray(PyObject* pylist) {
  if (!(pylist && PySequence_Check(pylist)))
    return NULL;
  Py_ssize_t len = PySequence_Size(pylist);
  double* result = calloc(len, sizeof(double));
  for(Py_ssize_t i = 0; i < len; ++i) {
    PyObject* item = PySequence_GetItem(pylist, i);
    result[i] = PyFloat_AsDouble(item);
    Py_DECREF(item); 
    if (result[i] == -1 && PyErr_Occurred()) {
      free(result);
      return NULL;
    }
  }
  return result;
}

double** PySequenceToDoubleArray2D(PyObject* pylist) {
  if (!(pylist && PySequence_Check(pylist)))
    return NULL;
  Py_ssize_t len = PySequence_Size(pylist);
  double** result = calloc(len, sizeof(long*));
  if (!result) return NULL;
  for(Py_ssize_t i = 0; i < len; ++i) {
    PyObject* item = PySequence_GetItem(pylist, i);
    result[i] = PySequenceToDoubleArray(item);
    Py_DECREF(item);
    if (!result[i]) {
      for(Py_ssize_t j = 0; j < i; ++j) {
	free(result[j]);
      }
      return NULL;
    }
  }
  return result;
}


long* getLongArray(PyObject* container, const char* attributeName) {
  PyObject* sequence = PyObject_GetAttrString(container, attributeName); 
  long* result = PySequenceToLongArray(sequence);
  Py_DECREF(sequence); 
  return result; 
}

double* getDoubleArray(PyObject* container, const char* attributeName) {
  PyObject* sequence = PyObject_GetAttrString(container, attributeName); 
  double* result = PySequenceToDoubleArray(sequence);
  Py_DECREF(sequence); 
  return result; 
}

long** getLongArray2D(PyObject* container, const char* attributeName) {
  PyObject* sequence = PyObject_GetAttrString(container, attributeName); 
  long** result = PySequenceToLongArray2D(sequence);
  Py_DECREF(sequence); 
  return result; 
}

double** getDoubleArray2D(PyObject* container, const char* attributeName) {
  PyObject* sequence = PyObject_GetAttrString(container, attributeName); 
  double** result = PySequenceToDoubleArray2D(sequence);
  Py_DECREF(sequence); 
  return result; 
}

/****** Conversion from C to Python **************/

// Sequence functions to be able to build the sequences from the C code
// They are initialized in PyInit_rockmate_csolver (see bottom of the file)
PyObject* LossFunc;
PyObject* ForwardEnableFunc;
PyObject* ForwardNogradFunc;
PyObject* ForwardCheckFunc;
PyObject* BackwardFunc;

// Implement this for easier compatibility with Python < 3.9
static PyObject*
Object_CallOneArg(PyObject* o, PyObject* arg) {
  PyObject* tuple = PyTuple_Pack(1, arg);
  PyObject* result = PyObject_CallObject(o, tuple);
  Py_DECREF(tuple);
  return result;
}

static inline PyObject*
call_with_bool(PyObject* func, PyObject* i, int keep_input) {
  PyObject* py_keep_input = keep_input ? Py_True : Py_False;
  Py_INCREF(py_keep_input);
  PyObject* tuple = PyTuple_Pack(2, i, py_keep_input);
  PyObject* result = PyObject_CallObject(func, tuple);
  Py_DECREF(tuple);
  return result;
}

static inline PyObject*
call_two_ints(PyObject* func, int x, int y) {
  return PyObject_CallFunction(func, "ii", x, y);
}

static inline PyObject*
call_one_int(PyObject* func, int x) {
  return PyObject_CallFunction(func, "i", x);
}

static inline PyObject*
call_with_bool_from_long(PyObject* func, long i, int keep_input) {
  PyObject* py_i = PyLong_FromLong(i);
  PyObject* result = call_with_bool(func, py_i, keep_input);
  Py_DECREF(py_i);
  return result;
}


/**** RkChain structure ******/


typedef struct rk_chain {
  long* nb_sol;
  long ln;

  double** fw;
  double** bw;
  long*  cw;
  long** cbw;
  long** fwd_tmp;
  long** bwd_tmp;
  long* ff_fwd_tmp;
  double* ff_fw;
} rk_chain;

static void
rk_chain_free(rk_chain* chain) {
  if (chain->fw) free(chain->fw);
  if (chain->bw) free(chain->bw);
  if (chain->cw) free(chain->cw);
  if (chain->cbw) free(chain->cbw);
  if (chain->fwd_tmp) free(chain->fwd_tmp);
  if (chain->bwd_tmp) free(chain->bwd_tmp);
  if (chain->ff_fwd_tmp) free(chain->ff_fwd_tmp);
  if (chain->ff_fw) free(chain->ff_fw);
}

static int
get_rk_chain(PyObject* chain_param, rk_chain* result) {

  result->fw = result->bw = NULL;
  result->cw = NULL;
  result->cbw = result->fwd_tmp = result->bwd_tmp = NULL;
  result->ff_fwd_tmp = NULL;
  result->ff_fw = NULL;
  
  result->fw = getDoubleArray2D(chain_param, "fw");
  if (!(result->fw)) goto on_error;

  result->bw = getDoubleArray2D(chain_param, "bw");
  if (!(result->bw)) goto on_error;

  result->cw = getLongArray(chain_param, "cw");
  if (!(result->cw)) goto on_error;

  result->cbw = getLongArray2D(chain_param, "cbw");
  if (!(result->cbw)) goto on_error;

  result->fwd_tmp = getLongArray2D(chain_param, "fwd_tmp");
  if (!(result->fwd_tmp)) goto on_error;

  result->bwd_tmp = getLongArray2D(chain_param, "bwd_tmp");
  if (!(result->bwd_tmp)) goto on_error;

  result->ff_fwd_tmp = getLongArray(chain_param, "ff_fwd_tmp");
  if (!(result->ff_fwd_tmp)) goto on_error;

  result->ff_fw = getDoubleArray(chain_param, "ff_fw");
  if (!(result->ff_fw)) goto on_error;
  
  PyObject* chain_length_param = PyObject_GetAttrString(chain_param, "ln");
  if (!chain_length_param) goto on_error;
  result->ln = PyLong_AsLong(chain_length_param);
  Py_DECREF(chain_length_param);
  if (result->ln == -1 && PyErr_Occurred())
    goto on_error;
  
  result->nb_sol = getLongArray(chain_param, "nb_sol");
  if (!result->nb_sol) goto on_error;

  return 0;

  
 on_error:
  rk_chain_free(result);
  return -1;
}

/************** Actual DynProg: Computing the table *******/

static inline int
tbl_index(int m, int a, int b, int ln) {
  assert ((a <= ln) && (b <= ln));
  return (m)*(ln+1)*(ln+1) + (a) * (ln+1) + (b);
}

// First version: compute all values in the table
static int*
compute_table(rk_chain* chain, int mmax)
{
  int ln = chain->ln;
  double* tbl_opt = calloc((1+mmax) * (ln+1) * (ln+1), sizeof(double));
  int* tbl_what = calloc((1+mmax) * (ln+1) * (ln+1), sizeof(int));
  // WHAT table: 0 means unfeasible
  //             v > 0 means (True, v-1)
  //             v < 0 means (False, -v-1)

  // Compute partial sums
  //  partial_sums_ff_fw[i] = sum(ff_fw[0:i])
  double* partial_sums_ff_fw = calloc(ln + 1, sizeof(double));
  double total = 0; 
  for(long i = 0; i < ln; ++i) {
    partial_sums_ff_fw[i] = total; 
    total += chain->ff_fw[i];
  }
  partial_sums_ff_fw[ln] = total; 
  
  // int nb_calls = 0;
  
  for (int m = 0; m <= mmax; ++m)
    for (int i = 0; i <= ln; ++i) {
      int best_k = -1;
      double best_k_value = INFINITY;
      for (int k = 0; k < chain->nb_sol[i]; ++k) 
	if ( (m >= chain->cw[i] + chain->cbw[i+1][k] + chain->bwd_tmp[i][k]) // No cw[i+1] ??
	     && (m >= chain->cw[i+1] + chain->cbw[i+1][k] + chain->fwd_tmp[i][k]) )
	  if (chain->fw[i][k] + chain->bw[i][k] < best_k_value) {
	    best_k_value = chain->fw[i][k] + chain->bw[i][k];
	    best_k = k;
	  }
      
      // Also correct if best_k == -1 and best_k_value == INFINITY
      tbl_opt[tbl_index(m, i, i, ln)] = best_k_value;
      tbl_what[tbl_index(m, i, i, ln)] = 1 + best_k;
      //nb_calls += 1;
    }

  
  for (int d = 1; d <= ln; ++d) {
    for (int a = 0; a <= ln - d; ++a) { // From a to b = a+d
      int b = a+d;
      // Compute minimum memory needed if storing nothing
      // Keep chain->cw[b+1] out of the max
      int mmin = chain->cw[a+1] + chain->ff_fwd_tmp[a];
      for (int j = a+1; j < b; ++j) {
	mmin = imax(mmin, chain->cw[j] + chain->cw[j+1] + chain->ff_fwd_tmp[j]);
      }
      mmin += chain->cw[b+1];

      //Unfeasible below mmin
      for (int m = 0; m < mmin && m <= mmax; ++m) {
	tbl_opt[tbl_index(m, a, b, ln)] = INFINITY;
	// nb_calls += 1;
      }

      for (int m = mmin; m <= mmax; ++m) {
	// Solution 1
	double best_later_value = INFINITY;
	int best_later_k = -1;
	for (int j = a+1; j <= b; ++j)
	  if (m >= chain->cw[j]) {
	    double val = partial_sums_ff_fw[j] - partial_sums_ff_fw[a]
	      + tbl_opt[tbl_index(m - chain->cw[j], j, b, ln)]
	      + tbl_opt[tbl_index(m, a, j-1, ln)];
	    if (val < best_later_value) {
	      best_later_value = val;
	      best_later_k = j;
	    }
	  }

	// Solution 2
	double best_now_value = INFINITY;
	int best_now_k = -1;
	for (int k = 0; k < chain->nb_sol[a]; ++k) {
	  if ( (m >= chain->cw[a+1] + chain->cbw[a+1][k] + chain->fwd_tmp[a][k])
	       && (m >= chain->cw[a] + chain->cbw[a+1][k] + chain->bwd_tmp[a][k]) ) {
	    double val = chain->fw[a][k] + chain->bw[a][k]
	      + tbl_opt[tbl_index(m - chain->cbw[a+1][k], a+1, b, ln)];
	    if (val < best_now_value) {
	      best_now_value = val;
	      best_now_k = k;
	    }
	  }
	}

	// Best of both solutions
	if (best_now_value < best_later_value) {
	  tbl_opt[tbl_index(m, a, b, ln)] = best_now_value;
	  tbl_what[tbl_index(m, a, b, ln)] = 1 + best_now_k;
	} else {
	  tbl_opt[tbl_index(m, a, b, ln)] = best_later_value;
	  tbl_what[tbl_index(m, a, b, ln)] = - best_later_k - 1;
	}
	//nb_calls += 1;
      }
    }
  }

  //  printf("C Nb calls: %d\n", nb_calls);
  
  free(partial_sums_ff_fw);
  free(tbl_opt);
  return tbl_what;
}

/********** RkTable: Python object representing the result of 
 **********  	     computation  **************/


typedef struct {
  PyObject_HEAD
  rk_chain chain;
  int* what;
  double* opt;
  int mmax;
} RkTable;

// Second version: compute only necessary values. The table is still
// large enough to contain all values.

static double
compute_table_base(rk_chain* chain, RkTable* tbl,
		   int mmax, int m, int i) {
  int best_k = -1;
  double best_k_value = INFINITY;
  int best_limit = 0;
  for (int k = 0; k < chain->nb_sol[i]; ++k) {
    int limit = imax(chain->cw[i] + chain->cbw[i+1][k] + chain->bwd_tmp[i][k],  // No cw[i+1] ??
		     chain->cw[i] + chain->cbw[i+1][k] + chain->fwd_tmp[i][k]);
    if (m >= limit)
      if (chain->fw[i][k] + chain->bw[i][k] < best_k_value) {
	best_k_value = chain->fw[i][k] + chain->bw[i][k];
	best_k = k;
	best_limit = limit;
      }
  }
  // Also correct if best_k == -1 and best_k_value == INFINITY
  tbl->opt[tbl_index(m, i, i, chain->ln)] = best_k_value;
  tbl->what[tbl_index(m, i, i, chain->ln)] = 1 + best_k;
  return best_k_value;
}

static void
do_compute_table_rec(rk_chain* chain, RkTable* tbl, int* mmin_values,
		     double* partial_sums_ff_fw,
		     int mmax, int m, int a, int b);

static inline void
compute_table_rec(rk_chain* chain, RkTable* tbl, int* mmin_values,
		  double* partial_sums_ff_fw,
		  int mmax, int m, int a, int b) {
  // Nothing to do if value was computed already.
  if (tbl->opt[tbl_index(m, a, b, chain->ln)] != 0)
    return;
  do_compute_table_rec(chain, tbl, mmin_values,
		       partial_sums_ff_fw,
		       mmax, m, a, b);
}

static void
do_compute_table_rec(rk_chain* chain, RkTable* tbl, int* mmin_values,
		     double* partial_sums_ff_fw,
		     int mmax, int m, int a, int b) {
  int ln = chain->ln;

  if (a == b) {
    compute_table_base(chain, tbl, mmax, m, a);
    return;
  }

  int mmin = mmin_values[a * ln + b];
  if (mmin == 0) {
    mmin = chain->cw[a+1] + chain->ff_fwd_tmp[a];
    for (int j = a+1; j < b; ++j) {
      mmin = imax(mmin, chain->cw[j] + chain->cw[j+1] + chain->ff_fwd_tmp[j]);
    }
    mmin += chain->cw[b+1];
    mmin_values[a * ln + b] = mmin;
  }

  //Unfeasible below mmin
  if (m < mmin) {
    tbl->opt[tbl_index(m, a, b, ln)] = INFINITY;
    return;
  }

  // Solution 1
  double best_later_value = INFINITY;
  int best_later_k = -1;
  for (int j = a+1; j <= b; ++j)
    if (m >= chain->cw[j]) {
      compute_table_rec(chain, tbl, mmin_values,
			partial_sums_ff_fw,
			mmax, m - chain->cw[j], j, b);
      compute_table_rec(chain, tbl, mmin_values,
			partial_sums_ff_fw,
			mmax, m, a, j-1);
      double val = partial_sums_ff_fw[j] - partial_sums_ff_fw[a]
	+ tbl->opt[tbl_index(m - chain->cw[j], j, b, ln)]
	+ tbl->opt[tbl_index(m, a, j-1, ln)];
      if (val < best_later_value) {
	best_later_value = val;
	best_later_k = j;
      }
    }

  // Solution 2
  double best_now_value = INFINITY;
  int best_now_k = -1;
  for (int k = 0; k < chain->nb_sol[a]; ++k) {
    int limit = imax(chain->cw[a+1] + chain->cbw[a+1][k] + chain->fwd_tmp[a][k],
		    chain->cw[a] + chain->cbw[a+1][k] + chain->bwd_tmp[a][k]);
    if (m >= limit) {
      compute_table_rec(chain, tbl, mmin_values,
			partial_sums_ff_fw,
			mmax, m - chain->cbw[a+1][k], a+1, b);
      double val = chain->fw[a][k] + chain->bw[a][k]
	+ tbl->opt[tbl_index(m - chain->cbw[a+1][k], a+1, b, ln)];
      if (val < best_now_value) {
	best_now_value = val;
	best_now_k = k;
      }
    }
  }
  // Best of both solutions
  if (best_now_value < best_later_value) {
    tbl->opt[tbl_index(m, a, b, ln)] = best_now_value;
    tbl->what[tbl_index(m, a, b, ln)] = 1 + best_now_k;
  } else {
    tbl->opt[tbl_index(m, a, b, ln)] = best_later_value;
    tbl->what[tbl_index(m, a, b, ln)] = - best_later_k - 1;
  }
}


static void
compute_table_v2(rk_chain* chain, RkTable* tbl, int mmax)
{
  int ln = chain->ln;
  // WHAT table: 0 means unfeasible
  //             v > 0 means (True, v-1)
  //             v < 0 means (False, -v-1)

  // Compute partial sums
  //  partial_sums_ff_fw[i] = sum(ff_fw[0:i])
  double* partial_sums_ff_fw = calloc(ln + 1, sizeof(double));
  double total = 0;
  for(long i = 0; i < ln; ++i) {
    partial_sums_ff_fw[i] = total;
    total += chain->ff_fw[i];
  }
  partial_sums_ff_fw[ln] = total;

  // Avoid recomputing mmin if already done
  int* mmin_values = calloc((ln+1) * (ln+1), sizeof(int));

  compute_table_rec(chain, tbl, mmin_values,
		    partial_sums_ff_fw,
		    mmax, mmax, 0, ln);

  free(partial_sums_ff_fw);
  free(mmin_values);
}


/************** Actual DynProg: Building the sequence *******/

static int
build_sequence(rk_chain* chain, int* what, int cmem,
	       int lmin, int lmax,
	       PyObject* result[], size_t* idx_result){

  assert (lmin >= 0);
  
  if (lmin > lmax)
    return 0;
  
  if ((cmem <= 0) || (what[tbl_index(cmem, lmin, lmax, chain->ln)] == 0)) {
    PyErr_SetString(PyExc_ValueError, "Can't find a feasible sequence with the given budget");
    return -1;
  }

  if(lmin == chain->ln) {
    result[(*idx_result)++] = PyObject_CallObject(LossFunc, NULL);
    return 0;
  }
  
  int w = what[tbl_index(cmem, lmin, lmax, chain->ln)];
  if (w > 0) {
    int k = w - 1;
    result[(*idx_result)++] = call_two_ints(ForwardEnableFunc, lmin, k);
    if (build_sequence(chain, what, cmem - chain->cbw[lmin+1][k],
		       lmin+1, lmax, result, idx_result))
      return -1;
    result[(*idx_result)++] = call_two_ints(BackwardFunc, lmin, k);
  } else {
    int j = -w - 1;
    result[(*idx_result)++] = call_one_int(ForwardCheckFunc, lmin);
    for (int k = lmin + 1; k < j; ++k)
      result[(*idx_result)++] = call_one_int(ForwardNogradFunc, k);
    if (build_sequence(chain, what, cmem - chain->cw[j],
		       j, lmax, result, idx_result))
      return -1;
    if (build_sequence(chain, what, cmem,
		       lmin, j - 1, result, idx_result))
      return -1;
  }
  return 0;
}



static void
RkTable_dealloc(RkTable *self)
{
  rk_chain_free(&self->chain);
  if(self->opt) free(self->opt);
  if(self->what) free(self->what);

  Py_TYPE(self)->tp_free((PyObject *) self);
}


static int
RkTable_init(RkTable* self, PyObject* args, PyObject* kwds) {
  PyObject* rk_chain_param;
  int mmax;

  self->opt = NULL;
  self->what = NULL;

  if (!PyArg_ParseTuple(args, "Oi", &rk_chain_param, & mmax))
    return -1;

  if (get_rk_chain(rk_chain_param, &self->chain))
    return -1;
  int ln = (self->chain).ln;

  self->opt = calloc((1+mmax) * (ln+1) * (ln+1), sizeof(double));
  self->what = calloc((1+mmax) * (ln+1) * (ln+1), sizeof(int));
  self->mmax = mmax;
  if (!self->opt || !self->what) {
    PyErr_NoMemory();
    return -1;
  }

  return 0;
}


static PyObject*
RkTable_get_opt(RkTable* self, PyObject* args) {
  int memory_limit;

  if (!PyArg_ParseTuple(args, "i", & memory_limit))
    return NULL;
  if (memory_limit < 0 || memory_limit > self->mmax) {
    return PyErr_Format(PyExc_ValueError, "Can not solve with limit %d, this table has mmax=%d", memory_limit, self->mmax);
  }

  compute_table_v2(&self->chain, self, memory_limit);
  double result = self->opt[tbl_index(memory_limit, 0, self->chain.ln, self->chain.ln)];
  PyObject* pyresult = PyFloat_FromDouble(result);
  return pyresult;
}

static PyObject*
RkTable_build_sequence(RkTable* self, PyObject* args) {
  int memory_limit;

  if (!PyArg_ParseTuple(args, "i", & memory_limit))
    return NULL;

  if (memory_limit < 0 || memory_limit > self->mmax) {
    return PyErr_Format(PyExc_ValueError, "Can not solve with limit %d, this table has mmax=%d", memory_limit, self->mmax);
  }

  compute_table_v2(&self->chain, self, memory_limit);

  // Make sure that the array is large enough
  PyObject* sequence[self->chain.ln * self->chain.ln];
  size_t sequence_length = 0;
  
  if (build_sequence(&self->chain, self->what, memory_limit,
		     0, self->chain.ln,
		     sequence, &sequence_length))
    return NULL;

  PyObject* result = PyList_New(sequence_length);
  for(size_t k = 0; k < sequence_length; ++k) {
    PyList_SET_ITEM(result, k, sequence[k]);
  }

  return result;
  // Py_RETURN_NONE;
}

static PyMethodDef RkTable_methods[] = {
    {"build_sequence", (PyCFunction) RkTable_build_sequence, METH_VARARGS,
     "Build a sequence from the table, given a memory limit."
    },
    {"get_opt", (PyCFunction) RkTable_get_opt, METH_VARARGS,
     "get optimal duration of the sequence, given a memory limit."
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject RkTableType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "rockmate.csolver.RkTable",
  .tp_doc = PyDoc_STR("Rockmate Table"),
  .tp_basicsize = sizeof(RkTable),
  .tp_itemsize = 0,
  .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  .tp_new = PyType_GenericNew,
  .tp_init = (initproc) RkTable_init,
  .tp_dealloc = (destructor) RkTable_dealloc,
  //.tp_members = Custom_members,
  .tp_methods = RkTable_methods,
};

    
static struct PyModuleDef rotor_solver_module = {
    PyModuleDef_HEAD_INIT,
    "rockmate.csolver",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
};

PyMODINIT_FUNC
PyInit_csolver(void)
{
  PyObject *m;
  if (PyType_Ready(&RkTableType) < 0)
    return NULL;

  m = PyModule_Create(&rotor_solver_module);
  if (m == NULL)
    return NULL;

  Py_INCREF(&RkTableType);
  if (PyModule_AddObject(m, "RkTable", (PyObject *) &RkTableType) < 0) {
    Py_DECREF(&RkTableType);
    Py_DECREF(m);
    return NULL;
  }

  PyObject* seq_module_name = PyUnicode_DecodeFSDefault("rockmate.csequence");
  PyObject* seq_module = PyImport_Import(seq_module_name);
  if (!seq_module)
    return NULL;

  LossFunc = PyObject_GetAttrString(seq_module, "SeqLoss");
  assert(LossFunc && PyCallable_Check(LossFunc));
  ForwardEnableFunc = PyObject_GetAttrString(seq_module, "SeqBlockFe");
  assert(ForwardEnableFunc && PyCallable_Check(ForwardEnableFunc));
  ForwardNogradFunc = PyObject_GetAttrString(seq_module, "SeqBlockFn");
  assert(ForwardNogradFunc && PyCallable_Check(ForwardNogradFunc));
  ForwardCheckFunc = PyObject_GetAttrString(seq_module, "SeqBlockFc");
  assert(ForwardCheckFunc && PyCallable_Check(ForwardCheckFunc));
  BackwardFunc = PyObject_GetAttrString(seq_module, "SeqBlockBwd");
  assert(BackwardFunc && PyCallable_Check(BackwardFunc));

  
  return m;
}

