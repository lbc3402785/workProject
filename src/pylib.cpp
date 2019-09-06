#ifndef NOPYTHONMODULE
#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB
#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>

namespace py = boost::python;
namespace np = boost::python::numpy;

using namespace std;

void coutxx(string a) {
	cout << a << endl;
}


BOOST_PYTHON_MODULE(XYZ) {
	using namespace boost::python;

	Py_Initialize();
	np::initialize();

	def("coutxx", coutxx);
	
}
#endif