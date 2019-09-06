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

#include "MMSolver.h"

void coutx(string a) {
	cout << a << endl;
}

#include "Dlib.h"
//#include "OpenCV.h"
//#include "FileUtil.h"



np::ndarray ToNP(Mat image)
{
	namespace p = boost::python;
	namespace np = boost::python::numpy;

	//TODO We should use opencv source code....
	//https://github.com/opencv/opencv/blob/master/modules/python/src2/cv2.cpp
	unsigned char *data = image.data;
	int W = image.cols;
	int H = image.rows;
	

	const np::dtype dt = np::dtype::get_builtin<unsigned char>();
	py::tuple shape = py::make_tuple(H, W, 3);
	py::tuple stride = py::make_tuple((int)image.step, 3, 1);
	const py::object own;
	np::ndarray data_ex = np::from_data<py::tuple>((void*)data, dt, shape, stride, own);

	return data_ex.copy();
	
}


Mat ToCV(np::ndarray arr)
{
	//TODO We should use opencv source code....
	//https://github.com/opencv/opencv/blob/master/modules/python/src2/cv2.cpp
	int nd = arr.get_nd();
	if (nd != 3)
		throw std::runtime_error("a must be 3-dimensional");
	size_t H = arr.shape(0);
	size_t W = arr.shape(1);
	size_t C = arr.shape(2);

	auto flag = arr.get_flags();
	if (flag && np::ndarray::C_CONTIGUOUS == 0)
		throw std::runtime_error("Not C_CONTIGUOUS");

	if (C != 3)
		throw std::runtime_error("Only 3 channesl");

	if (arr.get_dtype() != np::dtype::get_builtin<unsigned char>())
		throw std::runtime_error("Must be uint8 array");
	unsigned char *p = reinterpret_cast<unsigned char *>(arr.get_data());
	
	return Mat(H, W, CV_8UC3, p);

}

MatF ToEigen(np::ndarray arr)
{
	int nd = arr.get_nd();
	if (nd > 2)
		throw std::runtime_error("a must be 1 or 2-dimensional");

	auto flag = arr.get_flags();
	if (flag && np::ndarray::C_CONTIGUOUS == 0)
		throw std::runtime_error("Not C_CONTIGUOUS");

	if (arr.get_dtype() != np::dtype::get_builtin<float>())
		throw std::runtime_error("a must be float32 array");

	float *p = reinterpret_cast<float *>(arr.get_data());


	if (nd == 1)
	{
		return Map<MatF>(p, arr.shape(0), 1);
	}
	else
	{
		return Map<MatF>(p, arr.shape(0), arr.shape(1));
	}
}


void PyImshow(np::ndarray img)
{
	Mat image = ToCV(img);
	imshow("IMG", image);
	waitKey(1);
}




MMSolver PyMMS;
FaceModel PyFM;

void InitMMS(string fmkp, string fmfull)
{
	cout << fmkp << endl;
	cout << fmfull << endl;
	PyMMS.Initialize(fmkp, fmfull);
}


boost::python::tuple RunMMS(np::ndarray img, np::ndarray kp)
{
	cout << "OK" << endl;
	MatF KP = ToEigen(kp);
	cout << KP << endl;


	Mat image = ToCV(img);

	int W = 512;
	int H = 512;

	assert(KP.rows() == 68);
	assert(KP.cols() == 2);



	PyMMS.Solve(KP);
	Mat res = MMSDraw(image, PyMMS, KP);
	Mat texture = MMSTexture(image, PyMMS, W, H);
	Mat normal = MMSNormal(image, PyMMS, W, H);
	/*imshow("IMG", res);
	imshow("TEX", texture);
	waitKey(1);*/
	return boost::python::make_tuple(ToNP(res), ToNP(texture), ToNP(normal));

}

void InitFM(string headfm)
{
	cout << headfm << endl;
	PyFM.Initialize(headfm, true);
}

#include "Poisson.h"

Mat Pymsk;
Mat Pytarget;
Mat PyHT;

void InitImages(string mskimage, string textureimage, string headtexture)
{
	Mat msk0 = imread(mskimage, 0);
	Pymsk = (msk0 == 255);
	Pytarget = imread(textureimage);
	PyHT = imread(headtexture);
}
bool GenerateModel(np::ndarray img, np::ndarray kp, string outfolder, string filename)
{
	


	MatF KP;

	Mat image = ToCV(img);

	if (kp.shape(0) == 68 && kp.shape(1) == 2)
	{
		KP = ToEigen(kp);
	}
	else
	{
		vector<vector<Point>> keypoints;
		double S = 0.5;
		Mat simage;
		resize(image, simage, Size(), S, S);

		vector<Rect> rectangles;
		DlibInit("shape_predictor_68_face_landmarks.dat");
		DlibFace(simage, rectangles, keypoints);

		if (keypoints.size() <= 0)
		{
			errorcout << "NO POINTS" << endl;
			return false;
		}

		KP = ToEigen(keypoints[0]) * (1.0 / S);
	}

	assert(KP.rows() == 68);
	assert(KP.cols() == 2);



	PyMMS.Solve(KP);

	if (false)
	{
		Matrix3f m;
		m << 1, 0, 0,
			0, -1, 0,
			0, 0, -1;

		cout << PyMMS.params.R << endl;

		if ((m - PyMMS.params.R).norm() > 0.4)
		{
			//Only Use straight face.
			return false;
		}
	}


	MMSObj(image, PyMMS, outfolder, filename);

	int maxsize = max(image.cols, image.rows);
	double ratio = 512.0 / maxsize;
	MatF KPX = KP * (float)ratio;
	resize(image, image, Size(), ratio, ratio);

	Mat draw = MMSDraw(image, PyMMS, KPX);
	imwrite(outfolder + filename + "D.jpg", draw);


	string texname = outfolder + filename + ".png";
	Mat src = imread(texname);


	Scalar diff;

	Mat result;
	result = PoissonBlending(src, Pytarget, Pymsk, true, &diff);

	imwrite(texname, result * 255);


	PyFM.Generate(PyMMS.SX, PyMMS.EX);
	FMObj(PyHT + diff * 255, PyFM, outfolder, filename + "H");


	return true;
}


BOOST_PYTHON_MODULE(SimpleFace) {
	using namespace boost::python;

	Py_Initialize();
	np::initialize();

	def("coutx", coutx);
	def("imshow", PyImshow);

	def("DlibInit", DlibInit);
	def("InitImages", InitImages);

	def("InitMMS", InitMMS);
	def("RunMMS", RunMMS);

	def("InitFM", InitFM);
	def("GenerateModel", GenerateModel);
}
#endif