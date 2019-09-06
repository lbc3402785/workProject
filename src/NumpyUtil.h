#pragma once

#include"cnpy.h"
#include<iostream>
#include<string>



#include <Eigen/Dense>
#include "EigenUtil.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace Eigen;
using namespace cnpy;

typedef Eigen::Matrix<float, Dynamic, Dynamic, RowMajor> MatF;
typedef Eigen::Matrix<int, Dynamic, Dynamic, RowMajor> MatI;
typedef Eigen::Matrix<float, Dynamic, Dynamic, ColMajor> MatFC;
typedef Eigen::Matrix<int, Dynamic, Dynamic, ColMajor> MatIC;

inline Map<MatF> ToEigen(NpyArray &np)
{
	if (np.data_holder == nullptr)
	{
		throw "No Data";
	}

	auto shape = np.shape;
	auto L = np.shape.size();
	if (L > 2)
	{
		cout << "Error" << endl;
	}

	if (L == 1)
	{
		return Map<MatF>(np.data<float>(), shape[0], 1);
	}
	else
	{
		return Map<MatF>(np.data<float>(), shape[0], shape[1]);
	}
}

inline MatF ToEigenC(NpyArray &np)
{
	if (np.data_holder == nullptr)
	{
		throw "No Data";
	}

	auto shape = np.shape;
	auto L = np.shape.size();
	if (L > 2)
	{
		cout << "Error" << endl;
	}

	if (L == 1)
	{
		return Map<MatFC>(np.data<float>(), shape[0], 1);
	}
	else
	{
		return Map<MatFC>(np.data<float>(), shape[0], shape[1]);
	}
}

inline Map<MatI> ToEigenInt(NpyArray &np)
{
	if (np.data_holder == nullptr)
	{
		throw "No Data";
	}

	auto shape = np.shape;
	auto L = np.shape.size();
	if (L > 2)
	{
		cout << "Error" << endl;
	}

	if (L == 1)
	{
		return Map<MatI>(np.data<int>(), shape[0], 1);
	}
	else
	{
		return Map<MatI>(np.data<int>(), shape[0], shape[1]);
	}
}

inline MatF Slice(MatF A, vector<int> indexes)
{

	MatF Y = MatF::Zero(indexes.size() * 3, A.cols());
	for (size_t i = 0; i < indexes.size(); i++)
	{
		auto idx = indexes[i];
		Y.row(3 * i + 0) = A.row(3 * idx + 0);
		Y.row(3 * i + 1) = A.row(3 * idx + 1);
		Y.row(3 * i + 2) = A.row(3 * idx + 2);
	}

	return Y;
}

inline vector<int> ToVector(MatI &mat)
{
	vector<int> vec(mat.data(), mat.data() + mat.size());
	return vec;
}

inline Map<MatF> Reshape(MatF &vec, int cols)
{
	return Map<MatF>((float*)vec.data(), Eigen::Index(vec.size() / cols), Eigen::Index(cols));
}
