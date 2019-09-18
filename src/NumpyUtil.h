#pragma once

#include"cnpy.h"
#include<iostream>
#include<string>



#include <Eigen/Dense>
#include "EigenUtil.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <torch/torch.h>
using namespace std;
using namespace Eigen;
using namespace cnpy;

typedef Eigen::Matrix<float, Dynamic, Dynamic, RowMajor> MatF;
typedef Eigen::Matrix<int, Dynamic, Dynamic, RowMajor> MatI;
typedef Eigen::Matrix<float, Dynamic, Dynamic, ColMajor> MatFC;
typedef Eigen::Matrix<int, Dynamic, Dynamic, ColMajor> MatIC;
inline torch::Tensor ToTensor(NpyArray &np)
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
        return std::move(torch::from_blob(np.data<float>(), {(int64_t)shape[0], (int64_t)1}).clone());
    }
    else
    {
        return std::move(torch::from_blob(np.data<float>(), {(int64_t)shape[0], (int64_t)shape[1]}).clone());
    }
}
inline torch::Tensor ToTensor(std::vector<cv::Point> imagePoints)
{
        int N = imagePoints.size();

        torch::Tensor b=torch::zeros({N, 2});
        for (int i = 0; i < N; ++i)
        {
            b[i][0]=imagePoints[i].x;b[i][1]=imagePoints[i].y;
        }

        return std::move(b);
}
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
inline MatF ToEigen(std::vector<cv::Point> image_points)
{
        int N = image_points.size();

        MatF b(N, 2);
        for (int i = 0; i < N; ++i)
        {
                Eigen::Vector2f p = Eigen::Vector2f(image_points[i].x, image_points[i].y);
                b.block<1, 2>(i, 0) = p;
        }

        return b;
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

inline torch::Tensor ToTensorInt(NpyArray &np)
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
        return std::move(torch::from_blob(np.data<int>(), {(int64_t)shape[0], (int64_t)1},at::TensorOptions().dtype(torch::kInt)).clone());
    }
    else
    {
        return std::move(torch::from_blob(np.data<int>(), {(int64_t)shape[0], (int64_t)shape[1]},at::TensorOptions().dtype(torch::kInt)).clone());
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
