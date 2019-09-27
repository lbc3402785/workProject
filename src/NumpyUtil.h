#pragma once


#include<iostream>
#include<string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include"cnpy.h"

using namespace cnpy;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatF;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatI;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatFC;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatIC;
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
        std::cout << "Error" << std::endl;
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
inline Eigen::Map<MatF> ToEigen(NpyArray &np)
{
	if (np.data_holder == nullptr)
	{
		throw "No Data";
	}

	auto shape = np.shape;
	auto L = np.shape.size();
	if (L > 2)
	{
        std::cout << "Error" << std::endl;
	}

	if (L == 1)
	{
        return Eigen::Map<MatF>(np.data<float>(), shape[0], 1);
	}
	else
	{
        return Eigen::Map<MatF>(np.data<float>(), shape[0], shape[1]);
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
        std::cout << "Error" << std::endl;
	}

	if (L == 1)
	{
        return Eigen::Map<MatFC>(np.data<float>(), shape[0], 1);
	}
	else
	{
        return Eigen::Map<MatFC>(np.data<float>(), shape[0], shape[1]);
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
        std::cout << "Error" << std::endl;
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
inline Eigen::Map<MatI> ToEigenInt(NpyArray &np)
{
	if (np.data_holder == nullptr)
	{
		throw "No Data";
	}

	auto shape = np.shape;
	auto L = np.shape.size();
	if (L > 2)
	{
        std::cout << "Error" << std::endl;
	}

	if (L == 1)
	{
        return Eigen::Map<MatI>(np.data<int>(), shape[0], 1);
	}
	else
	{
        return Eigen::Map<MatI>(np.data<int>(), shape[0], shape[1]);
	}
}

inline MatF Slice(MatF A, std::vector<int> indexes)
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

inline std::vector<int> ToVector(MatI &mat)
{
    std::vector<int> vec(mat.data(), mat.data() + mat.size());
	return vec;
}

inline Eigen::Map<MatF> Reshape(MatF &vec, int cols)
{
    return Eigen::Map<MatF>((float*)vec.data(), Eigen::Index(vec.size() / cols), Eigen::Index(cols));
}
