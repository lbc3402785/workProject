#ifndef DATACONVERTOR_H
#define DATACONVERTOR_H
#include <cassert>
#include <torch/torch.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
namespace EigenToTorch {
    template<typename T,int _Options>
    using  MatrixType = Eigen::Matrix<T,Eigen::Dynamic, Eigen::Dynamic,_Options>;

    template<typename Scalar,int rank,int _Options, typename sizeType>
    auto EigenTensorToEigenMatrix(const Eigen::Tensor<Scalar,rank,_Options> &tensor,const sizeType rows,const sizeType cols)
    {
        return Eigen::Map<const MatrixType<Scalar,_Options>> (tensor.data(), rows,cols);
    }


    template<typename Scalar,int _Options, typename... Dims>
    auto EigenMatrixToEigenTensor(const MatrixType<Scalar,_Options> matrix, Dims... dims)
    {
        constexpr int rank = sizeof... (Dims);
        return std::move(Eigen::TensorMap<Eigen::Tensor<const Scalar, rank,_Options>>(matrix.data(), {dims...}));
    }
    template<typename Scalar,int rows,int cols,int _Options>
    auto EigenMatrixToEigenTensor(Eigen::Matrix<Scalar,rows,cols,_Options>& matrix)
    {
        return std::move(Eigen::TensorMap<Eigen::Tensor<const Scalar,2,_Options>>(matrix.data(), {rows,cols}));
    }
    template<typename... Dims>
    auto TorchTensorToEigenTensor(torch::Tensor& t, Dims... dims)
    {
        constexpr int rank = sizeof... (Dims);
        return std::move(Eigen::TensorMap<Eigen::Tensor<float, rank,Eigen::RowMajor>>(t.data<float>(), {dims...}));
    }
    static auto TorchTensorToEigenMatrix(torch::Tensor& t)
    {
        assert(t.dim()==2);
        return Eigen::Map<const MatrixType<float,Eigen::RowMajor>> (t.data<float>(), t.sizes()[0],t.sizes()[1]);
    }
    template<int rank>
    auto EigenTensorToTorchTensor(const Eigen::Tensor<float,rank,Eigen::RowMajor> &tensor){
        return std::move(torch::from_blob((void*)tensor.data(),tensor.dimensions()).clone());
    }
    static auto EigenMatrixToTorchTensor(const MatrixType<float,Eigen::RowMajor> &matrix)
    {
        const std::array<int64_t,2> sizes={matrix.rows(),matrix.cols()};
        return std::move(torch::from_blob((void*)matrix.data(),torch::IntList(sizes)).clone());
    }
}
#endif // DATACONVERTOR_H
