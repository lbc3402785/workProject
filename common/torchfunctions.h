#ifndef TORCHFUNCTIONS_H
#define TORCHFUNCTIONS_H
#include <torch/torch.h>
#include <Eigen/Sparse>
class TorchFunctions
{
public:
    TorchFunctions();
    static torch::Tensor rodrigues(torch::Tensor theta);
   static torch::Tensor batchRodrigues(torch::Tensor& thetas);
    static torch::Tensor unRodrigues(torch::Tensor& R);
    static void saveObj(std::string name,torch::Tensor m,torch::Tensor state);
    static void saveObj(std::string name,torch::Tensor m,std::vector<int64_t> faceIds);
    static void saveObj(std::string name,torch::Tensor m,torch::Tensor n,std::vector<int64_t> faceIds);
    static at::Tensor computeSingleNormal( at::Tensor &model,  std::vector<int64_t>& faceIds);
    static at::Tensor computeFaceNormals( at::Tensor model, at::Tensor f);
    static Eigen::SparseMatrix<float,Eigen::RowMajor> computeFaceNormalsWeight(int64_t& vnum, at::Tensor &f);
    static at::Tensor computeVertNormals( at::Tensor &model, at::Tensor &f);
    static at::Tensor computeVertNormals( at::Tensor &model, at::Tensor &f,Eigen::SparseMatrix<float, Eigen::RowMajor>& weigths);
    static at::Tensor Kronecker(at::Tensor &A, at::Tensor &B);
};

#endif // TORCHFUNCTIONS_H
