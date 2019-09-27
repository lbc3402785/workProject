#ifndef CERESNONLINEAR_HPP
#define CERESNONLINEAR_HPP
#include "common/torchfunctions.h"
#include "mmtensorsolver.h"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/transform.hpp"
namespace fitting {
struct LandmarkCost{
    LandmarkCost(FaceModelTensor& keyFace,FaceModelTensor& fullFace,torch::Tensor observed,int vertexId):keyFace(keyFace),fullFace(fullFace),observed(observed),vertexId(vertexId){};
    template <typename T>
    bool operator()(const T* const cameraRotation, const T* const cameraTranslation,
                    const T* const shapeCoeffs, const T* const blendshapeCoeffs, T* residual)const;
private:
    FaceModelTensor keyFace;
    FaceModelTensor fullFace;
    torch::Tensor observed;
    int vertexId;
};

template<typename T>
bool LandmarkCost::operator()(const T * const cameraRotation, const T * const cameraTranslation, const T * const shapeCoeffs, const T * const blendshapeCoeffs, T *residual)const
{
    using namespace glm;
    torch::Tensor shapeX=torch::from_blob((double*)shapeCoeffs,{keyFace.SB.size(1),1},at::TensorOptions().dtype(torch::kDouble));
    torch::Tensor blendShapeX=torch::from_blob((double*)blendshapeCoeffs,{keyFace.EB.size(1),1},at::TensorOptions().dtype(torch::kDouble));
    torch::Tensor FaceS= torch::matmul(keyFace.SB , shapeX.toType(torch::kFloat));
    torch::Tensor FaceE= torch::matmul(keyFace.EB , blendShapeX.toType(torch::kFloat));
    torch::Tensor currentKeyFace=keyFace.Face+FaceS.view({-1,3})+FaceE.view({-1,3});
    torch::Tensor keyPoint=currentKeyFace[vertexId];

    const tvec3<T> point3d(keyPoint[0].item().toFloat(), keyPoint[1].item().toFloat(), keyPoint[2].item().toFloat());
    const tquat<T> rotQuat(cameraRotation[0], cameraRotation[1], cameraRotation[2],
            cameraRotation[3]);
    const tmat4x4<T> rotMtx = glm::mat4_cast(rotQuat);
    const auto tMtx = glm::translate(tvec3<T>(cameraTranslation[0],cameraTranslation[1],0.0f));
    tvec4<T> tmp(point3d,(T)1);
    tmp= tMtx * rotMtx  *tmp;
    T proX=tmp.x*cameraTranslation[2];
    T proY=tmp.y*cameraTranslation[2];
    //    std::cout<<"proX:"<<proX<<std::endl;
    //    std::cout<<"proY:"<<proY<<std::endl;
    residual[0]=proX-T(observed[0].item().toFloat());
    residual[1]=proY-T(observed[1].item().toFloat());
    //    std::cout<<"residual[0]:"<<residual[0]<<std::endl;
    //    std::cout<<"residual[1]:"<<residual[1]<<std::endl;

    return true;
}
struct MultiLandmarkCost{
public:

    MultiLandmarkCost(FaceModelTensor& keyFace,FaceModelTensor& fullFace,torch::Tensor observed,int viewIndex,int vertexId,int stride):keyFace(keyFace),fullFace(fullFace),observed(observed),viewIndex(viewIndex),vertexId(vertexId),stride(stride){};
    template <typename T>
    bool operator()(T const* const* parameters, T* residuals)const;
private:
    FaceModelTensor keyFace;
    FaceModelTensor fullFace;
    torch::Tensor observed;
    int viewIndex;
    int vertexId;
    int stride;
};
template<typename T>
bool MultiLandmarkCost::operator()(T const* const* parameters,T *residuals) const
{
    const T * const cameras=parameters[0];
    const T * const shapeCoeffs=parameters[1];
    const T * const blendshapeCoeffs=parameters[2];
    torch::Tensor tmp=torch::from_blob((void*)shapeCoeffs,{keyFace.SB.size(1),1+stride},at::TensorOptions().dtype(torch::kDouble));
    torch::Tensor shapeX=tmp.select(1,0).toType(torch::kFloat);

    torch::Tensor tmp1=torch::zeros({keyFace.EB.size(1),stride+1},at::TensorOptions().dtype(torch::kDouble));
    torch::Tensor blendShapeX=tmp1.select(1,0).toType(torch::kFloat);
    torch::Tensor pS=torch::matmul(keyFace.SB.slice(0,3*vertexId,3*vertexId+3),shapeX);//3
    torch::Tensor pE=torch::matmul(keyFace.EB.slice(0,3*vertexId,3*vertexId+3),blendShapeX);//3
    torch::Tensor keyPoint=(keyFace.Face[vertexId]+pS+pE).toType(torch::kDouble);//3
    torch::Tensor tmpC=torch::from_blob((void*)(cameras+6*viewIndex),{6,1+stride},at::TensorOptions().dtype(torch::kDouble));
    torch::Tensor camera=tmpC.select(1,0);//6
//    std::cout<<"camera:"<<camera<<std::endl;
    torch::Tensor axis=camera.slice(0,0,3);//3
    double tx=camera[3].item().toDouble();
    double ty=camera[4].item().toDouble();
    double scale=camera[5].item().toDouble();
//        std::cout<<"tx:"<<tx<<" ty:"<<ty<<" scale:"<<scale<<std::endl;
    torch::Tensor R=TorchFunctions::rodrigues(axis.unsqueeze(-1));
//    std::cout<<"R:"<<R<<std::endl;
    torch::Tensor pro=torch::matmul(R,keyPoint);//3
    double proX=(pro[0].item().toDouble()+tx)*scale;
    double proY=(pro[1].item().toDouble()+ty)*scale;
    double difX=proX-double(observed[0].item().toFloat());
    double difY=proY-double(observed[1].item().toFloat());
    if(std::abs(difX)>10.0||std::abs(difY)>10.0){
        residuals[0]=T(0);
        residuals[1]=T(0);
    }else{
        residuals[0]=T(difX);
        residuals[1]=T(difY);
    }
//    residuals[0]=T(proX-double(observed[0].item().toFloat()));
//    residuals[1]=T(proY-double(observed[1].item().toFloat()));
    return true;
}
/**
 * Cost function for a prior on the parameters.
 *
 * Prior towards zero (0, 0...) for the parameters.
 * Note: The weight is inside the norm, so may not correspond to the "usual"
 * formulas. However I think it's equivalent up to a scaling factor, but it
 * should be checked.
 */
struct PriorCost
{

    /**
     * Creates a new prior object with set number of variables and a weight.
     *
     * @param[in] num_variables Number of variables that the parameter vector contains.
     * @param[in] weight A weight that the parameters are multiplied with.
     */
    PriorCost(int numVariables, double weight = 1.0) : numVariables(numVariables), weight(weight){};

    /**
     * Cost function implementation.
     *
     * @param[in] x An array of parameters.
     * @param[in] residual An array of the resulting residuals.
     * @return Returns true. The ceres documentation is not clear about that I think.
     */
    template <typename T>
    bool operator()(const T* const x, T* residual) const
    {
        for (int i = 0; i < numVariables; ++i)
        {
            residual[i] = weight * x[i];
        }
        return true;
    };


public:
    double PriorCost::getWeight() const
    {
        return weight;
    }

    void PriorCost::setWeight(double value)
    {
        weight = value;
    }

private:
    int numVariables;
    double weight;
};





}
#endif // CERESNONLINEAR_HPP
