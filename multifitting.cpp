#include "multifitting.h"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/transform.hpp"
#include "glmfunctions.h"
MultiFitting::MultiFitting()
{

}

void MultiFitting::fitShapeAndPose(std::vector<cv::Mat>& images,ContourLandmarks& contour,MMSolver& PyMMS,std::vector<MatF>& landMarks,MatF &shapeX,
                                   std::vector<MatF> &blendShapeXs, std::vector<MatF> &fittedImagePoints,int iterNum)
{
    int imageNum = static_cast<int>(landMarks.size());
    //std::cout<<"imageNum:"<<imageNum<<std::endl;
    std::vector<ProjectionParameters> params;
    std::vector<MatF> currentModelPoints;
    std::vector<MatF> currentMeshes;
    params.reserve(imageNum);
    blendShapeXs.reserve(imageNum);
    currentMeshes.reserve(imageNum);
    float Lambdas[4] = { 100.0, 30.0, 10.0, 5.0 };
    torch::Tensor modelMask=torch::ones((int64)landMarks[0].rows(),at::TensorOptions().dtype(torch::kByte));
    torch::Tensor leftContourIds=torch::from_blob(contour.leftContour.data(),{(int64)contour.leftContour.size()},at::TensorOptions().dtype(torch::kLong));
    torch::Tensor leftValue=torch::zeros(leftContourIds.size(0),at::TensorOptions().dtype(torch::kByte));
    torch::Tensor rightContourIds=torch::from_blob(contour.rightContour.data(),{(int64)contour.rightContour.size()},at::TensorOptions().dtype(torch::kLong));
    torch::Tensor rightValue=torch::zeros(rightContourIds.size(0),at::TensorOptions().dtype(torch::kByte));
    modelMask.index_put_(leftContourIds,leftValue);
    modelMask.index_put_(rightContourIds,rightValue);
//    std::cout<<modelMask.sizes()<<std::endl;
    torch::Tensor allModelMask=modelMask.expand({imageNum,modelMask.size(0)}).contiguous();
    torch::Tensor fixModelMask=allModelMask.clone();
//    std::cout<<allModelMask.sizes()<<std::endl;
    for(int i=0;i<imageNum;i++){
       ProjectionParameters param=PyMMS.SolveProjection(landMarks[i],PyMMS.FM.Face);
       params.emplace_back(param);
       MatF currentEX=PyMMS.SolveShape(param,landMarks[i],PyMMS.FM.Face,PyMMS.FM.EB,1.0f);
       blendShapeXs.emplace_back(currentEX);
       MatF FaceS = PyMMS.FM.EB * currentEX;
       MatF FaceFullS = PyMMS.FMFull.EB * currentEX;
       MatF S=Reshape(FaceS,3);
       MatF FullS=Reshape(FaceFullS,3);
       MatF currentModelPoint =PyMMS.FM.Face+S;
       MatF currentMesh =PyMMS.FMFull.Face+FullS;
       currentModelPoints.emplace_back(currentModelPoint);
       currentMeshes.emplace_back(currentMesh);
    }
    for(int iter=0;iter<iterNum;iter++){
        allModelMask=fixModelMask.clone();
        for(int j=0;j<imageNum;j++){
            torch::Tensor vertexIndices=allModelMask.select(0,j).nonzero();
            torch::Tensor inner2DIndex=vertexIndices.clone();
            torch::Tensor inner3DIndex=vertexIndices.clone();
            auto yawAngle =glm::degrees(glm::eulerAngles(GlmFunctions::RotationToQuat(params[j].R))[1]);
            torch::Tensor visual2DIndex;
            torch::Tensor visual3DIndex;
            std::tie(visual2DIndex,visual3DIndex)=getContourCorrespondences(params[j],contour,currentModelPoints[j],landMarks[j],yawAngle);
//            torch::Tensor current2DIndex=torch::cat({inner2DIndex,visual2DIndex},0);
//            torch::Tensor current3DIndex=torch::cat({inner3DIndex,visual3DIndex},0);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> MultiFitting::getContourCorrespondences(ProjectionParameters& param,ContourLandmarks &contour, MatF &modelPoint, MatF &landMark, float &yawAngle)
{
    torch::Tensor modelContourMask=torch::zeros((int64)landMark.rows(),at::TensorOptions().dtype(torch::kByte));
    selectContour(contour,yawAngle,modelContourMask);
    return getNearestContourCorrespondences(param,modelPoint,landMark,modelContourMask);
}

void MultiFitting::selectContour(ContourLandmarks &contour, float &yawAngle,torch::Tensor &modelContourMask, float frontalRangeThreshold)
{

    if (yawAngle >= -frontalRangeThreshold) // positive yaw = subject looking to the left
    {
        // ==> we use the right cnt-lms
        torch::Tensor rightContourIds=torch::from_blob(contour.rightContour.data(),{(int64)contour.rightContour.size()},at::TensorOptions().dtype(torch::kLong));
        torch::Tensor rightValue=torch::ones(rightContourIds.size(0),at::TensorOptions().dtype(torch::kByte));
        modelContourMask.index_put_(rightContourIds,rightValue);
    }
    if (yawAngle <= frontalRangeThreshold)
    {
        // ==> we use the left cnt-lms
        torch::Tensor leftContourIds=torch::from_blob(contour.leftContour.data(),{(int64)contour.leftContour.size()},at::TensorOptions().dtype(torch::kLong));
        torch::Tensor leftValue=torch::ones(leftContourIds.size(0),at::TensorOptions().dtype(torch::kByte));
        modelContourMask.index_put_(leftContourIds,leftValue);
    }
}

std::tuple<torch::Tensor, torch::Tensor> MultiFitting::getNearestContourCorrespondences(ProjectionParameters &param, MatF &modelPoint,MatF &landMark, at::Tensor &modelContourMask)
{
   at::Tensor indexes=modelContourMask.nonzero();
   MatF projected=Projection(param,modelPoint);
   torch::Tensor corIndex=torch::zeros(indexes.size(0),torch::dtype(torch::kLong));
   for(int i=0;i<indexes.size(0);i++){
       int index=indexes[i].item().toInt();
       Eigen::Vector2f imagePoint=landMark.row(index);
       std::vector<float> distances2d;
       for(int j=0;j<modelPoint.rows();j++){
           Eigen::Vector2f proModelPoint=projected.row(j);
           const double dist = (proModelPoint - imagePoint).norm();
           distances2d.emplace_back(dist);
       }
       const auto minEle = std::min_element(begin(distances2d), end(distances2d));
       const auto minEleIdx = std::distance(begin(distances2d), minEle);
       const Eigen::Vector4f closest(modelPoint.row(minEleIdx)[0],modelPoint.row(minEleIdx)[1],modelPoint.row(minEleIdx)[2],1);
       corIndex[i]=minEleIdx;
   }
   return std::make_tuple(indexes,corIndex);
}
