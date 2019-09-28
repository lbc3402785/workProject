#ifndef MULTIFITTING_H
#define MULTIFITTING_H
#include <iostream>
#include <vector>
#include <torch/script.h>
#include "NumpyUtil.h"
#include "mmtensorsolver.h"
#include "contour.h"
#include "common/torchfunctions.h"
//#include "common/dataconvertor.h"
class MultiFitting
{
public:
    MultiFitting();
    static std::vector<ProjectionTensor> fitShapeAndPose(std::vector<cv::Mat>& images,ContourLandmarks& contour,MMTensorSolver& PyMMS,std::vector<torch::Tensor>& landMarks,torch::Tensor &shapeX,torch::Tensor& blendShapeX,
                                std::vector<torch::Tensor> &blendShapeXs,int iterNum=4);
    static std::tuple<std::vector<torch::Tensor>,std::vector<torch::Tensor>> fitShapeAndPoseLinear(std::vector<cv::Mat> &images,std::vector<ProjectionTensor>& params,ContourLandmarks& contour,torch::Tensor &allModelMask,MMTensorSolver& PyMMS,std::vector<torch::Tensor>& landMarks,
                                                                                                                 std::vector<torch::Tensor>& modelMarks,std::vector<float>& yawAngles,torch::Tensor &shapeX,std::vector<torch::Tensor> &blendShapeXs,int iterNum);
    static void fitShapeAndPoseNonlinear(std::vector<ProjectionTensor>& params,MMTensorSolver& PyMMS,std::vector<torch::Tensor>& landMarks,std::vector<torch::Tensor>& visdual2Ds,std::vector<torch::Tensor>& visdual3Ds,std::vector<float>& yawAngles,
                                         torch::Tensor &shapeX,torch::Tensor &blendShapeX,int maxIterNum=50);
    static std::tuple<torch::Tensor, torch::Tensor> getContourCorrespondences(ProjectionTensor& param,ContourLandmarks& contour,torch::Tensor &modelMarkT,torch::Tensor& landMarkT,float& yawAngle);

    static cv::Mat render(std::vector<cv::Mat>& images,std::vector<ProjectionTensor>& params,torch::Tensor &shapeX,torch::Tensor &blendShapeX,std::vector<torch::Tensor> &blendShapeXs,ContourLandmarks &contour,MMTensorSolver& PyMMS,float offset=5.0f);
private:
    static cv::Mat merge(std::vector<cv::Mat> &images,std::vector<torch::Tensor> projecteds, std::vector<torch::Tensor> &weightTs,MMTensorSolver& PyMMS,int H,int W);
    static cv::Mat merge2(std::vector<cv::Mat> &images,std::vector<torch::Tensor> projecteds,std::vector<torch::Tensor>& weightTs,MMTensorSolver& PyMMS,int H,int W);
    static void selectContour(ContourLandmarks& contour,float& yawAngle,torch::Tensor &modelContourMask,float frontalRangeThreshold = 7.5f);
    static std::tuple<torch::Tensor, torch::Tensor> getNearestContourCorrespondences(ProjectionTensor& param,torch::Tensor &modelMarkT,torch::Tensor &landMarkT,torch::Tensor &modelContourMask);
    static void innerSelect(torch::Tensor modelMask,ContourLandmarks& contour);
};

#endif // MULTIFITTING_H
