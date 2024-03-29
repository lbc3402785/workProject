#ifndef MULTIFITTING_H
#define MULTIFITTING_H
#include <iostream>
#include <vector>
#include <Eigen/Sparse>
#include <torch/script.h>
#include "NumpyUtil.h"
#include "MMSolver.h"
#include "contour.h"
#include "common/torchfunctions.h"
#include "common/dataconvertor.h"
class MultiFitting
{
public:
    MultiFitting();
    static std::vector<ProjectionParameters> fitShapeAndPose(std::vector<cv::Mat>& images,ContourLandmarks& contour,MMSolver& PyMMS,std::vector<MatF>& landMarks,MatF &shapeX,
                                std::vector<MatF> &blendShapeXs,int iterNum=4);
    static std::tuple<torch::Tensor, torch::Tensor> getContourCorrespondences(ProjectionParameters& param,ContourLandmarks& contour,torch::Tensor &modelMarkT,torch::Tensor& landMarkT,float& yawAngle);
private:
    static void selectContour(ContourLandmarks& contour,float& yawAngle,torch::Tensor &modelContourMask,float frontalRangeThreshold = 7.5f);
    static std::tuple<torch::Tensor, torch::Tensor> getNearestContourCorrespondences(ProjectionParameters& param,torch::Tensor &modelMarkT,torch::Tensor &landMarkT,torch::Tensor &modelContourMask);
    static std::pair<torch::Tensor,torch::Tensor>  findOccludingEdgeCorrespondences(MatF& mesh,FaceModel& fmFull,ProjectionParameters& param,torch::Tensor &landMarkT,torch::Tensor& occluding2DIndex,
                                                 float distanceThreshold =4.0f);
    static std::vector<int> boundary3DIndex(MatF& proMesh,FaceModel& fmFull,torch::Tensor&faceNormals,bool performSelfOcclusionCheck = true);
    static torch::Tensor computeFaceNormal(MatF& mesh,MatI& TRI);
};

#endif // MULTIFITTING_H
