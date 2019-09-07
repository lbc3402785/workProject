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
    static void fitShapeAndPose(std::vector<cv::Mat>& images,ContourLandmarks& contour,MMSolver& PyMMS,std::vector<MatF>& landMarks,MatF &shapeX,
                                std::vector<MatF> &blendShapeXs, std::vector<MatF> &fittedImagePoints,int iterNum=1);
    static std::tuple<torch::Tensor, torch::Tensor> getContourCorrespondences(ProjectionParameters& param,ContourLandmarks& contour,MatF& modelPoint,MatF& landMark,float& yawAngle);
private:
    static void selectContour(ContourLandmarks& contour,float& yawAngle,torch::Tensor &modelContourMask,float frontalRangeThreshold = 7.5f);
    static std::tuple<torch::Tensor, torch::Tensor> getNearestContourCorrespondences(ProjectionParameters& param,MatF &modelPoint,MatF &landMark,torch::Tensor &modelContourMask);
    static void findOccludingEdgeCorrespondences(MatF& mesh,FaceModel& fmFull,ProjectionParameters& param,MatF& landMark,torch::Tensor& occluding2DIndex,
                                                 float distanceThreshold = 64.0f);
    static std::vector<int> boundary3DIndex(MatF& proMesh,FaceModel& fmFull,torch::Tensor&faceNormals,bool performSelfOcclusionCheck = true);
    static torch::Tensor computeFaceNormal(MatF& mesh,MatI& TRI);
};

#endif // MULTIFITTING_H
