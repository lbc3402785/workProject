#include "multifitting.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/transform.hpp"
#include "glmfunctions.h"
#include "common/knnsearch.h"
#include "common/raytriangleintersect.h"
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
       torch::Tensor imagePoints=EigenToTorch::EigenMatrixToTorchTensor(landMarks[i]);
       torch::Tensor modelPoints=EigenToTorch::EigenMatrixToTorchTensor(PyMMS.FM.Face);
       torch::Tensor selectIds=modelMask.nonzero();
       torch::Tensor relia2D=imagePoints.index_select(0,selectIds.squeeze(-1));
       std::cout<<"relia2D:"<<relia2D.sizes()<<std::endl;
       torch::Tensor relia3D=modelPoints.index_select(0,selectIds.squeeze(-1));
       MatF select2D=EigenToTorch::TorchTensorToEigenMatrix(relia2D);
       MatF select3D=EigenToTorch::TorchTensorToEigenMatrix(relia3D);
       //ProjectionParameters param=PyMMS.SolveProjection(landMarks[i],PyMMS.FM.Face);
       ProjectionParameters param=PyMMS.SolveProjection(select2D,select3D);
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
//       PyMMS.params=param;
//       cv::Mat m=MMSDraw(images[i],PyMMS,landMarks[i]);
//       cv::imshow("aa",m);
//       cv::waitKey();
    }
    for(int iter=0;iter<iterNum;iter++){
        allModelMask=fixModelMask.clone();
        for(int j=0;j<imageNum;j++){
            torch::Tensor mask2D=allModelMask.select(0,j).clone();
            torch::Tensor innerIndices=allModelMask.select(0,j).nonzero();
//            std::cout<<innerIndices<<std::endl;
            auto yawAngle =glm::degrees(glm::eulerAngles(GlmFunctions::RotationToQuat(params[j].R))[1]);
            std::cout<<"yawAngle:"<<yawAngle<<std::endl;
            torch::Tensor visual2DIndex;
            torch::Tensor visual3DIndex;
            std::tie(visual2DIndex,visual3DIndex)=getContourCorrespondences(params[j],contour,currentModelPoints[j],landMarks[j],yawAngle);
            torch::Tensor current2DIndex=torch::cat({innerIndices,visual2DIndex},0);
            torch::Tensor current3DIndex=torch::cat({innerIndices,visual3DIndex},0);
//            std::cout<<current2DIndex<<std::endl;
//            std::cout<<"------------"<<std::endl;
//            std::cout<<current3DIndex<<std::endl;
            torch::Tensor value=torch::ones(visual2DIndex.size(0),at::TensorOptions().dtype(torch::kByte));
            mask2D.index_put_(visual2DIndex.squeeze(-1),value);
            torch::Tensor occluding2DIndex=(1-mask2D).nonzero();
            findOccludingEdgeCorrespondences(currentMeshes[j],PyMMS.FMFull,params[j],landMarks[j],occluding2DIndex);
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
   torch::Tensor corIndex=torch::zeros({indexes.size(0),1},torch::dtype(torch::kLong));
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

void MultiFitting::findOccludingEdgeCorrespondences(MatF &mesh, FaceModel &fmFull, ProjectionParameters &param, MatF &landMark, at::Tensor &occluding2DIndex, float distanceThreshold)
{
//    std::cout<<"===============:"<<occluding2DIndex<<std::endl;
    if(occluding2DIndex.size(0)==0){
        return;
    }
    MatF rotated=Rotation(param,mesh);
//    std::cout<<"rotated:"<<rotated.rows()<<std::endl;
    torch::Tensor rotatedFaceNormals=computeFaceNormal(rotated,fmFull.TRI);
    MatF projected=Projection(param,mesh);
//    std::cout<<"rotatedFaceNormals:"<<rotatedFaceNormals.sizes()<<std::endl;
    std::vector<int> boundaries=boundary3DIndex(projected,fmFull,rotatedFaceNormals,true);
    std::cout<<"boundaries:"<<boundaries.size()<<std::endl;
    torch::Tensor landMarkPtsIndexes=EigenToTorch::EigenMatrixToTorchTensor(landMark);
    torch::Tensor landMarkPts=landMarkPtsIndexes.index_select(0,occluding2DIndex.squeeze(-1));
    KNNSearch tree(landMarkPts.data<float>(),landMarkPts.sizes()[0],2);
    std::vector<Eigen::Vector2f> boundriesPro;

    for(auto index:boundaries){
       boundriesPro.emplace_back(projected.row(index));
    }

}
/**
 * @brief MultiFitting::boundary3DIndex
 * @param mesh
 * @param fmFull
 * @param faceNormals   fnumx3x1
 * @return
 */
std::vector<int> MultiFitting::boundary3DIndex(MatF &proMesh, FaceModel &fmFull,torch::Tensor&faceNormals,bool performSelfOcclusionCheck)
{
    auto Ev=fmFull.Ev;
    auto Ef=fmFull.Ef;
    std::vector<int> edgeIdxs;
    std::vector<int> bound3DIndexes;
    for (size_t i = 0; i < Ef.rows(); i++){
        torch::Tensor fn1=faceNormals.select(0,Ef(i,0));
        torch::Tensor fn2=faceNormals.select(0,Ef(i,1));
        torch::Tensor r=torch::dot(fn1.squeeze(-1),fn2.squeeze(-1));
        if(r.item().toFloat()<0){
            edgeIdxs.push_back(i);
            bound3DIndexes.push_back(Ev(i,0));
            bound3DIndexes.push_back(Ev(i,1));
        }
    }
    // Remove duplicate vertex id's (std::unique works only on sorted sequences):
    std::sort(begin(bound3DIndexes), end(bound3DIndexes));
    bound3DIndexes.erase(std::unique(begin(bound3DIndexes), end(bound3DIndexes)),
                             end(bound3DIndexes));
    if (performSelfOcclusionCheck)
    {
        // Perform ray-casting to find out which vertices are not visible (i.e. self-occluded):
        std::vector<bool> visibility;
        const Eigen::Vector3f rayDirection(0.0f, 0.0f,1.0f); // we shoot rays from the vertex towards the camera
        auto TRI=fmFull.TRI;
        for(auto index:bound3DIndexes){
            const Eigen::Vector3f rayOrigin(proMesh.row(index));
            bool visible = true;
            for(int k=0;k<TRI.rows();k++)
            {
                const auto& v0=proMesh.row(TRI(k,0));
                const auto& v1=proMesh.row(TRI(k,1));
                const auto& v2=proMesh.row(TRI(k,2));
                const auto intersect=Render::rayTriangleIntersect(rayOrigin,rayDirection,v0,v1,v2,true);
                if (intersect.first == true)
                {
                    // We've hit a triangle. Ray hit its own triangle. If it's behind the ray origin, ignore
                    // the intersection:
                    // Note: Check if in front or behind?
                    if (intersect.second.get() <= 1e-4)
                    {
                        continue; // the intersection is behind the vertex, we don't care about it
                    }
                    // Otherwise, we've hit a genuine triangle, and the vertex is not visible:
                    visible = false;
                    break;
                }
            }
            visibility.push_back(visible);
        }
        // Remove vertices from occluding boundary list that are not visible:
        std::vector<int> finalVertexIds;
        for (int i = 0; i < bound3DIndexes.size(); ++i)
        {
            if (visibility[i] == true)
            {
                finalVertexIds.push_back(bound3DIndexes[i]);
            }
        }
        return finalVertexIds;
    }
    return bound3DIndexes;
}
/**
 * @brief MultiFitting::computeFaceNormal
 * @param mesh  Nx3
 * @param faces Mx3
 * @return
 */
at::Tensor MultiFitting::computeFaceNormal(MatF &mesh, MatI &TRI)
{
    at::Tensor model=EigenToTorch::EigenMatrixToTorchTensor(mesh).unsqueeze(-1);
    //std::cout<<"TRI:"<<TRI.rows()<<","<<TRI.cols()<<std::endl;
    at::Tensor faces=torch::from_blob(TRI.data(),{(int64_t)TRI.rows(),3},at::TensorOptions().dtype(torch::kInt)).toType(torch::kLong);
   // std::cout<<"faces:"<<faces<<std::endl;
    return TorchFunctions::computeFaceNormals(model,faces);
}
