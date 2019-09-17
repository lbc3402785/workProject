#include "multifitting.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/transform.hpp"
#include "glmfunctions.h"
#include "common/knnsearch.h"
#include "common/raytriangleintersect.h"
#include "common/eigenfunctions.h"
#include "facemorph.h"
template <class T>
inline auto concat(const std::vector<T>& vec_a, const std::vector<T>& vec_b)
{
    std::vector<T> concatenated_vec;
    concatenated_vec.reserve(vec_a.size() + vec_b.size());
    concatenated_vec.insert(std::end(concatenated_vec), std::begin(vec_a), std::end(vec_a));
    concatenated_vec.insert(std::end(concatenated_vec), std::begin(vec_b), std::end(vec_b));
    return concatenated_vec;
};
MultiFitting::MultiFitting()
{

}

void MultiFitting::innerSelect(torch::Tensor modelMask,ContourLandmarks& contour)
{
    torch::Tensor leftContourIds=torch::from_blob(contour.leftContour.data(),{(int64)contour.leftContour.size()},at::TensorOptions().dtype(torch::kLong));
    torch::Tensor leftValue=torch::zeros(leftContourIds.size(0),at::TensorOptions().dtype(torch::kByte));
    torch::Tensor rightContourIds=torch::from_blob(contour.rightContour.data(),{(int64)contour.rightContour.size()},at::TensorOptions().dtype(torch::kLong));
    torch::Tensor rightValue=torch::zeros(rightContourIds.size(0),at::TensorOptions().dtype(torch::kByte));
    modelMask.index_put_(leftContourIds,leftValue);
    modelMask.index_put_(rightContourIds,rightValue);
}

std::vector<ProjectionParameters> MultiFitting::fitShapeAndPose(std::vector<cv::Mat>& images,ContourLandmarks& contour,MMSolver& PyMMS,std::vector<MatF>& landMarks,MatF &shapeX,
                                   std::vector<MatF> &blendShapeXs,int iterNum)
{
    int imageNum = static_cast<int>(landMarks.size());
    //std::cout<<"imageNum:"<<imageNum<<std::endl;
    std::vector<ProjectionParameters> params;
    std::vector<MatF> currentMeshes;
    params.reserve(imageNum);
    blendShapeXs.reserve(imageNum);
    currentMeshes.reserve(imageNum);
    float eLambdas[8] = { 100.0, 100.0, 50.0, 50.0 , 50.0 , 50.0 , 50.0 , 50.0 };
    float Lambdas[8] = { 50.0, 100.0, 200.0, 300.0 , 400.0 , 500.0 , 600.0 , 700.0 };
    torch::Tensor modelMask=torch::ones((int64)landMarks[0].rows(),at::TensorOptions().dtype(torch::kByte));
    innerSelect(modelMask,contour);
    torch::Tensor allModelMask=modelMask.expand({imageNum,modelMask.size(0)}).contiguous();
    torch::Tensor fixModelMask=allModelMask.clone();
    std::vector<torch::Tensor> landMarkTs;
    std::vector<torch::Tensor> modelMarkTs;
    std::vector<float> angles(imageNum,0.0f);
    for(int i=0;i<imageNum;i++){
       torch::Tensor imagePoints=EigenToTorch::EigenMatrixToTorchTensor(landMarks[i]);
       landMarkTs.emplace_back(imagePoints);
       torch::Tensor modelPoints=EigenToTorch::EigenMatrixToTorchTensor(PyMMS.FM.Face);
       torch::Tensor selectIds=modelMask.nonzero();
       torch::Tensor relia2D=imagePoints.index_select(0,selectIds.squeeze(-1));
       torch::Tensor relia3D=modelPoints.index_select(0,selectIds.squeeze(-1));
       MatF select2D=EigenToTorch::TorchTensorToEigenMatrix(relia2D);
       MatF select3D=EigenToTorch::TorchTensorToEigenMatrix(relia3D);
       ProjectionParameters param=PyMMS.SolveProjection(select2D,select3D);
       params.emplace_back(param);
       MatF currentEX=PyMMS.SolveShape(param,landMarks[i],PyMMS.FM.Face,PyMMS.FM.EB,20.0f);
       blendShapeXs.emplace_back(currentEX);
       MatF FaceS = PyMMS.FM.EB * currentEX;
       MatF FaceFullS = PyMMS.FMFull.EB * currentEX;
       MatF S=Reshape(FaceS,3);
       MatF FullS=Reshape(FaceFullS,3);
       MatF currentModelPoint =PyMMS.FM.Face+S;
       modelPoints=EigenToTorch::EigenMatrixToTorchTensor(currentModelPoint);
       modelMarkTs.emplace_back(modelPoints);
       MatF currentMesh =PyMMS.FMFull.Face+FullS;
       currentMeshes.emplace_back(currentMesh);

    }
    std::vector<torch::Tensor> visdual2Ds,visdual3Ds;
    visdual2Ds.resize(imageNum);
    visdual3Ds.resize(imageNum);

    for(int iter=0;iter<iterNum;iter++){
        std::vector<MatF> imageMarks;
        std::vector<MatF> modelMarks;
        allModelMask=fixModelMask.clone();
        for(int j=0;j<imageNum;j++){
            torch::Tensor mask2D=allModelMask.select(0,j).clone();
            torch::Tensor innerIndices=allModelMask.select(0,j).nonzero();
            angles[j]=glm::eulerAngles(GlmFunctions::RotationToQuat(params[j].R))[1];
            auto yawAngle =glm::degrees(angles[j]);

            torch::Tensor visual2DIndex;
            torch::Tensor visual3DIndex;
            std::tie(visual2DIndex,visual3DIndex)=getContourCorrespondences(params[j],contour,modelMarkTs[j],landMarkTs[j],yawAngle);
            visual2DIndex=torch::cat({innerIndices.clone(),visual2DIndex},0);
            visual3DIndex=torch::cat({innerIndices.clone(),visual3DIndex},0);

            visdual2Ds[j]=visual2DIndex;
            visdual3Ds[j]=visual3DIndex;
            torch::Tensor imagePointsT=landMarkTs[j].index_select(0,visual2DIndex.squeeze(-1));
            torch::Tensor modelPointsT=modelMarkTs[j].index_select(0,visual3DIndex.squeeze(-1));

            imageMarks.emplace_back(EigenToTorch::TorchTensorToEigenMatrix(imagePointsT));
            modelMarks.emplace_back(EigenToTorch::TorchTensorToEigenMatrix(modelPointsT));
            ProjectionParameters param=PyMMS.SolveProjection(EigenToTorch::TorchTensorToEigenMatrix(imagePointsT),EigenToTorch::TorchTensorToEigenMatrix(modelPointsT));
            params[j]=param;

        }
        shapeX=PyMMS.SolveMultiShape(params,imageMarks,modelMarks,angles,PyMMS.FM.SB,Lambdas[iter]);
        for(int j=0;j<imageNum;j++)
        {
            MatF FaceS = PyMMS.FM.SB * shapeX;
            MatF FaceFullS = PyMMS.FMFull.SB * shapeX;
            MatF S=Reshape(FaceS,3);
            MatF FullS=Reshape(FaceFullS,3);
            MatF currentModelPoint =PyMMS.FM.Face+S;
            torch::Tensor modelPoints=EigenToTorch::EigenMatrixToTorchTensor(currentModelPoint);
            modelMarkTs[j]=(modelPoints);
            MatF currentMesh =PyMMS.FMFull.Face+FullS;
            torch::Tensor modelPointsT=modelMarkTs[j].index_select(0,visdual3Ds[j].squeeze(-1));

            blendShapeXs[j]=PyMMS.SolveShape(params[j],imageMarks[j],EigenToTorch::TorchTensorToEigenMatrix(modelPointsT),PyMMS.FM.EB,eLambdas[iter]);
            //if(iter%2==0){
//                string name=std::to_string(j);
//                PyMMS.params=params[j];
//                PyMMS.EX=blendShapeXs[j];
//                PyMMS.SX=shapeX;
//                cv::Mat m=MMSDraw(images[j],PyMMS,landMarks[j]);
//                cv::imwrite(name+".jpg",m);
//                cv::imshow(name,m);
//                cv::waitKey();
            //}

        }
    }
    return params;
}

std::tuple<torch::Tensor, torch::Tensor> MultiFitting::getContourCorrespondences(ProjectionParameters& param,ContourLandmarks &contour, torch::Tensor &modelMarkT, torch::Tensor &landMarkT, float &yawAngle)
{
    torch::Tensor modelContourMask=torch::zeros(landMarkT.size(0),at::TensorOptions().dtype(torch::kByte));
    selectContour(contour,yawAngle,modelContourMask);
//    std::cout<<"selectContour done!"<<std::endl<<std::flush;
    return getNearestContourCorrespondences(param,modelMarkT,landMarkT,modelContourMask);
}
void saveModel(MatF& Face,FaceModel&FM,std::string filename)
{
    std::stringstream ss;
    auto TRI = FM.TRI;
    {
        std::stringstream ss;

        int N = Face.rows();
        auto TRIUV = FM.TRIUV;
        for (size_t i = 0; i < N; i++)
        {
            ss << "v " << Face(i, 0) << " " << Face(i, 1) << " " << Face(i, 2) << endl;
        }


        N = TRI.rows();
        for (size_t i = 0; i < N; i++)
        {
            ss << "f " << TRI(i, 0) + 1 << "/" << TRIUV(i, 0) + 1 << " "
               << TRI(i, 1) + 1 << "/" << TRIUV(i, 1) + 1 << " "
               << TRI(i, 2) + 1 << "/" << TRIUV(i, 2) + 1 << " "
               << endl;
        }


        std::string input = ss.str();

        std::ofstream out(filename + ".obj", std::ofstream::out);
        out << input;
        out.close();
    }
}
inline float cosOfPosition(float x,float y)
{
    return x/std::sqrt(x*x+y*y);
}
torch::Tensor computeTexturesWeight(cv::Mat orig,MatF& frontModel,torch::Tensor& frontNormals,float yawAngle,std::array<float,4>& axisMetrics, MMSolver &MMS, int W, int H,bool flip=true,float frontalAngleThreshold = 7.5f)
{
    float cosRightZ,sinRightZ,cosLeftZ,sinLeftZ;
    cosRightZ=axisMetrics[0];
    sinRightZ=axisMetrics[1];
    cosLeftZ=axisMetrics[2];
    sinLeftZ=axisMetrics[3];

    auto TRI = MMS.FMFull.TRIUV;
    auto UV = MMS.FMFull.UV;
    torch::Tensor weightT=torch::zeros(TRI.rows());
    std::cout<<"yawAngle:"<<yawAngle<<std::endl;
    for (size_t t = 0; t < TRI.rows(); t++)
    {
        float weight=0.0f;
        vector<cv::Point2f> t1;
        vector<cv::Point2f> t2;
        Eigen::Vector3f frontAvg(0.0,0.0,0.0);
        for (size_t i = 0; i < 3; i++)
        {
            int j = TRI(t, i);
            frontAvg+=frontModel.row(j);
        }
        frontAvg/=3.0;
        float cosPosToZ=cosOfPosition(frontAvg[2],frontAvg[0]);
        float sinPosToZ=cosOfPosition(frontAvg[0],frontAvg[2]);
        float cosNormalToZ=cosOfPosition(frontNormals[t][2].item().toFloat(),frontNormals[t][0].item().toFloat());
        float sinNormalToZ=cosOfPosition(frontNormals[t][0].item().toFloat(),frontNormals[t][2].item().toFloat());
        if(std::abs(yawAngle)<frontalAngleThreshold){
            //looking to front for personal perspective
            if(frontAvg[2]<0){
                weightT[t]=0.0;
                continue;
            }
            weight=std::min(cosPosToZ,cosNormalToZ);
        }else if(yawAngle>frontalAngleThreshold){
            //looking to left for personal perspective,so pose the right face
            if(frontAvg[0]>0){
                weightT[t]=0.0;
                continue;
            }
            float cosPosToRightAxis=cosPosToZ*cosRightZ+sinPosToZ*sinRightZ;
            float cosNormalToRightAxis=cosNormalToZ*cosRightZ+sinNormalToZ*sinRightZ;
            weight=std::max(cosPosToRightAxis,cosNormalToRightAxis);
        }else{
            //looking to right for personal perspective,so pose the left face
            if(frontAvg[0]<0){
                weightT[t]=0.0;
                continue;
            }
            float cosPosToLeftAxis=cosPosToZ*cosLeftZ+sinPosToZ*sinLeftZ;
            float cosNormalToLeftAxis=cosNormalToZ*cosLeftZ+sinNormalToZ*sinLeftZ;
            weight=std::max(cosPosToLeftAxis,cosNormalToLeftAxis);
        }
        if(weight<0.0){
            weightT[t]=0.0;
            continue;
        }else{
            weightT[t]=weight;
        }

    }
    return std::move(weightT);
}
void MultiFitting::render(std::vector<cv::Mat>& images,std::vector<ProjectionParameters>& params,MatF &shapeX,std::vector<MatF> &blendShapeXs,ContourLandmarks &contour,MMSolver& PyMMS,float offset)
{
    MatF EX(PyMMS.FMFull.EB.cols(),1);
    EX*=0;
    MatF model=PyMMS.FMFull.Generate(shapeX,EX);
    MatF frontKeyModel=PyMMS.FM.Generate(shapeX,EX);
    size_t imageNum=images.size();
    Eigen::Vector3f rightBound=frontKeyModel.row(contour.rightContour[0]);
    Eigen::Vector3f leftBound=frontKeyModel.row(contour.leftContour[0]);

    float z=(rightBound[2]+leftBound[2])/2;
    MatF frontModel=model;
    frontModel.col(2).array()-=z;
    frontKeyModel.col(2).array()-=z;
    Eigen::Vector3f rightEye=frontKeyModel.row(36);
    if(offset>45)offset=45;
    if(offset<-5)offset=-5;
    double pi = 4 * atan(1.0);
    float cosOffset=std::cos(offset/180*pi);
    float sinOffset=std::sin(offset/180*pi);
    float cos90=0;
    float sin90=1;
    float cosRotation=cos90*cosOffset+sin90*sinOffset;
    float sinRotation=sin90*cosOffset-cos90*sinOffset;
    float cosRightEye=rightEye[2]/std::sqrt(rightEye[2]*rightEye[2]+rightEye[0]*rightEye[0]);
    float sinRightEye=rightEye[0]/std::sqrt(rightEye[2]*rightEye[2]+rightEye[0]*rightEye[0]);
    float cosRightZ=cosRightEye*cos90+sinRightEye*sin90;
    float sinRightZ=sinRightEye*cos90-cosRightEye*sin90;
    Eigen::Vector3f leftEye=frontKeyModel.row(45);
    float cosLeftEye=leftEye[2]/std::sqrt(leftEye[2]*leftEye[2]+leftEye[0]*leftEye[0]);
    float sinLeftEye=leftEye[0]/std::sqrt(leftEye[2]*leftEye[2]+leftEye[0]*leftEye[0]);
    float cosLeftZ=cosLeftEye*cosRotation-sinLeftEye*sinRotation;
    float sinLeftZ=sinLeftEye*cosRotation+cosLeftEye*sinRotation;
    std::array<float,4> axisMetrics={cosRightZ,sinRightZ,cosLeftZ,sinLeftZ};
    PyMMS.SX=shapeX;
    torch::Tensor frontNormals=computeFaceNormal(frontModel,PyMMS.FMFull.TRI);
    std::vector<at::Tensor> weightTs;
    std::vector<MatF> projecteds;
    for(size_t i=0;i<imageNum;i++){
        //rotate around (0,0,-z)
        MatF R=params[i].R;
        glm::tvec3<float> euler=glm::eulerAngles(GlmFunctions::RotationToQuat(params[i].R));
        float yawAngle=glm::degrees(euler.y);
        PyMMS.params=params[i];
        PyMMS.EX=blendShapeXs[i];
        auto result=computeTexturesWeight(images[i],frontModel,frontNormals,yawAngle,axisMetrics,PyMMS,1024,1024,true,7.5f);
        result*=params[i].s;
        weightTs.emplace_back(result);
//        cv::imwrite(std::to_string(i)+".isomap.png",result.first);
        auto Face2 = PyMMS.FMFull.Generate(shapeX, blendShapeXs[i]);
        MatF projected = Projection(params[i], Face2);
        projected.col(1).array()=images[i].rows-projected.col(1).array();
        projecteds.emplace_back(projected);
    }
    std::cout<<"weightTs.size():"<<weightTs.size()<<std::endl;
    merge(images,projecteds,weightTs,PyMMS,1024,1024);
}

void MultiFitting::merge(std::vector<cv::Mat> &images,std::vector<MatF> projecteds, std::vector<at::Tensor> &weightTs,MMSolver& PyMMS,int H,int W)
{
    auto TRI = PyMMS.FMFull.TRIUV;
    auto UV = PyMMS.FMFull.UV;
    cv::Mat result=cv::Mat::zeros(H,W,images[0].type());
    size_t imageNum=weightTs.size();
    at::Tensor sum=torch::zeros(TRI.rows());
    for(int k=0;k<imageNum;k++){
        sum.add_(weightTs[k]);
    }

    torch::Tensor mask=sum.lt(1e-6);
    sum.masked_fill_(mask,1.0);
    mask=sum.lt(1e-6);
    std::cout<<"nonzero:"<<mask.nonzero()<<std::endl;
    for(int k=0;k<imageNum;k++){
        weightTs[k].div_(sum);
    }


    for (size_t t = 0; t < TRI.rows(); t++)
    {
        std::vector<cv::Point2f> dstTri;
        std::vector<float> weights;
        for(int k=0;k<imageNum;k++){
            weights.push_back(weightTs[k][t].item().toFloat());
        }

        std::vector<std::vector<cv::Point2f>> srcTris;
        srcTris.resize(imageNum);
        for (size_t i = 0; i < 3; i++)
        {
            int j = TRI(t, i);

            auto u = (UV(j, 0)) * (W - 1);
            auto v = (1 - UV(j, 1)) * (H - 1);
            dstTri.push_back(cv::Point2f(u, v));
            for(int k=0;k<imageNum;k++){
                auto x = projecteds[k](j, 0);
                auto y = projecteds[k](j, 1);
                srcTris[k].push_back(cv::Point2f(x, y));
            }
        }

        FaceMorph::morphTriangle(images,result,srcTris,dstTri,weights);
    }
    cv::imwrite("result.isomap.png",result);
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

std::tuple<torch::Tensor, torch::Tensor> MultiFitting::getNearestContourCorrespondences(ProjectionParameters &param, torch::Tensor &modelMarkT,torch::Tensor &landMarkT, at::Tensor &modelContourMask)
{
   at::Tensor indexes=modelContourMask.nonzero();
   MatF projected=Projection(param,EigenToTorch::TorchTensorToEigenMatrix(modelMarkT));
   at::Tensor projectedT=EigenToTorch::EigenMatrixToTorchTensor(projected);
   torch::Tensor corIndex=torch::zeros({indexes.size(0),1},torch::dtype(torch::kLong));
   for(int i=0;i<indexes.size(0);i++){
       long long index=indexes[i].item().toLong();
       torch::Tensor dist=projectedT.sub(landMarkT[index]).norm(2,1);
       corIndex[i]=std::get<1>(dist.min(0,true));
   }
//   std::cout<<"getNearestContourCorrespondences done"<<std::endl<<std::flush;
   return std::make_tuple(std::move(indexes),std::move(corIndex));
}

std::pair<torch::Tensor,torch::Tensor> MultiFitting::findOccludingEdgeCorrespondences(MatF &mesh, FaceModel &fmFull, ProjectionParameters &param, torch::Tensor &landMarkT, at::Tensor &occluding2DIndex, float distanceThreshold)
{
//    std::cout<<"===============:"<<occluding2DIndex<<std::endl;
    if(occluding2DIndex.size(0)==0){
        return {};
    }

    MatF rotated=Rotation(param,mesh);
//    std::cout<<"rotated:"<<rotated.rows()<<std::endl;
    torch::Tensor rotatedFaceNormals=computeFaceNormal(rotated,fmFull.TRI);//Mx3x1
    MatF projected=Projection(param,mesh);
//    std::cout<<"rotatedFaceNormals:"<<rotatedFaceNormals.sizes()<<std::endl;
    std::vector<int> boundaries=boundary3DIndex(projected,fmFull,rotatedFaceNormals,true);
    torch::Tensor boundT=torch::from_blob(boundaries.data(),boundaries.size(),torch::TensorOptions().dtype(torch::kInt)).toType(torch::kLong);
    torch::Tensor projectedT=EigenToTorch::EigenMatrixToTorchTensor(projected);
    torch::Tensor occludinglandMarkPts=landMarkT.index_select(0,occluding2DIndex.squeeze(-1));
    KNNSearch tree(occludinglandMarkPts.data<float>(),occludinglandMarkPts.sizes()[0],2);
    std::vector<Eigen::Vector2f> imagePoints;
    std::vector<int> select3DIndexes;
    torch::Tensor dst=projectedT.index_select(0,boundT);
    tree.search(dst.data<float>(),dst.size(0),dst.size(1));
    torch::Tensor distanceT=torch::from_blob(tree.distances.data(),tree.distances.size());
    torch::Tensor maskT=distanceT.lt(distanceThreshold*param.s);
    torch::Tensor srcIndexT=torch::from_blob(tree.srcIndex.data(),tree.srcIndex.size(),torch::TensorOptions().dtype(torch::kInt)).toType(torch::kLong);

    torch::Tensor select2DIndexesT=srcIndexT.masked_select(maskT);
    torch::Tensor bound2DIndexesT=occluding2DIndex.index_select(0,select2DIndexesT).squeeze(-1);
    //std::cout<<"bound2DIndexesT:"<<bound2DIndexesT<<std::endl;
    torch::Tensor bound3DIndexesT=boundT.masked_select(maskT);
     //std::cout<<"bound3DIndexesT:"<<bound3DIndexesT<<std::endl;
    return std::make_pair(bound2DIndexesT,bound3DIndexesT);
}
/**
 * @brief MultiFitting::boundary3DIndex
 * @param mesh
 * @param fmFull
 * @param faceNormals   Mx3x1
 * @return
 */
std::vector<int> MultiFitting::boundary3DIndex(MatF &proMesh, FaceModel &fmFull,torch::Tensor&faceNormals,bool performSelfOcclusionCheck)
{
    torch::Tensor Ev=torch::from_blob(fmFull.Ev.data(),{fmFull.Ev.rows(),fmFull.Ev.cols()},torch::TensorOptions().dtype(torch::kInt));//NX2
//    std::cout<<"Ev.sizes():"<<Ev.sizes()<<std::endl;
    torch::Tensor Ef=torch::from_blob(fmFull.Ef.data(),{fmFull.Ef.rows(),fmFull.Ef.cols()},torch::TensorOptions().dtype(torch::kInt)).toType(torch::kLong);//NX2
    torch::Tensor if0=Ef.select(1,0);
    torch::Tensor if1=Ef.select(1,1);
    torch::Tensor check=if0.ge(0).nonzero();

    if0=if0.index_select(0,check.squeeze(-1));
    if1=if1.index_select(0,check.squeeze(-1));
    Ev=Ev.index_select(0,check.squeeze(-1));
    if(if1.lt(0).size(0)>0){
        check=if1.ge(0).nonzero();
        if0=if0.index_select(0,check.squeeze(-1));
        if1=if1.index_select(0,check.squeeze(-1));
        Ev=Ev.index_select(0,check.squeeze(-1));
    }

    torch::Tensor fn0=faceNormals.squeeze(-1).index_select(0,if0);//Nx3 N>=M
    torch::Tensor fn1=faceNormals.squeeze(-1).index_select(0,if1);//Nx3 N>=M
    torch::Tensor r=torch::mul(fn0,fn1).sum(1);//Nx1 float
    torch::Tensor boundE=r.lt(0).nonzero();//Kx1 klong
    if(boundE.size(0)==0)return std::vector<int>();
    torch::Tensor boundV=Ev.index_select(0,boundE.squeeze(-1)).view(-1);
    std::vector<int> bound3DIndexes(boundV.data<int>(),boundV.data<int>()+boundV.size(0));

    // Remove duplicate vertex id's (std::unique works only on sorted sequences):
    std::sort(begin(bound3DIndexes), end(bound3DIndexes));
    bound3DIndexes.erase(std::unique(begin(bound3DIndexes), end(bound3DIndexes)),
                             end(bound3DIndexes));
    if (performSelfOcclusionCheck)
    {
        // Perform ray-casting to find out which vertices are not visible (i.e. self-occluded):
        std::vector<bool> visibility;
        const Eigen::Vector3f rayDirection(0.0f, 0.0f,-1.0f); // we shoot rays from the vertex towards the camera
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
 * @return      Mx3x1
 */
at::Tensor MultiFitting::computeFaceNormal(MatF &mesh, MatI &TRI)
{
    at::Tensor model=EigenToTorch::EigenMatrixToTorchTensor(mesh).unsqueeze(-1);
    //std::cout<<"TRI:"<<TRI.rows()<<","<<TRI.cols()<<std::endl;
    at::Tensor faces=torch::from_blob(TRI.data(),{(int64_t)TRI.rows(),3},at::TensorOptions().dtype(torch::kInt)).toType(torch::kLong);
   // std::cout<<"faces:"<<faces<<std::endl;
    return TorchFunctions::computeFaceNormals(model,faces);
}

