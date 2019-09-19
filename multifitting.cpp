#include "multifitting.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/transform.hpp"
#include "glmfunctions.h"
#include "common/knnsearch.h"
//#include "common/eigenfunctions.h"
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

std::vector<ProjectionTensor> MultiFitting::fitShapeAndPose(std::vector<cv::Mat>& images,ContourLandmarks& contour,MMTensorSolver& PyMMS,std::vector<torch::Tensor>& landMarks,torch::Tensor &shapeX,
                                   std::vector<torch::Tensor> &blendShapeXs,int iterNum)
{
    int imageNum = static_cast<int>(landMarks.size());
    //std::cout<<"imageNum:"<<imageNum<<std::endl;
    std::vector<ProjectionTensor> params;
    std::vector<torch::Tensor> currentMeshes;
    params.reserve(imageNum);
    blendShapeXs.reserve(imageNum);
    currentMeshes.reserve(imageNum);
    float eLambdas[8] = { 100.0, 100.0, 50.0, 50.0 , 50.0 , 50.0 , 50.0 , 50.0 };
    float Lambdas[8] = { 50.0, 100.0, 200.0, 300.0 , 400.0 , 500.0 , 600.0 , 700.0 };
    torch::Tensor modelMask=torch::ones((int64)landMarks[0].size(0),at::TensorOptions().dtype(torch::kByte));
    innerSelect(modelMask,contour);
    torch::Tensor allModelMask=modelMask.expand({imageNum,modelMask.size(0)}).contiguous();
    torch::Tensor fixModelMask=allModelMask.clone();

    std::vector<float> angles(imageNum,0.0f);
    std::vector<torch::Tensor> modelMarkTs;
    for(int i=0;i<imageNum;i++){
       torch::Tensor selectIds=modelMask.nonzero();
       torch::Tensor relia2D=landMarks[i].index_select(0,selectIds.squeeze(-1));
       torch::Tensor relia3D=PyMMS.FM.Face.index_select(0,selectIds.squeeze(-1));

       ProjectionTensor param=PyMMS.SolveProjection(relia2D,relia3D);
       params.emplace_back(param);

       torch::Tensor currentEX=PyMMS.SolveShape(param,landMarks[i],PyMMS.FM.Face,PyMMS.FM.EB,20.0f);
       blendShapeXs.emplace_back(currentEX);

       torch::Tensor FaceS =torch::matmul( PyMMS.FM.EB , currentEX);
       torch::Tensor FaceFullS = torch::matmul(PyMMS.FMFull.EB , currentEX);
       torch::Tensor S=FaceS.view({-1,3});
       torch::Tensor FullS=FaceFullS.view({-1,3});
       torch::Tensor currentModelPoint =PyMMS.FM.Face+S;
       modelMarkTs.emplace_back(currentModelPoint);
       torch::Tensor currentMesh =PyMMS.FMFull.Face+FullS;
       currentMeshes.emplace_back(currentMesh);

    }

    std::vector<torch::Tensor> visdual2Ds,visdual3Ds;
    visdual2Ds.resize(imageNum);
    visdual3Ds.resize(imageNum);

    for(int iter=0;iter<iterNum;iter++){
        std::vector<torch::Tensor> imageMarks;
        std::vector<torch::Tensor> modelMarks;
        allModelMask=fixModelMask.clone();
        for(int j=0;j<imageNum;j++){
            torch::Tensor innerIndices=allModelMask.select(0,j).nonzero();

            angles[j]=glm::eulerAngles(GlmFunctions::RotationToQuat(params[j].R))[1];
            auto yawAngle =glm::degrees(angles[j]);

            torch::Tensor visual2DIndex;
            torch::Tensor visual3DIndex;
            std::tie(visual2DIndex,visual3DIndex)=getContourCorrespondences(params[j],contour,modelMarkTs[j],landMarks[j],yawAngle);
            visual2DIndex=torch::cat({innerIndices.clone(),visual2DIndex},0);
            visual3DIndex=torch::cat({innerIndices.clone(),visual3DIndex},0);

            visdual2Ds[j]=visual2DIndex;
            visdual3Ds[j]=visual3DIndex;
            torch::Tensor imagePointsT=landMarks[j].index_select(0,visual2DIndex.squeeze(-1));
            torch::Tensor modelPointsT=modelMarkTs[j].index_select(0,visual3DIndex.squeeze(-1));

            imageMarks.emplace_back(imagePointsT);
            modelMarks.emplace_back(modelPointsT);
            ProjectionTensor param=PyMMS.SolveProjection(imagePointsT,modelPointsT);
            params[j]=param;

        }

        shapeX=PyMMS.SolveMultiShape(params,imageMarks,modelMarks,angles,PyMMS.FM.SB,Lambdas[iter]);

        for(int j=0;j<imageNum;j++)
        {
            torch::Tensor FaceS = torch::matmul(PyMMS.FM.SB , shapeX);
            torch::Tensor FaceFullS =torch::matmul( PyMMS.FMFull.SB , shapeX);
            torch::Tensor S=FaceS.view({-1,3});
            torch::Tensor FullS=FaceFullS.view({-1,3});
            torch::Tensor currentModelPoint =PyMMS.FM.Face+S;
            modelMarkTs[j]=(currentModelPoint);
            torch::Tensor currentMesh =PyMMS.FMFull.Face+FullS;
            torch::Tensor modelPointsT=modelMarkTs[j].index_select(0,visdual3Ds[j].squeeze(-1));

            blendShapeXs[j]=PyMMS.SolveShape(params[j],imageMarks[j],modelPointsT,PyMMS.FM.EB,eLambdas[iter]);

            //if(iter%2==0){
//                string name=std::to_string(j);
//                PyMMS.params=params[j];
//                PyMMS.EX=blendShapeXs[j];
//                PyMMS.SX=shapeX;
//                cv::Mat m=MMSDraw(images[j],PyMMS,landMarks[j]);
//                cv::imwrite(name+".jpg",m);
//                cv::imshow(name,m);
//                cv::waitKey();
           // }

        }
    }
    return params;
}

std::tuple<torch::Tensor, torch::Tensor> MultiFitting::getContourCorrespondences(ProjectionTensor& param,ContourLandmarks &contour, torch::Tensor &modelMarkT, torch::Tensor &landMarkT, float &yawAngle)
{
    torch::Tensor modelContourMask=torch::zeros(landMarkT.size(0),at::TensorOptions().dtype(torch::kByte));
    selectContour(contour,yawAngle,modelContourMask);
//    std::cout<<"selectContour done!"<<std::endl<<std::flush;
    return getNearestContourCorrespondences(param,modelMarkT,landMarkT,modelContourMask);
}
void saveModel(torch::Tensor& Face,FaceModelTensor&FM,std::string filename)
{
    std::stringstream ss;
    auto TRI = FM.TRI;
    {
        std::stringstream ss;

        int N = Face.size(0);
        auto TRIUV = FM.TRIUV;
        for (size_t i = 0; i < N; i++)
        {
            ss << "v " << Face[i][0].item().toFloat() << " " << Face[i][1].item().toFloat() << " " << Face[i][2].item().toFloat() << endl;
        }


        N = TRI.size(0);
        for (size_t i = 0; i < N; i++)
        {
            ss << "f " << TRI[i][0].item().toInt() + 1 << "/" << TRIUV[i][0].item().toInt() + 1 << " "
               << TRI[i][1].item().toInt() + 1 << "/" << TRIUV[i][1].item().toInt() + 1 << " "
               << TRI[i][2].item().toInt() + 1 << "/" << TRIUV[i][2].item().toInt() + 1 << " "
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

std::vector<torch::Tensor> computeTextureWeights(size_t imageNum,torch::Tensor& frontModel,torch::Tensor& frontNormals,std::vector<float> yawAngles,std::array<float,4>& axisMetrics, MMTensorSolver &MMS, float frontalAngleThreshold = 17.5f)
{
    std::vector<torch::Tensor> result;
    result.resize(imageNum);
    auto TRI = MMS.FMFull.TRI;
    float cosRightZ,sinRightZ,cosLeftZ,sinLeftZ;
    cosRightZ=axisMetrics[0];
    sinRightZ=axisMetrics[1];
    cosLeftZ=axisMetrics[2];
    sinLeftZ=axisMetrics[3];
    std::cout<<"cosRightZ:"<<cosRightZ<<" sinRightZ:"<<sinRightZ<<" cosLeftZ:"<<cosLeftZ<<" sinLeftZ:"<<sinLeftZ<<std::endl;
    auto num=TRI.size(0);
    torch::Tensor cosRightZs=torch::ones(num)*cosRightZ;
    torch::Tensor sinRightZs=torch::ones(num)*sinRightZ;
    torch::Tensor cosLeftZs=torch::ones(num)*cosLeftZ;
    torch::Tensor sinLeftZs=torch::ones(num)*sinLeftZ;

    torch::Tensor centers=frontModel.index_select(0,TRI.toType(torch::kLong).view(-1)).view({-1,3,3}).mean(1);//tx3
    torch::Tensor centerXs=centers.select(1,0);//t
    torch::Tensor centerZs=centers.select(1,2);//t
    torch::Tensor dis=(centerXs.pow(2)+centerZs.pow(2)).sqrt();//t

    torch::Tensor cosPosToZs=centerZs.div(dis);//t
    torch::Tensor sinPosToZs=centerXs.div(dis);//t

    torch::Tensor centerNormmalXs=frontNormals.select(1,0);//t
    torch::Tensor centerNormmalZs=frontNormals.select(1,2);//t
    torch::Tensor normalDis=(centerNormmalXs.pow(2)+centerNormmalZs.pow(2)).sqrt();

    torch::Tensor cosNormalToZs=centerNormmalZs.div(normalDis);
    torch::Tensor sinNormalToZs=centerNormmalXs.div(normalDis);
    torch::Tensor minFronts=torch::min(cosPosToZs,cosNormalToZs);

//    std::cout<<"cosPosToZs[0]:"<<cosPosToZs[0]<<std::endl;
//    std::cout<<"cosNormalToZs[0]:"<<cosNormalToZs[0]<<std::endl;
//    std::cout<<"cosPosToZs[1]:"<<cosPosToZs[1]<<std::endl;
//    std::cout<<"cosNormalToZs[1]:"<<cosNormalToZs[1]<<std::endl;
//    std::cout<<"cosPosToZs[2]:"<<cosPosToZs[2]<<std::endl;
//    std::cout<<"cosNormalToZs[2]:"<<cosNormalToZs[2]<<std::endl;
    minFronts=minFronts*minFronts.gt(0).toType(torch::kFloat);

    torch::Tensor cosPosToRightAxiss=cosPosToZs*cosRightZs+sinPosToZs*sinRightZs;
    torch::Tensor cosNormalToRightAxiss=cosNormalToZs*cosRightZs+sinNormalToZs*sinRightZs;
    torch::Tensor maxRights=torch::max(cosPosToRightAxiss,cosNormalToRightAxiss);
    maxRights=maxRights*maxRights.gt(0).toType(torch::kFloat);

    torch::Tensor cosPosToLeftAxiss=cosPosToZs*cosLeftZs+sinPosToZs*sinLeftZs;
    torch::Tensor cosNormalToLeftAxiss=cosNormalToZs*cosLeftZs+sinNormalToZs*sinLeftZs;
    torch::Tensor maxLefts=torch::max(cosPosToLeftAxiss,cosNormalToLeftAxiss);
    maxLefts=maxLefts*maxLefts.gt(0).toType(torch::kFloat);
    for(size_t j=0;j<imageNum;j++)
    {

        if(std::abs(yawAngles[j])<frontalAngleThreshold){
            //looking to front for personal perspective
            std::cout<<"looking to front..."<<std::endl;
            result[j]=minFronts*centerZs.gt(0.0).toType(torch::kFloat);
        }else if(yawAngles[j]>frontalAngleThreshold){
            //looking to left for personal perspective,so pose the right face
            result[j]=maxRights*centerXs.lt(0.0).toType(torch::kFloat);
        }else{
            //looking to right for personal perspective,so pose the left face
            result[j]=maxLefts*centerXs.gt(0.0).toType(torch::kFloat);
        }

    }
    return std::move(result);
}
void MultiFitting::render(std::vector<cv::Mat>& images,std::vector<ProjectionTensor>& params,torch::Tensor &shapeX,std::vector<torch::Tensor> &blendShapeXs,ContourLandmarks &contour,MMTensorSolver& PyMMS,float offset)
{
    torch::Tensor EX=torch::zeros({PyMMS.FMFull.EB.size(1),1});
    PyMMS.FMFull.Generate(shapeX,EX);
    torch::Tensor frontModel=PyMMS.FMFull.GeneratedFace.clone();
    PyMMS.FM.Generate(shapeX,EX);
    torch::Tensor frontKeyModel=PyMMS.FM.GeneratedFace.clone();

    size_t imageNum=images.size();
    torch::Tensor rightBound=frontKeyModel[contour.rightContour[0]];
    torch::Tensor leftBound=frontKeyModel[contour.leftContour[0]];
    float zOffset=(rightBound[2]+leftBound[2]).item().toFloat()/2;

    frontModel.select(1,2).sub_(zOffset);
    frontKeyModel.select(1,2).sub_(zOffset);
    torch::Tensor rightEye=frontKeyModel[36];
    if(offset>45)offset=45;
    if(offset<-5)offset=-5;
    double pi = 4 * atan(1.0);
    float cosOffset=std::cos(offset/180*pi);
    float sinOffset=std::sin(offset/180*pi);
    float cos90=0;
    float sin90=1;
    float cosRotation=cos90*cosOffset+sin90*sinOffset;
    float sinRotation=sin90*cosOffset-cos90*sinOffset;

    float x,z;
    x=rightEye[0].item().toFloat();z=rightEye[2].item().toFloat();
    float cosRightEye=cosOfPosition(z,x);
    float sinRightEye=cosOfPosition(x,z);
    float cosRightZ=cosRightEye*cos90+sinRightEye*sin90;
    float sinRightZ=sinRightEye*cos90-cosRightEye*sin90;
    torch::Tensor leftEye=frontKeyModel[45];
    x=leftEye[0].item().toFloat();z=leftEye[2].item().toFloat();
    float cosLeftEye=cosOfPosition(z,x);
    float sinLeftEye=cosOfPosition(x,z);
    float cosLeftZ=cosLeftEye*cosRotation-sinLeftEye*sinRotation;
    float sinLeftZ=sinLeftEye*cosRotation+cosLeftEye*sinRotation;
    std::array<float,4> axisMetrics={cosRightZ,sinRightZ,cosLeftZ,sinLeftZ};
    PyMMS.SX=shapeX;

    std::cout<<frontModel[0]<<std::endl;
    torch::Tensor frontNormals=TorchFunctions::computeFaceNormals(frontModel,PyMMS.FMFull.TRI);

    std::vector<torch::Tensor> projecteds;
    std::vector<float> yawAngles;
    projecteds.resize(imageNum);
    yawAngles.resize(imageNum);
    for(size_t j=0;j<imageNum;j++){
        glm::tvec3<float> euler=glm::eulerAngles(GlmFunctions::RotationToQuat(params[j].R));
        yawAngles[j]=glm::degrees(euler.y);
        PyMMS.FMFull.Generate(shapeX, blendShapeXs[j]);
        torch::Tensor projected = Projection(params[j], PyMMS.FMFull.GeneratedFace);
        projected.select(1,1)=images[j].rows-projected.select(1,1);
        projecteds[j]=projected;
    }
    std::vector<at::Tensor> weightTs=computeTextureWeights(imageNum,frontModel,frontNormals,yawAngles,axisMetrics,PyMMS);

    merge(images,projecteds,weightTs,PyMMS,1024,1024);
}
/**
 * @brief MultiFitting::merge
 * @param images
 * @param projecteds    [NX2] SIZE=imageNum
 * @param weightTs
 * @param PyMMS
 * @param H
 * @param W
 * @return
 */
//cv::Mat MultiFitting::merge(std::vector<cv::Mat> &images,std::vector<torch::Tensor> projecteds, std::vector<at::Tensor> &weightTs,MMTensorSolver& PyMMS,int H,int W)
//{
//    auto TRI = PyMMS.FMFull.TRI;
//    auto TRIUV = PyMMS.FMFull.TRIUV;
//    auto UV = PyMMS.FMFull.UV;
//    cv::Mat result=cv::Mat::zeros(H,W,images[0].type());
//    size_t imageNum=weightTs.size();
//    at::Tensor sum=torch::zeros(TRI.size(0));
//    std::vector<torch::Tensor> triProjecteds;
//    triProjecteds.resize(imageNum);
//    for(size_t k=0;k<imageNum;k++){
//        sum.add_(weightTs[k]);
//        triProjecteds[k]=projecteds[k].index_select(0,TRI.toType(torch::kLong).view(-1)).view({-1,3,2});//tx3x2
//    }

//    at::Tensor triUVs=UV.index_select(0,TRIUV.toType(torch::kLong).view(-1)).view({-1,3,2});//tx3x2
//    sum.mul_(1e6);
//    torch::Tensor mask=sum.lt(1e-6);
//    sum.masked_fill_(mask,1.0);
//    sum.mul_(1e-6);
//    for(size_t k=0;k<imageNum;k++){
//        weightTs[k].div_(sum);
//    }

//    torch::TensorList wv(weightTs.data(),weightTs.size());
//    torch::Tensor ws=torch::stack(wv,0);//imageNum*TRI.size(0)
//    for (size_t t = 0; t < TRI.size(0); t++)
//    {
//        torch::Tensor srcTris=torch::zeros({(int64_t)imageNum,3,2});
//        torch::Tensor dstTri=triUVs[t];
//        dstTri.select(1,0).mul_(W-1);
//        dstTri.select(1,1).sub_(1.0);
//        dstTri.select(1,1).mul_(-(H - 1));
//        for(size_t k=0;k<imageNum;k++){
//            srcTris[k]=triProjecteds[k][t];
//        }


//        FaceMorph::morphTriangle(images,result,srcTris,dstTri,ws.select(1,t));
//    }


//    cv::imwrite("result.isomap.png",result);
//    return std::move(result);
//}
void MultiFitting::merge(std::vector<cv::Mat> &images,std::vector<torch::Tensor> projecteds, std::vector<at::Tensor> &weightTs,MMTensorSolver& PyMMS,int H,int W)
{
    auto TRI = PyMMS.FMFull.TRIUV;
    auto UV = PyMMS.FMFull.UV;
    cv::Mat result=cv::Mat::zeros(H,W,images[0].type());
    size_t imageNum=weightTs.size();
    at::Tensor sum=torch::zeros(TRI.size(0));
    for(int k=0;k<imageNum;k++){
        sum.add_(weightTs[k]);
    }

    torch::Tensor mask=sum.lt(1e-6);
    sum.masked_fill_(mask,1.0);

    for(int k=0;k<imageNum;k++){
        weightTs[k].div_(sum);
    }


    for (size_t t = 0; t < TRI.size(0); t++)
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
            int j = TRI[t][i].item().toInt();

            auto u = (UV[j][0].item().toFloat()) * (W - 1);
            auto v = (1 - UV[j][1].item().toFloat()) * (H - 1);
            dstTri.push_back(cv::Point2f(u, v));
            for(int k=0;k<imageNum;k++){
                auto x = projecteds[k][j][0].item().toFloat();
                auto y = projecteds[k][j][1].item().toFloat();
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

std::tuple<torch::Tensor, torch::Tensor> MultiFitting::getNearestContourCorrespondences(ProjectionTensor &param, torch::Tensor &modelMarkT,torch::Tensor &landMarkT, at::Tensor &modelContourMask)
{
   at::Tensor indexes=modelContourMask.nonzero();
   torch::Tensor projected=Projection(param,modelMarkT);
   torch::Tensor corIndex=torch::zeros({indexes.size(0),1},torch::dtype(torch::kLong));
   for(int i=0;i<indexes.size(0);i++){
       long long index=indexes[i].item().toLong();
       torch::Tensor dist=projected.sub(landMarkT[index]).norm(2,1);
       corIndex[i]=std::get<1>(dist.min(0,true));
   }
//   std::cout<<"getNearestContourCorrespondences done"<<std::endl<<std::flush;
   return std::make_tuple(std::move(indexes),std::move(corIndex));
}

