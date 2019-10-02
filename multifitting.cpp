#include "multifitting.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/transform.hpp"
#include "glmfunctions.h"
#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/rotation.h"
#include "common/torchfunctions.h"
#include "ceresnonlinear.hpp"
#include "facemorph.h"
#include "priorcostcallback.h"
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
std::vector<ProjectionTensor> MultiFitting::fitShapeAndPose(std::vector<cv::Mat> &images, ContourLandmarks &contour, MMTensorSolver &PyMMS, std::vector<at::Tensor> &landMarks, at::Tensor &shapeX,torch::Tensor& blendShapeX, std::vector<at::Tensor> &blendShapeXs, int iterNum)
{
    int imageNum = static_cast<int>(landMarks.size());
    std::vector<ProjectionTensor> params;
    params.reserve(imageNum);
    blendShapeXs.reserve(imageNum);
    torch::Tensor modelMask=torch::ones((int64)landMarks[0].size(0),at::TensorOptions().dtype(torch::kByte));
    innerSelect(modelMask,contour);
    torch::Tensor allModelMask=modelMask.expand({imageNum,modelMask.size(0)}).contiguous();
    shapeX=torch::zeros({PyMMS.FM.SB.size(1),1});
    std::vector<float> yawAngles(imageNum,0.0f);
    std::vector<torch::Tensor> modelMarks;

    for(int i=0;i<imageNum;i++){
       torch::Tensor selectIds=modelMask.nonzero();
       torch::Tensor relia2D=landMarks[i].index_select(0,selectIds.squeeze(-1));
       torch::Tensor relia3D=PyMMS.FM.Face.index_select(0,selectIds.squeeze(-1));
       ProjectionTensor param=PyMMS.SolveProjectionCenter(relia2D,relia3D,images[i].rows,images[i].cols);
//       ProjectionTensor param=PyMMS.SolveProjection(landMarks[i],PyMMS.FM.Face);
       float yawRadian=glm::eulerAngles(GlmFunctions::RotationToQuat(param.R))[1];
       yawAngles[i] =glm::degrees(yawRadian);
       params.emplace_back(param);
       std::cout<<"yawAngles[i]:"<<yawAngles[i]<<std::endl;
       torch::Tensor currentEX=PyMMS.SolveShape(param,landMarks[i],PyMMS.FM.Face,PyMMS.FM.EB,10.0f,true);
       blendShapeXs.emplace_back(currentEX);
       torch::Tensor FaceS =torch::matmul( PyMMS.FM.EB , currentEX);
       torch::Tensor S=FaceS.view({-1,3});
       torch::Tensor currentModelPoint =PyMMS.FM.Face+S;
       modelMarks.emplace_back(currentModelPoint);
    }
    std::vector<torch::Tensor> visdual2Ds,visdual3Ds;
    std::tie(visdual2Ds,visdual3Ds)=fitShapeAndPoseLinear(images,params,contour,allModelMask,PyMMS,landMarks,modelMarks,yawAngles,shapeX,blendShapeXs,iterNum);
    torch::Tensor FaceS = torch::matmul(PyMMS.FM.SB , shapeX);
    torch::Tensor S=FaceS.view({-1,3});
    torch::Tensor currentModelPoint =PyMMS.FM.Face+S;
    for(int j=0;j<imageNum;j++){
        modelMarks[j]=currentModelPoint;
    }
    PyMMS.USEWEIGHT=false;
    //blendShapeX=PyMMS.SolveMultiShape(params,landMarks,visdual2Ds,modelMarks,visdual3Ds,yawAngles,PyMMS.FM.EB,5.0);

    blendShapeX=blendShapeXs[0];
    for(size_t j=1;j<images.size();j++){
       blendShapeX+= blendShapeXs[j];
    }
    blendShapeX.div_((int64_t)images.size());
    fitShapeAndPoseNonlinear(params,PyMMS,landMarks,visdual2Ds,visdual3Ds,yawAngles,shapeX,blendShapeX,iterNum);
    return params;
}

std::tuple<std::vector<torch::Tensor>,std::vector<torch::Tensor>> MultiFitting::fitShapeAndPoseLinear(std::vector<cv::Mat> &images,std::vector<ProjectionTensor>& params,ContourLandmarks& contour,torch::Tensor &allModelMask,MMTensorSolver& PyMMS,std::vector<torch::Tensor>& landMarks,
                                                                                                      std::vector<torch::Tensor>& modelMarks,std::vector<float>& yawAngles,torch::Tensor &shapeX,std::vector<torch::Tensor> &blendShapeXs,int iterNum,bool show)
{
    int imageNum = static_cast<int>(landMarks.size());
    std::cout<<"image size:"<<imageNum<<std::endl;
    std::vector<torch::Tensor> visdual2Ds,visdual3Ds;
    visdual2Ds.resize(imageNum);
    visdual3Ds.resize(imageNum);
//    float eLambdas[8] = { 25.0, 25.0, 15.0,15.0 , 10.0 ,10.0 , 8.0 ,8.0 };
//    float Lambdas[8] = {10.0, 10.0,15.0,15.0 ,20 , 20 ,25.0 , 25.0 };
    float eLambdas[8] = { 5.0, 5.0, 4.0,4.0 , 3.0 ,3.0 , 2.0 ,2.0 };
    float Lambdas[8] = {5.0, 5.0,6.0,6.0 ,7 , 7 , 8.0 , 8.0 };
    for(int iter=0;iter<iterNum;iter++){
        //#pragma omp parallel for
        for(int j=0;j<imageNum;j++){
            torch::Tensor innerIndices=allModelMask.select(0,j).nonzero();
            torch::Tensor visual2DIndex;
            torch::Tensor visual3DIndex;
            std::tie(visual2DIndex,visual3DIndex)=getContourCorrespondences(params[j],contour,modelMarks[j],landMarks[j],yawAngles[j]);
            visual2DIndex=torch::cat({innerIndices.clone(),visual2DIndex},0);
            visual3DIndex=torch::cat({innerIndices.clone(),visual3DIndex},0);

            visdual2Ds[j]=visual2DIndex;
            visdual3Ds[j]=visual3DIndex;
            torch::Tensor imagePointsT=landMarks[j].index_select(0,visual2DIndex.squeeze(-1));
            torch::Tensor modelPointsT=modelMarks[j].index_select(0,visual3DIndex.squeeze(-1));
            ProjectionTensor param=PyMMS.SolveProjectionCenter(imagePointsT,modelPointsT,images[j].rows,images[j].cols);
            params[j]=param;
        }
        PyMMS.USEWEIGHT=true;
        shapeX=PyMMS.SolveMultiShape(params,landMarks,visdual2Ds,modelMarks,visdual3Ds,yawAngles,PyMMS.FM.SB,Lambdas[iter%8]);
        torch::Tensor FaceS = torch::matmul(PyMMS.FM.SB , shapeX);
        torch::Tensor S=FaceS.view({-1,3});
        torch::Tensor currentModelPoint =PyMMS.FM.Face+S;
        //#pragma omp parallel for
        for(int j=0;j<imageNum;j++)
        {
            modelMarks[j]=(currentModelPoint);
            blendShapeXs[j]=PyMMS.SolveSelectShape(params[j],landMarks[j],visdual2Ds[j],modelMarks[j],visdual3Ds[j],PyMMS.FM.EB,eLambdas[iter%8]);
            torch::Tensor FaceES = torch::matmul(PyMMS.FM.EB , blendShapeXs[j]);
            torch::Tensor ES=FaceES.view({-1,3});
            modelMarks[j]=currentModelPoint+ES;
            if(show&&iter%2==0){
                string name=std::to_string(j);
                PyMMS.params=params[j];
                PyMMS.EX=blendShapeXs[j];
                PyMMS.SX=shapeX;
                PyMMS.params.centerX=images[j].cols%2==0?images[j].cols/2:images[j].cols/2+0.5;
                PyMMS.params.centerY=images[j].rows%2==0?images[j].rows/2:images[j].rows/2+0.5;
                PyMMS.params.height=images[j].rows;
                cv::Mat m=MMSDraw(images[j],PyMMS,landMarks[j]);
                cv::imwrite(name+".jpg",m);
                cv::imshow(name,m);
                cv::waitKey();
            }
        }
    }

    return make_tuple(visdual2Ds,visdual3Ds);
}

void MultiFitting::fitShapeAndPoseNonlinear(std::vector<ProjectionTensor> &params, MMTensorSolver &PyMMS, std::vector<at::Tensor> &landMarks,std::vector<torch::Tensor>& visdual2Ds,std::vector<torch::Tensor>& visdual3Ds,std::vector<float>& yawAngles, at::Tensor &shapeX, at::Tensor &blendShapeX, int maxIterNum)
{
    int imageNum = static_cast<int>(landMarks.size());
    int numOfShapeCoef=PyMMS.FM.SB.size(1);
    int numOfBlendShapeCoef=PyMMS.FM.EB.size(1);
    int numOfCameraParam=6;
    torch::Tensor cameras=torch::zeros({imageNum,numOfCameraParam},torch::TensorOptions().dtype(torch::kDouble));
    for(int i=0;i<imageNum;i++){
        torch::Tensor axis=TorchFunctions::unRodrigues(params[i].R);
        torch::Tensor t=torch::zeros({3,1});
        t[0]=params[i].tx;t[1]=params[i].ty;t[2]=params[i].s;
        torch::Tensor camera=torch::cat({axis,t}).squeeze(-1).toType(torch::kDouble);
        cameras[i]=camera;
    }

    at::Tensor oldShapeX=shapeX.clone();
    at::Tensor centers=torch::zeros({imageNum,2},torch::TensorOptions().dtype(torch::kDouble));
    for(int i=0;i<imageNum;i++){
        centers[i][0]=params[i].centerX;
        centers[i][1]=params[i].centerY;
    }
    at::Tensor tmpShapeX=shapeX.clone().toType(torch::kDouble);
    at::Tensor tmpBlendShapeX=blendShapeX.clone().toType(torch::kDouble);
    ceres::Problem fittingCostfunction;
    std::vector<double*> parameters(4,0);
    parameters[0]=cameras.data<double>();
    parameters[1]=centers.data<double>();
    parameters[2]=tmpShapeX.data<double>();
    parameters[3]=tmpBlendShapeX.data<double>();

    for(int i=0;i<imageNum;i++){
        for(int j=0;j<visdual2Ds[i].size(0);j++){
            long long observedId=visdual2Ds[i][j].item().toLong();
            long long vertexId=visdual3Ds[i][j].item().toLong();
            fitting::MultiLandmarkCost* cost=new fitting::MultiLandmarkCost(PyMMS.FM,PyMMS.FMFull,landMarks[i][observedId],i,vertexId,4);
//            cost->centerX=params[i].centerX;
//            cost->centerY=params[i].centerY;
            cost->height=params[i].height;
             ceres::DynamicAutoDiffCostFunction<fitting::MultiLandmarkCost,4>* costFunction=new ceres::DynamicAutoDiffCostFunction<fitting::MultiLandmarkCost,4>(cost);
             costFunction->AddParameterBlock(6*imageNum);
             costFunction->AddParameterBlock(2*imageNum);
             costFunction->AddParameterBlock(numOfShapeCoef);
             costFunction->AddParameterBlock(numOfBlendShapeCoef);
             costFunction->SetNumResiduals(2);
             fittingCostfunction.AddResidualBlock(costFunction, new ceres::CauchyLoss(0.5),parameters);
        }
    }
    // Shape prior:
    fitting::PriorCost *shapePrior=new fitting::PriorCost(numOfShapeCoef, 6.0);
    ceres::CostFunction* shapePriorCost =
            new ceres::AutoDiffCostFunction<fitting::PriorCost, 199 /* num residuals */,
            199 /* shape-coeffs */>(
                shapePrior);
    fittingCostfunction.AddResidualBlock(shapePriorCost, NULL, parameters[2]);
    // Prior and constraints on blendshapes:
    fitting::PriorCost *blendShapePrior=new fitting::PriorCost(numOfBlendShapeCoef, 4.0);
        ceres::CostFunction* blendshapesPriorCost =
            new ceres::AutoDiffCostFunction<fitting::PriorCost, 100 /* num residuals */,
                100 /* bs-coeffs */>(
                blendShapePrior);
    fittingCostfunction.AddResidualBlock(blendshapesPriorCost, NULL, parameters[3]);

    ceres::Solver::Options solverOptions;
    solverOptions.linear_solver_type = ceres::SPARSE_SCHUR;
    solverOptions.num_threads = 1;
    solverOptions.max_num_iterations=200;
    solverOptions.callbacks.push_back(new PriorCostCallBack(shapePrior));
    solverOptions.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary solverSummary;
    Solve(solverOptions, &fittingCostfunction, &solverSummary);
    std::cout << solverSummary.BriefReport() << "\n";
    //std::cout<<"after tmpShapeX:"<<std::endl;
    //std::cout<<tmpShapeX<<std::endl;
    centers=centers.toType(torch::kFloat);
    for(int i=0;i<imageNum;i++){
        params[i].centerX=centers[i][0].item().toFloat();
        params[i].centerY=centers[i][1].item().toFloat();
    }
    shapeX=std::move(tmpShapeX.toType(torch::kFloat));
    blendShapeX=std::move(tmpBlendShapeX.toType(torch::kFloat));
    std::cout<<(shapeX-oldShapeX).norm()<<std::endl;
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
            ss << "v " << Face[i][0].item().toFloat() << " " << Face[i][1].item().toFloat() << " " << Face[i][2].item().toFloat() << std::endl;
        }


        N = TRI.size(0);
        for (size_t i = 0; i < N; i++)
        {
            ss << "f " << TRI[i][0].item().toInt() + 1 << "/" << TRIUV[i][0].item().toInt() + 1 << " "
               << TRI[i][1].item().toInt() + 1 << "/" << TRIUV[i][1].item().toInt() + 1 << " "
               << TRI[i][2].item().toInt() + 1 << "/" << TRIUV[i][2].item().toInt() + 1 << " "
               << std::endl;
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
//    std::cout<<"cosRightZ:"<<cosRightZ<<" sinRightZ:"<<sinRightZ<<" cosLeftZ:"<<cosLeftZ<<" sinLeftZ:"<<sinLeftZ<<std::endl;
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
    minFronts=minFronts*minFronts.gt(0).toType(torch::kFloat);

    torch::Tensor cosPosToRightAxiss=cosPosToZs*cosRightZs+sinPosToZs*sinRightZs;
    torch::Tensor cosNormalToRightAxiss=cosNormalToZs*cosRightZs+sinNormalToZs*sinRightZs;
    torch::Tensor maxRights=torch::max(cosPosToRightAxiss,cosNormalToRightAxiss);
    maxRights=maxRights*maxRights.gt(0).toType(torch::kFloat);

    torch::Tensor cosPosToLeftAxiss=cosPosToZs*cosLeftZs+sinPosToZs*sinLeftZs;
    torch::Tensor cosNormalToLeftAxiss=cosNormalToZs*cosLeftZs+sinNormalToZs*sinLeftZs;
    torch::Tensor maxLefts=torch::max(cosPosToLeftAxiss,cosNormalToLeftAxiss);
    maxLefts=maxLefts*maxLefts.gt(0).toType(torch::kFloat);
    //#pragma omp parallel for
    for(int j=0;j<imageNum;j++)
    {

        if(std::abs(yawAngles[j])<frontalAngleThreshold){
            //looking to front for personal perspective
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

cv::Mat MultiFitting::render(std::vector<cv::Mat>& images,std::vector<ProjectionTensor>& params,torch::Tensor &shapeX,torch::Tensor &blendShapeX,std::vector<torch::Tensor> &blendShapeXs,ContourLandmarks &contour,MMTensorSolver& PyMMS,float offset)
{
    torch::Tensor EX=torch::zeros({PyMMS.FMFull.EB.size(1),1});
    torch::Tensor frontModel=PyMMS.FMFull.Generate(shapeX,EX);
    torch::Tensor frontKeyModel= PyMMS.FM.Generate(shapeX,EX);

    size_t imageNum=images.size();
    torch::Tensor rightBound=frontKeyModel[contour.rightContour[0]];
    torch::Tensor leftBound=frontKeyModel[contour.leftContour[0]];
    float zOffset=(rightBound[2]+leftBound[2]).item().toFloat()/2;
    float r=2.0;
    frontModel.select(1,2).sub_(zOffset*r);
    frontKeyModel.select(1,2).sub_(zOffset*r);
    torch::Tensor rightEye=frontKeyModel[36];
    if(offset>45)offset=45;
    if(offset<-45)offset=-45;
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
    float cosRightZ=cosRightEye*cosRotation+sinRightEye*sinRotation;
    float sinRightZ=sinRightEye*cosRotation-cosRightEye*sinRotation;
    torch::Tensor leftEye=frontKeyModel[45];
    x=leftEye[0].item().toFloat();z=leftEye[2].item().toFloat();
    float cosLeftEye=cosOfPosition(z,x);
    float sinLeftEye=cosOfPosition(x,z);
    float cosLeftZ=cosLeftEye*cosRotation-sinLeftEye*sinRotation;
    float sinLeftZ=sinLeftEye*cosRotation+cosLeftEye*sinRotation;
    std::array<float,4> axisMetrics={cosRightZ,sinRightZ,cosLeftZ,sinLeftZ};
    PyMMS.SX=shapeX;
    torch::Tensor frontNormals=TorchFunctions::computeFaceNormals(frontModel,PyMMS.FMFull.TRI);
    std::vector<torch::Tensor> projecteds;
    std::vector<float> yawAngles;
    projecteds.resize(imageNum);
    yawAngles.resize(imageNum);
    torch::Tensor model=PyMMS.FMFull.Generate(shapeX, blendShapeX);
    //#pragma omp parallel for
    for(int j=0;j<imageNum;j++){
        glm::tvec3<float> euler=glm::eulerAngles(GlmFunctions::RotationToQuat(params[j].R));
        yawAngles[j]=glm::degrees(euler.y);    
        torch::Tensor projected = ProjectionCenter(params[j], model);
        projecteds[j]=projected;
    }
    std::vector<at::Tensor> weightTs=computeTextureWeights(imageNum,frontModel,frontNormals,yawAngles,axisMetrics,PyMMS);
    return merge(images,projecteds,weightTs,PyMMS,1024,1024);
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
cv::Mat MultiFitting::merge(std::vector<cv::Mat> &images,std::vector<torch::Tensor> projecteds, std::vector<at::Tensor> &weightTs,MMTensorSolver& PyMMS,int H,int W)
{
    auto TRI = PyMMS.FMFull.TRI;
    auto TRIUV = PyMMS.FMFull.TRIUV;
    auto UV = PyMMS.FMFull.UV;
    cv::Mat result=cv::Mat::zeros(H,W,images[0].type());
    size_t imageNum=weightTs.size();
    at::Tensor sum=torch::zeros(TRI.size(0));

    for(size_t k=0;k<imageNum;k++){
        sum.add_(weightTs[k]);
    }
    sum.mul_(1000);
    torch::Tensor mask=sum.lt(1e-6);
    sum.masked_fill_(mask,1.0);
    sum.mul_(0.001);
    std::vector<torch::Tensor> triProjecteds;
    triProjecteds.resize(imageNum);
    for(size_t k=0;k<imageNum;k++){
        weightTs[k].div_(sum);
        triProjecteds[k]=projecteds[k].index_select(0,TRI.toType(torch::kLong).view(-1)).view({-1,3,2});//tx3x2
    }
    at::Tensor triUVs=UV.index_select(0,TRIUV.toType(torch::kLong).view(-1)).view({-1,3,2});//tx3x2
    triUVs.select(2,0).mul_((float)(W - 1));
    triUVs.select(2,1).sub_(1.0f);
    triUVs.select(2,1).mul_(-(float)(H - 1));
    torch::TensorList wv(weightTs.data(),weightTs.size());
    torch::Tensor ws=torch::stack(wv,0);//imageNum*TRI.size(0)
    torch::TensorList tv(triProjecteds.data(),triProjecteds.size());
    torch::Tensor ts=torch::stack(tv,1);//tximageNumx3x2
//    std::cout<<"ts.sizes():"<<ts.sizes()<<std::endl;
    //#pragma omp parallel for
    for (int t = 0; t < TRI.size(0); t++)
    {
        torch::Tensor srcTris=ts[t];
        torch::Tensor dstTri=triUVs[t];
        FaceMorph::morphTriangle(images,result,srcTris,dstTri,ws.select(1,t));
    }
    //cv::imwrite("result.isomap.png",result);
    return std::move(result);
}
cv::Mat MultiFitting::merge2(std::vector<cv::Mat> &images,std::vector<torch::Tensor> projecteds, std::vector<at::Tensor> &weightTs,MMTensorSolver& PyMMS,int H,int W)
{
    auto TRI = PyMMS.FMFull.TRI;
    auto TRIUV = PyMMS.FMFull.TRIUV;
    auto UV = PyMMS.FMFull.UV;
    cv::Mat result=cv::Mat::zeros(H,W,images[0].type());
    size_t imageNum=weightTs.size();
    at::Tensor sum=torch::zeros(TRI.size(0));
    for(int k=0;k<imageNum;k++){
        sum.add_(weightTs[k]);
    }
    sum.mul_(1000);
    torch::Tensor mask=sum.lt(1e-6);
    sum.masked_fill_(mask,1.0);
    sum.mul_(0.001);
    std::vector<torch::Tensor> triProjecteds;
    triProjecteds.resize(imageNum);
    for(int k=0;k<imageNum;k++){
        weightTs[k].div_(sum);
        triProjecteds[k]=projecteds[k].index_select(0,TRI.toType(torch::kLong).view(-1)).view({-1,3,2});//tx3x2
    }
    at::Tensor triUVs=UV.index_select(0,TRIUV.toType(torch::kLong).view(-1)).view({-1,3,2});//tx3x2

    //#pragma omp parallel for
    for (int t = 0; t < TRI.size(0); t++)
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
//                auto x = projecteds[k][j][0].item().toFloat();
//                auto y = projecteds[k][j][1].item().toFloat();
                auto x =triProjecteds[k][t][i][0].item().toFloat();
                auto y =triProjecteds[k][t][i][1].item().toFloat();
                srcTris[k].push_back(cv::Point2f(x, y));
            }
        }
        FaceMorph::morphTriangle(images,result,srcTris,dstTri,weights);

    }
    //cv::imwrite("result.isomap.png",result);
    return std::move(result);
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
   torch::Tensor projected=ProjectionCenter(param,modelMarkT);
   torch::Tensor corIndex=torch::zeros({indexes.size(0),1},torch::dtype(torch::kLong));
   for(int i=0;i<indexes.size(0);i++){
       long long index=indexes[i].item().toLong();
       torch::Tensor dist=projected.sub(landMarkT[index]).norm(2,1);
       corIndex[i]=std::get<1>(dist.min(0,true));
   }
//   std::cout<<"getNearestContourCorrespondences done"<<std::endl<<std::flush;
   return std::make_tuple(std::move(indexes),std::move(corIndex));
}

