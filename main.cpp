#include "boost/filesystem.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <torch/torch.h>

#include "MMSolver.h"
#include "Dlib.h"
#include "contour.h"
#include "multifitting.h"
string dir_path = "./data/";
MMSolver PyMMS;
FaceModel PyFM;
void testMul()
{
   torch::Tensor t1=torch::rand({5,4});
   t1.select(1,0) *= 0;
   std::cout<<t1<<std::endl;
}
void testSvd()
{
//    torch::Tensor t1=torch::rand({3,3});
//    std::cout<<t1<<std::endl;
//    torch::Tensor U,S,V;
//    std::tie(U,S,V)=torch::svd(t1);
//    std::cout<<"U:"<<U<<std::endl;
//    std::cout<<"------"<<std::endl;
//    std::cout<<"S:"<<S<<std::endl;
//    std::cout<<"------"<<std::endl;
//    std::cout<<"V:"<<V<<std::endl;
//    std::cout<<"------"<<std::endl;
//    Eigen::JacobiSVD<Eigen::Matrix3f, Eigen::NoQRPreconditioner> svd(EigenToTorch::TorchTensorToEigenMatrix(t1), Eigen::ComputeFullU | Eigen::ComputeFullV);
//    Eigen::Matrix3f U1 = svd.matrixU();
//    std::cout<<"U1:"<<U1<<std::endl;
//    std::cout<<"------"<<std::endl;
//    const Eigen::Matrix3f V1 = svd.matrixV();
//    std::cout<<"V1:"<<V1<<std::endl;
//    std::cout<<"------"<<std::endl;
    MatF input(4,3);
    input(0,0)=1;input(0,1)=2;input(0,2)=3;
    input(1,0)=4;input(1,1)=5;input(1,2)=6;
    input(2,0)=7;input(2,1)=8;input(2,2)=9;
    input(3,0)=10;input(3,1)=11;input(3,2)=12;
    std::cout<<input.colwise().mean()<<std::endl;
}
void testMat()
{
    MatF f(4,3);
    f(0,1)=5;f(0,2)=6;
    f(1,0)=5;f(1,2)=7;
    f(2,0)=-3;f(2,2)=-7;
    f(3,0)=-3;f(3,2)=2;
    f(4,1)=-3;f(4,2)=-2;
    std::cout<<f<<std::endl;
//    f.col(2).array()-=5;
    std::cout<<"---------"<<std::endl;

    MatF f1=f;
    f1.col(2).array()-=5;
    std::cout<<f<<std::endl;
}
void InitMMS(std::string fmkp, std::string fmfull)
{
    cout << fmkp << endl;
    cout << fmfull << endl;
    PyMMS.Initialize(fmkp, fmfull);
}
void readfolder(boost::filesystem::path& imageDir,std::vector<std::string>& result) {
    boost::filesystem::directory_iterator endIter;
    for (boost::filesystem::directory_iterator iter(imageDir); iter != endIter; iter++) {
      if (boost::filesystem::is_directory(*iter)) {
        //cout << "is dir" << endl;
        //cout << iter->path().string() << endl;
      } else {
        //cout << "is a file" << endl;
        //std::cout << iter->path().string() << std::endl;
        result.emplace_back(iter->path().string());
      }
    }
}
bool KeypointDetectgion(cv::Mat image, MatF &KP)
{
    std::vector<std::vector<cv::Point>> keypoints;
    double S = 0.5;
    cv::Mat simage;
    cv::resize(image, simage, cv::Size(), S, S);

    std::vector<cv::Rect> rectangles;
    DlibInit(dir_path + "shape_predictor_68_face_landmarks.dat");
    DlibFace(simage, rectangles, keypoints);

    if (keypoints.size() <= 0)
    {
        errorcout << "NO POINTS" << endl;
        return false;
    }
    KP = ToEigen(keypoints[0]) * (1.0 / S);

    return true;
}

std::vector<std::string> loadImages(std::vector<std::string>& imageFiles,std::vector<cv::Mat>& images,std::vector<MatF>& landMarks,bool flip=true)
{
    std::vector<std::string> outputImages;
    for(const auto& imageFile:imageFiles){
        cv::Mat image = cv::imread(imageFile);
        if(image.empty())continue;
        std::string pts=imageFile.substr(0,imageFile.find_last_of("."))+".pts";
        MatF KP;
        bool success = KeypointDetectgion(image, KP);
        if(success){
            std::cout<<imageFile<<std::endl;
            images.emplace_back(image);
            if(flip){
                KP.col(1).array()=image.rows-KP.col(1).array();
            }
            landMarks.emplace_back(KP);
            outputImages.push_back(imageFile);
        }
    }
    return outputImages;
}
void testMajor()
{
    MatF a(3,3);
    std::cout<<a<<std::endl;
    a.row(1).array()+=2;
    std::cout<<a<<std::endl;
}
void testMToT()
{
    MatI aa(3,2);
    aa(0,0)=0;aa(0,1)=2;
    aa(1,0)=1;aa(1,1)=3;
    aa(2,0)=5;aa(2,1)=184510;
    std::cout<<aa<<std::endl;
    std::cout<<"-------"<<std::endl;
    torch::Tensor t=torch::from_blob(aa.data(),{3,2},torch::TensorOptions().dtype(torch::kInt));
    std::cout<<t<<std::endl;
}
void MakeDir(string path)
{
//    namespace fs = std::filesystem; // C++17
//    std::error_code ec;
//    bool success = fs::create_directories(path, ec);
    boost::filesystem::path dir(path);
    boost::filesystem::create_directory(dir);
}
int main(int argc, char *argv[])
{
    if(argc<3){
        std::cout<<"useage:exe modelPath imagePath";
        exit(EXIT_FAILURE);
    }
//    std::string modelPath=argv[1];
//    std::string imagePath=argv[2];
//    std::string fmkp =modelPath+"BFM2017KP.npz";
//    std::string fmfull =modelPath+ "BFMUV.obj.npz";
//    std::string dlibModel=modelPath+ "shape_predictor_68_face_landmarks.dat";
//    std::string mappingsfile=modelPath+"contour.json";
//    InitMMS(fmkp, fmfull);
//    DlibInit(dlibModel);
//    ContourLandmarks contour=ContourLandmarks::load(mappingsfile);
//    boost::filesystem::path imageDir(imagePath);
//    std::vector<std::string> imageFiles;
//    readfolder(imageDir,imageFiles);
//    std::vector<cv::Mat> images;
//    std::vector<MatF> landMarks;
//    std::vector<std::string> outputImages=loadImages(imageFiles,images,landMarks);
//    std::vector<int> image_widths;
//    std::vector<int> image_heights;
//    for (const auto& image : images)
//    {
//        image_widths.push_back(image.cols);
//        image_heights.push_back(image.rows);
//    }
//    MatF shapeX;
//    std::vector<MatF> blendShapeXs;
//    std::vector<ProjectionParameters> params=MultiFitting::fitShapeAndPose(images,contour,PyMMS,landMarks,shapeX,blendShapeXs,4);
//    int W = 512;
//    int H = 512;
//    std::vector<cv::Mat> textures;
//    for(size_t j=0;j<images.size();j++){
//        PyMMS.params=params[j];
//        PyMMS.EX=blendShapeXs[j];
//        PyMMS.SX=shapeX;
//        cv::Mat texture = MMSTexture(images[j], PyMMS, W, H);
//        boost::filesystem::path outputfile(outputImages[j]);
//        outputfile.replace_extension(".isomap.png");
//        cv::imwrite(outputfile.string(),texture);
//        textures.emplace_back(texture);
//    }
//    MatF blendShapeX=blendShapeXs[0];
//    for(size_t j=1;j<images.size();j++){
//       blendShapeX+= blendShapeXs[j];
//    }
//    blendShapeX/=images.size();
//    PyMMS.params=params[2];
//    PyMMS.EX=blendShapeX;
//    PyMMS.SX=shapeX;
//    string outfolder = "./output/";
//    string filename = "TestObj";
//    std::cout<<"---------------"<<std::endl;
//    MakeDir(outfolder);
    //MMSObj(images[2], PyMMS, outfolder, filename);
//    MultiFitting::render(images,params,shapeX,blendShapeXs,contour,PyMMS,5.0f);
    testMul();
    return 0;
}
