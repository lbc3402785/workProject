#include "test.h"
#include "boost/filesystem.hpp"
#include <vector>

#include <fstream>
#include <string>
#include <torch/torch.h>

#include "mmtensorsolver.h"
#include "Dlib.h"
#include "contour.h"
#include "multifitting.h"

string dir_path = "./data/";
MMTensorSolver PyMMS;
FaceModelTensor PyFM;

void InitMMS(std::string fmkp, std::string fmfull)
{
    std::cout << fmkp << std::endl;
    std::cout << fmfull << std::endl;
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
bool readNpyKeyPonts(std::string filename,torch::Tensor& KP)
{
    try {
        NpyArray npy=cnpy::npy_load(filename);
        KP=ToTensor(npy);
        return true;
    } catch (std::exception&e) {
        std::cout<<e.what()<<std::endl;
        return false;
    }
}
bool KeypointDetectgion(cv::Mat image, torch::Tensor &KP,std::string pts="")
{
//    std::vector<std::vector<cv::Point>> keypoints;
//    double S = 0.5;
//    cv::Mat simage;
//    cv::resize(image, simage, cv::Size(), S, S);

//    std::vector<cv::Rect> rectangles;
//    DlibInit(dir_path + "shape_predictor_68_face_landmarks.dat");
//    if(!DlibFace(simage, rectangles, keypoints))return false;

//    if (keypoints.size() <= 0)
//    {
//        std::cout << "NO POINTS" << std::endl;
//        return false;
//    }
//    KP = ToTensor(keypoints[0]) * (1.0 / S);
    return readNpyKeyPonts(pts,KP);
}

std::vector<std::string> loadImages(std::vector<std::string>& imageFiles,std::vector<cv::Mat>& images,std::vector<torch::Tensor>& landMarks)
{
    std::vector<std::string> outputImages;
    for(const auto& imageFile:imageFiles){
        cv::Mat image = cv::imread(imageFile);
        if(image.empty())continue;
        std::string pts=imageFile.substr(0,imageFile.find_last_of("."))+".png.npy";
        torch::Tensor KP;
        bool success = KeypointDetectgion(image, KP,pts);
        if(success){
            images.emplace_back(image);
            landMarks.emplace_back(KP);
            outputImages.push_back(imageFile);
        }
    }
    return outputImages;
}

void MakeDir(string path)
{
    boost::filesystem::path dir(path);
    boost::filesystem::create_directory(dir);
}
void Test::testCeres(std::string modelPath, std::string imagePath)
{
    std::string fmkp =modelPath+"BFM2017KP.npz";
    std::string fmfull =modelPath+ "BFMUV.obj.npz";
    std::string dlibModel=modelPath+ "shape_predictor_68_face_landmarks.dat";
    std::string mappingsfile=modelPath+"contour.json";
    InitMMS(fmkp, fmfull);

    DlibInit(dlibModel);

    ContourLandmarks contour=ContourLandmarks::load(mappingsfile);

    boost::filesystem::path imageDir(imagePath);
    std::vector<std::string> imageFiles;
    readfolder(imageDir,imageFiles);

    std::vector<cv::Mat> images;
    std::vector<torch::Tensor> landMarks;
    std::vector<std::string> outputImages=loadImages(imageFiles,images,landMarks);
    if(images.empty()){
        std::cout<<"no avaliable image!";
        exit(EXIT_FAILURE);
    }
    torch::Tensor shapeX;
    torch::Tensor blendShapeX;
    std::vector<torch::Tensor> blendShapeXs;
    std::vector<ProjectionTensor> params=MultiFitting::fitShapeAndPose(images,contour,PyMMS,landMarks,shapeX,blendShapeX,blendShapeXs,4);
    std::cout<<"fitShapeAndPose done!"<<std::endl;
    string outfolder = "./output/";
    string filename = "TestObj";
    MakeDir(outfolder);
    if(images.size()>1){
        PyMMS.EX=blendShapeX;
        PyMMS.SX=shapeX;
        std::cout<<"begin render ..."<<std::endl;
        cv::Mat texture=MultiFitting::render(images,params,shapeX,blendShapeX,blendShapeXs,contour,PyMMS,5.0f);
         std::cout<<"render done!"<<std::endl;
        MMSObjWithTexture(texture, PyMMS, outfolder, filename);
        std::cout<<"MMSObjWithTexture done!"<<std::endl;
    }else{
        PyMMS.params=params[0];
        PyMMS.EX=blendShapeXs[0];
        PyMMS.SX=shapeX;
        MMSObj(images[0], PyMMS, outfolder, filename);
    }

}
