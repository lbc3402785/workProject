#include "boost/filesystem.hpp"
#include <vector>
#include <iostream>
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
void testMul()
{
   torch::Tensor t1=torch::rand({5,4});
   t1.select(1,0) *= 0;
   std::cout<<t1<<std::endl;
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
bool KeypointDetectgion(cv::Mat image, torch::Tensor &KP)
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
        //std::cout << "NO POINTS" << endl;
        return false;
    }
    KP = ToTensor(keypoints[0]) * (1.0 / S);

    return true;
}

std::vector<std::string> loadImages(std::vector<std::string>& imageFiles,std::vector<cv::Mat>& images,std::vector<torch::Tensor>& landMarks,bool flip=true)
{
    std::vector<std::string> outputImages;
    for(const auto& imageFile:imageFiles){
        cv::Mat image = cv::imread(imageFile);
        if(image.empty())continue;
        std::string pts=imageFile.substr(0,imageFile.find_last_of("."))+".pts";
        torch::Tensor KP;
        bool success = KeypointDetectgion(image, KP);
        if(success){
            images.emplace_back(image);
            if(flip){
                KP.select(1,1)=image.rows-KP.select(1,1);
            }
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
int main(int argc, char *argv[])
{
//    testMul();
    if(argc<3){
        std::cout<<"useage:exe modelPath imagePath";
        exit(EXIT_FAILURE);
    }
    std::string modelPath=argv[1];
    std::string imagePath=argv[2];
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
    std::vector<int> image_widths;
    std::vector<int> image_heights;
    for (const auto& image : images)
    {
        image_widths.push_back(image.cols);
        image_heights.push_back(image.rows);
    }
    torch::Tensor shapeX;
    std::vector<torch::Tensor> blendShapeXs;
    std::vector<ProjectionTensor> params=MultiFitting::fitShapeAndPose(images,contour,PyMMS,landMarks,shapeX,blendShapeXs,4);
    int W = 512;
    int H = 512;

    torch::Tensor blendShapeX=blendShapeXs[0];
    for(size_t j=1;j<images.size();j++){
       blendShapeX+= blendShapeXs[j];
    }
    blendShapeX.div_((int64_t)images.size());
    PyMMS.params=params[2];
    PyMMS.EX=blendShapeX;
    PyMMS.SX=shapeX;
    string outfolder = "./output/";
    string filename = "TestObj";

    MakeDir(outfolder);

    cv::Mat texture=MultiFitting::render(images,params,shapeX,blendShapeXs,contour,PyMMS,5.0f);

    MMSObjWithTexture(texture, PyMMS, outfolder, filename);

//    testMul();
    return 0;
}
