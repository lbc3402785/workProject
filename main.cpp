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
void loadImages(std::vector<std::string>& imageFiles,std::vector<cv::Mat>& images,std::vector<MatF>& landMarks)
{
    for(const auto& imageFile:imageFiles){
        cv::Mat image = cv::imread(imageFile);
        if(image.empty())continue;
        MatF KP;
        bool success = KeypointDetectgion(image, KP);
        if(success){
            std::cout<<imageFile<<std::endl;
            images.emplace_back(image);
            landMarks.emplace_back(KP);
        }
    }
}
int main(int argc, char *argv[])
{
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
    std::vector<MatF> landMarks;
    loadImages(imageFiles,images,landMarks);
    std::vector<int> image_widths;
    std::vector<int> image_heights;
    for (const auto& image : images)
    {
        image_widths.push_back(image.cols);
        image_heights.push_back(image.rows);
    }
    MatF shapeX;
    std::vector<MatF> blendShapeXs;
    std::vector<MatF> fittedImagePoints;
    MultiFitting::fitShapeAndPose(images,contour,PyMMS,landMarks,shapeX,blendShapeXs,fittedImagePoints);

    return 0;
}
