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

std::vector<std::string> loadImages(std::vector<std::string>& imageFiles,std::vector<cv::Mat>& images,std::vector<MatF>& landMarks)
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
    std::vector<std::string> outputImages=loadImages(imageFiles,images,landMarks);
    std::vector<int> image_widths;
    std::vector<int> image_heights;
    for (const auto& image : images)
    {
        image_widths.push_back(image.cols);
        image_heights.push_back(image.rows);
    }
    MatF shapeX;
    std::vector<MatF> blendShapeXs;
    std::vector<ProjectionParameters> params=MultiFitting::fitShapeAndPose(images,contour,PyMMS,landMarks,shapeX,blendShapeXs,4);
    int W = 512;
    int H = 512;
    std::vector<cv::Mat> textures;
    for(size_t j=0;j<images.size();j++){
        PyMMS.params=params[j];
        PyMMS.EX=blendShapeXs[j];
        PyMMS.SX=shapeX;
        cv::Mat texture = MMSTexture(images[j], PyMMS, W, H);
        boost::filesystem::path outputfile(outputImages[j]);
        outputfile.replace_extension(".isomap.png");
        cv::imwrite(outputfile.string(),texture);
        textures.emplace_back(texture);
    }
    PyMMS.params=params[2];
    PyMMS.EX=blendShapeXs[2];
    PyMMS.SX=shapeX;
    string outfolder = "./output/";
    string filename = "TestObj";
    std::cout<<"---------------"<<std::endl;
    MakeDir(outfolder);
    MMSObj(images[2], PyMMS, outfolder, filename);
    std::cout<<"done!"<<std::endl;
    //testMajor();

    return 0;
}
