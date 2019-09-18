#ifndef FACEMORPH_H
#define FACEMORPH_H
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <torch/torch.h>
class FaceMorph
{
public:
    FaceMorph();
    static void morphTriangle(std::vector<cv::Mat> &imgs, cv::Mat &target, torch::Tensor &triangles, torch::Tensor &targetTri, torch::Tensor weights);
    static void morphTriangle(std::vector<cv::Mat> &imgs, cv::Mat &target, std::vector<std::vector<cv::Point2f>> &triangles, std::vector<cv::Point2f> &targetTri, std::vector<float>& weights);
private:
    static void applyAffineTransform(cv::Mat &warpImage, cv::Mat &src, std::vector<cv::Point2f> &srcTri, std::vector<cv::Point2f> &dstTri);
};

#endif // FACEMORPH_H
