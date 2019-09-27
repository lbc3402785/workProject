#pragma once
#include <opencv2/highgui/highgui.hpp>


void DlibInit(std::string filepath);
bool DlibFace(cv::Mat img, std::vector<cv::Rect> &rectangles, std::vector<std::vector<cv::Point>> &keypoints);
