#pragma once
#include <opencv2/highgui/highgui.hpp>


void DlibInit(std::string filepath);
void DlibFace(cv::Mat img, vector<cv::Rect> &rectangles, std::vector<std::vector<cv::Point>> &keypoints);