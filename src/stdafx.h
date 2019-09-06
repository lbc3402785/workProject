#pragma once
//#include "eos/core/Image.hpp"
//#include "eos/core/image/opencv_interop.hpp"
//#include "eos/core/Landmark.hpp"
//#include "eos/core/LandmarkMapper.hpp"
//#include "eos/core/read_pts_landmarks.hpp"
//#include "eos/fitting/fitting.hpp"
//#include "eos/morphablemodel/Blendshape.hpp"
//#include "eos/morphablemodel/MorphableModel.hpp"
//#include "eos/render/draw_utils.hpp"
//#include "eos/render/texture_extraction.hpp"
//#include "eos/cpp17/optional.hpp"

#include "Eigen/Core"

//#include "boost/filesystem.hpp"
//#include "boost/program_options.hpp"
#ifdef _WIN32
#include <WinSock2.h>
#endif
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <vector>

//#include "eos/fitting/closest_edge_fitting.hpp"
//
//using namespace eos;
//namespace po = boost::program_options;
//namespace fs = boost::filesystem;
//using eos::core::Landmark;
//using eos::core::LandmarkCollection;
using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::vector;
