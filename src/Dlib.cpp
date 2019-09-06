#include "stdafx.h"
#include <string>
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace cv;
using namespace std;
const string folder = "";
//string folder = "/storage/emulated/0/";
//using namespace dlib;


bool USEDNN = false;
bool DlibInitialized = false;
dlib::frontal_face_detector detector;
dlib::shape_predictor pose_model;
CascadeClassifier face_cascade;

#ifdef CVDNN
#include <opencv2/dnn.hpp>
using namespace cv::dnn;
const std::string tensorflowConfigFile = folder + "opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = folder + "opencv_face_detector_uint8.pb";
Net net;
bool USECVDETECT = true;
#else

bool USECVDETECT = false;
#endif

void InitDNN()
{
	if (USEDNN) {
#ifdef CVDNN
		net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
#endif
	}
	else {
		face_cascade.load(folder + "haarcascade_frontalface_alt.xml");
	}
}

std::vector<Rect> DetectDNN(Mat frame)
{
	std::vector<Rect> faces;


#ifdef CVDNN
	const size_t inWidth = 300;
	const size_t inHeight = 300;
	const double inScaleFactor = 1.0;
	const float confidenceThreshold = 0.7;
	const cv::Scalar meanVal(104.0, 177.0, 123.0);

	int frameHeight = frame.rows;
	int frameWidth = frame.cols;

	cv::Mat inputBlob = cv::dnn::blobFromImage(frame, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);

	net.setInput(inputBlob, "data");
	cv::Mat detection = net.forward("detection_out");

	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > confidenceThreshold)
		{
			int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
			int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
			int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
			int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

			faces.push_back(Rect(x1, y1, x2 - x1, y2 - y1));
			break;
		}
	}
#endif
	return faces;
}

inline Rect Scale(Rect r, float s)
{
	return Rect(r.x * s, r.y* s, r.width* s, r.height* s);

}

void DlibInit(string filepath)
{
	if (!DlibInitialized)
	{
		cout << "DLIB INITIALIZE" << endl;
		detector = dlib::get_frontal_face_detector();
		dlib::deserialize(filepath) >> pose_model;
		if (USECVDETECT)
		{
			InitDNN();
		}
		DlibInitialized = true;
	}

}




void DlibFace(Mat img, vector<Rect> &rectangles, vector<vector<Point>> &keypoints)
{
	dlib::cv_image<dlib::bgr_pixel> cimg(img);

	// Detect faces
	std::vector<dlib::rectangle> faces;

	if (USECVDETECT)
	{
		std::vector<Rect> faces0;

		if (USEDNN) {
			faces0 = DetectDNN(img);
			for (size_t i = 0; i < faces0.size(); i++) {
				auto f = faces0[i];
				faces.push_back(dlib::rectangle(f.x, f.y, f.x + f.width, f.y + f.height));
			}
		}
		else {

			Mat frame_gray;
			cvtColor(img, frame_gray, COLOR_BGR2GRAY);
			resize(frame_gray, frame_gray, Size(), 0.5, 0.5);
			equalizeHist(frame_gray, frame_gray);
			face_cascade.detectMultiScale(frame_gray, faces0, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(100, 100));

			for (size_t i = 0; i < faces0.size(); i++)
			{
				auto f = faces0[i];//
				f = Scale(f, 2);
				faces.push_back(dlib::rectangle(f.x, f.y, f.x + f.width, f.y + f.height));
			}
		}


	}
	else
	{
		faces = detector(cimg);
	}

    //cout << faces.size() << endl;

	// Find the pose of each face.
	std::vector<dlib::full_object_detection> shapes;
	for (unsigned long i = 0; i < faces.size(); ++i)
		shapes.push_back(pose_model(cimg, faces[i]));

	rectangles = vector<Rect>(faces.size());
	for (size_t i = 0; i < faces.size(); i++)
	{
		auto r = faces[i];
		rectangles[i] = Rect(r.left(), r.top(), r.width(), r.height());
	}

	keypoints = vector<vector<Point>>(faces.size());

	for (size_t i = 0; i < shapes.size(); i++)
	{
		auto shape = shapes[i];
		int N = shape.num_parts();
		keypoints[i] = vector<Point>(N);
		for (size_t j = 0; j < N; j++)
		{
			dlib::point p = shape.part(j);

			keypoints[i][j] = Point(p.x(), p.y());
		}

	}


}
