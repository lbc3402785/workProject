#pragma once
//#include "OpenCV.h"
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"


#define MASK_BACKGROUND (1<<7)
#define MAX_ITERATION 300

inline uchar clampValue(float val) {
	if (val > 255) return 255;
	else if (val < 0) return 0;
	else return (uchar)val;
}

inline void Poisson(cv::Mat srcImage, cv::Mat destImage, cv::Mat maskImage) {

	using namespace std;
	using namespace cv;

	if (srcImage.empty() || destImage.empty() || maskImage.empty()) {
		cerr << "cannot find image file" << endl;
		exit(-1);
	}

	int w = srcImage.cols;
	int h = srcImage.rows;

	if (w != maskImage.cols || h != maskImage.rows) {
		cerr << "mask size doesn't match src size" << endl;
		exit(-1);
	}

	vector<int> destPoints;
	vector<int> constants;

	int maskStride = maskImage.cols;
	int destStride = destImage.cols * 3;
	int srcStride = srcImage.cols * 3;

	int size = 0;
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int maskIndex = maskStride * y + x;
			uchar c = maskImage.data[maskIndex];
			//printf("%d\n", (int)c);
			//if (!(c & MASK_BACKGROUND)) {
			if (c & MASK_BACKGROUND) {
				//printf("input (%d,%d)\n", x, y);
				int destIndex = destStride * y + (x * 3);
				destPoints.push_back(destIndex);

				int srcIndex = srcStride * y + (x * 3);

				int constant[3] = { 0 };
				int sum1[3] = { 0 };
				int sum2[3] = { 0 };
				// right
				if (x < srcImage.cols - 1) {
					for (int i = 0; i < 3; i++) {
						int val1 = (uchar)(destImage.data[destIndex + 3 + i]) - (uchar)(destImage.data[destIndex + i]);
						int val2 = (uchar)(srcImage.data[srcIndex + 3 + i]) - (uchar)(srcImage.data[srcIndex + i]);
						//printf("%d,%d\n", val1, val2);
						//constant[i] += min(val1, val2);
						sum1[i] += val1;
						sum2[i] += val2;
						//constant[i] += val2;
					}
				}
				// left
				if (x > 0) {
					for (int i = 0; i < 3; i++) {
						int val1 = (uchar)(destImage.data[destIndex - 3 + i]) - (uchar)(destImage.data[destIndex + i]);
						int val2 = (uchar)(srcImage.data[srcIndex - 3 + i]) - (uchar)(srcImage.data[srcIndex + i]);
						//printf("%d,%d\n", val1, val2);
						//constant[i] += min(val1, val2);
						sum1[i] += val1;
						sum2[i] += val2;
						//constant[i] += val2;
					}
				}
				// top
				if (y > 0) {
					for (int i = 0; i < 3; i++) {
						int val1 = (uchar)(destImage.data[destStride*(y - 1) + (x * 3) + i]) - (uchar)(destImage.data[destIndex + i]);
						int val2 = (uchar)(srcImage.data[srcStride*(y - 1) + (x * 3) + i]) - (uchar)(srcImage.data[srcIndex + i]);
						//printf("%d,%d\n", val1, val2);
						//constant[i] += min(val1, val2);
						sum1[i] += val1;
						sum2[i] += val2;
						//constant[i] += val2;
					}
				}
				// bottom
				if (y < srcImage.rows - 1) {
					for (int i = 0; i < 3; i++) {
						int val1 = (uchar)(destImage.data[destStride*(y + 1) + (x * 3) + i]) - (uchar)(destImage.data[destIndex + i]);
						int val2 = (uchar)(srcImage.data[srcStride*(y + 1) + (x * 3) + i]) - (uchar)(srcImage.data[srcIndex + i]);
						//printf("%d,%d\n", val1, val2);
						//constant[i] += min(val1, val2);
						sum1[i] += val1;
						sum2[i] += val2;
						//constant[i] += val2;
					}
				}
				for (int i = 0; i < 3; i++) {
					//constants.push_back(constant[i]);
					//if (abs(sum1[i]) > abs(sum2[i])){
					//  constants.push_back(sum1[i]);
					//} else {
					constants.push_back(sum2[i]);
					//}
				}
			}
			// int offset = srcStride * y + (x * 3);
			size++;
		}
	}

	printf("destPoints size=%d\n", (int)destPoints.size());
	printf("constants size=%d\n", (int)constants.size());
	uchar* destImageData = (uchar*)destImage.data;
	cv::Mat final(3 * destImage.cols, destImage.rows, DataType<float>::type);
	for (int x = 0; x < destImage.cols; x++) for (int y = 0; y < destImage.rows; y++)
		for (int i = 0; i < 3; i++)
			final.at<float>(x * 3 + i, y) = ((uchar)destImage.data[destStride*y + 3 * x + i]);

	//for (int x=0; x<destImage.cols; x++) for (int y=0; y<destImage.rows; y++) 
	//  printf("(%f,%f,%f)\n", final.at<float>(x*3+0, y), final.at<float>(x*3+1, y), final.at<float>(x*3+2, y));

	// ヤコビ法で連立一次方程式を解く
	for (int loop = 0; loop < MAX_ITERATION; loop++) {
		int n = destPoints.size();
		for (int p = 0; p < n; p++) {
			int destIndex = destPoints[p];
			int y = destIndex / (destStride);
			int x = (destIndex % (destStride)) / 3;
			//printf("check (%d,%d)\n", x, y);
			float values[3] = { 0 };
			// right
			int count = 0;
			if (x < destImage.cols - 1) {
				count++;
				for (int i = 0; i < 3; i++) {
					//values[i] += (uchar)(destImageData[destIndex+3+i]);
					values[i] += final.at<float>((3 * (x + 1)) + i, y);
				}
			}
			// left
			if (x > 0) {
				count++;
				for (int i = 0; i < 3; i++) {
					//values[i] += (uchar)(destImageData[destIndex-3+i]);
					values[i] += final.at<float>((3 * (x - 1)) + i, y);
				}
			}
			// top
			if (y > 0) {
				count++;
				for (int i = 0; i < 3; i++) {
					//values[i] += (uchar)(destImageData[destStride*(y-1)+(x*3)+i]);
					values[i] += final.at<float>((3 * x) + i, y - 1);
				}
			}
			// bottom
			if (y < destImage.rows - 1) {
				count++;
				for (int i = 0; i < 3; i++) {
					//values[i] += (uchar)(destImageData[destStride*(y+1)+(x*3)+i]);
					values[i] += final.at<float>((3 * x) + i, y + 1);
				}
			}
			//for (int i=0; i<3; i++) {
			//  values[i] -= count*((uchar)(destImageData[destIndex+i]));
			//}
			// 更新
			for (int j = 0; j < 3; j++) {
				float newval = (values[j] - constants[p * 3 + j]) / count;
				float oldval = final.at<float>((3 * x) + j, y);
				//if (newval >= 256) newval = 255;
				//if (newval < 0) newval = 0;
				//destImageData[destIndex+j] = (uchar)newval;
				//printf("%f->%f\n", oldval, newval);
				final.at<float>((3 * x) + j, y) = newval;
			}
		}
	}

	{
		int n = destPoints.size();
		for (int p = 0; p < n; p++) {
			int destIndex = destPoints[p];
			int y = destIndex / (destStride);
			int x = (destIndex % (destStride)) / 3;
			//printf("set (%d,%d)\n", x, y);
			for (int i = 0; i < 3; i++) {
				//printf(" %d->%d\n", x, y, (uchar)destImage.data[destIndex + i], clampValue(final.at<float>(x*3+i, y)));
				destImage.data[destIndex + i] = clampValue(final.at<float>(x * 3 + i, y));
			}
		}
	}
	return;
}




inline cv::Mat PoissonBlending(cv::Mat src, cv::Mat target, cv::Mat msk, bool UsePoisson, cv::Scalar* _diff = nullptr)
{
	using namespace std;
	using namespace cv;

	{

		Mat msk1 = msk.clone();
		Mat msk2 = 255 - msk;

		int dilation_size = 20;
		Mat element = getStructuringElement(MORPH_ELLIPSE,
			Size(2 * dilation_size + 1, 2 * dilation_size + 1),
			Point(dilation_size, dilation_size));


		if (UsePoisson)
		{
			erode(msk1, msk1, element);
			Poisson(src, target, msk1);
		}
		

		src.convertTo(src, CV_32FC3, 1.0 / 255);
		target.convertTo(target, CV_32FC3, 1.0 / 255);


		msk = msk1;

		dilation_size = 60;
		element = getStructuringElement(MORPH_ELLIPSE,
			Size(2 * dilation_size + 1, 2 * dilation_size + 1),
			Point(dilation_size, dilation_size));
		erode(msk, msk, element);
		GaussianBlur(msk, msk, Size(2 * dilation_size + 1, 2 * dilation_size + 1), dilation_size / 2);


		msk.convertTo(msk, CV_32FC1, 1.0 / 255); // 

		cvtColor(msk, msk, CV_GRAY2BGR);


		if (true)
		{
			Mat diff = (src - target).mul(msk);
			cv::Scalar s = cv::sum(diff);
			cv::Scalar w = cv::sum(msk);

			s /= w[0];


			target += s;

			if (_diff != nullptr)
			{
				*_diff = s;
			}
		}


		Mat result = src.mul(msk) + target.mul(-msk + Scalar::all(1.0));

		return result;
	}

}