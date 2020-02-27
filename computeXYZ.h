#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace std;
//void  parameters_get(cv::Mat leftIntrinsic, cv::Mat rightIntrinsic, cv::Mat rightRotation, cv::Mat rightTranslation);
extern cv::Mat leftIntrinsic, rightIntrinsic, rightRotation, rightTranslation;
cv::Mat computeXYZ(cv::Point2f uvLeft, cv::Point2f uvRight);

