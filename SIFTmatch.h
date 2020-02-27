#pragma once
#include "pch.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
using namespace cv;
using namespace xfeatures2d;
using namespace std;
extern cv::Mat Intrinsic;
void feature_sift(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& m1, cv::Mat& m2, vector <Point2f>& p01, vector <Point2f>& p02);
