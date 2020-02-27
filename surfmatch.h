#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>
using namespace std;
using namespace cv::xfeatures2d;
void surf_match(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& uvmark);