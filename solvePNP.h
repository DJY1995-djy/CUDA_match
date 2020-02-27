#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
extern cv::Mat img1, img2;
extern cv::Mat img1_rectify, img2_rectify;
struct UV_XYZ
{
	cv::Mat uvcache;
	std::vector<cv::Point3f> XYZcache;
	//cv::Mat XYZcache = cv::Mat_<cv::Point3f>();
	//std::vector<cv::Point2f> uvcache;
};
extern struct UV_XYZ mylist;