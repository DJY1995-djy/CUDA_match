/*read image from the path ,to get the worldpoint's coordinate ( u,v) */
/*the points are four black mark*/
#include "pch.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctype.h>
#include <string>
void uvmaker(std::string& imgfile,cv::Mat& point)
{
	cv::Mat points = cv::Mat_<double>(4, 2);
	std::string  filepath= "G:\\ZEDphotostest\\L\\L1.jpg";
	cv::Mat img = cv::imread(filepath);
	cv::namedWindow("img",1);
	cv::resizeWindow("img", (int)(2204 / 3), (int)(1242 / 3));
	cv::imshow("img",img);
	/*第一个点*/
	points.at<double>(0, 0) = 0;    //// u
	points.at<double>(0, 1) = 1;    //// v
	/*第二个点*/
	points.at<double>(1, 0) = 0;
	points.at<double>(1, 1) = 1;
	/*第三个点*/
	points.at<double>(2, 0) = 0;
	points.at<double>(2, 1) = 1;
	/*第四个点*/
	points.at<double>(3, 0) = 0;
	points.at<double>(3, 1) = 1;
}
