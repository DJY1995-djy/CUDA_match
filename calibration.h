#pragma once
#include "pch.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <string>
extern cv::Mat cameraMatrix_F;
/* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
extern cv::Mat distCoeffs_F;
//void FindRT(cv::Mat& img0, cv::Mat& img1, cv::Mat& R, cv::Mat& T,cv::Mat& uvpix);
void FindRT(cv::Mat& img0, cv::Mat& img1, cv::Mat& R, cv::Mat& T);