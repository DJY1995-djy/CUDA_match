#include "pch.h"
#include <iostream>
#include <opencv2\opencv.hpp>
#include "vector"
using namespace std;
using namespace cv;
Mat flowdata;
int scale = 10;
std::vector<cv::Mat> channels_filt(3);
cv::Mat uvleft;   //// 记录uv坐标 
cv::Mat uvright;   //// 记录uv坐标

int flowmatch(cv::Mat& preframe,cv::Mat& frame,cv::Mat& uvmark)
{
	int flag=0;
	cv::Mat gray, pregray;
	/*cvtColor(preframe, pregray, CV_BGR2GRAY);   /// 灰度图  前一帧
	cvtColor(frame, gray, CV_BGR2GRAY);         /// 当前帧  灰度图*/
	pregray = preframe;
	gray = frame;
	///////////////// pregray 前一帧， gray 当前帧//////////////////
	calcOpticalFlowFarneback(pregray, gray, flowdata, 0.5, 3, 15, 3, 5, 1.5, 0);
	//cvtColor(pregray, preframe, CV_GRAY2BGR);
	//cout << "flow match completed" << endl;
	cv::Mat blank_ch = cv::Mat::zeros(cv::Size(gray.cols, gray.rows), CV_8UC1);  /// 空白列
	channels_filt[0] = pregray;    ////// R
	channels_filt[1] = blank_ch;   ////   G
	channels_filt[2] = gray;       ////   B
	Mat dstimg;
	cv::merge(channels_filt, dstimg);
	for (int row = 0; row < preframe.rows; row++)
	{
		for (int col = 0; col < preframe.cols; col++)
		{
			const Point2f fxy = flowdata.at<Point2f>(row, col);
			if (fxy.x > 2 || fxy.y > 2)
			{
				Point2i uvpre = Point(col, row);
				Point2i uvnow;
				uvnow.x = (int)(uvpre.x + fxy.x);
				uvnow.y = (int)(uvpre.y + fxy.y);
				if (flag % 1 == 0&uvnow.x > 0& uvnow.y > 0)
				{
					//uvmark.at<Point2i>(flag, 0) = uvpre;
					//uvmark.at<Point2i>(flag, 1) = uvnow;
					uvleft.push_back(uvpre);
					uvright.push_back(uvnow);
					//cout << uvleft << endl;
					cv::line(dstimg, uvpre, uvnow, cv::Scalar(0, 255, 0), 3, 4);// BGR
					circle(dstimg, uvpre, 2, Scalar(255, 0, 0), 2);   //BGR
					circle(dstimg, uvnow, 2, Scalar(0, 0, 255), 2);   //BGR
				}
				flag++;
			}
		}
	}
	cv::Mat uvtrans = Mat_<Point2i>(uvleft.rows, 2);
	//cout <<uvtrans.col(0) << endl;
	uvleft.col(0).copyTo(uvtrans.col(0));
	uvright.col(0).copyTo(uvtrans.col(1));
	/*uvleft.copyTo(uvtrans.col(0));
	uvright.copyTo(uvtrans.col(1));*/
	cout << "this" << endl;
	uvmark = uvtrans;
	//cout <<uvmark<< endl;
	cout << "flow work all done" << endl;
	cv::namedWindow("flowmatch",0);
	cv::resizeWindow("flowmatch",(int)(gray.cols/scale),(int)(gray.rows/scale));
	cv::imshow("flowmatch", dstimg);
	return 0;
}
