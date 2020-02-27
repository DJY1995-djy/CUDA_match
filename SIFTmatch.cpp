// solvePNP.cpp : ���ļ����� "main" ����������ִ�н��ڴ˴���ʼ��������
#include "pch.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <ctype.h>
#include <string>
#include "core/core.hpp"
#include <stdio.h>
#include "read.h"
#include "getuv.h"
#include "8points.h"
using namespace cv;
using namespace std;
using namespace xfeatures2d;
cv::Mat Intrinsic;
////////  m1 ��8�㡣 m2 ��8��///////////// 
void feature_sift(const cv::Mat& img1, const cv::Mat& img2,  cv::Mat& m1, cv::Mat& m2, vector <Point2f>& p01, vector <Point2f>& p02)
{
	//sift������ȡ
	Ptr<xfeatures2d::SiftFeatureDetector> detector = xfeatures2d::SiftFeatureDetector::create();

	vector<KeyPoint> keyPoint1, keyPoint2;
	detector->detect(img1, keyPoint1);
	detector->detect(img2, keyPoint2);
	cout << "Number of KeyPoint1:" << keyPoint1.size() << endl;
	cout << "Number of KeyPoint2:" << keyPoint2.size() << endl;

	//sift���������Ӽ���
	Ptr<xfeatures2d::SiftDescriptorExtractor> desExtractor = xfeatures2d::SiftDescriptorExtractor::create();
	Mat des1, des2;
	desExtractor->compute(img1, keyPoint1, des1);
	desExtractor->compute(img2, keyPoint2, des2);

	//sift������(������)ƥ��
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(des1, des2, matches);

	//nth_element(matches.begin(), matches.begin() + 19, matches.end());
	//matches.erase(matches.begin() + 19, matches.end());

	Mat img_match;
	drawMatches(img1, keyPoint1, img2, keyPoint2, matches, img_match);
	//imwrite("img_match.jpg", img_match);
	//����ƥ�����о������;�����Сֵ
	double min_dist = matches[0].distance, max_dist = matches[0].distance;
	for (int m = 0; m < matches.size(); m++)
	{
		if (matches[m].distance < min_dist)
		{
			min_dist = matches[m].distance;
		}
		if (matches[m].distance > max_dist)
		{
			max_dist = matches[m].distance;
		}
	}
	//cout << "min dist=" << min_dist << endl;
	//cout << "max dist=" << max_dist << endl;
	//ɸѡ���Ϻõ�ƥ���
	vector<DMatch> goodMatches;
	for (int m = 0; m < matches.size(); m++)
	{
		if (matches[m].distance < 0.6*max_dist)
		{
			goodMatches.push_back(matches[m]);
		}
	}
	cout << "The number of good matches:" << goodMatches.size() << endl;
	//����ƥ����
	Mat img_out;
	//��ɫ���ӵ���ƥ���������������ɫ���ӵ���δƥ�����������
	//matchColor �C Color of matches (lines and connected keypoints). If matchColor==Scalar::all(-1) , the color is generated randomly.
	//singlePointColor �C Color of single keypoints(circles), which means that keypoints do not have the matches.If singlePointColor == Scalar::all(-1), the color is generated randomly.
	//CV_RGB(0, 255, 0)�洢˳��ΪR-G-B,��ʾ��ɫ
	drawMatches(img1, keyPoint1, img2, keyPoint2, goodMatches, img_out, Scalar::all(-1), CV_RGB(0, 0, 255), Mat(), 2);
	//imshow("good Matches", img_out);

	//RANSACƥ�����
	vector<DMatch> m_Matches;
	m_Matches = goodMatches;
	int ptCount = goodMatches.size();
	if (ptCount < 100)
	{
		cout << "Don't find enough match points" << endl;
	}

	//����ת��Ϊfloat����
	vector <KeyPoint> RAN_KP1, RAN_KP2;
	//size_t�Ǳ�׼C���ж���ģ�ӦΪunsigned int����64λϵͳ��Ϊlong unsigned int,��C++��Ϊ����Ӧ��ͬ��ƽ̨�����ӿ���ֲ�ԡ�
	for (size_t i = 0; i < m_Matches.size(); i++)
	{
		RAN_KP1.push_back(keyPoint1[goodMatches[i].queryIdx]);
		RAN_KP2.push_back(keyPoint2[goodMatches[i].trainIdx]);
		//RAN_KP1��Ҫ�洢img01������img02ƥ��ĵ�
		//goodMatches�洢����Щƥ���Ե�img01��img02������ֵ
	}
	//����任
	for (size_t i = 0; i < m_Matches.size(); i++)
	{
		p01.push_back(RAN_KP1[i].pt);
		p02.push_back(RAN_KP2[i].pt);
	}
	vector<uchar> RansacStatus;

	//////////////// ��������////////////////////////////////////
	Mat Fundamental = findFundamentalMat(p01, p02, RansacStatus, FM_RANSAC);

	/*�ֽ�F���� Ϊ R T  */
	/*Mat E = Intrinsic.t()* Fundamental*Intrinsic;
	SVD svd(E);
	Mat W = Mat::eye(3, 3, CV_64FC1);
	W.at<double>(0, 1) = -1;
	W.at<double>(1, 0) = 1;
	W.at<double>(2, 2) = 1;
	Mat_<double> R = svd.u*W*svd.vt;
	Mat_<double> t = svd.u.col(2);
	cout << " API ֱ�Ӽ���R=" << R << endl;
	printf("\n");
	cout << " APIֱ�Ӽ��� t=" << t << endl;*/

	//���¶���ؼ���RR_KP��RR_matches���洢�µĹؼ���ͻ�������ͨ��RansacStatus��ɾ����ƥ���
	vector <KeyPoint> RR_KP1, RR_KP2;
	vector <DMatch> RR_matches;
	int index = 0;
	for (size_t i = 0; i < m_Matches.size(); i++)
	{
		if (RansacStatus[i] != 0)
		{
			RR_KP1.push_back(RAN_KP1[i]);
			RR_KP2.push_back(RAN_KP2[i]);
			m_Matches[i].queryIdx = index;
			m_Matches[i].trainIdx = index;
			RR_matches.push_back(m_Matches[i]);
			index++;
		}
	}
	cout << "RANSAC��ƥ�����" << RR_matches.size() << endl;
	Mat img_RR_matches;
	drawMatches(img1, RR_KP1, img2, RR_KP2, RR_matches, img_RR_matches);

	nth_element(RR_matches.begin(), RR_matches.begin() + 8, RR_matches.end());
	RR_matches.erase(RR_matches.begin() + 8, RR_matches.end());

	cv::Point2f ref_pt, cur_pt;
	//vector<Point2f> v1, v2;
	cv::Mat v1, v2;
	int idx = 0;
	while (idx < 8) 
	{
		ref_pt = RR_KP1[RR_matches[idx].queryIdx].pt;
		v1.push_back(ref_pt);
		cur_pt = RR_KP2[RR_matches[idx].trainIdx].pt;
		v2.push_back(cur_pt);
		idx++;
	}
    m1 = Mat(v1);
	m2 = Mat(v2);
}














//#include "pch.h"
//#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
//using namespace cv;
//using namespace xfeatures2d;
//using namespace std;
//void feature_sift(const cv::Mat& img1,const cv::Mat& img2,const cv::Mat& points_81 ,const cv::Mat& points_82)
//{
//	//sift������ȡ
//	SiftFeatureDetector detector;
//	//Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
//	vector<KeyPoint> keyPoint1, keyPoint2;
//	detector.detect(img1, keyPoint1);
//	detector.detect(img2, keyPoint2);
//	cout << "Number of KeyPoint1:" << keyPoint1.size() << endl;
//	cout << "Number of KeyPoint2:" << keyPoint2.size() << endl;
//
//	//sift���������Ӽ���
//	SiftDescriptorExtractor desExtractor;
//	Mat des1, des2;
//	desExtractor.compute(img1, keyPoint1, des1);
//	desExtractor.compute(img2, keyPoint2, des2);
//
//	//sift������(������)ƥ��
//	BFMatcher matcher(NORM_L2);
//	vector<DMatch> matches;
//	matcher.match(des1, des2, matches);
//	Mat img_match;
//	drawMatches(img1, keyPoint1, img2, keyPoint2, matches, img_match);
//	imshow("img_match", img_match);
//}