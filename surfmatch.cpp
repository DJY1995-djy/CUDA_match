#include "pch.h"
#include <iostream>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include "surfmatch.h"
using namespace std;
using namespace cv::xfeatures2d;
/****** surf ƥ�� vector<Point2f>********/
int scale = 3;   //��ʾ����
void surf_match(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& uvmark) {
	cv::cuda::GpuMat gmat1(3648, 5472, CV_8UC1);//����һ������ͼƬ�Ŀ�gpumat
	cv::cuda::GpuMat gmat2(3648, 5472, CV_8UC1);
	cv::cuda::GpuMat gmat01;
	cv::cuda::GpuMat gmat02;

	gmat01.upload(img1);
	gmat02.upload(img2);
	//ROI �鸳ֵ
	/*gmat01(cv::Rect(3087, 513, 1193, 1130)).copyTo(gmat1(cv::Rect(3087, 513, 1193, 1130)));
	gmat02(cv::Rect(2187, 493, 1311, 1130)).copyTo(gmat2(cv::Rect(2187, 493, 1311, 1130)));*/
	gmat01.copyTo(gmat1);
	gmat02.copyTo(gmat2);
	/*gmat1.download(imgdemo);

	cv::namedWindow("img1", 0);
	cv::resizeWindow("img1", (int)(img1.size().width / 10), (int)(img1.size().height / 10));
	cv::imshow("img1", img1);

	cv::namedWindow("demo", 0);
	cv::resizeWindow("demo", (int)(img1.size().width / 10), (int)(img1.size().height / 10));
	cv::imshow("demo", imgdemo);
	cv::waitKey(0);
	cv::destroyAllWindows();*/

	/*�������������ԭ���ǣ�
	explicit SURF_CUDA(double
		_hessianThreshold, //SURF��ɭ��������ֵ
		int _nOctaves=4, //�߶Ƚ���������
		int _nOctaveLayers=2, //ÿһ���߶Ƚ���������
		bool _extended=false, //���true��ô�õ�����������128ά��������64ά
		float _keypointsRatio=0.01f,
		bool _upright = false
		);
	Ҫ����⼸�������漰SURF��ԭ��*/
	cv::cuda::SURF_CUDA surf(10, 4, 3);

	/*�������漸��GpuMat�洢keypoint����Ӧ��descriptor*/
	cv::cuda::GpuMat keypt1, keypt2;
	cv::cuda::GpuMat desc1, desc2;

	/*���������*/
	surf(gmat1, cv::cuda::GpuMat(), keypt1, desc1);
	surf(gmat2, cv::cuda::GpuMat(), keypt2, desc2);

	/*ƥ�䣬�����ƥ�䲿�ֺ�CPU��matchû��̫������,�����½�һ��Brute-Force Matcher��һ��descriptor��L2����С��0.1����Ϊƥ��*/
	auto matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
	//vector< vector< DMatch> > match_vec;
	vector<cv::DMatch> match_vec;
	matcher->match(desc1, desc2, match_vec);
	/*int count = 0;
	for (auto & d : match_vec) {
		if (d.distance < 0.1)
			count++;
	}*/
	// downloading results  Gpu -> Cpu
	vector< cv::KeyPoint> keypoints1, keypoints2;
	vector< float> descriptors1, descriptors2;
	surf.downloadKeypoints(keypt1, keypoints1);
	surf.downloadKeypoints(keypt2, keypoints2);
	surf.downloadDescriptors(desc1, descriptors1);
	surf.downloadDescriptors(desc2, descriptors2);

	cout << "surf done" << endl;
	int ptcount = (int)match_vec.size();
	cv::Mat p1(ptcount, 2, CV_32F);
	cv::Mat p2(ptcount, 2, CV_32F);

	//change keypoint to mat
	cv::Point2f pt;
	vector<cv::Point2f> p01, p02;
	for (int i = 0; i < ptcount; i++)
	{
		pt = keypoints1[match_vec[i].queryIdx].pt;
		p01.push_back(pt);
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = keypoints2[match_vec[i].trainIdx].pt;
		p02.push_back(pt);
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}
	//use RANSAC to calculate F
	cv::Mat fundamental;
	vector <uchar> RANSACStatus;
	fundamental = findFundamentalMat(p1, p2, RANSACStatus, cv::FM_RANSAC, 10, 0.99);
	//����Ĵ���ΪRANSAC�Ż����������ƥ��Ч��
		//calculate the number of outliner
	int outlinerCount = 0;
	for (int i = 0; i < ptcount; i++)
	{
		if (RANSACStatus[i] == 0)
			outlinerCount++;
	}
	//calculate inLiner
	vector<cv::Point2f> inliner1, inliner2;
	vector<cv::DMatch> inlierMatches;
	int inlinerCount = ptcount - outlinerCount;
	inliner1.resize(inlinerCount);
	inliner2.resize(inlinerCount);
	inlierMatches.resize(inlinerCount);
	int inlinerMatchesCount = 0;
	for (int i = 0; i < ptcount; i++)
	{
		if (RANSACStatus[i] != 0){
			double diss = abs(p1.at<float>(i, 1) - p2.at<float>(i, 1));
			//cout <<"diss is: "<< diss << endl;   // ������С��100����������һ��ˮƽ�ߵĲ�����ȷƥ���
			if (diss < 100)
			{
				inliner1[inlinerMatchesCount].x = p1.at<float>(i, 0);
				inliner1[inlinerMatchesCount].y = p1.at<float>(i, 1);
				inliner2[inlinerMatchesCount].x = p2.at<float>(i, 0);
				inliner2[inlinerMatchesCount].y = p2.at<float>(i, 1);
				inlierMatches[inlinerMatchesCount].queryIdx = inlinerMatchesCount;
				inlierMatches[inlinerMatchesCount].trainIdx = inlinerMatchesCount;
				inlinerMatchesCount++;
			}
		}
	}
	inliner1.resize(inlinerMatchesCount);
	inliner2.resize(inlinerMatchesCount);
	inlierMatches.resize(inlinerMatchesCount);

	vector<cv::KeyPoint> key1(inlinerMatchesCount);
	vector<cv::KeyPoint> key2(inlinerMatchesCount);
	cv::KeyPoint::convert(inliner1, key1);
	cv::KeyPoint::convert(inliner2, key2);
	cv::Mat out;
	cv::Mat uvtrans = cv::Mat_<cv::Point2f>(inliner1.size(), 2);
	//vector<Point2f> p001, p002;
	for (int i = 0; i < inliner1.size(); i++)
	{
		uvtrans.at<cv::Point2f>(i, 0) = inliner1[i];
		//p001.push_back(inliner1[i]);
		uvtrans.at<cv::Point2f>(i, 1) = inliner2[i];
		//p002.push_back(inliner2[i]);
	}
	uvmark = uvtrans;
	//cout << uvmark << endl;
	drawMatches(img1, key1, img2, key2, inlierMatches, out);
	cv::namedWindow("good match result", 0);
	cv::resizeWindow("good match result", (int)(out.size().width / scale), (int)(out.size().height / scale));
	cv::imshow("good match result", out);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
