// solvePNP.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//#pragma comment(lib, "User32.lib")
//#pragma comment(lib, "gdi32.lib")
#include "pch.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <ctype.h>
#include <string>
#include "core/core.hpp"
#include <stdio.h>
#include "read.h"
#include "computeXYZ.h"
#include "surfmatch.h"
#include "calibration.h"
#include "solvePNP.h"
#include "XYZfindUV.h"
#include <iostream>

#include <pcl/common/common_headers.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
using namespace std;
/****************step0: 4张3次重建 ,从视频文件夹中读取    *****************/
/****************step1: 棋盘检测solve 计算RT    ****************/
/****************step2：光流稠密匹配每两帧之间的对应点     ****************/
/****************step3: 用最小二乘求解稠密对应点的3维坐标  ****************/
/****************step4：利用每两帧之间的RT矩阵统一9次重建的坐标系**********/

/*********** 全局变量定义 bump **************/

//保存像素坐标和世界坐标的结构体 其成员声明在头文件中
UV_XYZ mylist;
cv::Mat img1, img2;
cv::Mat img1_rectify, img2_rectify;
/********************************************/
/********************* txt文本 write function ***********************************/
static void saveXYZ(const char* filename, std::vector<cv::Point3f>& xyz , int num)
{
	//cout <<"txt size： "<< xyz.size() << endl;
	FILE* fp = fopen(filename, "w");
	for (int i = 0; i < num; i++)
	{
		fprintf(fp, "%f;%f;%f\n", xyz[i].x, xyz[i].y, xyz[i].z);
	}
	fclose(fp);
}
/**********************************************************************/
/********************   main 函数    **********************************/
int main()
{
	//////////////////camera  intristic///////////////////////
	cv::Mat cameraMatrix = cv::Mat::eye(3, 3, cv::DataType<double>::type);  //相机内参矩阵
	double *p;
	string path = "interc1.txt";
	p = get_para(path);
	int n = 0;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
		{
			cameraMatrix.at<double>(i, j) = *(p + n);
			n++;
		}
	leftIntrinsic  = cameraMatrix;
	rightIntrinsic = cameraMatrix;
	cameraMatrix_F = cameraMatrix;
	std::cout << "内参:\n" << cameraMatrix << endl;
	cv::Mat distCoeffs(5, 1, cv::DataType<double>::type);  //畸变参数
	path = "distort1.txt";
	p = get_para(path);
	n = 0;
	for (int i = 0; i < 5; i++)
	{
		distCoeffs.at<double>(i) = *(p + n);
		n++;
	}
	distCoeffs_F = distCoeffs;
	std::cout << "畸变系数:\n" << distCoeffs << endl;
	cv::Mat img0= cv::imread("E:\\py\\ZEDsingle\\Leftcamera\\1.bmp",0); //Read Img_gray   
	cv::Mat view, rview, map1, map2;
	cv::Size imageSize;
	imageSize = img0.size();
	/*计算校正矩阵 map1 map2*/
	initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),
		getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
		imageSize, CV_16SC2, map1, map2);


	/**************数据定义 bump **************/
	cv::Mat XYZ = cv::Mat_<double>(3,1);
	cv::Mat XYZ1= cv::Mat_<double>(4,1);   ///// 储存XYZ齐次坐标
	XYZ1.at<double>(3) = 1;           ///// 齐次构造         
	cv::Mat imgtemp;
	vector <cv::Mat> Ri(3);
	vector <cv::Mat> Ti(3);
	vector <cv::Point2f> p1, p2;
	cv::Mat R_temp, t_temp;
	bool main_ret;
	int RTsize;
	int pixel_num=0;                   /////// 统计所有重建出来的点数
	//std::vector<vector<Point3f>> xyz_txt;
	std::vector<cv::Point3f> XYZmat;
	std::vector<cv::Point3f> XYZ_txt;
	int succeed = 0;
	cv::Mat R_n;
	/***************************************************************************/
	/*************起始图像index为1 已经预读取。现在从第二帧开始******************/
	/* process */
	/***** 顺序读入图片，相邻重建。失败：img1继续成为前帧 成功:img2成为前帧*****/
	/*每轮中处理的是 img1 img2 */
	//img0.copyTo(img1);
	/*************************  star   ********************************/
 	for (int i = 2; i < 3; i++)
	{
		if (i == 2)
			img0.copyTo(img1);
		else
			imgtemp.copyTo(img1);
		string filepath = "E:\\py\\ZEDsingle\\Leftcamera\\";
		filepath += to_string(i);
		filepath += ".bmp";
		img2 = cv::imread(filepath,0);    /// 读取灰度图
		/*solvePnP 计算RT*/
		cv::Mat R, t;
		FindRT(img1, img2, R, t);
		//cout<<"succeed get uv"<<uvpixels<<endl;
		R.copyTo(rightRotation);
		t.copyTo(rightTranslation);
		/*缓存RT*/
		Ri[i-2]=R;
		Ti[i-2]=t; 
		/*************光流稠密匹配  找到对应点*************/
		//cv::Mat uvpixels;
		//int ret = flowmatch(img1,img2,uvpixels);
		/**************************************************/
		/*surf 稀疏匹配*/
		int scale = 3;
		remap(img1, img1_rectify, map1, map2, cv::INTER_LINEAR);   //校正
		remap(img2, img2_rectify, map1, map2, cv::INTER_LINEAR);   //校正
		//undistort(img1, img1_rectify, cameraMatrix, distCoeffs);
		//undistort(img2, img2_rectify, cameraMatrix, distCoeffs);

		//cv::namedWindow("undistortion 0", 0);
		//cv::resizeWindow("undistortion 0", cv::Size((int)(img1_rectify.cols / scale), int(img1_rectify.rows / scale)));
		//cv::imshow("undistortion 0", img1_rectify);//显示图片
		//cv::waitKey(0);
		//cv::destroyAllWindows();
		cv::Mat uvpixels;
		//FindRT(img1_rectify, img2_rectify, R_temp, t_temp, uvpixels);
		surf_match(img1_rectify, img2_rectify, uvpixels);
		//cout << uvpixels << endl;
		/*********************计算世界坐标****************************/
		//cv::Mat XYZmat = Mat_<double>(uvpixels.size().height, 3);
		
		//*利用 [u,v] 计算 XYZ *///
		//RTsize = Ri.size();
		// 进入一轮重建的所有匹配点的遍历
		for (int counter = 0; counter < uvpixels.rows; counter++)
		{
			cv::Point3f XYZpoint;
			XYZ = computeXYZ(uvpixels.at<cv::Point2f>(counter, 0), uvpixels.at<cv::Point2f>(counter, 1));
			if(XYZ.at<double>(2)>0 & XYZ.at<double>(2)<1500)
			{	
				/*RT只有一组重建只一次 直接保存XYZ*/
				if (i==2)
				{
					XYZpoint.x = XYZ.at<double>(0);
					XYZpoint.y = XYZ.at<double>(1);
					XYZpoint.z = XYZ.at<double>(2);
					//cout << uvpixels.rowRange(counter, counter+2) << endl;
					//cout << uvpixels.row(counter) << endl;
					XYZmat.push_back(XYZpoint);
					///// 保存左右视图的(u,v)和对应的XYZ

					mylist.uvcache.push_back(uvpixels.row(counter).clone());
					mylist.XYZcache.push_back(XYZpoint);
				}
				/*用缓存的RT 归一化坐标系*/
				else
				{	
					/*坐标系迭代*/
					for (int j = (i-3); j>=0; j--)
					{
						R_n = Ri[j].t();
						XYZ = R_n * (XYZ - Ti[j]);
					}
					XYZpoint.x = XYZ.at<double>(0);
					XYZpoint.y = XYZ.at<double>(1);
					XYZpoint.z = XYZ.at<double>(2);
					XYZmat.push_back(XYZpoint);
				}
			}	
		}
		//cout <<"XYZ demo; "<< XYZmat << endl;
		//pixel_num += XYZmat.size();
		succeed += 1;
		//xyz_txt.push_back(XYZmat);
		std::cout<<"succeed : " <<succeed<<" time restruct!!!"<<endl;
		img2.copyTo(imgtemp);
	}
	std::cout << "重建总点数：" << XYZmat.size() << endl;
	// 构造 PCL点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < XYZmat.size(); i++) {
		cloud->push_back(pcl::PointXYZ(XYZmat[i].x, XYZmat[i].y, XYZmat[i].z));
	}
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud);
	sor.setMeanK(10);
	sor.setStddevMulThresh(1.0);
	sor.filter(*cloud_filtered);
	//pcl::PointCloud<pcl::PointXYZ> cloud_mat;
	//cloud_mat = *cloud; 
	// 点云转化为vector
	for (int i = 0; i < cloud_filtered->size(); i++) {
		cv::Point3f XYZ_trans;
		XYZ_trans.x = cloud_filtered.get()->at(i).x;
		XYZ_trans.y = cloud_filtered.get()->at(i).y;
		XYZ_trans.z = cloud_filtered.get()->at(i).z;
		XYZ_txt.push_back(XYZ_trans);   // XYZ_txt 阔以用来保存为文本和进行下一步点云生长处理
	}
	// 匹配XYZ的对应坐标  构成数据体。
	XYZsearchpixs search;   // 创建实例对象
	search.setXYZ(XYZ_txt); // 传入滤波后的XYZ
	search.cleanXYZ();      // 引用方法成员清洗初始的XYZ（找到滤波后的点云的像素索引）
	//pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");//直接创造一个显示窗口
	//viewer.showCloud(cloud_filtered);//窗口显示点云
	//while (!viewer.wasStopped()) {}
	//保存滤波后的点云
	//saveXYZ("xyz.txt", XYZ_txt, XYZ_txt.size());
	return 0;
}
	