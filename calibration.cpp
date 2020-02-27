// calibration.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
#include "pch.h"
#include "read.h"
#include "calibration.h"
using namespace std;
/*内参数*/
cv::Mat cameraMatrix_F;
/* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
cv::Mat distCoeffs_F;
cv::Mat UVpix = cv::Mat_<cv::Point2f>(88,2);
//void FindRT(cv::Mat& img0, cv::Mat& img1, cv::Mat& R, cv::Mat& T,cv::Mat& uvpix)
void FindRT(cv::Mat& img0, cv::Mat& img1, cv::Mat& R, cv::Mat& T)
{
	int scale = 3;
	int ni = 1;
	cv::Mat R0, R1;
	cv::Mat img0_resize, img1_resize;
	/*棋盘三维信息*/
	cv::Size square_size = cv::Size(15, 15);    /* 实际测量得到的标定板上每个棋盘格的大小 */
	cv::Size board_size  = cv::Size(11, 8);    /* 标定板上每行、列的角点数 */
	cv::Mat RT0 = cv::Mat_<double>(4, 4);
	cv::Mat RT1 = cv::Mat_<double>(4, 4);
	cv::Mat RT  = cv::Mat_<double>(4, 4);
	//读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化	
	cout << "开始提取角点………………"<<endl;
	int  image_count = 2;  /* 图像数量   */
	cv::Size image_size;       /* 图像的尺寸 */
	vector<cv::Point2f> image_points_buf;  /* 缓存每幅图像上检测到的角点  */
	vector<vector<cv::Point2f>> image_points_seq; /* 保存检测到的所有角点 */
	image_size.width  = img0.cols;
	image_size.height = img0.rows;
	cout << "image_size.width  = " << image_size.width  << endl;
	cout << "image_size.height = " << image_size.height << endl;
	/* 提取 第一幅图片 角点 */
	cv::resize(img0,img0_resize, cv::Size(img0.cols/ni,img0.rows/ni));
	//cout << "resize:\n"<<img0_resize.size().width << endl;
	if (0 == findChessboardCorners(img0_resize, board_size, image_points_buf))
	{
		cout << "can not find chessboard corners!\n"; //找不到角点
		exit(1);
	}
	else
	{
		cv::Mat view_gray;
		img0.copyTo(view_gray);
		// 恢复坐标
		//cout << "压缩坐标:\n"<< image_points_buf << endl;
		for (int k = 0; k < image_points_buf.size(); k++)
		{
			image_points_buf[k] = ni * image_points_buf[k];
		}
		//cout << "原始坐标:\n" << image_points_buf << endl;
		/* 亚像素精确化 */
		find4QuadCornerSubpix(view_gray, image_points_buf, cv::Size(5, 5)); //对粗提取的角点进行精确化
		image_points_seq.push_back(image_points_buf);  //保存亚像素角点
		/* 在图像上显示角点位置 */
		drawChessboardCorners(view_gray, board_size, image_points_buf, true); //用于在图片中标记角点
		cv::namedWindow("Camera Calibration 0", 0);
		cv::resizeWindow("Camera Calibration 0",cv::Size((int)(image_size.width/scale),int(image_size.height/scale)));
		imshow("Camera Calibration 0", view_gray);//显示图片
		cv::imwrite("demo.jpg", view_gray);
	}
	/* 提取 第二幅图片 角点 */
	cv::resize(img1, img1_resize, cv::Size(img1.cols / ni, img1.rows / ni));
	if (0 == findChessboardCorners(img1_resize, board_size, image_points_buf))
	{
		cout << "can not find chessboard corners!\n"; //找不到角点
		exit(1);
	}
	else
	{
		cv::Mat view_gray;
		img1.copyTo(view_gray);
		// 恢复坐标
		for (int k = 0; k < image_points_buf.size(); k++)
		{
			image_points_buf[k] = ni * image_points_buf[k];
		}
		/* 亚像素精确化 */
		find4QuadCornerSubpix(view_gray, image_points_buf, cv::Size(5, 5)); //对粗提取的角点进行精确化
		image_points_seq.push_back(image_points_buf);  //保存亚像素角点
		///* 在图像上显示角点位置 */
		drawChessboardCorners(view_gray, board_size, image_points_buf, true); //用于在图片中标记角点
		cv::namedWindow("Camera Calibration 1", 0);
		cv::resizeWindow("Camera Calibration 1", cv::Size((int)(image_size.width / scale), int(image_size.height / scale)));
		cv::imshow("Camera Calibration 1", view_gray);//显示图片
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
	int total = image_points_seq.size();

	cout << "total = " <<total<< endl;
	int CornerNum = board_size.width*board_size.height;  //每张图片上总的角点数
	cout << "角点提取完成！" << endl;
	//以下是摄像机标定
	cout << "开始计算RT………………"<<endl;
	vector<vector<cv::Point3f>> object_points; /* 保存标定板上角点的三维坐标 */

	vector<int> point_counts;  // 每幅图像中角点的数量
	
	cv::Mat rvecsMat0, rvecsMat1;  /* 每幅图像的旋转向量 */
	cv::Mat tvecsMat0, tvecsMat1;  /* 每幅图像的平移向量 */
	/* 初始化标定板上角点的三维坐标 */
	int i, j, t;
	for (t = 0; t < image_count; t++)
	{
		vector<cv::Point3f> tempPointSet;
		for (i = 0; i < board_size.height; i++)
		{
			for (j = 0; j < board_size.width; j++)
			{
				cv::Point3f realPoint;
				/* 假设标定板放在世界坐标系中z=0的平面上 */
				realPoint.x = j * square_size.width;
				realPoint.y = i * square_size.height;
				realPoint.z = 0;
				/* 假设标定板放在世界坐标系中z平面成60°夹角 */
				/*realPoint.x = j * square_size.width / 2;
				realPoint.y = i * square_size.height;
				realPoint.z = j * square_size.width*0.8660254037844;*/
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	//cout << "世界坐标点：\n" << object_points[0] << endl;
    //保存角点坐标
	/*for (int i = 0; i < 88; i++)
	{
		UVpix.at<Point2f>(i, 0) = image_points_seq[0].at(i);
		UVpix.at<Point2f>(i, 1) = image_points_seq[1].at(i);
	}
	uvpix = UVpix;*/
	/* 开始标定 */
	//calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix_F, distCoeffs_F, rvecsMat, tvecsMat, 0);
	cv::solvePnPRansac(object_points[0], image_points_seq[0], cameraMatrix_F, distCoeffs_F, rvecsMat0, tvecsMat0, true, 1000, 1, 0.99);
	cv::solvePnPRansac(object_points[1], image_points_seq[1], cameraMatrix_F, distCoeffs_F, rvecsMat1, tvecsMat1, true, 1000, 1, 0.99);
	/* 将旋转向量转换为相对应的旋转矩阵 */
	Rodrigues(rvecsMat0, R0);
	Rodrigues(rvecsMat1, R1);
	R0 = R0.t();
	R = R1 * R0;
	T = tvecsMat1 - R * tvecsMat0;
	cout << "R: \n" << R << endl;
	cout << "T: \n" << T << endl;
}

