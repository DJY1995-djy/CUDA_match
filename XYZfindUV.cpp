#include "pch.h"
#include <algorithm>
#include "XYZfindUV.h"
cv::Mat newuv;
double Dthx = 1;
double Dthy = 1;
double Dthz = 1;
//////////////////////////
vector<cv::Point3d> wordnew; // 一个点的生长集
cv::Mat uv_new;  // 对应的像素坐标
//////////////////////////
// 储存新的生长点和对应的世界坐标系
struct  growdata
{
	cv::Mat growuv;
	cv::Mat field_8_left  = cv::Mat_<cv::Point2i>(3, 3);  // 8 邻域 ROI 左视图
	cv::Mat field_8_right = cv::Mat_<cv::Point2i>(3, 3);  // 8 邻域 ROI 右视图
	cv::Mat field_16_left = cv::Mat_<cv::Point2i>(4, 4);  // 16邻域 ROI 左视图
	cv::Mat field_16_right= cv::Mat_<cv::Point2i>(4, 4);  // 16邻域 ROI 右视图
	std::vector<cv::Point3f> growXYZ;    // 缓存生长的点云
};
growdata mynewdata;
void XYZsearchpixs::setXYZ(std::vector<cv::Point3f>& xyz_input) {
	XYZ = xyz_input;      //滤波后的三维坐标
}
void XYZsearchpixs::cleanXYZ() {
	//cout << "demo " << mylist.XYZcache[0] << endl;
	//cout << "uvcache " << mylist.uvcache.row(0) << endl;
	//遍历整个结构体   找到对应的XYZ的下标，用此索引去find出uv坐标  重新放入新的结构体使得XYZ ，uv 索引一致
	vector<cv::Point3f>::iterator it;
	for (int i = 0; i < XYZ.size(); i++) {
		it = find(mylist.XYZcache.begin(), mylist.XYZcache.end(), XYZ[i]);
		if (it != mylist.XYZcache.end()) {
			//cout << *it << endl;// 打印内容
			int index = &*it - &mylist.XYZcache[0];     //  计算索引
			newuv.push_back(mylist.uvcache.row(index)); //  XYZ对应newuv
			cout << "index: "<<index << endl;
		}
		else
			cout << "can not find" << endl;
	}
}
// 核心算法(auto one point grow) 
void XYZsearchpixs::pointgrow() {
	// 对每一个点 XYZ(i) 进行生长    
	for (int i = 0; i < XYZ.size(); i++) {
		cv::Point2i lefttemp, righttemp;
		lefttemp  = newuv.row(i).at<cv::Point2i>(0);                        // 左视图中的像素坐标
		righttemp = newuv.row(i).at<cv::Point2i>(1);                        // 右视图中的像素坐标
		int result0 = field_8_struct(lefttemp, mynewdata.field_8_left);     // 构造左视图的八邻域
		int result1 = field_8_struct(righttemp, mynewdata.field_8_right);   // 构造右视图的八邻域
		if (result0 != 0 & result1 != 0) {
			int flag0 = 1;  // 1 表示还能继续生长
			int flag1 = 1; 
			// 两层循环，外层是不遇到已有点集 继续生长，内层循环 处理邻域的对应点
			while (flag0!=0) {
				for (int m = 0; m < 8; m++) {
					lefttemp = mynewdata.field_8_left.at<cv::Point2i>(m);
					for (int n = 0; n < 8; n++) {
						righttemp= mynewdata.field_8_right.at<cv::Point2i>(n);
						cv::Mat XYZTEMP = computeXYZ(lefttemp, righttemp);
						double Dx = abs(XYZ[i].x-XYZTEMP.at<double>(0));   // 空间距离
						double Dy = abs(XYZ[i].y-XYZTEMP.at<double>(1));
						double Dz = abs(XYZ[i].z-XYZTEMP.at<double>(2));
						if (Dx < Dthx&Dy < Dthy&Dz < Dthz) {
							vector<cv::Point3d>::iterator it0;
							vector<cv::Point3f>::iterator it1;
							it0 = find(wordnew.begin(), wordnew.end(), XYZTEMP); // 在新生成的点集里查找重复
							it1 = find(XYZ.begin(), XYZ.end(), XYZTEMP);         // 在最初的点云里查找重复
							if (it0 != wordnew.end()) {
								continue;
							}
							else {
								cv::Mat uv_temp = cv::Mat_<cv::Point2i>(1, 2);
								wordnew.push_back(cv::Point3d(XYZTEMP.at<double>(0), XYZTEMP.at<double>(1), XYZTEMP.at<double>(2)));
								uv_temp.at<cv::Point2i>(0, 0) = lefttemp;
								uv_temp.at<cv::Point2i>(0, 1) = righttemp;
								uv_new.push_back(uv_temp);
							}															
						}		
					}
				}					
			}
		}
	}
}
// 构造8邻域  返回0 则没有八邻域   返回1 则构造8邻域
int XYZsearchpixs::field_8_struct(cv::Point2i& Pixels,cv::Mat& field_8) {
	//首先判断一个点是不是有8邻域边界点没有8邻域   不做计算
	cv::Size imagesize = cv::Size(img1.size().width,img1.size().height);   // 图片的分辨率
	if (Pixels.x == 0 || Pixels.x == imagesize.width) {
		return 0;
	}
	if (Pixels.y == 0 || Pixels.y == imagesize.height) {
		return 0;
	}
	// 构造
	cv::Mat field = cv::Mat_ <cv::Point2d >(8,1);//横向遍历
	field.at <cv::Point2d>(0)= cv::Point2d(Pixels.x-1,Pixels.y-1);
	field.at <cv::Point2d>(1)= cv::Point2d(Pixels.x,Pixels.y-1);
	field.at <cv::Point2d>(2)= cv::Point2d(Pixels.x+1,Pixels.y-1);
	field.at <cv::Point2d>(3)= cv::Point2d(Pixels.x-1,Pixels.y);
	field.at <cv::Point2d>(4)= cv::Point2d(Pixels.x+1,Pixels.y);
	field.at <cv::Point2d>(5)= cv::Point2d(Pixels.x-1,Pixels.y+1);
	field.at <cv::Point2d>(6)= cv::Point2d(Pixels.x,Pixels.y+1);
	field.at <cv::Point2d>(7)= cv::Point2d(Pixels.x+1,Pixels.y+1);
	field_8 = field;
	return 1;
}
