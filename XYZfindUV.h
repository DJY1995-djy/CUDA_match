#pragma once
#include "pch.h"
#include "computeXYZ.h"    // 最小二乘计算XYZ的方法
#include "solvePNP.h"      // 传递保存XYZ和 pixs的结构体   mylist 
#include <iostream>
using namespace std;
// 用来清洗原生态世界坐标。
class XYZsearchpixs
{
public:
	void setXYZ(std::vector<cv::Point3f>& xyz_input); // 传递滤波点云坐标
	void cleanXYZ();                 // 清洗点云坐标  从初始的点云结构体里找到滤波过的点云和其像素坐标
	void pointgrow();                // 点云生长
	int  field_8_struct(cv::Point2i& Pixels, cv::Mat& field_8);           // 用一个像素坐标构造8邻域
private:
	std::vector<cv::Point3f> XYZ;   
};
// 点生长
