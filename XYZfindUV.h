#pragma once
#include "pch.h"
#include "computeXYZ.h"    // ��С���˼���XYZ�ķ���
#include "solvePNP.h"      // ���ݱ���XYZ�� pixs�Ľṹ��   mylist 
#include <iostream>
using namespace std;
// ������ϴԭ��̬�������ꡣ
class XYZsearchpixs
{
public:
	void setXYZ(std::vector<cv::Point3f>& xyz_input); // �����˲���������
	void cleanXYZ();                 // ��ϴ��������  �ӳ�ʼ�ĵ��ƽṹ�����ҵ��˲����ĵ��ƺ�����������
	void pointgrow();                // ��������
	int  field_8_struct(cv::Point2i& Pixels, cv::Mat& field_8);           // ��һ���������깹��8����
private:
	std::vector<cv::Point3f> XYZ;   
};
// ������
