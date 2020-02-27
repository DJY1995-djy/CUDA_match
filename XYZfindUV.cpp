#include "pch.h"
#include <algorithm>
#include "XYZfindUV.h"
cv::Mat newuv;
double Dthx = 1;
double Dthy = 1;
double Dthz = 1;
//////////////////////////
vector<cv::Point3d> wordnew; // һ�����������
cv::Mat uv_new;  // ��Ӧ����������
//////////////////////////
// �����µ�������Ͷ�Ӧ����������ϵ
struct  growdata
{
	cv::Mat growuv;
	cv::Mat field_8_left  = cv::Mat_<cv::Point2i>(3, 3);  // 8 ���� ROI ����ͼ
	cv::Mat field_8_right = cv::Mat_<cv::Point2i>(3, 3);  // 8 ���� ROI ����ͼ
	cv::Mat field_16_left = cv::Mat_<cv::Point2i>(4, 4);  // 16���� ROI ����ͼ
	cv::Mat field_16_right= cv::Mat_<cv::Point2i>(4, 4);  // 16���� ROI ����ͼ
	std::vector<cv::Point3f> growXYZ;    // ���������ĵ���
};
growdata mynewdata;
void XYZsearchpixs::setXYZ(std::vector<cv::Point3f>& xyz_input) {
	XYZ = xyz_input;      //�˲������ά����
}
void XYZsearchpixs::cleanXYZ() {
	//cout << "demo " << mylist.XYZcache[0] << endl;
	//cout << "uvcache " << mylist.uvcache.row(0) << endl;
	//���������ṹ��   �ҵ���Ӧ��XYZ���±꣬�ô�����ȥfind��uv����  ���·����µĽṹ��ʹ��XYZ ��uv ����һ��
	vector<cv::Point3f>::iterator it;
	for (int i = 0; i < XYZ.size(); i++) {
		it = find(mylist.XYZcache.begin(), mylist.XYZcache.end(), XYZ[i]);
		if (it != mylist.XYZcache.end()) {
			//cout << *it << endl;// ��ӡ����
			int index = &*it - &mylist.XYZcache[0];     //  ��������
			newuv.push_back(mylist.uvcache.row(index)); //  XYZ��Ӧnewuv
			cout << "index: "<<index << endl;
		}
		else
			cout << "can not find" << endl;
	}
}
// �����㷨(auto one point grow) 
void XYZsearchpixs::pointgrow() {
	// ��ÿһ���� XYZ(i) ��������    
	for (int i = 0; i < XYZ.size(); i++) {
		cv::Point2i lefttemp, righttemp;
		lefttemp  = newuv.row(i).at<cv::Point2i>(0);                        // ����ͼ�е���������
		righttemp = newuv.row(i).at<cv::Point2i>(1);                        // ����ͼ�е���������
		int result0 = field_8_struct(lefttemp, mynewdata.field_8_left);     // ��������ͼ�İ�����
		int result1 = field_8_struct(righttemp, mynewdata.field_8_right);   // ��������ͼ�İ�����
		if (result0 != 0 & result1 != 0) {
			int flag0 = 1;  // 1 ��ʾ���ܼ�������
			int flag1 = 1; 
			// ����ѭ��������ǲ��������е㼯 �����������ڲ�ѭ�� ��������Ķ�Ӧ��
			while (flag0!=0) {
				for (int m = 0; m < 8; m++) {
					lefttemp = mynewdata.field_8_left.at<cv::Point2i>(m);
					for (int n = 0; n < 8; n++) {
						righttemp= mynewdata.field_8_right.at<cv::Point2i>(n);
						cv::Mat XYZTEMP = computeXYZ(lefttemp, righttemp);
						double Dx = abs(XYZ[i].x-XYZTEMP.at<double>(0));   // �ռ����
						double Dy = abs(XYZ[i].y-XYZTEMP.at<double>(1));
						double Dz = abs(XYZ[i].z-XYZTEMP.at<double>(2));
						if (Dx < Dthx&Dy < Dthy&Dz < Dthz) {
							vector<cv::Point3d>::iterator it0;
							vector<cv::Point3f>::iterator it1;
							it0 = find(wordnew.begin(), wordnew.end(), XYZTEMP); // �������ɵĵ㼯������ظ�
							it1 = find(XYZ.begin(), XYZ.end(), XYZTEMP);         // ������ĵ���������ظ�
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
// ����8����  ����0 ��û�а�����   ����1 ����8����
int XYZsearchpixs::field_8_struct(cv::Point2i& Pixels,cv::Mat& field_8) {
	//�����ж�һ�����ǲ�����8����߽��û��8����   ��������
	cv::Size imagesize = cv::Size(img1.size().width,img1.size().height);   // ͼƬ�ķֱ���
	if (Pixels.x == 0 || Pixels.x == imagesize.width) {
		return 0;
	}
	if (Pixels.y == 0 || Pixels.y == imagesize.height) {
		return 0;
	}
	// ����
	cv::Mat field = cv::Mat_ <cv::Point2d >(8,1);//�������
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
