#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <fstream>
#define ATD at<double>
using namespace cv;
using namespace std;
//����֡ͼƬ���Ix
Mat get_Ix(Mat &src1, Mat &src2)
{
	Mat Ix;
	Mat kernal = Mat::ones(2, 2, CV_64FC1);
	kernal.ATD(0, 0) = -1.0;
	kernal.ATD(1, 0) = -1.0;

	Mat dst1, dst2;
	filter2D(src1, dst1, -1, kernal);
	filter2D(src2, dst2, -1, kernal);

	Ix = dst1 + dst2;
	return Ix;
}
//����֡ͼƬ���Iy
Mat get_Iy(Mat &src1, Mat &src2)
{
	Mat Iy;
	Mat kernal = Mat::ones(2, 2, CV_64FC1);
	kernal.ATD(0, 0) = -1.0;
	kernal.ATD(0, 1) = -1.0;

	Mat dst1, dst2;
	filter2D(src1, dst1, -1, kernal);
	filter2D(src2, dst2, -1, kernal);

	Iy = dst1 + dst2;
	return Iy;
}
//����֡ͼƬ���It
Mat get_It(Mat &src1, Mat &src2)
{
	Mat It;
	Mat kernal = Mat::ones(2, 2, CV_64FC1);
	kernal = kernal.mul(-1);

	Mat dst1, dst2;
	filter2D(src1, dst1, -1, kernal);
	kernal = kernal.mul(-1);
	filter2D(src2, dst2, -1, kernal);

	It = dst1 + dst2;
	return It;
}

//ȡ3*3�Ĵ��ڣ���9��ֵ�ĺ�
Mat get_sum9(Mat &m)
{
	Mat sum9;
	Mat kernal = Mat::ones(3, 3, CV_64FC1);

	filter2D(m, sum9, -1, kernal);
	return sum9;
}

//LK�㷨ʵ��
//����:��֡ͼƬ img1��img2
//���:������ u��x�����������v��y���������
void getLucasKanadeOpticalFlow(Mat &img1, Mat &img2, Mat &u, Mat &v)
{
	Mat Ix = get_Ix(img1, img2);
	Mat Iy = get_Iy(img1, img2);
	Mat It = get_It(img1, img2);
	Mat Ix2 = Ix.mul(Ix);
	Mat Iy2 = Iy.mul(Iy);
	Mat IxIy = Ix.mul(Iy);
	Mat IxIt = Ix.mul(It);
	Mat IyIt = Iy.mul(It);
	Mat Ix2_sum9 = get_sum9(Ix2);
	Mat Iy2_sum9 = get_sum9(Iy2);
	Mat IxIy_sum9 = get_sum9(IxIy);
	Mat IxIt_sum9 = get_sum9(IxIt);
	Mat IyIt_sum9 = get_sum9(IyIt);
	Mat det = (Ix2_sum9.mul(Iy2_sum9) - IxIy_sum9.mul(IxIy_sum9)) * 10;//A������ʽ���㣨���ף�������*10��Ϊ�˹����޷�
	u = IxIy_sum9.mul(IyIt_sum9) - Iy2_sum9.mul(IxIt_sum9);//���u*det
	v = IxIy_sum9.mul(IxIt_sum9) - Ix2_sum9.mul(IyIt_sum9);//���v*det
	divide(u, det, u);//���u
	divide(v, det, v);//���v
}

//��һ��ͼƬimg�ϸ���u��v����������
void draw_optical_flow(Mat &img, Mat &u, Mat &v)
{
	int width = img.cols;//y
	int height = img.rows;//x

	int newi, newj;//����������������
	for (int i = 0; i < height; i++)//����x
	{
		for (int j = 0; j < width; j++)//����y
		{
			newi = i + int(u.ATD(i, j));//�������֮�����x����
			newj = j + int(v.ATD(i, j));//�������֮�����y����

			if (newi >= 0 && newi < height && newj >= 0 && newj < width)//�Ա߽�������ƣ���ֹ�ڴ����
			{
				if ((u.ATD(i, j) + v.ATD(i, j) > 2) && (u.ATD(i, j) + v.ATD(i, j) < 40))//�����˲�������2-40
				{
					circle(img, Point(newj, newi), 1, Scalar(0, 0, 255));//��������㻭Բ���뾶Ϊ1����ɫ��ɫ
					line(img, Point(j, i), Point(newj, newi), Scalar(0, 0, 255));//������֮�仭�ߣ���ɫ��ɫ
				}
			}

		}
	}
}
void main()
{
	//��ǰ����֡ͼƬ������
	Mat img1 = imread("frame08.png", 0);//������ͼƬ���ƣ�0�ԻҶȷ�ʽ����
	Mat img2 = imread("frame09.png", 0);

	img1.convertTo(img1, CV_64FC1, 1.0 / 255, 0);//�����imshow()�Ļ�����Ҫд��1.0/255
	img2.convertTo(img2, CV_64FC1, 1.0 / 255, 0);//�����imwrite()�Ļ�����Ҫд��1�� ����Ӱ��ʵ�ʹ���������

	Mat u = Mat::zeros(img1.rows, img1.cols, CV_64FC1);
	Mat v = Mat::zeros(img1.rows, img1.cols, CV_64FC1);
	getLucasKanadeOpticalFlow(img1, img2, u, v);
	imshow("u", u);
	imshow("v", v);

	Mat img = imread("frame08.png");//���¶����ɫͼƬ
	draw_optical_flow(img, u, v);//������ͼƬ�ϻ����������
	imshow("optical_flowflow", img);
	imwrite("optical_flow.jpg", img);
	waitKey(0);
}
