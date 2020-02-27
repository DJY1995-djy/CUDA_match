// calibration.cpp : ���ļ����� "main" ����������ִ�н��ڴ˴���ʼ��������
#include "pch.h"
#include "read.h"
#include "calibration.h"
using namespace std;
/*�ڲ���*/
cv::Mat cameraMatrix_F;
/* �������5������ϵ����k1,k2,p1,p2,k3 */
cv::Mat distCoeffs_F;
cv::Mat UVpix = cv::Mat_<cv::Point2f>(88,2);
//void FindRT(cv::Mat& img0, cv::Mat& img1, cv::Mat& R, cv::Mat& T,cv::Mat& uvpix)
void FindRT(cv::Mat& img0, cv::Mat& img1, cv::Mat& R, cv::Mat& T)
{
	int scale = 3;
	int ni = 1;
	cv::Mat R0, R1;
	cv::Mat img0_resize, img1_resize;
	/*������ά��Ϣ*/
	cv::Size square_size = cv::Size(15, 15);    /* ʵ�ʲ����õ��ı궨����ÿ�����̸�Ĵ�С */
	cv::Size board_size  = cv::Size(11, 8);    /* �궨����ÿ�С��еĽǵ��� */
	cv::Mat RT0 = cv::Mat_<double>(4, 4);
	cv::Mat RT1 = cv::Mat_<double>(4, 4);
	cv::Mat RT  = cv::Mat_<double>(4, 4);
	//��ȡÿһ��ͼ�񣬴�����ȡ���ǵ㣬Ȼ��Խǵ���������ؾ�ȷ��	
	cout << "��ʼ��ȡ�ǵ㡭����������"<<endl;
	int  image_count = 2;  /* ͼ������   */
	cv::Size image_size;       /* ͼ��ĳߴ� */
	vector<cv::Point2f> image_points_buf;  /* ����ÿ��ͼ���ϼ�⵽�Ľǵ�  */
	vector<vector<cv::Point2f>> image_points_seq; /* �����⵽�����нǵ� */
	image_size.width  = img0.cols;
	image_size.height = img0.rows;
	cout << "image_size.width  = " << image_size.width  << endl;
	cout << "image_size.height = " << image_size.height << endl;
	/* ��ȡ ��һ��ͼƬ �ǵ� */
	cv::resize(img0,img0_resize, cv::Size(img0.cols/ni,img0.rows/ni));
	//cout << "resize:\n"<<img0_resize.size().width << endl;
	if (0 == findChessboardCorners(img0_resize, board_size, image_points_buf))
	{
		cout << "can not find chessboard corners!\n"; //�Ҳ����ǵ�
		exit(1);
	}
	else
	{
		cv::Mat view_gray;
		img0.copyTo(view_gray);
		// �ָ�����
		//cout << "ѹ������:\n"<< image_points_buf << endl;
		for (int k = 0; k < image_points_buf.size(); k++)
		{
			image_points_buf[k] = ni * image_points_buf[k];
		}
		//cout << "ԭʼ����:\n" << image_points_buf << endl;
		/* �����ؾ�ȷ�� */
		find4QuadCornerSubpix(view_gray, image_points_buf, cv::Size(5, 5)); //�Դ���ȡ�Ľǵ���о�ȷ��
		image_points_seq.push_back(image_points_buf);  //���������ؽǵ�
		/* ��ͼ������ʾ�ǵ�λ�� */
		drawChessboardCorners(view_gray, board_size, image_points_buf, true); //������ͼƬ�б�ǽǵ�
		cv::namedWindow("Camera Calibration 0", 0);
		cv::resizeWindow("Camera Calibration 0",cv::Size((int)(image_size.width/scale),int(image_size.height/scale)));
		imshow("Camera Calibration 0", view_gray);//��ʾͼƬ
		cv::imwrite("demo.jpg", view_gray);
	}
	/* ��ȡ �ڶ���ͼƬ �ǵ� */
	cv::resize(img1, img1_resize, cv::Size(img1.cols / ni, img1.rows / ni));
	if (0 == findChessboardCorners(img1_resize, board_size, image_points_buf))
	{
		cout << "can not find chessboard corners!\n"; //�Ҳ����ǵ�
		exit(1);
	}
	else
	{
		cv::Mat view_gray;
		img1.copyTo(view_gray);
		// �ָ�����
		for (int k = 0; k < image_points_buf.size(); k++)
		{
			image_points_buf[k] = ni * image_points_buf[k];
		}
		/* �����ؾ�ȷ�� */
		find4QuadCornerSubpix(view_gray, image_points_buf, cv::Size(5, 5)); //�Դ���ȡ�Ľǵ���о�ȷ��
		image_points_seq.push_back(image_points_buf);  //���������ؽǵ�
		///* ��ͼ������ʾ�ǵ�λ�� */
		drawChessboardCorners(view_gray, board_size, image_points_buf, true); //������ͼƬ�б�ǽǵ�
		cv::namedWindow("Camera Calibration 1", 0);
		cv::resizeWindow("Camera Calibration 1", cv::Size((int)(image_size.width / scale), int(image_size.height / scale)));
		cv::imshow("Camera Calibration 1", view_gray);//��ʾͼƬ
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
	int total = image_points_seq.size();

	cout << "total = " <<total<< endl;
	int CornerNum = board_size.width*board_size.height;  //ÿ��ͼƬ���ܵĽǵ���
	cout << "�ǵ���ȡ��ɣ�" << endl;
	//������������궨
	cout << "��ʼ����RT������������"<<endl;
	vector<vector<cv::Point3f>> object_points; /* ����궨���Ͻǵ����ά���� */

	vector<int> point_counts;  // ÿ��ͼ���нǵ������
	
	cv::Mat rvecsMat0, rvecsMat1;  /* ÿ��ͼ�����ת���� */
	cv::Mat tvecsMat0, tvecsMat1;  /* ÿ��ͼ���ƽ������ */
	/* ��ʼ���궨���Ͻǵ����ά���� */
	int i, j, t;
	for (t = 0; t < image_count; t++)
	{
		vector<cv::Point3f> tempPointSet;
		for (i = 0; i < board_size.height; i++)
		{
			for (j = 0; j < board_size.width; j++)
			{
				cv::Point3f realPoint;
				/* ����궨�������������ϵ��z=0��ƽ���� */
				realPoint.x = j * square_size.width;
				realPoint.y = i * square_size.height;
				realPoint.z = 0;
				/* ����궨�������������ϵ��zƽ���60��н� */
				/*realPoint.x = j * square_size.width / 2;
				realPoint.y = i * square_size.height;
				realPoint.z = j * square_size.width*0.8660254037844;*/
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	//cout << "��������㣺\n" << object_points[0] << endl;
    //����ǵ�����
	/*for (int i = 0; i < 88; i++)
	{
		UVpix.at<Point2f>(i, 0) = image_points_seq[0].at(i);
		UVpix.at<Point2f>(i, 1) = image_points_seq[1].at(i);
	}
	uvpix = UVpix;*/
	/* ��ʼ�궨 */
	//calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix_F, distCoeffs_F, rvecsMat, tvecsMat, 0);
	cv::solvePnPRansac(object_points[0], image_points_seq[0], cameraMatrix_F, distCoeffs_F, rvecsMat0, tvecsMat0, true, 1000, 1, 0.99);
	cv::solvePnPRansac(object_points[1], image_points_seq[1], cameraMatrix_F, distCoeffs_F, rvecsMat1, tvecsMat1, true, 1000, 1, 0.99);
	/* ����ת����ת��Ϊ���Ӧ����ת���� */
	Rodrigues(rvecsMat0, R0);
	Rodrigues(rvecsMat1, R1);
	R0 = R0.t();
	R = R1 * R0;
	T = tvecsMat1 - R * tvecsMat0;
	cout << "R: \n" << R << endl;
	cout << "T: \n" << T << endl;
}

