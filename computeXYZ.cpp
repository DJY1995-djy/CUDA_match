#include "pch.h"
#include <algorithm>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include "computeXYZ.h"
//�������ת����  
cv::Mat leftRotation    = (cv::Mat_<double>(3, 3) << 1, 0, 0,0,1,0,0,0,1);
//�����ƽ������  
cv::Mat leftTranslation = (cv::Mat_<double>(3, 1) << 0 , 0,  0);
using namespace std;
//using namespace cv;
cv::Mat leftIntrinsic, rightIntrinsic, rightRotation, rightTranslation;
/**/
/////////// ��С���˷�����XYZ ����Ϊ�������� �� ��u��v������ ��u��v��   point2f
cv::Mat computeXYZ(cv::Point2f uvLeft, cv::Point2f uvRight)
{
	cv::Mat mLeftRotation = cv::Mat_<double>(3, 3);
	mLeftRotation = leftRotation;
	cv::Mat mLeftTranslation = cv::Mat_<double>(3, 1);
	mLeftTranslation = leftTranslation;
	cv::Mat mLeftRT = cv::Mat_<double>(3, 4);//�����RT����
	hconcat(mLeftRotation, mLeftTranslation, mLeftRT);
	
	cv::Mat mLeftIntrinsic = cv::Mat_<double>(3, 3);
	mLeftIntrinsic=leftIntrinsic;   // Mat(3, 3, CV_32F, leftIntrinsic);
	cv::Mat mLeftM = mLeftIntrinsic * mLeftRT;
	//cout << "M��"<<mLeftM << endl;
	
	
	cv::Mat mRightRotation = rightRotation;       // Mat(3, 3, CV_32F, rightRotation);
	cv::Mat mRightTranslation = rightTranslation; // Mat(3, 1, CV_32F, rightTranslation);
	cv::Mat mRightRT = cv::Mat_<double>(3, 4);        //�����M����
	hconcat(mRightRotation, mRightTranslation, mRightRT);
	
	cv::Mat mRightIntrinsic = rightIntrinsic;     // Mat(3, 3, CV_32F, rightIntrinsic);
	cv::Mat mRightM = mRightIntrinsic * mRightRT;
	//cout<<"�����M���� = "<<mRightM<<endl;

	//��С���˷�A����
	cv::Mat A = cv::Mat_<double>(4, 3);
	A.at<double>(0, 0) = uvLeft.x * mLeftM.at<double>(2, 0) - mLeftM.at<double>(0, 0);
	A.at<double>(0, 1) = uvLeft.x * mLeftM.at<double>(2, 1) - mLeftM.at<double>(0, 1);
	A.at<double>(0, 2) = uvLeft.x * mLeftM.at<double>(2, 2) - mLeftM.at<double>(0, 2);

	A.at<double>(1, 0) = uvLeft.y * mLeftM.at<double>(2, 0) - mLeftM.at<double>(1, 0);
	A.at<double>(1, 1) = uvLeft.y * mLeftM.at<double>(2, 1) - mLeftM.at<double>(1, 1);
	A.at<double>(1, 2) = uvLeft.y * mLeftM.at<double>(2, 2) - mLeftM.at<double>(1, 2);

	A.at<double>(2, 0) = uvRight.x * mRightM.at<double>(2, 0) - mRightM.at<double>(0, 0);
	A.at<double>(2, 1) = uvRight.x * mRightM.at<double>(2, 1) - mRightM.at<double>(0, 1);
	A.at<double>(2, 2) = uvRight.x * mRightM.at<double>(2, 2) - mRightM.at<double>(0, 2);

	A.at<double>(3, 0) = uvRight.y * mRightM.at<double>(2, 0) - mRightM.at<double>(1, 0);
	A.at<double>(3, 1) = uvRight.y * mRightM.at<double>(2, 1) - mRightM.at<double>(1, 1);
	A.at<double>(3, 2) = uvRight.y * mRightM.at<double>(2, 2) - mRightM.at<double>(1, 2);

	//��С���˷�B����
	cv::Mat B = cv::Mat_<double>(4, 1);
	B.at<double>(0, 0) = mLeftM.at<double>(0, 3) - uvLeft.x * mLeftM.at<double>(2, 3);
	B.at<double>(1, 0) = mLeftM.at<double>(1, 3) - uvLeft.y * mLeftM.at<double>(2, 3);
	B.at<double>(2, 0) = mRightM.at<double>(0, 3) - uvRight.x * mRightM.at<double>(2, 3);
	B.at<double>(3, 0) = mRightM.at<double>(1, 3) - uvRight.y * mRightM.at<double>(2, 3);
	
	cv::Mat XYZ = cv::Mat_<double>(3, 1);
	//����SVD��С���˷����XYZ
	cv::solve(A, B, XYZ, cv::DECOMP_SVD);
	return XYZ;
}

