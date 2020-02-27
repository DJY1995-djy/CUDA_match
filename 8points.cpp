#include "pch.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <ctype.h>
#include <string>
#include "core/core.hpp"
#include <stdio.h>
using namespace std;
int run8Point( cv::Mat& _m1, cv::Mat& _m2, cv::Mat& _fmatrix)
{
	std::cout << _m1.size << std::endl;
	std::cout << _m2.size << std::endl;
	cv::Point2d m1c, m2c;
	double t, scale1 = 0, scale2 = 0;
	const cv::Point2f* m1 = _m1.ptr<cv::Point2f>();
	const cv::Point2f* m2 = _m2.ptr<cv::Point2f>();
	//CV_Assert((_m1.cols == 1 || _m1.rows == 1) && _m1.size() == _m2.size());
	int count = _m1.checkVector(2);
	int i;
	std::cout<<"count is :"<<count<<std::endl;
	// compute centers and average distances for each of the two point sets
	for (i = 0; i < count; i++)
	{
		m1c += cv::Point2d(m1[i]);
		m2c += cv::Point2d(m2[i]);
	}
	// calculate the normalizing transformations for each of the point sets:
	// after the transformation each set will have the mass center at the coordinate origin
	// and the average distance from the origin will be ~sqrt(2).
	t = 1. / count;
	m1c *= t;
	m2c *= t;

	for (i = 0; i < count; i++)
	{
		scale1 += cv::norm(cv::Point2d(m1[i].x - m1c.x, m1[i].y - m1c.y));
		scale2 += cv::norm(cv::Point2d(m2[i].x - m2c.x, m2[i].y - m2c.y));
	}

	scale1 *= t;
	scale2 *= t;

	if (scale1 < FLT_EPSILON || scale2 < FLT_EPSILON)
		return 0;
	scale1 = std::sqrt(2.) / scale1;
	scale2 = std::sqrt(2.) / scale2;
	cv::Matx<double, 9, 9> A;
	// form a linear system Ax=0: for each selected pair of points m1 & m2,
	// the row of A(=a) represents the coefficients of equation: (m2, 1)'*F*(m1, 1) = 0
	// to save computation time, we compute (At*A) instead of A and then solve (At*A)x=0.
	for (i = 0; i < count; i++)
	{
		double x1 = (m1[i].x - m1c.x)*scale1;
		double y1 = (m1[i].y - m1c.y)*scale1;
		double x2 = (m2[i].x - m2c.x)*scale2;
		double y2 = (m2[i].y - m2c.y)*scale2;
		cv::Vec<double, 9> r(x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1);
		A += r * r.t();
	}

	cv::Vec<double, 9> W;
	cv::Matx<double, 9, 9> V;
	cv::eigen(A, W, V);

	for (i = 0; i < 9; i++)
	{
		if (fabs(W[i]) < DBL_EPSILON)
			break;
	}

	if (i < 8)
		return 0;

	cv::Matx33d F0(V.val + 9 * 8); // take the last column of v as a solution of Af = 0

	// make F0 singular (of rank 2) by decomposing it with SVD,
	// zeroing the last diagonal element of W and then composing the matrices back.

	cv::Vec3d w;
	cv::Matx33d U;
	cv::Matx33d Vt;

	cv::SVD::compute(F0, w, U, Vt);
	w[2] = 0.;
	F0 = U * cv::Matx33d::diag(w)*Vt;

	// apply the transformation that is inverse
	// to what we used to normalize the point coordinates
	cv::Matx33d T1(scale1, 0, -scale1 * m1c.x, 0, scale1, -scale1 * m1c.y, 0, 0, 1);
	cv::Matx33d T2(scale2, 0, -scale2 * m2c.x, 0, scale2, -scale2 * m2c.y, 0, 0, 1);

	F0 = T2.t()*F0*T1;

	// make F(3,3) = 1
	if (fabs(F0(2, 2)) > FLT_EPSILON)
		F0 *= 1. / F0(2, 2);
	cv::Mat(F0).copyTo(_fmatrix);/////////// F matrix
	//std::cout <<"子函数调用结果："<< _fmatrix << std::endl;
	return 1;
}