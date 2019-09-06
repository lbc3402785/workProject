#pragma once

#include <Eigen/Dense>

using namespace Eigen;

typedef Eigen::Matrix<float, 3, 3, RowMajor> Mat33;


inline Mat33 EulerAngleToRotation(float x, float y, float z)
{
	// Calculate rotation about x axis
	Mat33 R_x;
	R_x <<
		1, 0, 0,
		0, cos(x), -sin(x),
		0, sin(x), cos(x)
		;

	// Calculate rotation about y axis
	Mat33 R_y;
	R_y  <<
		cos(y), 0, sin(y),
		0, 1, 0,
		-sin(y), 0, cos(y)
		;

	// Calculate rotation about z axis
	Mat33 R_z;
	R_z <<
		cos(z), -sin(z), 0,
		sin(z), cos(z), 0,
		0, 0, 1;

	// Combined rotation matrix
	Mat33 R = R_z * R_y * R_x;

	return R;

}