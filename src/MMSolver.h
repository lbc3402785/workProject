#pragma once
#define TODO
#define errorcout cout
#define debugcout cout
#define CV_AA -1
#include "cnpy.h"
#include "NumpyUtil.h"


//using namespace cv;
inline cv::Point2f ImageCoordinate(MatF face, int i)
{
	float s = 2.0;
	float x = face(3 * i + 0) * s;
	float y = face(3 * i + 1) * s;

    cv::Point2f p(x, -y);
    cv::Point2f offset(250, 250);

	return p + offset;
}

inline cv::Point Shape(MatF A)
{
    return cv::Point(A.rows(), A.cols());
}

inline cv::Point Shape(MatI A)
{
    return cv::Point(A.rows(), A.cols());
}

inline Eigen::Matrix3f Orthogonalize(MatF R)
{
	// Set R to the closest orthonormal matrix to the estimated affine transform:
	Eigen::JacobiSVD<Eigen::Matrix3f, Eigen::NoQRPreconditioner> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3f U = svd.matrixU();
	const Eigen::Matrix3f V = svd.matrixV();
	Eigen::Matrix3f R_ortho = U * V.transpose();

	// The determinant of R must be 1 for it to be a valid rotation matrix
	if (R_ortho.determinant() < 0)
	{
		U.block<1, 3>(2, 0) = -U.block<1, 3>(2, 0); // not tested
		R_ortho = U * V.transpose();
	}

	return R_ortho;
}


inline Eigen::VectorXf SolveLinear(MatF A, MatF B, float lambda)
{
	//lambda = 1.0
	// Solve d[(Ax-b)^2 + lambda * x^2 ]/dx = 0 
		// https://math.stackexchange.com/questions/725185/minimize-a-x-b
	Eigen::MatrixXf Diagonal = Eigen::MatrixXf::Identity(A.cols(), A.cols()) * lambda;
	auto AA = A.transpose() * A + Diagonal;
	Eigen::VectorXf X = AA.colPivHouseholderQr().solve(A.transpose() * B);
	return X;
}

inline Eigen::VectorXf SolveLinear(MatF A, MatF B)
{
	Eigen::VectorXf X = A.colPivHouseholderQr().solve(B);
	return X;
}




class ProjectionParameters
{
public:
	Eigen::Matrix3f R; ///< 3x3 rotation matrix
	float tx, ty; ///< x and y translation
	float s;      ///< Scaling

	//Need Transpose
    cv::Mat GenerateCVProj()
	{
		Matrix3f Rt = R.transpose() * s;

        cv::Mat P = cv::Mat::eye(4, 4, CV_32F);

		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				P.at<float>(i, j) = Rt(i, j);
			}
		}

		P.at<float>(0, 3) = tx * s;
		P.at<float>(1, 3) = ty * s;

		return P;
	}
};


inline MatF Projection(ProjectionParameters p, MatF model_points)
{
	MatF R = p.R * p.s;
    R = R.block(0, 0, 3, 2);

    MatF rotated = model_points * R;

	auto sTx = p.tx * p.s;
	auto sTy = p.ty * p.s;

	int N = model_points.rows();
	rotated.col(0).array() += sTx;
	rotated.col(1).array() += sTy;

	return rotated;// .block(0, 0, N, 2);
}

inline MatF Rotation(ProjectionParameters p, MatF model_points)
{
	MatF R = p.R;
	R = R.block(0, 0, 3, 3);
	MatF rotated = model_points * R;
	return rotated;// .block(0, 0, N, 2);
}


inline MatF ToEigen(std::vector<cv::Point> image_points)
{
	int N = image_points.size();

	MatF b(N, 2);
	for (int i = 0; i < N; ++i)
	{
		Eigen::Vector2f p = Eigen::Vector2f(image_points[i].x, image_points[i].y);
		b.block<1, 2>(i, 0) = p;
	}

	return b;
}

inline Matrix<float, 1, Eigen::Dynamic> GetMean(MatF &input)
{
	return input.colwise().mean();
}

inline void SubtractMean(MatF &input)
{
	input.rowwise() -= GetMean(input);
}





class FaceModel
{
public:
	MatF SM;
    MatF SB;//204x199 PCA BASIS
    MatF EM;
    MatF EB;//204x100 PCA BASIS

	MatI TRI;
	MatI TRIUV;
	MatI Ef;
	MatI Ev;

	MatF UV;

    MatF Face;//68x3
	MatF GeneratedFace;
	Matrix<float, 1, Eigen::Dynamic> Mean;

	FaceModel()
	{

	}

	void Initialize(string file, bool LoadEdge)
	{
		cnpy::npz_t npz = cnpy::npz_load(file);

		SM = ToEigen(npz["SM"]);
		SB = ToEigen(npz["SB"]);

		EM = ToEigen(npz["EM"]);
		EB = ToEigen(npz["EB"]);

		MatF FaceFlat = SM + EM; // 204 * 1 
		Face = Reshape(FaceFlat, 3);

		/*Mean = GetMean(Face);
		Face.rowwise() -= Mean; */

		if (LoadEdge)
		{
			TRI = ToEigenInt(npz["TRI"]);

			try
			{
				TRIUV = ToEigenInt(npz["TRIUV"]);
			}
			catch (...)
			{
				TRIUV = TRI;
			}
			Ef = ToEigenInt(npz["Ef"]);
			Ev = ToEigenInt(npz["Ev"]);

			UV = ToEigen(npz["UV"]);
		}
	}


	MatF Generate(MatF SX, MatF EX)
	{
        MatF FaceS = SB * SX;
        MatF S = Reshape(FaceS, 3);

        MatF FaceE = EB * EX;
        MatF E = Reshape(FaceE, 3);

        GeneratedFace =  Face + S + E;
		return GeneratedFace;
	}
};




class MMSolver
{
public:
	bool USEWEIGHT = true;
	float WEIGHT = 1.0;
	//vector<int> SkipList = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  11, 12, 13, 14, 15, 16 };
	//vector<int> SkipList = { 0, 1, 2, 3, 4,   5, 6, 7,    9,10,  11, 12,  13, 14, 15, 16 };
	vector<int> SkipList = {8, };


	FaceModel FM;
	FaceModel FMFull;
	bool FIXFIRSTSHAPE = true;

	void Initialize(string file, string file2)
	{
		FM.Initialize(file, false);
		//FMFull.Initialize(file2, FM.Mean, true);
		FMFull.Initialize(file2, true);


		if (FIXFIRSTSHAPE)
		{
			FM.SB.col(0) *= 0;
			FMFull.SB.col(0) *= 0;
		}
	}

	MatF SolveShape(ProjectionParameters p, MatF image_points, MatF M, MatF SB, float lambda)
	{
		//cout << Shape(SB) << endl;

		MatF R = p.R * p.s;
		R = R.block(0, 0, 3, 2);

		MatF rotated = Projection(p, M);

		MatF error = image_points - rotated;
		error = Reshape(error, 1);

		int N = M.rows();
		int N2 = SB.rows();
		int L = SB.cols();

		assert(N2 == N * 3);

		MatF SBX(N * 2, L);
		MatF Rt = R.transpose();
		for (size_t i = 0; i < N; i++)
		{
			SBX.block(i * 2, 0, 2, L) = Rt * SB.block(i * 3, 0, 3, L);
		}

		if (USEWEIGHT)
		{
			Matrix<float, Eigen::Dynamic, 1> W = Matrix<float, Eigen::Dynamic, 1>::Ones(2 * N, 1);
			for (size_t i = 0; i < SkipList.size(); i++)
			{
				W(2 * SkipList[i] + 0, 0) = WEIGHT;
				W(2 * SkipList[i] + 1, 0) = WEIGHT;
			}
			SBX = W.asDiagonal() * SBX;
			error = W.asDiagonal() * error;
		}

		auto X = SolveLinear(SBX, error, lambda);

		//cout << (error - SBX * X).norm() << endl;

		return X;
		//MatF rotated = (model_points + Ax) * R;
	}

	ProjectionParameters SolveProjection(MatF image_points, MatF model_points)
	{
		//########## Mean should be subtracted from model_points ############

		Matrix<float, 1, Eigen::Dynamic> Mean = GetMean(model_points);
		model_points.rowwise() -= Mean;
		

		using Eigen::Matrix;
		int N = image_points.rows();

		assert(image_points.rows() == model_points.rows());
		assert(2 == image_points.cols());
		assert(3 == model_points.cols());

		model_points.conservativeResize(N, 4);
		model_points.col(3).setOnes();

		Matrix<float, Eigen::Dynamic, 8> A = Matrix<float, Eigen::Dynamic, 8>::Zero(2 * N, 8);
		for (int i = 0; i < N; ++i)
		{
			Eigen::Vector4f P = model_points.row(i);// .transpose();//Eigen::Vector4f();
			A.block<1, 4>(2 * i, 0) = P;       // even row - copy to left side (first row is row 0)
			A.block<1, 4>((2 * i) + 1, 4) = P; // odd row - copy to right side
		} // 4th coord (homogeneous) is already 1


		/*Matrix<float, 1, Eigen::Dynamic> Mean = image_points.colwise().mean();
		image_points.rowwise() -= Mean;*/

		MatF b = Reshape(image_points, 1);

		if (USEWEIGHT)
		{
			Matrix<float, Eigen::Dynamic, 1> W = Matrix<float, Eigen::Dynamic, 1>::Ones(2 * N, 1);
			for (size_t i = 0; i < SkipList.size(); i++)
			{
				W(2 * SkipList[i] + 0, 0) = WEIGHT;
				W(2 * SkipList[i] + 1, 0) = WEIGHT;
			}
			A = W.asDiagonal() * A;
			b = W.asDiagonal() * b;
		}

		const Matrix<float, 8, 1> k = SolveLinear(A, b); // resulting affine matrix (8x1)

		// Extract all values from the estimated affine parameters k:
		const Eigen::Vector3f R1 = k.segment<3>(0);
		const Eigen::Vector3f R2 = k.segment<3>(4);
		Eigen::Matrix3f R;
		Eigen::Vector3f r1 = R1.normalized(); // Not sure why R1.normalize() (in-place) produces a compiler error.
		Eigen::Vector3f r2 = R2.normalized();
		R.block<1, 3>(0, 0) = r1;
		R.block<1, 3>(1, 0) = r2;
		R.block<1, 3>(2, 0) = r1.cross(r2);

		float sTx = k(3);
		float sTy = k(7);

		//sTx += Mean(0);
		//sTy += Mean(1);

		const auto s = (R1.norm() + R2.norm()) / 2.0f;



		Eigen::Matrix3f R_ortho = Orthogonalize(R);

		MatF T = Mean * R_ortho;
		// Remove the scale from the translations:
		const auto t1 = sTx / s - T(0);
		const auto t2 = sTy / s - T(1);

		auto error = (A*k - b).norm();


		return ProjectionParameters{ R_ortho, t1, t2, s };
	}

	MatF SX;
	MatF EX;
	ProjectionParameters params;

	bool FixShape = false;
	MatF SX0;
	

	

	void Solve(MatF KP)
	{
		MatF Face = FM.Face;
		MatF S = Face * 0;
		MatF E = Face * 0;


		float Lambdas[4] = { 100.0, 30.0, 10.0, 5.0 };


		for (size_t i = 0; i < 4; i++)
		{
			params = SolveProjection(KP, Face);

			if (FixShape)
			{
				SX = SX0;
			}
			else
			{
				SX = SolveShape(params, KP, FM.Face + E, FM.SB, Lambdas[i] * 5);
				if (FIXFIRSTSHAPE)
				{
					SX(0, 0) = 0;
				}
			}
			MatF FaceS = FM.SB * SX;
			S = Reshape(FaceS, 3);

			EX = SolveShape(params, KP, FM.Face + S, FM.EB, Lambdas[i] * 1);
			
			MatF FaceE = FM.EB * EX;
			E = Reshape(FaceE, 3);

			Face = FM.Face + S + E;

			
		}

	}
};





inline cv::Mat MMSDraw(cv::Mat orig, MMSolver &MMS, MatF &KP)
{

	auto params = MMS.params;
	auto Face2 = MMS.FMFull.Generate(MMS.SX, MMS.EX);
	MatF projected = Projection(params, Face2);

	auto image = orig.clone();
	auto image2 = orig.clone();

	auto Ev = MMS.FMFull.Ev;

	for (size_t i = 0; i < Ev.rows(); i++)
	{
		int i1 = Ev(i, 0);
		int i2 = Ev(i, 1);

		auto x = projected(i1, 0);
		auto y = projected(i1, 1);

		auto x2 = projected(i2, 0);
		auto y2 = projected(i2, 1);

        line(image, cv::Point(x, y), cv::Point(x2, y2), cv::Scalar(0, 0, 255, 255), 1);
		//image.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
		//circle(image, Point(x, y), 1, Scalar(0, 0, 255), -1);
	}


	auto Face = MMS.FM.Generate(MMS.SX, MMS.EX);
	projected = Projection(params, Face);

	for (size_t i = 0; i < projected.rows(); i++)
	{
		auto x = projected(i, 0);
		auto y = projected(i, 1);
        circle(image, cv::Point(x, y), 2, cv::Scalar(255, 0, 0, 255), -1, CV_AA);
	}

	if (MMS.USEWEIGHT)
	{
		for (size_t i = 0; i < MMS.SkipList.size(); i++)
		{
			int i2 = MMS.SkipList[i];
			auto x = projected(i2, 0);
			auto y = projected(i2, 1);
            circle(image, cv::Point(x, y), 6, cv::Scalar(255, 0, 0, 255), 1, CV_AA);
		}
	}

	for (size_t i = 0; i < KP.rows(); i++)
	{
		auto x = KP(i, 0);
		auto y = KP(i, 1);

        circle(image, cv::Point(x, y), 2, cv::Scalar(0, 255, 0, 255), -1, CV_AA);
	}

	//imshow("IMG", image / 2 + image2 / 2);
	return image / 2 + image2 / 2;
}





// Apply affine transform calculated using srcTri and dstTri to src
inline void applyAffineTransform(cv::Mat &warpImage, cv::Mat &src, std::vector<cv::Point2f> &srcTri, std::vector<cv::Point2f> &dstTri)
{
	// Given a pair of triangles, find the affine transform.
    cv::Mat warpMat = getAffineTransform(srcTri, dstTri);

	// Apply the Affine Transform just found to the src image
    warpAffine(src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR,cv::BORDER_REFLECT_101);
}


// Warps and alpha blends triangular regions from img1 and img2 to img
inline void warpTriangle(cv::Mat &img1, cv::Mat &img2, vector<cv::Point2f> &t1, vector<cv::Point2f> &t2)
{
	TODO // Need to make sure rect is in Mat.
	try
	{
        cv::Rect r1 = boundingRect(t1);
        cv::Rect r2 = boundingRect(t2);

		//cout << r1 << r2 << endl;
		// Offset points by left top corner of the respective rectangles
        vector<cv::Point2f> t1Rect, t2Rect;
        vector<cv::Point> t2RectInt;
		for (int i = 0; i < 3; i++)
		{

            t1Rect.push_back(cv::Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
            t2Rect.push_back(cv::Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
            t2RectInt.push_back(cv::Point(t2[i].x - r2.x, t2[i].y - r2.y)); // for fillConvexPoly

		}

		// Get mask by filling triangle
        cv::Mat mask = cv::Mat::zeros(r2.height, r2.width, CV_8UC3);
        fillConvexPoly(mask, t2RectInt, cv::Scalar(1.0, 1.0, 1.0), 8, 0);

		// Apply warpImage to small rectangular patches
        cv::Mat img1Rect;
		img1(r1).copyTo(img1Rect);

        cv::Mat img2Rect = cv::Mat::zeros(r2.height, r2.width, img1Rect.type());

		applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);

		multiply(img2Rect, mask, img2Rect);
        multiply(img2(r2), cv::Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
		img2(r2) = img2(r2) + img2Rect;
	}
	catch (const std::exception& e) { 
		std::cout << e.what(); 
		//fillConvexPoly(img2, t2, Scalar(0, 0, 255), 16, 0);
	}
	


}


inline cv::Mat MMSTexture(cv::Mat orig, MMSolver &MMS, int W, int H, bool doubleface = true)
{
	auto image = orig.clone();
    cv::Mat texture = cv::Mat::zeros(H, W, CV_8UC3);

	auto params = MMS.params;
	auto Face2 = MMS.FMFull.Generate(MMS.SX, MMS.EX);
	MatF projected = Projection(params, Face2);


	auto TRI = MMS.FMFull.TRIUV;
	auto UV = MMS.FMFull.UV;

	for (size_t t = 0; t < TRI.rows(); t++)
	{

        vector<cv::Point2f> t1;
        vector<cv::Point2f> t2;
		for (size_t i = 0; i < 3; i++)
		{
			int j = TRI(t, i);
			auto x = projected(j, 0);
			auto y = projected(j, 1);

			auto u = (UV(j, 0)) * (W - 1);
			auto v = (1 - UV(j, 1)) * (H - 1);
            t1.push_back(cv::Point2f(x, y));
            t2.push_back(cv::Point2f(u, v));

			//cout << Point2f(x, y) << Point2f(u, v) << endl;
		}

		auto c = (t1[2] - t1[0]).cross(t1[1] - t1[0]);
		
		if (doubleface || c > 0)
		{
			warpTriangle(image, texture, t1, t2);
		}
	}

	return texture;

}


inline cv::Mat MMSNormal(cv::Mat orig, MMSolver &MMS, int W, int H)
{
	auto image = orig.clone();
    cv::Mat texture = cv::Mat::zeros(H, W, CV_8UC3);

	auto params = MMS.params;
	auto Face2 = MMS.FMFull.Generate(MMS.SX, MMS.EX);
	Face2 = Rotation(params, Face2);

	auto TRI = MMS.FMFull.TRIUV;
	auto UV = MMS.FMFull.UV;

	for (size_t t = 0; t < TRI.rows(); t++)
	{

        std::vector<cv::Point3f> t1;
        std::vector<cv::Point> t2;
		for (size_t i = 0; i < 3; i++)
		{
			int j = TRI(t, i);
			auto x = Face2(j, 0);
			auto y = Face2(j, 1);
			auto z = Face2(j, 2);

			auto u = (UV(j, 0)) * (W - 1);
			auto v = (1.0 - UV(j, 1)) * (H - 1);
            t1.push_back(cv::Point3f(x, y, z));
            t2.push_back(cv::Point(u, v));

			//cout << Point2f(x, y) << Point2f(u, v) << endl;
		}

        cv::Point3f c = (t1[2] - t1[0]).cross(t1[1] - t1[0]);
		auto normal = c / norm(c);

		/*if (c.z > 0)
		{
			float n = ((c.z / norm(c))) * 255;

			fillConvexPoly(texture, t2, Scalar(n,n,n), 8, 0);
		}*/

		{
            fillConvexPoly(texture, t2, cv::Scalar(normal.x + 1.0, normal.y + 1.0, normal.z + 1.0) * 128, 8, 0);
		}

	}

	return texture;

}



#include <sstream>
#include <fstream>  
#include "cnpy.h"


inline void FMObj(cv::Mat texture, FaceModel &FM, string folder, string filename0)
{
	string filename = folder + filename0;
	//Mat texture = MMSTexture(orig, MMS, 1024, 1024);
	imwrite(filename + ".png", texture);

	//cout << Shape(MMS.SX) << Shape(MMS.EX) << endl;
	auto Face = FM.GeneratedFace;
	auto TRI = FM.TRI;
	auto TRIUV = FM.TRIUV;
	auto UV = FM.UV;

	//string numpyfile = filename + ".npz";


	//cnpy::npz_save(numpyfile, "SX", MMS.SX.data(), { (unsigned long long)MMS.SX.rows(), (unsigned long long)MMS.SX.cols() }, "w"); //"w" overwrites any existing file
	//cnpy::npz_save(numpyfile, "EX", MMS.EX.data(), { (unsigned long long)MMS.EX.rows(), (unsigned long long)MMS.EX.cols() }, "a"); //"a" appends to the file we created above

	{
		std::stringstream ss;


		ss << "mtllib " << filename0 << ".mtl" << endl;
		ss << "o FaceObject" << endl;

		int N = Face.rows();
		for (size_t i = 0; i < N; i++)
		{
			ss << "v " << Face(i, 0) << " " << Face(i, 1) << " " << Face(i, 2) << endl;
		}

		N = UV.rows();
		for (size_t i = 0; i < N; i++)
		{
			ss << "vt " << UV(i, 0) << " " << UV(i, 1) << endl;
		}

		ss << "usemtl material_0" << endl;
		ss << "s 1" << endl;

		N = TRI.rows();
		for (size_t i = 0; i < N; i++)
		{
			ss << "f " << TRI(i, 0) + 1 << "/" << TRIUV(i, 0) + 1 << " "
				<< TRI(i, 1) + 1 << "/" << TRIUV(i, 1) + 1 << " "
				<< TRI(i, 2) + 1 << "/" << TRIUV(i, 2) + 1 << " "
				<< endl;
		}


		std::string input = ss.str();

		std::ofstream out(filename + ".obj", std::ofstream::out);
		out << input;
		out.close();
	}

	{
		std::stringstream ss;


		ss << "mtllib " << filename0 << ".mtl" << endl;
		ss << "o FaceObject" << endl;

		ss << "newmtl material_0" << endl;
		ss << "	Ns 0.000000" << endl;
		ss << "Ka 0.200000 0.200000 0.200000" << endl;
		ss << "Kd 0.639216 0.639216 0.639216" << endl;
		ss << "Ks 1.000000 1.000000 1.000000" << endl;
		ss << "Ke 0.000000 0.000000 0.000000" << endl;
		ss << "Ni 1.000000" << endl;
		ss << "d 1.000000" << endl;
		ss << "illum 2" << endl;
		ss << "map_Kd " << filename0 + ".png" << endl;



		std::string input = ss.str();

		std::ofstream out(filename + ".mtl", std::ofstream::out);
		out << input;
		out.close();
	}


}


inline void MMSObj(cv::Mat orig, MMSolver &MMS, string folder, string filename0)
{
	string filename = folder + filename0;
    cv::Mat texture = MMSTexture(orig, MMS, 1024,1024);
	imwrite(filename + ".png", texture);

	auto Face = MMS.FMFull.Generate(MMS.SX, MMS.EX);
	auto TRI = MMS.FMFull.TRI;
	auto TRIUV = MMS.FMFull.TRIUV;
	auto UV = MMS.FMFull.UV;

	string numpyfile = filename + ".npz";
	

	cnpy::npz_save(numpyfile, "SX", MMS.SX.data(), { (unsigned long long)MMS.SX.rows(), (unsigned long long)MMS.SX.cols() }, "w"); //"w" overwrites any existing file
	cnpy::npz_save(numpyfile, "EX", MMS.EX.data(), { (unsigned long long)MMS.EX.rows(), (unsigned long long)MMS.EX.cols() }, "a"); //"a" appends to the file we created above

	{
		std::stringstream ss;


		ss << "mtllib " << filename0 << ".mtl" << endl;
		ss << "o FaceObject" << endl;

		int N = Face.rows();
		for (size_t i = 0; i < N; i++)
		{
			ss << "v " << Face(i, 0) << " " << Face(i, 1) << " " << Face(i, 2) << endl;
			ss << "vt " << UV(i, 0) << " " << UV(i, 1) << endl;
		}

		ss << "usemtl material_0" << endl;
		ss << "s 1" << endl;

		N = TRI.rows();
		for (size_t i = 0; i < N; i++)
		{
			ss << "f " << TRI(i, 0) + 1 << "/" << TRIUV(i, 0) + 1 << " "
				<< TRI(i, 1) + 1 << "/" << TRIUV(i, 1) + 1 << " "
				<< TRI(i, 2) + 1 << "/" << TRIUV(i, 2) + 1 << " "
				<< endl;
		}


		std::string input = ss.str();

		std::ofstream out(filename + ".obj", std::ofstream::out);
		out << input;
		out.close();
	}

	{
		std::stringstream ss;


		ss << "mtllib " << filename0 << ".mtl" << endl;
		ss << "o FaceObject" << endl;

		ss << "newmtl material_0" << endl;
		ss << "	Ns 0.000000" << endl;
		ss << "Ka 0.200000 0.200000 0.200000" << endl;
		ss << "Kd 0.639216 0.639216 0.639216" << endl;
		ss << "Ks 1.000000 1.000000 1.000000" << endl;
		ss << "Ke 0.000000 0.000000 0.000000" << endl;
		ss << "Ni 1.000000" << endl;
		ss << "d 1.000000" << endl;
		ss << "illum 2" << endl;
		ss << "map_Kd " << filename0 + ".png" << endl;



		std::string input = ss.str();

		std::ofstream out(filename + ".mtl", std::ofstream::out);
		out << input;
		out.close();
	}


}


