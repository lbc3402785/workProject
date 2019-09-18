#ifndef MMTENSORSOLVER_H
#define MMTENSORSOLVER_H
#define CV_AA -1
#include "cnpy.h"
#include "NumpyUtil.h"
#include "common/dataconvertor.h"
#include <torch/torch.h>
inline torch::Tensor Orthogonalize(torch::Tensor R)
{
//    std::cout<<"R:" << R << std::endl;
    // Set R to the closest orthonormal matrix to the estimated affine transform:
    torch::Tensor U,S,V;
    std::tie(U,S,V)=torch::svd(R);
//    std::cout<<"U:" << U << std::endl;
//    std::cout<<"V:" << V << std::endl;
    torch::Tensor R_ortho = torch::matmul(U , V.transpose(0,1));

    // The determinant of R must be 1 for it to be a valid rotation matrix
    if (R_ortho.det().item().toFloat() < 0)
    {
        U.select(0,2)=  -U.select(0,2);
        R_ortho = U * V.transpose(0,1);
    }

    return std::move(R_ortho);
}
inline torch::Tensor SolveLinear(torch::Tensor A, torch::Tensor B, float lambda)
{
    //lambda = 1.0
    // Solve d[(Ax-b)^2 + lambda * x^2 ]/dx = 0
    // https://math.stackexchange.com/questions/725185/minimize-a-x-b
//    MatF MA=EigenToTorch::TorchTensorToEigenMatrix(A);
//    MatF MB=EigenToTorch::TorchTensorToEigenMatrix(B);
//    MatF Diagonal = Eigen::MatrixXf::Identity(MA.cols(), MA.cols()) * lambda;
//    auto AA = MA.transpose() * MA + Diagonal;
//    MatF X = AA.colPivHouseholderQr().solve(MA.transpose() * MB);
//    return EigenToTorch::EigenMatrixToTorchTensor(X);
    torch::Tensor W=torch::matmul(A.transpose(0,1),A)+torch::eye(A.size(1))*lambda;
    torch::Tensor b=torch::matmul(A.transpose(0,1),B);
    torch::Tensor X,QR;
    std::tie(X,QR)=torch::gels(b,W);
    return std::move(X);
}
inline torch::Tensor SolveLinear(torch::Tensor A, torch::Tensor B)
{

//    MatF MA=EigenToTorch::TorchTensorToEigenMatrix(A);
//    MatF MB=EigenToTorch::TorchTensorToEigenMatrix(B);
//    std::cout << MB.row(0) << std::endl;
//        std::cout << MB.row(1) << std::endl;
//    MatF X = MA.colPivHouseholderQr().solve(MB);
//    return EigenToTorch::EigenMatrixToTorchTensor(X);
    torch::Tensor X,QR;
    std::tie(X,QR)=torch::gels(B,A);
    return std::move(X);
}



class ProjectionTensor
{
public:
    torch::Tensor R;
    float tx,ty;
    float s;
};
inline torch::Tensor Projection(ProjectionTensor p, torch::Tensor model_points)
{
    torch::Tensor R = p.R.transpose(0,1) * p.s;
    R = R.slice(1,0,2);

    torch::Tensor rotated =torch::matmul( model_points, R);

    auto sTx = p.tx * p.s;
    auto sTy = p.ty * p.s;

    rotated.select(1,0)+= sTx;
    rotated.select(1,1)+= sTy;

    return std::move(rotated);
}
inline torch::Tensor Rotation(ProjectionTensor p, torch::Tensor model_points)
{
    torch::Tensor R = p.R;
    R = R.slice(1,0,2);
    torch::Tensor rotated =torch::matmul( model_points, R);
    return std::move(rotated);// .block(0, 0, N, 2);
}
inline torch::Tensor GetMean(torch::Tensor &input)
{
    return std::move(input.mean(0,true));
}
class FaceModelTensor
{
public:
    torch::Tensor SM;
    torch::Tensor SB;
    torch::Tensor EM;
    torch::Tensor EB;

    torch::Tensor TRI;
    torch::Tensor TRIUV;
    torch::Tensor Ef;
    torch::Tensor Ev;

    torch::Tensor UV;

    torch::Tensor Face;
    torch::Tensor GeneratedFace;
    torch::Tensor Mean;//1XN

    FaceModelTensor()
    {

    }

    void Initialize(string file, bool LoadEdge)
    {
        cnpy::npz_t npz = cnpy::npz_load(file);

        SM = ToTensor(npz["SM"]);

        SB = ToTensor(npz["SB"]);

        EM = ToTensor(npz["EM"]);
        EB = ToTensor(npz["EB"]);

        torch::Tensor FaceFlat = SM + EM; // 204 * 1
        Face = std::move(FaceFlat.view({-1,3}));

        /*Mean = GetMean(Face);
        Face.rowwise() -= Mean; */

        if (LoadEdge)
        {
            TRI = ToTensorInt(npz["TRI"]);
            try
            {
                TRIUV = ToTensorInt(npz["TRIUV"]);
            }
            catch (...)
            {
                TRIUV = TRI;
            }
            Ef = ToTensorInt(npz["Ef"]);
            Ev = ToTensorInt(npz["Ev"]);

            UV = ToTensor(npz["UV"]);
        }
    }


    void Generate(torch::Tensor SX, torch::Tensor EX)
    {
        torch::Tensor FaceS = torch::matmul(SB , SX);
        torch::Tensor S = FaceS.view({-1,3});

        torch::Tensor FaceE = torch::matmul(EB , EX);
        torch::Tensor E = FaceE.view({-1, 3});

        GeneratedFace =  std::move(Face + S + E);
    }
};

class MMTensorSolver
{
public:
    bool USEWEIGHT = true;
    float WEIGHT = 1.0;
    //vector<int> SkipList = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  11, 12, 13, 14, 15, 16 };
    //vector<int> SkipList = { 0, 1, 2, 3, 4,   5, 6, 7,    9,10,  11, 12,  13, 14, 15, 16 };
    std::vector<int> SkipList = {8, };


    FaceModelTensor FM;
    FaceModelTensor FMFull;
    bool FIXFIRSTSHAPE = true;

    void Initialize(string file, string file2)
    {
        FM.Initialize(file, false);
        //FMFull.Initialize(file2, FM.Mean, true);
        FMFull.Initialize(file2, true);

        if (FIXFIRSTSHAPE)
        {
            FM.SB.select(1,0) *= 0;
            FMFull.SB.select(1,0) *= 0;
        }
    }
    torch::Tensor SolveMultiShape(std::vector<ProjectionTensor> params, std::vector<torch::Tensor> landMarks, std::vector<torch::Tensor> modelMarks,std::vector<float>& angles, torch::Tensor SB, float lambda)
    {
        //cout << Shape(SB) << endl;

        const int imageNum=params.size();
        int totalLandMarkNum=0;
        for(size_t i=0;i<imageNum;i++)
        {
            totalLandMarkNum+=landMarks[i].size(0);
        }

        int rowIndex=0;
        int L = SB.size(1);
        torch::Tensor SBX=torch::zeros({totalLandMarkNum * 2, L});
        torch::Tensor error=torch::zeros({totalLandMarkNum * 2,1});
        for(size_t i=0;i<imageNum;i++)
        {
            float weighti=std::abs(std::cos(angles[i]));
            torch::Tensor Ri = params[i].R * params[i].s;
            Ri = Ri.slice(1,0,2);

            torch::Tensor rotatedi = Projection(params[i], modelMarks[i]);

            torch::Tensor errori = landMarks[i] - rotatedi;

            errori = errori.view({-1, 1});
            int Ni = modelMarks[i].size(0);
            torch::Tensor SBXi=torch::zeros({Ni * 2, L});
            torch::Tensor Rit = Ri.transpose(0,1);
            for (size_t ii = 0; ii < Ni; ii++)
            {
                SBXi.slice(0,ii * 2,ii * 2+2)=torch::matmul( Rit , SB.slice(0,ii * 3, ii * 3+3));
            }

            if (USEWEIGHT)
            {
                torch::Tensor W = torch::ones(2 * Ni);
                for (size_t j = 0; j < SkipList.size(); j++)
                {
                    W[2 * SkipList[j] + 0] = WEIGHT;
                    W[2 * SkipList[j] + 1] = WEIGHT;
                }
                SBXi =  torch::matmul(W.diag(), SBXi);
                errori = torch::matmul( W.diag(), errori);
            }
            SBXi*=weighti/params[i].s;
            errori*=weighti/params[i].s;
            error.slice(0,rowIndex,rowIndex+Ni*2)=errori;
            SBX.slice(0,rowIndex,rowIndex+Ni*2)=SBXi;
            rowIndex+=Ni*2;
        }
        return SolveLinear(SBX, error, lambda);
    }
    torch::Tensor SolveShape(ProjectionTensor& p, torch::Tensor& imagePoints, torch::Tensor M, torch::Tensor SB, float lambda)
    {
        //cout << Shape(SB) << endl;
//        std::cout << p.R << endl;
        torch::Tensor R = p.R.transpose(0,1)  * p.s;
        R = R.slice(1,0,2);

        torch::Tensor rotated = Projection(p, M);

        torch::Tensor error = imagePoints - rotated;
        error = error.view({-1, 1});

        int N = M.size(0);
        int N2 = SB.size(0);
        int L = SB.size(1);

        assert(N2 == N * 3);
        auto sTx = p.tx * p.s;
        auto sTy = p.ty * p.s;
        torch::Tensor SBX=torch::zeros({N * 2, L});
        torch::Tensor Rt = R.transpose(0,1);
        for (size_t i = 0; i < N; i++)
        {
            SBX.slice(0,i * 2,i * 2+2) =torch::matmul( Rt , SB.slice(0,i * 3, i * 3+3));
        }

        if (USEWEIGHT)
        {
            torch::Tensor W = torch::ones(2 * N);
            for (size_t i = 0; i < SkipList.size(); i++)
            {
                W[2 * SkipList[i] + 0] = WEIGHT;
                W[2 * SkipList[i] + 1] = WEIGHT;
            }
            SBX = torch::matmul(W.diag(), SBX);
            error =torch::matmul( W.diag(), error);
        }
        return SolveLinear(SBX, error, lambda);
    }

    ProjectionTensor SolveProjection(torch::Tensor& imagePoints, torch::Tensor& modelPoints)
    {
        //########## Mean should be subtracted from model_points ############
        int N = imagePoints.size(0);

        torch::Tensor vmodelPoints=torch::cat({modelPoints,torch::ones({modelPoints.size(0),1})},1);

        torch::Tensor A = torch::zeros({(int64_t)2 * N, (int64_t)8});
        for (int i = 0; i < N; ++i)
        {
            torch::Tensor P = vmodelPoints.select(0,i);// .transpose();//Eigen::Vector4f();
            A[2*i].slice(0,0,4) = P;       // even row - copy to left side (first row is row 0)
            A[2*i+1].slice(0,4,8) = P; // odd row - copy to right side
        } // 4th coord (homogeneous) is already 1

        /*Matrix<float, 1, Eigen::Dynamic> Mean = image_points.colwise().mean();
        image_points.rowwise() -= Mean;*/

        torch::Tensor b =imagePoints.view({-1,1});

        if (USEWEIGHT)
        {
            torch::Tensor W = torch::ones(2 * N);
            for (size_t i = 0; i < SkipList.size(); i++)
            {
                W[2 * SkipList[i] + 0] = WEIGHT;
                W[2 * SkipList[i] + 1] = WEIGHT;
            }
            A = torch::matmul(W.diag() , A);
            b = torch::matmul(W.diag() , b);
        }

        const torch::Tensor k = SolveLinear(A, b); // resulting affine matrix (8x1)
//        std::cout<<"k:"<<k<<std::endl;
        // Extract all values from the estimated affine parameters k:
        torch::Tensor R1 = k.slice(0,0,3).transpose(0,1);
        torch::Tensor R2 = k.slice(0,4,7).transpose(0,1);
        torch::Tensor R=torch::eye(3);
        torch::Tensor r1=R1.div(R1.norm(2,1)).squeeze(0);
        torch::Tensor r2=R2.div(R2.norm(2,1)).squeeze(0);
        R[0] = r1;
        R[1] = r2;
        R[2] = torch::cross(r1,r2);
        float sTx = k[3][0].item().toFloat();
        float sTy = k[7][0].item().toFloat();

        //sTx += Mean(0);
        //sTy += Mean(1);

        const auto s = (R1.norm(2,1).item().toFloat() + R2.norm(2,1).item().toFloat()) / 2.0f;

        torch::Tensor R_ortho = Orthogonalize(R);
        // Remove the scale from the translations:
        const auto t1 = sTx / s /*- T(0)*/;
        const auto t2 = sTy / s/* - T(1)*/;
        return std::move(ProjectionTensor{ std::move(R_ortho), t1, t2, s });
    }

    torch::Tensor SX;
    torch::Tensor EX;
    ProjectionTensor params;

    bool FixShape = false;
    torch::Tensor SX0;




    void Solve(torch::Tensor& KP)
    {
        torch::Tensor Face = FM.Face;
        torch::Tensor S = Face * 0;
        torch::Tensor E = Face * 0;


        float Lambdas[7] = { 100.0, 30.0, 10.0, 5.0,4.0,3.0,2.0};


        for (size_t i = 0; i < 1; i++)
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
                    SX[0][0] = 0;
                }
            }
            torch::Tensor FaceS = torch::matmul(FM.SB , SX);
            S = FaceS.view({-1,3});

            EX = SolveShape(params, KP, FM.Face + S, FM.EB, Lambdas[i] * 1);

            torch::Tensor FaceE = torch::matmul(FM.EB , EX);
            E = FaceE.view({-1,3});
            Face = FM.Face + S + E;
        }

    }
};

inline cv::Mat MMSDraw(cv::Mat orig, MMTensorSolver &MMS,const torch::Tensor &KP,bool flip=true)
{

    auto params = MMS.params;
    MMS.FMFull.Generate(MMS.SX, MMS.EX);
    auto imagePoints=KP.clone();
    torch::Tensor projected = Projection(params, MMS.FMFull.GeneratedFace);
    if(flip){
        imagePoints.select(1,1)=orig.rows-imagePoints.select(1,1);
        projected.select(1,1)=orig.rows-projected.select(1,1);
    }
    auto image = orig.clone();
    auto image2 = orig.clone();

    auto Ev = MMS.FMFull.Ev;

    for (size_t i = 0; i < Ev.size(0); i++)
    {
        int i1 = Ev[i][0].item().toInt();
        int i2 = Ev[i][1].item().toInt();
        auto x = projected[i1][0].item().toFloat();
        auto y = projected[i1][1].item().toFloat();

        auto x2 = projected[i2][0].item().toFloat();
        auto y2 = projected[i2][1].item().toFloat();

        cv::line(image, cv::Point(x, y), cv::Point(x2, y2), cv::Scalar(0, 0, 255, 255), 1);
        //image.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
        //circle(image, Point(x, y), 1, Scalar(0, 0, 255), -1);
    }

    MMS.FM.Generate(MMS.SX, MMS.EX);
    projected = Projection(params, MMS.FM.GeneratedFace);
    if(flip){
        projected.select(1,1)=orig.rows-projected.select(1,1);
    }
    for (size_t i = 0; i < projected.size(0); i++)
    {
        auto x = projected[i][0].item().toFloat();
        auto y = projected[i][1].item().toFloat();
        circle(image, cv::Point(x, y), 2, cv::Scalar(255, 0, 0, 255), -1, CV_AA);
    }

    if (MMS.USEWEIGHT)
    {
        for (size_t i = 0; i < MMS.SkipList.size(); i++)
        {
            int i2 = MMS.SkipList[i];
            auto x = projected[i2][0].item().toFloat();
            auto y = projected[i2][1].item().toFloat();
            cv::circle(image, cv::Point(x, y), 6, cv::Scalar(255, 0, 0, 255), 1, CV_AA);
        }
    }

    for (size_t i = 0; i < imagePoints.size(0); i++)
    {
        auto x = imagePoints[i][0].item().toFloat();
        auto y = imagePoints[i][1].item().toFloat();

        circle(image, cv::Point(x, y), 2, cv::Scalar(0, 255, 0, 255), -1, CV_AA);
    }

    //imshow("IMG", image / 2 + image2 / 2);
    return image / 2 + image2 / 2;
}



// Apply affine transform calculated using srcTri and dstTri to src
inline void applyAffineTransform(cv::Mat &warpImage, cv::Mat &src, vector<cv::Point2f> &srcTri, vector<cv::Point2f> &dstTri)
{
    // Given a pair of triangles, find the affine transform.
    cv::Mat warpMat = getAffineTransform(srcTri, dstTri);

    // Apply the Affine Transform just found to the src image
    warpAffine(src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
}


// Warps and alpha blends triangular regions from img1 and img2 to img
inline void warpTriangle(cv::Mat &img1, cv::Mat &img2, vector<cv::Point2f> &t1, vector<cv::Point2f> &t2)
{
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


inline cv::Mat MMSTexture(cv::Mat orig, MMTensorSolver &MMS, int W, int H, bool doubleface = false,bool flip=true)
{
    auto image = orig.clone();
    cv::Mat texture = cv::Mat::zeros(H, W, CV_8UC3);

    auto params = MMS.params;
    MMS.FMFull.Generate(MMS.SX, MMS.EX);
    torch::Tensor projected = Projection(params, MMS.FMFull.GeneratedFace);
    if(flip){
        projected.select(1,1)=orig.rows-projected.select(1,1);
    }

    auto TRI = MMS.FMFull.TRIUV;
    auto UV = MMS.FMFull.UV;
    for (size_t t = 0; t < TRI.size(0); t++)
    {

        vector<cv::Point2f> t1;
        vector<cv::Point2f> t2;
        for (size_t i = 0; i < 3; i++)
        {
            int j = TRI[t][i].item().toInt();
            auto x = projected[j][0].item().toFloat();
            auto y = projected[j][1].item().toFloat();

            auto u = (UV[j][0].item().toFloat()) * (W - 1);
            auto v = (1 - UV[j][1].item().toFloat()) * (H - 1);
            t1.push_back(cv::Point2f(x, y));
            t2.push_back(cv::Point2f(u, v));

            //cout << Point2f(x, y) << Point2f(u, v) << endl;
        }

        auto c = (t1[2] - t1[0]).cross(t1[1] - t1[0]);

        if ( c > 0)
        {
            warpTriangle(image, texture, t1, t2);
        }
    }

    return texture;

}


inline cv::Mat MMSNormal(cv::Mat orig, MMTensorSolver &MMS, int W, int H)
{
    auto image = orig.clone();
    cv::Mat texture = cv::Mat::zeros(H, W, CV_8UC3);

    auto params = MMS.params;
    MMS.FMFull.Generate(MMS.SX, MMS.EX);
    auto Face2 = Rotation(params, MMS.FMFull.GeneratedFace);
    auto TRI = MMS.FMFull.TRIUV;
    auto UV = MMS.FMFull.UV;

    for (size_t t = 0; t < TRI.size(0); t++)
    {

        vector<cv::Point3f> t1;
        vector<cv::Point> t2;
        for (size_t i = 0; i < 3; i++)
        {
            int j = TRI[t][i].item().toInt();
            auto x = Face2[j][0].item().toFloat();
            auto y = Face2[j][1].item().toFloat();
            auto z = Face2[j][2].item().toFloat();

            auto u = (UV[j][0].item().toFloat()) * (W - 1);
            auto v = (1.0 - UV[j][1].item().toFloat()) * (H - 1);
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


inline void FMObj(cv::Mat texture, FaceModelTensor &FM, string folder, string filename0)
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

        int N = Face.size(0);
        for (size_t i = 0; i < N; i++)
        {
            ss << "v " << Face[i][0].item().toFloat() << " " << Face[i][1].item().toFloat() << " " << Face[i][2].item().toFloat() << endl;
        }

        N = UV.size(0);
        for (size_t i = 0; i < N; i++)
        {
            ss << "vt " << UV[i][0].item().toFloat() << " " << UV[i][1].item().toFloat() << endl;
        }

        ss << "usemtl material_0" << endl;
        ss << "s 1" << endl;

        N = TRI.size(0);
        for (size_t i = 0; i < N; i++)
        {
            ss << "f " << TRI[i][0].item().toInt()  + 1 << "/" << TRIUV[i][0].item().toInt() + 1 << " "
               << TRI[i][1].item().toInt()  + 1 << "/" << TRIUV[i][1].item().toInt() + 1 << " "
               << TRI[i][2].item().toInt()  + 1 << "/" << TRIUV[i][2].item().toInt() + 1 << " "
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


inline void MMSObj(cv::Mat orig, MMTensorSolver &MMS, string folder, string filename0)
{
    string filename = folder + filename0;
    cv::Mat texture = MMSTexture(orig, MMS, 1024,1024);
    imwrite(filename + ".png", texture);

    MMS.FMFull.Generate(MMS.SX, MMS.EX);
    auto Face=MMS.FMFull.GeneratedFace;
    auto TRI = MMS.FMFull.TRI;
    auto TRIUV = MMS.FMFull.TRIUV;
    auto UV = MMS.FMFull.UV;

    string numpyfile = filename + ".npz";


    cnpy::npz_save(numpyfile, "SX", MMS.SX.data<float>(), { (unsigned long long)MMS.SX.size(0), (unsigned long long)MMS.SX.size(1) }, "w"); //"w" overwrites any existing file
    cnpy::npz_save(numpyfile, "EX", MMS.EX.data<float>(), { (unsigned long long)MMS.EX.size(0), (unsigned long long)MMS.EX.size(1) }, "a"); //"a" appends to the file we created above

    {
        std::stringstream ss;


        ss << "mtllib " << filename0 << ".mtl" << endl;
        ss << "o FaceObject" << endl;

        int N = Face.size(0);
        for (size_t i = 0; i < N; i++)
        {
            ss << "v " << Face[i][0].item().toFloat() << " " << Face[i][1].item().toFloat() << " " << Face[i][2].item().toFloat() << endl;
            ss << "vt " << UV[i][0].item().toFloat()<< " " << UV[i][1].item().toFloat() << endl;
        }

        ss << "usemtl material_0" << endl;
        ss << "s 1" << endl;

        N = TRI.size(0);
        for (size_t i = 0; i < N; i++)
        {
            ss << "f " << TRI[i][0].item().toInt() + 1 << "/" << TRIUV[i][0].item().toInt()  + 1 << " "
               << TRI[i][1].item().toInt() + 1 << "/" << TRIUV[i][1].item().toInt()  + 1 << " "
               << TRI[i][2].item().toInt() + 1 << "/" << TRIUV[i][2].item().toInt()  + 1 << " "
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
#endif // MMTENSORSOLVER_H
