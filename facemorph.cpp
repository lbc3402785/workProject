#include "facemorph.h"

FaceMorph::FaceMorph()
{

}
/**
 * @brief boundingRectOfTensor
 * @param tri 3x2
 * @return
 */
inline cv::Rect boundingRectOfTensor(torch::Tensor tri)
{
//    torch::Tensor xs=tri.select(1,0);
//    torch::Tensor ys=tri.select(1,1);
//    float minx=xs.min().item().toFloat();
//    float maxx=xs.max().item().toFloat();
//    float miny=ys.min().item().toFloat();
//    float maxy=ys.max().item().toFloat();
//    int width=std::max((int)std::round(maxx-minx),1);
//    int height=std::max((int)std::round(maxy-miny),1);
    std::vector<cv::Point2f> pv;
    pv.resize(3);
    pv[0]=cv::Point2f(tri[0][0].item().toFloat(),tri[0][1].item().toFloat());
    pv[1]=cv::Point2f(tri[1][0].item().toFloat(),tri[1][1].item().toFloat());
    pv[2]=cv::Point2f(tri[2][0].item().toFloat(),tri[2][1].item().toFloat());
    return boundingRect(pv);
}
/**
 * @brief FaceMorph::morphTriangle
 * @param imgs
 * @param target
 * @param srcTris   imageNumx3x2
 * @param dstTri    3x2
 * @param weights   imageNum
 */
void FaceMorph::morphTriangle(std::vector<cv::Mat> &imgs, cv::Mat &target, at::Tensor &srcTris, at::Tensor &dstTri, at::Tensor weights)
{
    size_t imageNum=imgs.size();
    cv::Rect dstRect = boundingRectOfTensor(dstTri);
    std::vector<cv::Rect> srcRects;
    srcRects.resize(imageNum);
    torch::Tensor rsT=torch::zeros({(int64_t)imageNum,2});
    for(size_t j=0;j<imageNum;j++)
    {
        srcRects[j]=boundingRectOfTensor(srcTris[j]);
        rsT[j][0]=srcRects[j].x;rsT[j][1]=srcRects[j].y;
    }
    std::vector<cv::Point2f> localDstRect;
    std::vector<cv::Point> localDstRectInt;
    std::vector<std::vector<cv::Point2f>> localSrcRects;
    localSrcRects.resize(imageNum);
    torch::Tensor dstTriRectXs=dstTri.select(1,0) - dstRect.x;//3
    torch::Tensor dstTriRectYs=dstTri.select(1,1) - dstRect.y;//3
    for(size_t i = 0; i < 3; i++)
    {
        localDstRect.push_back( cv::Point2f( dstTriRectXs[i].item().toFloat(), dstTriRectYs[i].item().toFloat()) );
        localDstRectInt.push_back( cv::Point(dstTriRectXs[i].item().toFloat(), dstTriRectYs[i].item().toFloat()) );
    }

    for(size_t j=0;j<imageNum;j++)
    {
        torch::Tensor temp=srcTris[j]-rsT[j];//3x2-2-->3x2
        localSrcRects[j].push_back( cv::Point2f( temp[0][0].item().toFloat(), temp[0][1].item().toFloat()) );
        localSrcRects[j].push_back( cv::Point2f( temp[1][0].item().toFloat(), temp[1][1].item().toFloat()) );
        localSrcRects[j].push_back( cv::Point2f( temp[2][0].item().toFloat(), temp[2][1].item().toFloat()) );
    }

    // Get mask by filling triangle
    cv::Mat mask = cv::Mat::zeros(dstRect.height, dstRect.width, imgs[0].type());
    fillConvexPoly(mask, localDstRectInt, cv::Scalar(1.0, 1.0, 1.0), 16, 0);//antialiased line

    std::vector<cv::Mat> srcImgRects;
    std::vector<cv::Mat> dstImgRects;
    srcImgRects.resize(imageNum);
    dstImgRects.resize(imageNum);
    //#pragma omp parallel for
    for(int j=0;j<imageNum;j++)
    {
        cv::Mat temp;
        imgs[j](srcRects[j]).copyTo(temp);
        srcImgRects[j]=(temp);
        cv::Mat warpImagej = cv::Mat::zeros(dstRect.height, dstRect.width, temp.type());
        applyAffineTransform(warpImagej, srcImgRects[j], localSrcRects[j], localDstRect);
        dstImgRects[j]=(warpImagej);
    }

    // Alpha blend rectangular patches
    cv::Mat imgRect= cv::Mat::zeros(dstRect.height, dstRect.width, dstImgRects[0].type());
    for(size_t j=0;j<imageNum;j++)
    {
        imgRect+=dstImgRects[j]*weights[j].item().toFloat();
    }
    //    // Copy triangular region of the rectangular patch to the output image
    multiply(imgRect,mask, imgRect);//填充三角形（锯齿形）内部部
    multiply(target(dstRect), cv::Scalar(1.0,1.0,1.0) - mask, target(dstRect));//保留三角形外部
    target(dstRect) = target(dstRect) + imgRect;
}

void FaceMorph::applyAffineTransform(cv::Mat &warpImage, cv::Mat &src, std::vector<cv::Point2f> &srcTri, std::vector<cv::Point2f> &dstTri)
{
    // Given a pair of triangles, find the affine transform.
    cv::Mat warpMat = cv::getAffineTransform( srcTri, dstTri );
    // Apply the Affine Transform just found to the src image
    cv::warpAffine( src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
}

void FaceMorph::morphTriangle(std::vector<cv::Mat> &imgs, cv::Mat &target, std::vector<std::vector<cv::Point2f> > &srcTris, std::vector<cv::Point2f> &dstTri, std::vector<float> &weights)
{

    size_t imageNum=imgs.size();
    cv::Rect dstRect = cv::boundingRect(dstTri);
    std::vector<cv::Rect> srcRects;
    srcRects.resize(imageNum);
    for(size_t j=0;j<imageNum;j++)
    {
        srcRects[j]=boundingRect(srcTris[j]);
    }

    std::vector<cv::Point2f> localDstRect;
    std::vector<cv::Point> localDstRectInt;
    std::vector<std::vector<cv::Point2f>> localSrcRects;
    localSrcRects.resize(imageNum);
    for(size_t i = 0; i < 3; i++)
    {
        localDstRect.push_back( cv::Point2f( dstTri[i].x - dstRect.x, dstTri[i].y -  dstRect.y) );
        localDstRectInt.push_back( cv::Point(dstTri[i].x - dstRect.x, dstTri[i].y - dstRect.y) );
        for(size_t j=0;j<imageNum;j++)
        {
            localSrcRects[j].push_back( cv::Point2f( srcTris[j][i].x - srcRects[j].x, srcTris[j][i].y -  srcRects[j].y) );
        }
    }

    // Get mask by filling triangle
    cv::Mat mask = cv::Mat::zeros(dstRect.height, dstRect.width, imgs[0].type());

    fillConvexPoly(mask, localDstRectInt, cv::Scalar(1.0, 1.0, 1.0), 16, 0);//antialiased line

    std::vector<cv::Mat> srcImgRects;
    std::vector<cv::Mat> dstImgRects;
    for(size_t j=0;j<imageNum;j++)
    {
        cv::Mat temp;
        try {
            imgs[j](srcRects[j]).copyTo(temp);
        } catch (std::exception& e) {
            std::cout<<srcRects[j]<<std::endl;
            std::cout<<e.what()<<std::endl;
            exit(EXIT_FAILURE);
        }

        srcImgRects.push_back(temp);
        cv::Mat warpImagej = cv::Mat::zeros(dstRect.height, dstRect.width, temp.type());
        applyAffineTransform(warpImagej, srcImgRects[j], localSrcRects[j], localDstRect);
        dstImgRects.emplace_back(warpImagej);
    }
    // Alpha blend rectangular patches
    cv::Mat imgRect= cv::Mat::zeros(dstRect.height, dstRect.width, dstImgRects[0].type());
    for(size_t j=0;j<imageNum;j++)
    {
        imgRect+=dstImgRects[j]*weights[j];
    }
//    // Copy triangular region of the rectangular patch to the output image
    multiply(imgRect,mask, imgRect);//填充三角形（锯齿形）内部部
    multiply(target(dstRect), cv::Scalar(1.0,1.0,1.0) - mask, target(dstRect));//保留三角形外部
    target(dstRect) = target(dstRect) + imgRect;
}
