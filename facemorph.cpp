#include "facemorph.h"

FaceMorph::FaceMorph()
{

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
    int imageNum=imgs.size();
    cv::Rect r = cv::boundingRect(dstTri);
    std::vector<cv::Rect> rs;
    rs.resize(imageNum);
    for(int j=0;j<imageNum;j++)
    {
        rs[j]=boundingRect(srcTris[j]);
    }

    std::vector<cv::Point2f> dstRect;
    std::vector<cv::Point> dstRectInt;
    std::vector<std::vector<cv::Point2f>> srcRects;
    srcRects.resize(imageNum);
    for(int i = 0; i < 3; i++)
    {
        dstRect.push_back( cv::Point2f( dstTri[i].x - r.x, dstTri[i].y -  r.y) );
        dstRectInt.push_back( cv::Point(dstTri[i].x - r.x, dstTri[i].y - r.y) );
        for(int j=0;j<imageNum;j++)
        {
            srcRects[j].push_back( cv::Point2f( srcTris[j][i].x - rs[j].x, srcTris[j][i].y -  rs[j].y) );
        }
    }

    // Get mask by filling triangle
    cv::Mat mask = cv::Mat::zeros(r.height, r.width, imgs[0].type());
    fillConvexPoly(mask, dstRectInt, cv::Scalar(1.0, 1.0, 1.0), 16, 0);//antialiased line

    std::vector<cv::Mat> srcImgRects;
    std::vector<cv::Mat> dstImgRects;
    for(int j=0;j<imageNum;j++)
    {
        cv::Mat temp;
        imgs[j](rs[j]).copyTo(temp);
        srcImgRects.push_back(temp);
        cv::Mat warpImagej = cv::Mat::zeros(r.height, r.width, temp.type());
        applyAffineTransform(warpImagej, srcImgRects[j], srcRects[j], dstRect);
        dstImgRects.emplace_back(warpImagej);
    }
    // Alpha blend rectangular patches
    cv::Mat imgRect= cv::Mat::zeros(r.height, r.width, dstImgRects[0].type());
    for(int j=0;j<imageNum;j++)
    {

        imgRect+=dstImgRects[j]*weights[j];
    }
//    // Copy triangular region of the rectangular patch to the output image
    multiply(imgRect,mask, imgRect);//填充三角形（锯齿形）内部部
    multiply(target(r), cv::Scalar(1.0,1.0,1.0) - mask, target(r));//保留三角形外部
    target(r) = target(r) + imgRect;
}
