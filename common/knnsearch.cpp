#include "knnsearch.h"
#include <torch/torch.h>
#include <QDebug>
#include <QDir>
#include <QTime>
KNNSearch::KNNSearch(float *data,int64_t rows,int64_t cols):rows(rows),cols(cols)
{
   dataset.resize(rows*cols);
   memcpy(dataset.data(),data,sizeof(float)*rows*cols);
   flann::Matrix<float> flann_dataset(dataset.data(), rows, cols);
   flannIndex=new flann::Index<flann::MinkowskiDistance<float> >(flann_dataset, flann::KDTreeIndexParams(1), flann::MinkowskiDistance<float>(1));
   flannIndex->buildIndex();
//   std::cout<<"kdtree construct!"<<std::endl;
}

void KNNSearch::search(float *dst, int64_t dstRows, int64_t dstCols)
{
    assert(dstCols==this->cols);
    srcIndex.clear();
    distances.clear();
    srcIndex.resize(dstRows);
    std::vector<int> tempIndex(dstRows,0);
    std::vector<float> tempDistances(dstRows,0.0);
    flann::Matrix<float> query(dst,dstRows,dstCols);
    flann::Matrix<int> indices(tempIndex.data(), dstRows, 1);
    flann::Matrix<float> dists(tempDistances.data(), dstRows, 1);

    flann::SearchParams param(flann::FLANN_CHECKS_UNLIMITED);
    param.cores=4;
//    param.matrices_in_gpu_ram=true;
//    QTime timedebuge;//声明一个时钟对象
//    timedebuge.start();//开始计时

    flannIndex->knnSearch(query, indices, dists, 1, param);

//    qDebug()<<"consume :"<<timedebuge.elapsed()/1000.0<<"s";
    srcIndex.swap(tempIndex);
    distances.swap(tempDistances);
}

void KNNSearch::reset()
{
    if(flannIndex){ delete flannIndex; flannIndex=nullptr;}
}
