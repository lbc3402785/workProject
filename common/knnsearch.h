#ifndef KNNSEARCH_H
#define KNNSEARCH_H
//#define FLANN_USE_CUDA
#include <flann/flann.hpp>
class KNNSearch
{
public:
    KNNSearch(float *data,int64_t rows,int64_t cols=3);
    ~KNNSearch(){if(flannIndex){ delete flannIndex; flannIndex=nullptr;}}
    std::vector<int> srcIndex;
//    std::vector<int> dstIndex;
    std::vector<float> distances;
    void search(float* dst,int64_t dstRows,int64_t dstCols);
private:
    int64_t rows;
    int64_t cols;
    std::vector<float> dataset;
    void reset();
    flann::Index<flann::MinkowskiDistance<float>>* flannIndex=nullptr;
};

#endif // KNNSEARCH_H
