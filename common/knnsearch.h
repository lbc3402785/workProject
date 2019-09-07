#ifndef KNNSEARCH_H
#define KNNSEARCH_H
//#define FLANN_USE_CUDA
#include <flann/flann.hpp>
class KNNSearch
{
public:
    KNNSearch(float *data,int rows,int cols=3);
    ~KNNSearch(){if(flannIndex){ delete flannIndex; flannIndex=nullptr;}}
    std::vector<int> srcIndex;
//    std::vector<int> dstIndex;
    std::vector<float> distances;
    void search(float* dst,int dstRows,int dstCols);
private:
    int rows;
    int cols;
    std::vector<float> dataset;
    void reset();
    flann::Index<flann::MinkowskiDistance<float>>* flannIndex=nullptr;
};

#endif // KNNSEARCH_H
