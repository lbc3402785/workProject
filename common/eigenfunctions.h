#ifndef EIGENFUNCTIONS_H
#define EIGENFUNCTIONS_H
#include <Eigen/Dense>
#include <vector>
template<typename Type>
class EigenFunctions{
public:
    static void saveObj(std::string name, Eigen::Matrix<Type,3,Eigen::Dynamic>& verts, std::vector<int64_t> &faceIds);
    static void saveObj(std::string name, Eigen::Matrix<Type,3,Eigen::Dynamic,Eigen::RowMajor>& verts, std::vector<int64_t> faceIds);
};
template<typename Type>
void EigenFunctions<Type>::saveObj(std::string name, Eigen::Matrix<Type, 3, Eigen::Dynamic>& verts, std::vector<int64_t> &faceIds)
{
    std::ofstream out(name.c_str());
    if(out){
        for(int i=0;i<verts.cols();i++){
            out<<"v ";
            for(int j=0;j<verts.rows();j++){
                out<<" "<<verts(j,i);
            }
            out<<"\n";
        }
        for(int i=0;i<faceIds.size()/3;i++){
            out<<"f "<<faceIds[3*i]+1<<" "<<faceIds[3*i+1]+1<<" "<<faceIds[3*i+2]+1<<"\n";
        }
    }
}
template<typename Type>
void EigenFunctions<Type>::saveObj(std::string name, Eigen::Matrix<Type, 3, Eigen::Dynamic,Eigen::RowMajor>& verts, std::vector<int64_t> faceIds)
{
    std::ofstream out(name.c_str());
    if(out){
        for(int i=0;i<verts.cols();i++){
            out<<"v ";
            for(int j=0;j<verts.rows();j++){
                out<<" "<<verts(j,i);
            }
            out<<"\n";
        }
        for(int i=0;i<faceIds.size()/3;i++){
            out<<"f "<<faceIds[3*i]+1<<" "<<faceIds[3*i+1]+1<<" "<<faceIds[3*i+2]+1<<"\n";
        }
    }
}
#endif // EIGENFUNCTIONS_H


