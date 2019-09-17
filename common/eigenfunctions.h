#ifndef EIGENFUNCTIONS_H
#define EIGENFUNCTIONS_H
#include <Eigen/Dense>
#include <vector>
#include <glm/gtx/euler_angles.hpp>
template<typename Type>
class EigenFunctions{
public:
    static void saveObj(std::string name, Eigen::Matrix<Type,3,Eigen::Dynamic>& verts, std::vector<int64_t> &faceIds);
    static void saveObj(std::string name, Eigen::Matrix<Type,3,Eigen::Dynamic,Eigen::RowMajor>& verts, std::vector<int64_t> faceIds);
    template <int _Options>
    static Eigen::Matrix<Type,3,3,_Options> eulerAngleY(Type const& angleY);
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


template <typename Type>
template <int _Options>
Eigen::Matrix<Type,3,3,_Options> EigenFunctions<Type>::eulerAngleY(const Type &angleY)
{
    Type cosY = glm::cos(angleY);
    Type sinY = glm::sin(angleY);
    Eigen::Matrix<Type,3,3,_Options> e;
    e<<cosY,	Type(0),	sinY,
       Type(0),	Type(1),	Type(0),
       -sinY,	Type(0),	cosY;
    return e;
}
#endif // EIGENFUNCTIONS_H



