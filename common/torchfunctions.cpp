#include "torchfunctions.h"
#include <cassert>
#include <ATen/Functions.h>
#include <Eigen/Dense>

#include "dataconvertor.h"
TorchFunctions::TorchFunctions()
{

}
/**
 * @brief TorchFunctions::rodrigues
 * @param theta 3x1
 * @return      3x3
 */
at::Tensor TorchFunctions::rodrigues(at::Tensor &theta)
{
    assert(theta.dim()==2);
    assert(theta.sizes() == torch::IntList({3,1}));
    torch::Tensor angle=theta.norm(2,0);//1
    float f=angle.data<float>()[0];
    if(f<1e-6){
        return std::move(torch::eye(3));
    }

    torch::Tensor normalized=torch::div(theta,angle);//3x1
    torch::Tensor nxs=normalized[0];//1
    torch::Tensor nys=normalized[1];//1
    torch::Tensor nzs=normalized[2];//1
    torch::Tensor stick=torch::zeros(1);//1
    torch::Tensor lr=torch::stack({stick,-nzs,nys,
                                  nzs,stick,-nxs,
                                  -nys,nxs,stick},1);//1x9
    torch::Tensor k=torch::reshape(lr,{3,3});//3x3
    torch::Tensor I=torch::eye(3);
    torch::Tensor dot=torch::matmul(normalized,normalized.t());
    torch::Tensor cos=torch::cos(angle);//1
    torch::Tensor sin=torch::sin(angle);//1
    torch::Tensor R=cos*I+(1-cos)*dot+sin*k;
    return std::move(R);
}
/**
 * @brief TorchFunctions::batchRodrigues
 * @param thetas    Mx3x1
 * @return          Mx3x3
 */
at::Tensor TorchFunctions::batchRodrigues(at::Tensor &thetas)
{
    torch::Tensor llnorm=torch::norm(thetas+ 1e-8,2,1);//(batch_num*24)
    torch::Tensor angle=torch::unsqueeze(llnorm,-1);//(batch_num*24)x1
    torch::Tensor normalized=torch::div(thetas,angle);//(batch_num*24)x3---(batch_num*24)x(x,y,z)
    torch::Tensor nxs=normalized.slice(1,0,1).squeeze(1);//(batch_num*24)
    torch::Tensor nys=normalized.slice(1,1,2).squeeze(1);//(batch_num*24)
    torch::Tensor nzs=normalized.slice(1,2,3).squeeze(1);//(batch_num*24)
    int num=thetas.sizes()[0];
    torch::Tensor stick=torch::zeros(num);//(batch_num*24)
    torch::Tensor lr=torch::stack({stick,-nzs,nys,
                                  nzs,stick,-nxs,
                                  -nys,nxs,stick},1);//(batch_num*24)x9
    torch::Tensor k=torch::reshape(lr,{-1,3,3});//(batch_num*24)x3x3
    torch::Tensor I=torch::eye(3).unsqueeze(0)+torch::zeros({num,3,3});//(batch_num*24)x3x3
    torch::Tensor A=normalized.unsqueeze(-1);//(batch_num*24)x3x1
    torch::Tensor AT=normalized.unsqueeze(1);//(batch_num*24)x1x3
    torch::Tensor dot=torch::matmul(A,AT);//(batch_num*24)x3x3
    torch::Tensor cos=torch::cos(angle.unsqueeze(-1));//(batch_num*24)x1x1
    torch::Tensor sin=torch::sin(angle.unsqueeze(-1));//(batch_num*24)x1x1
    torch::Tensor R=cos*I+(1-cos)*dot+sin*k;//(batch_num*24)x3x3
    return std::move(R);
}
/**
 * @brief TorchFunctions::unRodrigues
 * @param R 3x3
 * @return  3x1
 */
at::Tensor TorchFunctions::unRodrigues(at::Tensor &R)
{
    at::Tensor temp=(R-R.t())/2;
    at::Tensor v=torch::stack({temp[2][1],temp[0][2],temp[1][0]},0).unsqueeze(-1);
    at::Tensor sin=v.norm(2,0);
    at::Tensor theta=torch::asin(sin);
    if(theta.item().toFloat()<1e-6){
        return std::move(v);
    }else{
        return std::move(v/sin*theta);
    }
}

void TorchFunctions::saveObj(std::string name, at::Tensor m,torch::Tensor state)
{
    std::ofstream out(name.c_str());
    if(out){
        for(int i=0;i<m.sizes()[0];i++){
            int si=state[i].item().toInt();
            if(si==0)continue;
            out<<"v ";
            for(int j=0;j<m.sizes()[1];j++){
                out<<" "<<m[i][j].item().toFloat();
            }
            out<<"\n";
        }
    }
}
/**
 * @brief TorchFunctions::saveObj
 * @param m        6890x3x1
 * @param faceIds
 */
void TorchFunctions::saveObj(std::string name,at::Tensor m, std::vector<int64_t> faceIds)
{
    std::ofstream out(name.c_str());
    if(out){
        for(int i=0;i<m.sizes()[0];i++){
            out<<"v ";
            for(int j=0;j<m.sizes()[1];j++){
                out<<" "<<m[i][j].item().toFloat();
            }
            out<<"\n";
        }
        for(int i=0;i<faceIds.size()/3;i++){
            out<<"f "<<faceIds[3*i]+1<<" "<<faceIds[3*i+1]+1<<" "<<faceIds[3*i+2]+1<<"\n";
        }
    }
}

void TorchFunctions::saveObj(std::string name, at::Tensor m, at::Tensor n, std::vector<int64_t> faceIds)
{
    std::ofstream out(name.c_str());
    if(out){
        for(int i=0;i<m.sizes()[0];i++){
            out<<"v ";
            for(int j=0;j<m.sizes()[1];j++){
                out<<" "<<m[i][j].item().toFloat();
            }
            out<<"\n";
        }
        for(int i=0;i<n.sizes()[0];i++){
            out<<"vn ";
            float l=n[i].norm(2).item().toFloat();
            for(int j=0;j<n.sizes()[1];j++){
                out<<" "<<n[i][j].item().toFloat()/l;
            }
            out<<"\n";
        }
        for(int i=0;i<faceIds.size()/3;i++){
            out<<"f "<<faceIds[3*i]+1<<" "<<faceIds[3*i+1]+1<<" "<<faceIds[3*i+2]+1<<"\n";
        }
    }
}

at::Tensor TorchFunctions::computeSingleNormal( at::Tensor &model,  std::vector<int64_t> &faceIds)
{
    int facesNum=faceIds.size()/3;
    int pointsNum=model.sizes()[0];
    torch::Tensor pointNomrals=torch::zeros({pointsNum,3,1});
    std::vector<int> counts(pointsNum,0);

    for(int i=0;i<facesNum;i++){
        int i1=faceIds[3*i];
        int i2=faceIds[3*i+1];
        int i3=faceIds[3*i+2];
        torch::Tensor v1=model[i1];
        torch::Tensor v2=model[i2];
        torch::Tensor v3=model[i3];
        torch::Tensor v=torch::cross(v2-v1,v3-v1);//3x1

        for(int  j=0;j<3;j++){
            pointNomrals[i1][j][0]+=v[j][0].item().toFloat();
            pointNomrals[i2][j][0]+=v[j][0].item().toFloat();
            pointNomrals[i3][j][0]+=v[j][0].item().toFloat();
        }
       // pointNomrals[i1][0][0]+=v[0][0].item().toFloat();pointNomrals[i1][1][0]+=v[1][0].item().toFloat();pointNomrals[i1][2][0]+=v[1][0].item().toFloat();
//        pointNomrals[i2]+=std::move(v);
//        pointNomrals[i3]+=std::move(v);
        counts[i1]+=1;
        counts[i2]+=1;
        counts[i3]+=1;
    }

    for(int j=0;j<pointsNum;j++){
        if(counts[j]>0){
            pointNomrals[j]/=counts[j];
        }
        float norm=torch::norm(pointNomrals[j],2,0).item().toFloat();
        pointNomrals[j]/=norm;
    }
    return std::move(pointNomrals);
}
/**
 * @brief TorchFunctions::computeFaceNormals
 * @param model     6890x3x1
 * @param f         fnumx3
 * @return          fnumx3x1
 */
at::Tensor TorchFunctions::computeFaceNormals(at::Tensor &model, at::Tensor &f)
{
    at::Tensor i0=f.select(1,0);
    at::Tensor i1=f.select(1,1);
    at::Tensor i2=f.select(1,2);
    std::cout<<"i2:"<<i2.sizes()<<std::endl;
    at::Tensor v1=torch::index_select(model,0,i0);//fnumx3x1
    at::Tensor v2=torch::index_select(model,0,i1);//fnumx3x1
    at::Tensor v3=torch::index_select(model,0,i2);//fnumx3x1
    at::Tensor fn=torch::cross(v2-v1,v3-v1);//fnumx3x1
//    at::Tensor l=fn.norm(2,1);//fnum
//    at::Tensor index=l.lt(0.00001);//fnum
//    l.add_(index.toType(l.type()));
    return std::move(fn);
}

Eigen::SparseMatrix<float, Eigen::RowMajor> TorchFunctions::computeFaceNormalsWeight(int64_t& vnum, at::Tensor &f)
{
    int64_t fnum=f.sizes()[0];
    Eigen::SparseMatrix<float,Eigen::RowMajor> weights;
    weights.resize(vnum,fnum);
    std::vector<Eigen::Triplet<float>> trips;
    trips.resize(fnum*3);
    for(int64_t i=0;i<fnum;i++){
        int r0=f[i][0].item().toInt();
        int r1=f[i][1].item().toInt();
        int r2=f[i][2].item().toInt();
        trips[3*i]=Eigen::Triplet<float>(r0,i,1.0);
        trips[3*i+1]=Eigen::Triplet<float>(r1,i,1.0);
        trips[3*i+2]=Eigen::Triplet<float>(r2,i,1.0);
    }
    weights.setFromTriplets(trips.begin(),trips.end());//vnumxfnum
    return std::move(weights);
}
/**
 * @brief TorchFunctions::computeVertNormals
 * @param model 6890x3
 * @param f     fnumx3
 * @return      vnumx3
 */
at::Tensor TorchFunctions::computeVertNormals(at::Tensor &model, at::Tensor &f)
{
    int64_t vnum=model.sizes()[0];
    Eigen::SparseMatrix<float, Eigen::RowMajor> weigths=computeFaceNormalsWeight(vnum,f);
    return computeVertNormals(model,f,weigths);
}
/**
 * @brief TorchFunctions::computeVertNormals
 * @param model     6890x3x1
 * @param f         fnumx3
 * @param weigths   vnumxfnum
 * @return          vnumx3x1
 */
at::Tensor TorchFunctions::computeVertNormals( at::Tensor &model,at::Tensor &f,Eigen::SparseMatrix<float, Eigen::RowMajor>& weigths)
{
    at::Tensor fn=computeFaceNormals(model,f).squeeze(-1).contiguous();//fnumx3x1->fnumx3
    Eigen::Matrix<float ,Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> fnormals=EigenToTorch::TorchTensorToEigenMatrix(fn);
    Eigen::Matrix<float ,Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> vnormals=weigths*fnormals;//vnumxfnum * fnumx3=vnumx3
    at::Tensor vn=EigenToTorch::EigenMatrixToTorchTensor(vnormals);//vnumx3
    at::Tensor l=vn.norm(2,1,true);//vnumx1
    at::Tensor index=l.lt(0.00001);//vnumx1
    l.add_(index.toType(l.type()));//vnumx1
    vn.div_(l);//vnumx3
    vn.unsqueeze_(-1);//vnumx3x1
    return std::move(vn);
}
/**
 * @brief TorchFunctions::Kronecker
 * @param A
 * @param B
 * @return
 */
at::Tensor TorchFunctions::Kronecker(at::Tensor &A, at::Tensor &B)
{
    assert(A.dim()==2&&B.dim()==2);
    torch::Tensor r=torch::zeros({A.sizes()[0]*B.sizes()[0],A.sizes()[1]*B.sizes()[1]});
    for(int64_t i=0;i<A.sizes()[0];i++){
        for(int64_t j=0;j<A.sizes()[1];j++){
            r.slice(0,i*B.sizes()[0],(i+1)*B.sizes()[0]).slice(1,j*B.sizes()[1],(j+1)*B.sizes()[1])=A[i][j]*B;
        }
    }
    return std::move(r);
}
