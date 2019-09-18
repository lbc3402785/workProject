#ifndef GLMFUNCTIONS_H
#define GLMFUNCTIONS_H
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/transform.hpp"
#include <Eigen/Dense>
#include <torch/torch.h>
class GlmFunctions
{
public:
    GlmFunctions();
    static glm::quat GlmFunctions::RotationToQuat(torch::Tensor &ROrtho);
    static glm::quat RotationToQuat(Eigen::Matrix3f& R);
};

#endif // GLMFUNCTIONS_H
