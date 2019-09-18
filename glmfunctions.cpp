#include "glmfunctions.h"

GlmFunctions::GlmFunctions()
{

}

glm::quat GlmFunctions::RotationToQuat(torch::Tensor &ROrtho)
{
    glm::mat3x3 RGlm; // identity
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            RGlm[c][r] = ROrtho[r][c].item().toFloat();
        }
    }
    return std::move(glm::quat(RGlm));
}
glm::quat GlmFunctions::RotationToQuat(Eigen::Matrix3f &ROrtho)
{
    glm::mat3x3 RGlm; // identity
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            RGlm[c][r] = ROrtho(r, c);
        }
    }
    return std::move(glm::quat(RGlm));
}
