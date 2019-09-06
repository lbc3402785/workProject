#include "glmfunctions.h"

GlmFunctions::GlmFunctions()
{

}

glm::quat GlmFunctions::RotationToQuat(Eigen::Matrix3f &R_ortho)
{
    glm::mat3x3 R_glm; // identity
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            R_glm[c][r] = R_ortho(r, c);
        }
    }
    return std::move(glm::quat(R_glm));
}
