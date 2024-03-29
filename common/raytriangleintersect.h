#ifndef RAYTRIANGLEINTERSECT_H
#define RAYTRIANGLEINTERSECT_H
#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/optional.hpp>

namespace Render {
/**
 * @brief Computes the intersection of the given ray with the given triangle.
 *
 * Uses the M鰈ler-Trumbore algorithm algorithm "Fast Minimum Storage
 * Ray/Triangle Intersection". Independent implementation, inspired by:
 * http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
 * The default eps (1e-6f) is from the paper.
 * When culling is on, rays intersecting triangles from the back will be discarded -
 * otherwise, the triangles normal direction w.r.t. the ray direction is just ignored.
 *
 * Todo: We don't need the pair<> here, we could just return optional<float>. Also, I hope optional wouldn't
 * be a performance problem here, as this function is called loads of times.
 *
 * @param[in] ray_origin Ray origin.
 * @param[in] ray_direction Ray direction.
 * @param[in] v0 First vertex of a triangle.
 * @param[in] v1 Second vertex of a triangle.
 * @param[in] v2 Third vertex of a triangle.
 * @param[in] enable_backculling When culling is on, rays intersecting triangles from the back will be
 *                               discarded.
 * @return Whether the ray intersects the triangle, and if yes, including the distance.
 */
inline std::pair<bool, boost::optional<float>>
rayTriangleIntersect(const Eigen::Vector3f& ray_origin, const Eigen::Vector3f& ray_direction, const Eigen::Vector3f& v0,
                       const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, bool enable_backculling)
{
    const float epsilon = 1e-6f;

    const Eigen::Vector3f v0v1 = v1 - v0;
    const Eigen::Vector3f v0v2 = v2 - v0;

    const Eigen::Vector3f pvec = ray_direction.cross(v0v2);

    const float det = v0v1.dot(pvec);
    if (enable_backculling)
    {
        // If det is negative, the triangle is back-facing.
        // If det is close to 0, the ray misses the triangle.
        if (det < epsilon)
            return {false, boost::optional<float>{}};
    } else
    {
        // If det is close to 0, the ray and triangle are parallel.
        if (std::abs(det) < epsilon)
            return {false, boost::optional<float>{}};
    }
    const float inv_det = 1 / det;

    const Eigen::Vector3f tvec = ray_origin - v0;
    const auto u = tvec.dot(pvec) * inv_det;
    if (u < 0 || u > 1)
        return {false, boost::optional<float>{}};

    const Eigen::Vector3f qvec = tvec.cross(v0v1);
    const auto v = ray_direction.dot(qvec) * inv_det;
    if (v < 0 || u + v > 1)
        return {false, boost::optional<float>{}};

    const auto t = v0v2.dot(qvec) * inv_det;

    return {true, t};
};


}
#endif // RAYTRIANGLEINTERSECT_H
