#include "../common/field_impls.hpp"

#include "allocate.hpp"
#include "projective_update.hpp"
#include "raycast.hpp"

#include <supereight/backend/backend.hpp>
#include <supereight/utils/cuda_util.hpp>

#include <cuda_runtime.h>

namespace se {

void Backend::allocate_(Image<float>& depth, const Eigen::Vector4f& k,
    const Eigen::Matrix4f& pose, const Eigen::Vector2i& computation_size,
    float mu) {
    float voxel_size    = octree_.dim() / octree_.size();
    int num_vox_per_pix = octree_.dim() / (BLOCK_SIDE * voxel_size);
    size_t total =
        num_vox_per_pix * computation_size.x() * computation_size.y();

    if (allocation_list_used_ == nullptr)
        safeCall(cudaMallocManaged(&allocation_list_used_, sizeof(int)));


    Buffer<se::key_t> allocation_list_;
    allocation_list_.resize(total);

    int allocated =
        se::buildAllocationList(allocation_list_.accessor(Device::CUDA),
            octree_, allocation_list_used_, pose, getCameraMatrix(k),
            depth.accessor(Device::CUDA), computation_size, mu);

    octree_.allocate(allocation_list_.accessor(Device::CPU).data(), allocated);
}

void Backend::update_(Image<float>& depth, const Sophus::SE3f& Tcw,
    const Eigen::Vector4f& k, const Eigen::Vector2i& computation_size, float mu,
    int frame) {
    float voxel_size = octree_.dim() / octree_.size();
    float timestamp  = (1.f / 30.f) * frame;

    voxel_traits<FieldType>::update_func_type func(
        depth.accessor(Device::CUDA).data(), computation_size, mu, timestamp,
        voxel_size);

    se::projectiveUpdate(
        octree_, func, Tcw, getCameraMatrix(k), computation_size);
}

void Backend::raycast_(Image<Eigen::Vector3f>& vertex,
    Image<Eigen::Vector3f>& normal, const Eigen::Vector4f& k,
    const Eigen::Matrix4f& pose, float mu) {
    if (normal.dim() != vertex.dim()) return;

    auto dim  = normal.dim();
    auto view = pose * getInverseCameraMatrix(k);

    vertex.resize(dim.prod());
    normal.resize(dim.prod());

    float step = octree_.dim() / octree_.size();
    se::raycast(octree_, vertex.accessor(Device::CUDA).data(),
        normal.accessor(Device::CUDA).data(), dim, view,
        Eigen::Vector2f(nearPlane, farPlane), mu, step);
}

void Backend::render_(unsigned char* out, const Eigen::Vector2i& output_size,
    const Eigen::Vector4f& k, const Eigen::Matrix4f& pose, float, float mu,
    Image<Eigen::Vector3f>& vertex, Image<Eigen::Vector3f>& normal,
    const Eigen::Matrix4f& raycast_view) {
    float step = octree_.dim() / octree_.size();

    Eigen::Matrix4f render_view = pose * getInverseCameraMatrix(k);

    Image<Eigen::Vector3f> new_vertex(output_size);
    Image<Eigen::Vector3f> new_normal(output_size);

    auto* r_vertex = &vertex;
    auto* r_normal = &normal;

    if (!render_view.isApprox(raycast_view) || !(output_size == vertex.dim())) {
        new_vertex.resize(output_size.prod());
        new_normal.resize(output_size.prod());

        se::raycast(octree_, new_vertex.accessor(Device::CUDA).data(),
            new_normal.accessor(Device::CUDA).data(), output_size, render_view,
            Eigen::Vector2f(nearPlane, farPlane), mu, step);

        r_vertex = &new_vertex;
        r_normal = &new_normal;
    }

    render_out_.resize(output_size.prod());
    se::render(render_out_.accessor(Device::CUDA).data(), output_size,
        r_vertex->accessor(Device::CUDA).data(),
        r_normal->accessor(Device::CUDA).data(), pose.topRightCorner<3, 1>(),
        ambient);

    safeCall(cudaMemcpy(out, render_out_.accessor(Device::CUDA).data(),
        output_size.prod() * 4, cudaMemcpyDeviceToHost));
}

} // namespace se
