#pragma once

#include <supereight/memory/buffer_enums.hpp>
#include <supereight/utils/cuda_util.hpp>

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>

namespace se {

template<Device D>
struct buffer_traits {};

template<>
struct buffer_traits<Device::CPU> {
    template<typename T>
    static T* allocate(std::size_t size) {
        return static_cast<T*>(std::malloc(size * sizeof(T)));
    }

    template<typename T>
    static void free(T* buf) {
        std::free(buf);
    }

    template<typename T, CopyDir D>
    static void copy(T* dst, T* src, std::size_t size) {
        std::memcpy(dst, src, size * sizeof(T));
    }
};

template<>
struct buffer_traits<Device::CUDA> {
    template<typename T>
    static T* allocate(std::size_t size) {
        T* buf;
        safeCall(cudaMalloc(&buf, size * sizeof(T)));

        return buf;
    }

    template<typename T>
    static void free(T* buf) {
        safeCall(cudaFree(buf));
    }

    template<typename T, CopyDir D>
    struct copy_impl {
        static void copy(T* dst, T* src, std::size_t size);
    };

    template<typename T>
    struct copy_impl<T, CopyDir::FromCPU> {
        static void copy(T* dst, T* src, std::size_t size) {
            safeCall(
                cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyHostToDevice));
        }
    };

    template<typename T>
    struct copy_impl<T, CopyDir::ToCPU> {
        static void copy(T* dst, T* src, std::size_t size) {
            safeCall(
                cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost));
        }
    };

    template<typename T, CopyDir D>
    static void copy(T* dst, T* src, std::size_t size) {
        copy_impl<T, D>::copy(dst, src, size);
    }
};

} // namespace se
