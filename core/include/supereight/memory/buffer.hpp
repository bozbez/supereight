#pragma once

#include <supereight/memory/buffer_enums.hpp>
#include <supereight/memory/buffer_traits.hpp>

#include <supereight/shared/commons.h>

#include <functional>

namespace se {

template<typename T>
class BufferAccessor;

template<typename T>
class Buffer {
public:
    Buffer() {}
    Buffer(std::size_t size) : size_{size} {}

    Buffer(Buffer&& other) : Buffer(other) {
        other.buffer_    = nullptr;
        other.allocated_ = false;
    };

    ~Buffer() {
        if (allocated_) free_(buffer_);
    }

    __attribute__((always_inline)) BufferAccessor<T> accessor(Device device) {
        switch (device) {
        case Device::CPU: return this->template accessor<Device::CPU>();
        case Device::CUDA: return this->template accessor<Device::CUDA>();
        }

        std::cout << "Unknown buffer device: "
                  << static_cast<std::underlying_type<Device>::type>(device)
                  << std::endl;

        exit(-1);
    }

    template<Device D>
    BufferAccessor<T> accessor() {
        if (allocated_ && device_ == D)
            return BufferAccessor<T>(buffer_, size_);

        if (size_ == 0) return BufferAccessor<T>(nullptr, 0);

        if (!allocated_) {
            buffer_    = buffer_traits<D>::template allocate<T>(size_);
            allocated_ = true;
        } else {
            move_buffer<D>();
        }

        device_ = D;

        free_   = &buffer_traits<D>::template free<T>;
        to_cpu_ = &buffer_traits<D>::template copy<T, CopyDir::ToCPU>;

        return BufferAccessor<T>(buffer_, size_);
    }

    void resize(std::size_t size) {
        if (size == size_) return;

        if (allocated_) free_(buffer_);
        allocated_ = false;

        size_ = size;
    }

    std::size_t size() const { return size_; }

private:
    Buffer(const Buffer&) = default;

    template<Device D>
    void move_buffer();

    T* buffer_        = nullptr;
    std::size_t size_ = 0;

    Device device_  = Device::CPU;
    bool allocated_ = false;

    void (*free_)(T*);
    void (*to_cpu_)(T*, T*, std::size_t);
};

template<typename T>
template<Device D>
inline void Buffer<T>::move_buffer() {
    std::printf("moving %d -> %d, size = %d (%d bytes)\n", D, device_, size_,
        size_ * sizeof(T));

    if (D == Device::CPU) {
        T* cpu_buffer = buffer_traits<Device::CPU>::allocate<T>(size_);
        to_cpu_(cpu_buffer, buffer_, size_);
        free_(buffer_);

        buffer_ = cpu_buffer;

        return;
    }

    if (device_ == Device::CPU) {
        T* new_buffer = buffer_traits<D>::template allocate<T>(size_);
        buffer_traits<D>::template copy<T, CopyDir::FromCPU>(
            new_buffer, buffer_, size_);

        free_(buffer_);
        buffer_ = new_buffer;

        return;
    }

    T* cpu_buffer = buffer_traits<Device::CPU>::template allocate<T>(size_);
    to_cpu_(cpu_buffer, buffer_, size_);
    free_(buffer_);

    buffer_ = buffer_traits<D>::template allocate<T>(size_);
    buffer_traits<D>::template copy<T, CopyDir::FromCPU>(
        buffer_, cpu_buffer, size_);
}

template<typename T>
class BufferAccessor {
public:
    BufferAccessor() = delete;

    SE_DEVICE_FUNC
    int size() const { return size_; }

    SE_DEVICE_FUNC
    T* data() const { return buffer_; }

    SE_DEVICE_FUNC
    T& operator[](std::size_t i) { return buffer_[i]; }

    SE_DEVICE_FUNC
    const T& operator[](std::size_t i) const { return buffer_[i]; }

private:
    BufferAccessor(T* buffer, std::size_t size)
        : buffer_{buffer}, size_{size} {}

    T* buffer_;
    std::size_t size_;

    friend class Buffer<T>;
};

} // namespace se
