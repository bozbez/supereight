#pragma once

#include <supereight/memory/buffer.hpp>
#include <supereight/shared/commons.h>

#include <cassert>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/StdVector>

namespace se {

template<typename T>
class Image : public Buffer<T> {
public:
    Image(const Eigen::Vector2i& size) : Image(size.x(), size.y()) {}

    Image(const unsigned w, const unsigned h)
        : Buffer<T>(w * h), width_(w), height_(h) {
        assert(width_ > 0 && height_ > 0);
    }

    Image(const unsigned w, const unsigned h, const T& val)
        : Buffer<T>(w * h), width_(w), height_(h) {
        assert(width_ > 0 && height_ > 0);
    }

    Image(const Image&)  = delete;
    Image(Image&& other) = default;

    Eigen::Vector2i dim() const { return Eigen::Vector2i(width(), height()); }

    int width() const { return width_; };
    int height() const { return height_; };

private:
    const int width_;
    const int height_;
};

} // end namespace se
