#pragma once

#include "core/types.h"
#include <cstddef>
#include <vector>
#include <span>

namespace ng {

// GPU buffer wrapper for SSBOs (Shader Storage Buffer Objects)
class GPUBuffer {
public:
    GPUBuffer() = default;
    ~GPUBuffer();

    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;
    GPUBuffer(GPUBuffer&& o) noexcept;
    GPUBuffer& operator=(GPUBuffer&& o) noexcept;

    // Create buffer with optional initial data
    void create(size_t size_bytes, const void* data = nullptr);

    // Bind to an indexed SSBO binding point (for compute shader access)
    void bind_base(u32 binding) const;

    // Upload data (full or partial)
    void upload(const void* data, size_t size_bytes, size_t offset = 0);

    // Download data from GPU to CPU
    void download(void* dest, size_t size_bytes, size_t offset = 0) const;

    // Convenience: download entire buffer into a vector
    template<typename T>
    std::vector<T> download_all() const {
        std::vector<T> result(size_ / sizeof(T));
        download(result.data(), size_);
        return result;
    }

    // Clear buffer to zero
    void clear();

    // Clear buffer to specific value (u32)
    void clear_u32(u32 value);

    u32 handle() const { return handle_; }
    size_t size() const { return size_; }

private:
    u32 handle_ = 0;
    size_t size_ = 0;
};

} // namespace ng
