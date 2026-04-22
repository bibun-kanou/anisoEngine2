#include "gpu/buffer.h"
#include "core/log.h"

#include <glad/gl.h>
#include <cstring>

namespace ng {

GPUBuffer::~GPUBuffer() {
    if (handle_) glDeleteBuffers(1, &handle_);
}

GPUBuffer::GPUBuffer(GPUBuffer&& o) noexcept : handle_(o.handle_), size_(o.size_) {
    o.handle_ = 0;
    o.size_ = 0;
}

GPUBuffer& GPUBuffer::operator=(GPUBuffer&& o) noexcept {
    if (this != &o) {
        if (handle_) glDeleteBuffers(1, &handle_);
        handle_ = o.handle_;
        size_ = o.size_;
        o.handle_ = 0;
        o.size_ = 0;
    }
    return *this;
}

void GPUBuffer::create(size_t size_bytes, const void* data) {
    if (handle_) glDeleteBuffers(1, &handle_);

    glCreateBuffers(1, &handle_);
    // GL_DYNAMIC_STORAGE_BIT allows glBufferSubData. Without MAP bits we rely
    // on explicit upload/download calls.
    glNamedBufferStorage(handle_, static_cast<GLsizeiptr>(size_bytes), data,
        GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT);
    size_ = size_bytes;
}

void GPUBuffer::bind_base(u32 binding) const {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, handle_);
}

void GPUBuffer::upload(const void* data, size_t size_bytes, size_t offset) {
    glNamedBufferSubData(handle_, static_cast<GLintptr>(offset),
        static_cast<GLsizeiptr>(size_bytes), data);
}

void GPUBuffer::download(void* dest, size_t size_bytes, size_t offset) const {
    glGetNamedBufferSubData(handle_, static_cast<GLintptr>(offset),
        static_cast<GLsizeiptr>(size_bytes), dest);
}

void GPUBuffer::clear() {
    u32 zero = 0;
    glClearNamedBufferData(handle_, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
}

void GPUBuffer::clear_u32(u32 value) {
    glClearNamedBufferData(handle_, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &value);
}

} // namespace ng
