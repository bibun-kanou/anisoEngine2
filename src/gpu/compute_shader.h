#pragma once

#include "core/types.h"
#include <string>
#include <unordered_map>

namespace ng {

class ComputeShader {
public:
    ComputeShader() = default;
    ~ComputeShader();

    ComputeShader(const ComputeShader&) = delete;
    ComputeShader& operator=(const ComputeShader&) = delete;
    ComputeShader(ComputeShader&& o) noexcept;
    ComputeShader& operator=(ComputeShader&& o) noexcept;

    // Load compute shader from file. Supports #include "common/foo.glsl" via
    // a shader_dir root (default: "shaders/")
    bool load(const std::string& path, const std::string& shader_dir = "shaders/");

    // Load from source string directly
    bool load_source(const std::string& source);

    void bind() const;
    u32 id() const { return program_; }

    // Dispatch compute work. group_count = ceil(total / local_size) per axis.
    void dispatch(u32 groups_x, u32 groups_y = 1, u32 groups_z = 1) const;

    // Dispatch with automatic group count calculation
    void dispatch_1d(u32 total, u32 local_size = 256) const;
    void dispatch_2d(u32 total_x, u32 total_y, u32 local_x = 16, u32 local_y = 16) const;

    // Insert memory barrier after dispatch
    static void barrier_ssbo();    // GL_SHADER_STORAGE_BARRIER_BIT
    static void barrier_image();   // GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
    static void barrier_all();     // GL_ALL_BARRIER_BITS

    // Uniforms
    void set_int(const char* name, i32 v);
    void set_uint(const char* name, u32 v);
    void set_float(const char* name, f32 v);
    void set_vec2(const char* name, vec2 v);
    void set_ivec2(const char* name, ivec2 v);

private:
    u32 program_ = 0;
    mutable std::unordered_map<std::string, i32> uniform_cache_;

    i32 get_uniform_loc(const char* name) const;
    static std::string read_and_process(const std::string& path, const std::string& shader_dir);
};

} // namespace ng
