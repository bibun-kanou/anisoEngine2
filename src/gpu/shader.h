#pragma once

#include "core/types.h"
#include <string>
#include <unordered_map>

namespace ng {

class Shader {
public:
    Shader() = default;
    ~Shader();

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;
    Shader(Shader&& o) noexcept;
    Shader& operator=(Shader&& o) noexcept;

    // Load and link a vertex+fragment shader program
    bool load(const std::string& vert_path, const std::string& frag_path);

    void bind() const;
    void unbind() const;
    u32 id() const { return program_; }

    // Uniforms
    void set_int(const char* name, i32 v);
    void set_float(const char* name, f32 v);
    void set_vec2(const char* name, vec2 v);
    void set_vec3(const char* name, vec3 v);
    void set_vec4(const char* name, vec4 v);
    void set_mat4(const char* name, const mat4& m);

private:
    u32 program_ = 0;
    mutable std::unordered_map<std::string, i32> uniform_cache_;

    i32 get_uniform_loc(const char* name) const;
    static u32 compile_shader(u32 type, const std::string& source);
    static std::string read_file(const std::string& path);
};

} // namespace ng
