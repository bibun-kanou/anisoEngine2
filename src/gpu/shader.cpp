#include "gpu/shader.h"
#include "core/log.h"

#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <sstream>

namespace ng {

Shader::~Shader() {
    if (program_) glDeleteProgram(program_);
}

Shader::Shader(Shader&& o) noexcept : program_(o.program_), uniform_cache_(std::move(o.uniform_cache_)) {
    o.program_ = 0;
}

Shader& Shader::operator=(Shader&& o) noexcept {
    if (this != &o) {
        if (program_) glDeleteProgram(program_);
        program_ = o.program_;
        uniform_cache_ = std::move(o.uniform_cache_);
        o.program_ = 0;
    }
    return *this;
}

std::string Shader::read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_ERROR("Cannot open shader file: %s", path.c_str());
        return "";
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

u32 Shader::compile_shader(u32 type, const std::string& source) {
    u32 shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info[1024];
        glGetShaderInfoLog(shader, sizeof(info), nullptr, info);
        const char* type_str = (type == GL_VERTEX_SHADER) ? "VERTEX" :
                               (type == GL_FRAGMENT_SHADER) ? "FRAGMENT" : "UNKNOWN";
        LOG_ERROR("%s shader compilation failed:\n%s", type_str, info);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

bool Shader::load(const std::string& vert_path, const std::string& frag_path) {
    std::string vert_src = read_file(vert_path);
    std::string frag_src = read_file(frag_path);
    if (vert_src.empty() || frag_src.empty()) return false;

    u32 vs = compile_shader(GL_VERTEX_SHADER, vert_src);
    u32 fs = compile_shader(GL_FRAGMENT_SHADER, frag_src);
    if (!vs || !fs) {
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        return false;
    }

    program_ = glCreateProgram();
    glAttachShader(program_, vs);
    glAttachShader(program_, fs);
    glLinkProgram(program_);

    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint success;
    glGetProgramiv(program_, GL_LINK_STATUS, &success);
    if (!success) {
        char info[1024];
        glGetProgramInfoLog(program_, sizeof(info), nullptr, info);
        LOG_ERROR("Shader link failed:\n%s", info);
        glDeleteProgram(program_);
        program_ = 0;
        return false;
    }

    uniform_cache_.clear();
    LOG_INFO("Shader linked: %s + %s", vert_path.c_str(), frag_path.c_str());
    return true;
}

void Shader::bind() const {
    glUseProgram(program_);
}

void Shader::unbind() const {
    glUseProgram(0);
}

i32 Shader::get_uniform_loc(const char* name) const {
    auto it = uniform_cache_.find(name);
    if (it != uniform_cache_.end()) return it->second;
    i32 loc = glGetUniformLocation(program_, name);
    uniform_cache_[name] = loc;
    return loc;
}

void Shader::set_int(const char* name, i32 v) {
    glUniform1i(get_uniform_loc(name), v);
}

void Shader::set_float(const char* name, f32 v) {
    glUniform1f(get_uniform_loc(name), v);
}

void Shader::set_vec2(const char* name, vec2 v) {
    glUniform2fv(get_uniform_loc(name), 1, glm::value_ptr(v));
}

void Shader::set_vec3(const char* name, vec3 v) {
    glUniform3fv(get_uniform_loc(name), 1, glm::value_ptr(v));
}

void Shader::set_vec4(const char* name, vec4 v) {
    glUniform4fv(get_uniform_loc(name), 1, glm::value_ptr(v));
}

void Shader::set_mat4(const char* name, const mat4& m) {
    glUniformMatrix4fv(get_uniform_loc(name), 1, GL_FALSE, glm::value_ptr(m));
}

} // namespace ng
