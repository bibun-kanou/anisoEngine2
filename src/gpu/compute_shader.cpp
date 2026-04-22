#include "gpu/compute_shader.h"
#include "core/log.h"

#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace ng {

ComputeShader::~ComputeShader() {
    if (program_) glDeleteProgram(program_);
}

ComputeShader::ComputeShader(ComputeShader&& o) noexcept
    : program_(o.program_), uniform_cache_(std::move(o.uniform_cache_)) {
    o.program_ = 0;
}

ComputeShader& ComputeShader::operator=(ComputeShader&& o) noexcept {
    if (this != &o) {
        if (program_) glDeleteProgram(program_);
        program_ = o.program_;
        uniform_cache_ = std::move(o.uniform_cache_);
        o.program_ = 0;
    }
    return *this;
}

std::string ComputeShader::read_and_process(const std::string& path, const std::string& shader_dir) {
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_ERROR("Cannot open compute shader: %s", path.c_str());
        return "";
    }

    std::stringstream result;
    std::string line;

    while (std::getline(file, line)) {
        // Simple #include "..." parser (no regex for MSVC compat)
        auto trimmed = line;
        auto start = trimmed.find_first_not_of(" \t");
        if (start != std::string::npos && trimmed.substr(start, 9) == "#include " ) {
            auto q1 = trimmed.find('"', start + 9);
            auto q2 = (q1 != std::string::npos) ? trimmed.find('"', q1 + 1) : std::string::npos;
            if (q1 != std::string::npos && q2 != std::string::npos) {
                std::string inc_file = trimmed.substr(q1 + 1, q2 - q1 - 1);
                std::string inc_path = shader_dir + inc_file;
                std::string inc_src = read_and_process(inc_path, shader_dir);
                if (inc_src.empty()) {
                    LOG_ERROR("Failed to include: %s (from %s)", inc_path.c_str(), path.c_str());
                    return "";
                }
                result << "// --- begin include " << inc_file << " ---\n";
                result << inc_src << "\n";
                result << "// --- end include " << inc_file << " ---\n";
                continue;
            }
        }
        result << line << "\n";
    }
    return result.str();
}

bool ComputeShader::load_source(const std::string& source) {
    u32 shader = glCreateShader(GL_COMPUTE_SHADER);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info[2048];
        glGetShaderInfoLog(shader, sizeof(info), nullptr, info);
        LOG_ERROR("Compute shader compilation failed:\n%s", info);
        glDeleteShader(shader);
        return false;
    }

    program_ = glCreateProgram();
    glAttachShader(program_, shader);
    glLinkProgram(program_);
    glDeleteShader(shader);

    glGetProgramiv(program_, GL_LINK_STATUS, &success);
    if (!success) {
        char info[2048];
        glGetProgramInfoLog(program_, sizeof(info), nullptr, info);
        LOG_ERROR("Compute shader link failed:\n%s", info);
        glDeleteProgram(program_);
        program_ = 0;
        return false;
    }

    uniform_cache_.clear();
    return true;
}

bool ComputeShader::load(const std::string& path, const std::string& shader_dir) {
    std::string source = read_and_process(path, shader_dir);
    if (source.empty()) return false;

    bool ok = load_source(source);
    if (ok) LOG_INFO("Compute shader loaded: %s", path.c_str());
    return ok;
}

void ComputeShader::bind() const {
    glUseProgram(program_);
}

void ComputeShader::dispatch(u32 gx, u32 gy, u32 gz) const {
    glDispatchCompute(gx, gy, gz);
}

void ComputeShader::dispatch_1d(u32 total, u32 local_size) const {
    u32 groups = (total + local_size - 1) / local_size;
    glDispatchCompute(groups, 1, 1);
}

void ComputeShader::dispatch_2d(u32 tx, u32 ty, u32 lx, u32 ly) const {
    u32 gx = (tx + lx - 1) / lx;
    u32 gy = (ty + ly - 1) / ly;
    glDispatchCompute(gx, gy, 1);
}

void ComputeShader::barrier_ssbo() {
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void ComputeShader::barrier_image() {
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void ComputeShader::barrier_all() {
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

i32 ComputeShader::get_uniform_loc(const char* name) const {
    auto it = uniform_cache_.find(name);
    if (it != uniform_cache_.end()) return it->second;
    i32 loc = glGetUniformLocation(program_, name);
    uniform_cache_[name] = loc;
    return loc;
}

void ComputeShader::set_int(const char* name, i32 v) {
    glUniform1i(get_uniform_loc(name), v);
}

void ComputeShader::set_uint(const char* name, u32 v) {
    glUniform1ui(get_uniform_loc(name), v);
}

void ComputeShader::set_float(const char* name, f32 v) {
    glUniform1f(get_uniform_loc(name), v);
}

void ComputeShader::set_vec2(const char* name, vec2 v) {
    glUniform2fv(get_uniform_loc(name), 1, glm::value_ptr(v));
}

void ComputeShader::set_ivec2(const char* name, ivec2 v) {
    glUniform2iv(get_uniform_loc(name), 1, glm::value_ptr(v));
}

} // namespace ng
