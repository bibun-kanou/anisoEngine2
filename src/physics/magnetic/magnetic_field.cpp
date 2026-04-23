#include "physics/magnetic/magnetic_field.h"

#include "physics/common/particle_buffer.h"
#include "physics/sdf/sdf_field.h"
#include "core/log.h"

#include <glad/gl.h>
#include <algorithm>
#include <cmath>

namespace ng {

namespace {

constexpr u32 kObjectBinding = 18u;
constexpr u32 kParticleMagXBinding = 19u;
constexpr u32 kParticleMagYBinding = 20u;
constexpr u32 kParticleOccBinding = 21u;

void init_scalar_texture(u32& texture, ivec2 resolution) {
    if (texture) glDeleteTextures(1, &texture);
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);
    glTextureStorage2D(texture, 1, GL_R32F, resolution.x, resolution.y);
    glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void init_vec2_texture(u32& texture, ivec2 resolution) {
    if (texture) glDeleteTextures(1, &texture);
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);
    glTextureStorage2D(texture, 1, GL_RG32F, resolution.x, resolution.y);
    glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void init_vec4_texture(u32& texture, ivec2 resolution) {
    if (texture) glDeleteTextures(1, &texture);
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);
    glTextureStorage2D(texture, 1, GL_RGBA32F, resolution.x, resolution.y);
    glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

} // namespace

void MagneticField::init(const Config& config) {
    resolution_ = config.resolution;
    world_min_ = config.world_min;
    world_max_ = config.world_max;

    init_vec4_texture(magnetization_tex_, resolution_);
    init_scalar_texture(source_tex_, resolution_);
    init_scalar_texture(phi_tex_, resolution_);
    init_scalar_texture(phi2_tex_, resolution_);
    init_vec4_texture(drive_field_tex_, resolution_);
    init_vec4_texture(field_tex_, resolution_);

    object_buffer_.create(sizeof(ObjectMagneticGPU));
    particle_magnet_x_buf_.create(static_cast<size_t>(resolution_.x * resolution_.y) * sizeof(i32));
    particle_magnet_y_buf_.create(static_cast<size_t>(resolution_.x * resolution_.y) * sizeof(i32));
    particle_occ_buf_.create(static_cast<size_t>(resolution_.x * resolution_.y) * sizeof(i32));

    raster_shader_.load("shaders/physics/magnetic_rasterize.comp");
    particle_shader_.load("shaders/physics/magnetic_particles.comp");
    compose_shader_.load("shaders/physics/magnetic_compose.comp");
    source_shader_.load("shaders/physics/magnetic_source.comp");
    jacobi_shader_.load("shaders/physics/magnetic_jacobi.comp");
    field_shader_.load("shaders/physics/magnetic_field.comp");

    clear_textures();

    LOG_INFO("MagneticField: %dx%d, world [%.1f,%.1f]-[%.1f,%.1f]",
             resolution_.x, resolution_.y,
             world_min_.x, world_min_.y, world_max_.x, world_max_.y);
}

void MagneticField::clear_textures() {
    const float zero = 0.0f;
    const float zero_rgba[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    glClearTexImage(magnetization_tex_, 0, GL_RGBA, GL_FLOAT, zero_rgba);
    glClearTexImage(source_tex_, 0, GL_RED, GL_FLOAT, &zero);
    glClearTexImage(phi_tex_, 0, GL_RED, GL_FLOAT, &zero);
    glClearTexImage(phi2_tex_, 0, GL_RED, GL_FLOAT, &zero);
    glClearTexImage(drive_field_tex_, 0, GL_RGBA, GL_FLOAT, zero_rgba);
    glClearTexImage(field_tex_, 0, GL_RGBA, GL_FLOAT, zero_rgba);
    debug_cache_dirty_ = true;
}

void MagneticField::upload_object_magnetics(const SDFField& sdf) {
    const auto& objects = sdf.objects();
    if (objects.empty()) {
        ObjectMagneticGPU dummy{};
        object_buffer_.upload(&dummy, sizeof(dummy));
        return;
    }

    std::vector<ObjectMagneticGPU> gpu(objects.size());
    for (size_t i = 0; i < objects.size(); ++i) {
        const auto& material = objects[i].material;
        auto& dst = gpu[i];
        dst.mode = static_cast<u32>(material.magnetic_mode);
        dst.strength = material.magnetic_strength;
        dst.susceptibility = material.magnetic_susceptibility;
        vec2 dir = material.magnetic_dir;
        if (glm::length(dir) < 1e-5f) dir = vec2(1.0f, 0.0f);
        dst.dir = glm::normalize(dir);
    }

    const size_t bytes = gpu.size() * sizeof(ObjectMagneticGPU);
    if (bytes > object_buffer_.size()) {
        object_buffer_.create(bytes);
    }
    object_buffer_.upload(gpu.data(), bytes);
}

void MagneticField::step(const SDFField& sdf, ParticleBuffer* particles) {
    // Early-out must happen BEFORE clear_textures, otherwise the frame
    // immediately following a toggle-on of debug_force_active (or any
    // other activation path) renders from an all-zero field for one
    // frame. With clear moved inside the active branch, the texture
    // simply retains its last solved state while inactive and gets
    // overwritten as soon as the solve runs again.
    if (!active()) {
        return;
    }
    clear_textures();

    upload_object_magnetics(sdf);
    object_buffer_.bind_base(kObjectBinding);

    vec2 cursor_dir = params_.cursor_dir;
    if (glm::length(cursor_dir) < 1e-5f) cursor_dir = vec2(0.0f, 1.0f);
    const float zero = 0.0f;
    const float zero_rgba[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    auto rasterize_magnetization = [&](bool include_soft, u32 prev_field_tex) {
        glClearTexImage(magnetization_tex_, 0, GL_RGBA, GL_FLOAT, zero_rgba);
        glBindImageTexture(0, magnetization_tex_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glBindTextureUnit(0, sdf.texture());
        glBindTextureUnit(1, sdf.object_id_texture());
        glBindTextureUnit(2, prev_field_tex);
        raster_shader_.bind();
        raster_shader_.set_ivec2("u_resolution", resolution_);
        raster_shader_.set_vec2("u_world_min", world_min_);
        raster_shader_.set_vec2("u_world_max", world_max_);
        raster_shader_.set_vec2("u_sdf_world_min", sdf.world_min());
        raster_shader_.set_vec2("u_sdf_world_max", sdf.world_max());
        raster_shader_.set_ivec2("u_sdf_resolution", sdf.resolution());
        raster_shader_.set_uint("u_object_count", static_cast<u32>(sdf.objects().size()));
        raster_shader_.set_float("u_source_scale", params_.source_scale);
        raster_shader_.set_float("u_rigid_permanent_scale", params_.rigid_permanent_scale);
        raster_shader_.set_float("u_rigid_soft_scale", params_.rigid_soft_scale);
        // Cursor = dragged bar magnet: injected as vector M here, not as scalar source.
        raster_shader_.set_int("u_use_cursor",
            std::abs(params_.cursor_strength) > 1e-4f ? 1 : 0);
        raster_shader_.set_vec2("u_cursor_pos", params_.cursor_pos);
        raster_shader_.set_vec2("u_cursor_dir", glm::normalize(cursor_dir));
        raster_shader_.set_float("u_cursor_inner_radius", params_.cursor_radius);
        raster_shader_.set_float("u_cursor_falloff_radius", std::max(params_.cursor_falloff_radius, params_.cursor_radius));
        raster_shader_.set_float("u_cursor_strength", params_.cursor_strength);
        raster_shader_.set_int("u_cursor_field_type", static_cast<i32>(params_.cursor_field_type));
        raster_shader_.set_int("u_include_soft", include_soft ? 1 : 0);
        // Scene magnets (permanent + soft iron) participate when either
        // the user enables Real Magnetics OR the debug-force-active
        // override is on (so scene-wide field viz works without the
        // user having to hold M or toggle Real Magnetics).
        const int scene_flag = (params_.enabled || params_.debug_force_active) ? 1 : 0;
        raster_shader_.set_int("u_include_scene", scene_flag);
        raster_shader_.dispatch_2d(static_cast<u32>(resolution_.x), static_cast<u32>(resolution_.y));
        ComputeShader::barrier_image();
    };

    auto rasterize_particle_magnetization = [&](u32 prev_field_tex) {
        if (!particles) return;
        const auto& range = particles->range(SolverType::MPM);
        if (range.count == 0) return;

        particle_magnet_x_buf_.clear();
        particle_magnet_y_buf_.clear();
        particle_occ_buf_.clear();
        particle_magnet_x_buf_.bind_base(kParticleMagXBinding);
        particle_magnet_y_buf_.bind_base(kParticleMagYBinding);
        particle_occ_buf_.bind_base(kParticleOccBinding);
        particles->positions().bind_base(Binding::POSITION);
        particles->temperatures().bind_base(Binding::TEMPERATURE);
        particles->material_ids().bind_base(Binding::MATERIAL_ID);

        glBindTextureUnit(0, prev_field_tex);
        particle_shader_.bind();
        particle_shader_.set_uint("u_offset", range.offset);
        particle_shader_.set_uint("u_count", range.count);
        particle_shader_.set_ivec2("u_resolution", resolution_);
        particle_shader_.set_vec2("u_world_min", world_min_);
        particle_shader_.set_vec2("u_world_max", world_max_);
        particle_shader_.set_float("u_source_scale", params_.source_scale);
        particle_shader_.dispatch_1d(range.count);
        ComputeShader::barrier_ssbo();

        glBindImageTexture(0, magnetization_tex_, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        compose_shader_.bind();
        compose_shader_.set_ivec2("u_resolution", resolution_);
        compose_shader_.set_float("u_accum_scale", 1.0f / 256.0f);
        compose_shader_.dispatch_2d(static_cast<u32>(resolution_.x), static_cast<u32>(resolution_.y));
        ComputeShader::barrier_image();
    };

    auto solve_field_from_magnetization = [&](u32 target_tex) {
        glBindImageTexture(1, source_tex_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glBindTextureUnit(0, magnetization_tex_);
        source_shader_.bind();
        source_shader_.set_ivec2("u_resolution", resolution_);
        source_shader_.set_vec2("u_world_min", world_min_);
        source_shader_.set_vec2("u_world_max", world_max_);
        source_shader_.dispatch_2d(static_cast<u32>(resolution_.x), static_cast<u32>(resolution_.y));
        ComputeShader::barrier_image();

        glClearTexImage(phi_tex_, 0, GL_RED, GL_FLOAT, &zero);
        glClearTexImage(phi2_tex_, 0, GL_RED, GL_FLOAT, &zero);
        const i32 iterations = std::max(params_.jacobi_iterations, 1);
        for (i32 i = 0; i < iterations; ++i) {
            glBindTextureUnit(0, phi_tex_);
            glBindTextureUnit(1, source_tex_);
            glBindImageTexture(2, phi2_tex_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
            jacobi_shader_.bind();
            jacobi_shader_.set_ivec2("u_resolution", resolution_);
            jacobi_shader_.set_vec2("u_world_min", world_min_);
            jacobi_shader_.set_vec2("u_world_max", world_max_);
            jacobi_shader_.dispatch_2d(static_cast<u32>(resolution_.x), static_cast<u32>(resolution_.y));
            ComputeShader::barrier_image();
            std::swap(phi_tex_, phi2_tex_);
        }

        glBindTextureUnit(0, phi_tex_);
        glBindImageTexture(1, target_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        field_shader_.bind();
        field_shader_.set_ivec2("u_resolution", resolution_);
        field_shader_.set_vec2("u_world_min", world_min_);
        field_shader_.set_vec2("u_world_max", world_max_);
        // Added to every cell after grad(-phi). Seeds the ferrofluid so a
        // free-standing puddle can magnetize and form Rosensweig spikes
        // without any external magnet.
        field_shader_.set_vec2("u_ambient_H", params_.ambient_H);
        field_shader_.dispatch_2d(static_cast<u32>(resolution_.x), static_cast<u32>(resolution_.y));
        ComputeShader::barrier_image();
    };

    rasterize_magnetization(false, drive_field_tex_);
    solve_field_from_magnetization(drive_field_tex_);

    const i32 induction_iterations = std::max(params_.induction_iterations, 0);
    for (i32 i = 0; i < induction_iterations; ++i) {
        // Scene drive field: permanent magnets + induced soft SDF. On the final
        // induction iteration we fold in particle (ferrofluid) magnetization so
        // the drive field particles actually read contains the demagnetizing
        // contribution from the fluid itself. Without this the Rosensweig spike
        // spacing is not self-regulated — spikes can pack denser than physical
        // or collapse under perturbation because there's no magnetic volume cost.
        rasterize_magnetization(true, drive_field_tex_);
        if (i == induction_iterations - 1) {
            rasterize_particle_magnetization(drive_field_tex_);
        }
        solve_field_from_magnetization(drive_field_tex_);
    }

    // If the user configured zero induction iterations, still add one pass that
    // folds in particle magnetization so drive includes demag feedback.
    if (induction_iterations == 0) {
        rasterize_magnetization(true, drive_field_tex_);
        rasterize_particle_magnetization(drive_field_tex_);
        solve_field_from_magnetization(drive_field_tex_);
    }

    // Debug/visual field: scene drive plus induced particle magnetization.
    rasterize_magnetization(true, drive_field_tex_);
    rasterize_particle_magnetization(drive_field_tex_);
    solve_field_from_magnetization(field_tex_);

    debug_cache_dirty_ = true;
}

void MagneticField::bind_field_for_read(u32 unit) const {
    glBindTextureUnit(unit, drive_field_tex_);
}

void MagneticField::bind_total_field_for_read(u32 unit) const {
    glBindTextureUnit(unit, field_tex_);
}

void MagneticField::bind_magnetization_for_read(u32 unit) const {
    glBindTextureUnit(unit, magnetization_tex_);
}

void MagneticField::ensure_debug_cache() {
    if (!debug_cache_dirty_) return;
    debug_field_cache_.resize(static_cast<size_t>(resolution_.x) * static_cast<size_t>(resolution_.y));
    debug_total_field_cache_.resize(static_cast<size_t>(resolution_.x) * static_cast<size_t>(resolution_.y));
    debug_magnetization_cache_.resize(static_cast<size_t>(resolution_.x) * static_cast<size_t>(resolution_.y));
    debug_source_cache_.resize(static_cast<size_t>(resolution_.x) * static_cast<size_t>(resolution_.y));
    glGetTextureImage(drive_field_tex_, 0, GL_RGBA, GL_FLOAT,
                      static_cast<GLsizei>(debug_field_cache_.size() * sizeof(vec4)),
                      debug_field_cache_.data());
    glGetTextureImage(field_tex_, 0, GL_RGBA, GL_FLOAT,
                      static_cast<GLsizei>(debug_total_field_cache_.size() * sizeof(vec4)),
                      debug_total_field_cache_.data());
    glGetTextureImage(magnetization_tex_, 0, GL_RG, GL_FLOAT,
                      static_cast<GLsizei>(debug_magnetization_cache_.size() * sizeof(vec2)),
                      debug_magnetization_cache_.data());
    glGetTextureImage(source_tex_, 0, GL_RED, GL_FLOAT,
                      static_cast<GLsizei>(debug_source_cache_.size() * sizeof(float)),
                      debug_source_cache_.data());
    debug_cache_dirty_ = false;
}

vec4 MagneticField::sample_debug(vec2 world_pos) {
    if (resolution_.x <= 0 || resolution_.y <= 0) return vec4(0.0f);
    ensure_debug_cache();

    vec2 uv = (world_pos - world_min_) / (world_max_ - world_min_);
    uv = glm::clamp(uv, vec2(0.0f), vec2(1.0f));
    vec2 tex = glm::clamp(uv * vec2(resolution_ - ivec2(1)), vec2(0.0f), vec2(resolution_ - ivec2(1)));
    ivec2 i0 = ivec2(glm::floor(tex));
    ivec2 i1 = glm::min(i0 + ivec2(1), resolution_ - ivec2(1));
    vec2 f = glm::fract(tex);

    auto fetch = [&](ivec2 p) -> vec4 {
        size_t idx = static_cast<size_t>(p.y) * static_cast<size_t>(resolution_.x) + static_cast<size_t>(p.x);
        if (idx >= debug_field_cache_.size()) return vec4(0.0f);
        return debug_field_cache_[idx];
    };

    vec4 c00 = fetch(i0);
    vec4 c10 = fetch(ivec2(i1.x, i0.y));
    vec4 c01 = fetch(ivec2(i0.x, i1.y));
    vec4 c11 = fetch(i1);
    vec4 cx0 = glm::mix(c00, c10, f.x);
    vec4 cx1 = glm::mix(c01, c11, f.x);
    return glm::mix(cx0, cx1, f.y);
}

vec4 MagneticField::sample_total_debug(vec2 world_pos) {
    if (resolution_.x <= 0 || resolution_.y <= 0) return vec4(0.0f);
    ensure_debug_cache();

    vec2 uv = (world_pos - world_min_) / (world_max_ - world_min_);
    uv = glm::clamp(uv, vec2(0.0f), vec2(1.0f));
    vec2 tex = glm::clamp(uv * vec2(resolution_ - ivec2(1)), vec2(0.0f), vec2(resolution_ - ivec2(1)));
    ivec2 i0 = ivec2(glm::floor(tex));
    ivec2 i1 = glm::min(i0 + ivec2(1), resolution_ - ivec2(1));
    vec2 f = glm::fract(tex);

    auto fetch = [&](ivec2 p) -> vec4 {
        size_t idx = static_cast<size_t>(p.y) * static_cast<size_t>(resolution_.x) + static_cast<size_t>(p.x);
        if (idx >= debug_total_field_cache_.size()) return vec4(0.0f);
        return debug_total_field_cache_[idx];
    };

    vec4 c00 = fetch(i0);
    vec4 c10 = fetch(ivec2(i1.x, i0.y));
    vec4 c01 = fetch(ivec2(i0.x, i1.y));
    vec4 c11 = fetch(i1);
    vec4 cx0 = glm::mix(c00, c10, f.x);
    vec4 cx1 = glm::mix(c01, c11, f.x);
    return glm::mix(cx0, cx1, f.y);
}

vec4 MagneticField::sample_magnetization_debug(vec2 world_pos) {
    if (resolution_.x <= 0 || resolution_.y <= 0) return vec4(0.0f);
    ensure_debug_cache();

    vec2 uv = (world_pos - world_min_) / (world_max_ - world_min_);
    uv = glm::clamp(uv, vec2(0.0f), vec2(1.0f));
    vec2 tex = glm::clamp(uv * vec2(resolution_ - ivec2(1)), vec2(0.0f), vec2(resolution_ - ivec2(1)));
    ivec2 i0 = ivec2(glm::floor(tex));
    ivec2 i1 = glm::min(i0 + ivec2(1), resolution_ - ivec2(1));
    vec2 f = glm::fract(tex);

    auto fetch = [&](ivec2 p) -> vec2 {
        size_t idx = static_cast<size_t>(p.y) * static_cast<size_t>(resolution_.x) + static_cast<size_t>(p.x);
        if (idx >= debug_magnetization_cache_.size()) return vec2(0.0f);
        return debug_magnetization_cache_[idx];
    };

    vec2 c00 = fetch(i0);
    vec2 c10 = fetch(ivec2(i1.x, i0.y));
    vec2 c01 = fetch(ivec2(i0.x, i1.y));
    vec2 c11 = fetch(i1);
    vec2 cx0 = glm::mix(c00, c10, f.x);
    vec2 cx1 = glm::mix(c01, c11, f.x);
    vec2 m = glm::mix(cx0, cx1, f.y);
    float mag = glm::length(m);
    return vec4(m, mag, mag * mag);
}

float MagneticField::sample_source_debug(vec2 world_pos) {
    if (resolution_.x <= 0 || resolution_.y <= 0) return 0.0f;
    ensure_debug_cache();

    vec2 uv = (world_pos - world_min_) / (world_max_ - world_min_);
    uv = glm::clamp(uv, vec2(0.0f), vec2(1.0f));
    vec2 tex = glm::clamp(uv * vec2(resolution_ - ivec2(1)), vec2(0.0f), vec2(resolution_ - ivec2(1)));
    ivec2 i0 = ivec2(glm::floor(tex));
    ivec2 i1 = glm::min(i0 + ivec2(1), resolution_ - ivec2(1));
    vec2 f = glm::fract(tex);

    auto fetch = [&](ivec2 p) -> float {
        size_t idx = static_cast<size_t>(p.y) * static_cast<size_t>(resolution_.x) + static_cast<size_t>(p.x);
        if (idx >= debug_source_cache_.size()) return 0.0f;
        return debug_source_cache_[idx];
    };

    float c00 = fetch(i0);
    float c10 = fetch(ivec2(i1.x, i0.y));
    float c01 = fetch(ivec2(i0.x, i1.y));
    float c11 = fetch(i1);
    float cx0 = glm::mix(c00, c10, f.x);
    float cx1 = glm::mix(c01, c11, f.x);
    return glm::mix(cx0, cx1, f.y);
}

} // namespace ng
