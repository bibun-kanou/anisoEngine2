#include "physics/electrostatic/electrostatic_field.h"

#include "physics/common/particle_buffer.h"
#include "physics/sdf/sdf_field.h"
#include "core/log.h"

#include <glad/gl.h>
#include <algorithm>
#include <cmath>

namespace ng {

namespace {

// SSBO bindings — chosen to not collide with magnetic (18-22).
constexpr u32 kParticleChargeAccumBinding = 23u;
constexpr u32 kParticleChargeBinding      = 24u;
// Per-particle charge buffer is sized like MagneticField's M_prev.
constexpr u32 kMaxParticles = 500000u;

void init_scalar_texture(u32& texture, ivec2 resolution) {
    if (texture) glDeleteTextures(1, &texture);
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);
    glTextureStorage2D(texture, 1, GL_R32F, resolution.x, resolution.y);
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

void ElectrostaticField::init(const Config& config) {
    resolution_ = config.resolution;
    world_min_ = config.world_min;
    world_max_ = config.world_max;

    init_scalar_texture(charge_tex_, resolution_);
    init_scalar_texture(source_tex_, resolution_);
    init_scalar_texture(phi_tex_, resolution_);
    init_scalar_texture(phi2_tex_, resolution_);
    init_vec4_texture(e_field_tex_, resolution_);

    particle_charge_accum_buf_.create(static_cast<size_t>(resolution_.x * resolution_.y) * sizeof(i32));
    particle_charge_buf_.create(static_cast<size_t>(kMaxParticles) * sizeof(f32));
    particle_charge_buf_.clear();

    rasterize_shader_.load("shaders/physics/charge_rasterize.comp");
    compose_shader_.load("shaders/physics/charge_compose.comp");
    // Jacobi + field are shared with the magnetic solver — Poisson is
    // Poisson, whether the potential is magnetic or electric.
    jacobi_shader_.load("shaders/physics/magnetic_jacobi.comp");
    field_shader_.load("shaders/physics/magnetic_field.comp");

    // Zero textures so a first-frame overlay reads sane data.
    const float zero = 0.0f;
    const float zero_rgba[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    glClearTexImage(charge_tex_, 0, GL_RED, GL_FLOAT, &zero);
    glClearTexImage(source_tex_, 0, GL_RED, GL_FLOAT, &zero);
    glClearTexImage(phi_tex_, 0, GL_RED, GL_FLOAT, &zero);
    glClearTexImage(phi2_tex_, 0, GL_RED, GL_FLOAT, &zero);
    glClearTexImage(e_field_tex_, 0, GL_RGBA, GL_FLOAT, zero_rgba);

    LOG_INFO("ElectrostaticField: %dx%d, world [%.1f,%.1f]-[%.1f,%.1f]",
             resolution_.x, resolution_.y,
             world_min_.x, world_min_.y, world_max_.x, world_max_.y);
}

void ElectrostaticField::step(const SDFField& /*sdf*/, ParticleBuffer* particles) {
    if (!active()) return;

    const float zero = 0.0f;
    glClearTexImage(charge_tex_, 0, GL_RED, GL_FLOAT, &zero);
    glClearTexImage(source_tex_, 0, GL_RED, GL_FLOAT, &zero);
    particle_charge_accum_buf_.clear();

    // Particle charge rasterization.
    if (particles) {
        const auto& range = particles->range(SolverType::MPM);
        if (range.count > 0) {
            particle_charge_accum_buf_.bind_base(kParticleChargeAccumBinding);
            particle_charge_buf_.bind_base(kParticleChargeBinding);
            particles->positions().bind_base(Binding::POSITION);
            particles->material_ids().bind_base(Binding::MATERIAL_ID);

            rasterize_shader_.bind();
            rasterize_shader_.set_uint("u_offset", range.offset);
            rasterize_shader_.set_uint("u_count", range.count);
            rasterize_shader_.set_ivec2("u_resolution", resolution_);
            rasterize_shader_.set_vec2("u_world_min", world_min_);
            rasterize_shader_.set_vec2("u_world_max", world_max_);
            rasterize_shader_.dispatch_1d(range.count);
            ComputeShader::barrier_ssbo();

            // Compose: decode accumulator → source_tex with -source_scale * ρ.
            glBindImageTexture(0, source_tex_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
            compose_shader_.bind();
            compose_shader_.set_ivec2("u_resolution", resolution_);
            // ∇²φ = -ρ/ε₀  →  source = -ρ * source_scale.
            // Negated + scaled in the compose shader.
            compose_shader_.set_float("u_accum_scale", -params_.source_scale / 256.0f);
            compose_shader_.dispatch_2d(static_cast<u32>(resolution_.x), static_cast<u32>(resolution_.y));
            ComputeShader::barrier_image();
        }
    }

    // Jacobi iterations — solve ∇²φ = source.
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

    // E = -∇φ + ambient. Reuses the magnetic_field.comp shader which
    // writes vec4(E.x, E.y, |E|, |E|²) — perfect for the Coulomb force
    // and the overlay.
    glBindTextureUnit(0, phi_tex_);
    glBindImageTexture(1, e_field_tex_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    field_shader_.bind();
    field_shader_.set_ivec2("u_resolution", resolution_);
    field_shader_.set_vec2("u_world_min", world_min_);
    field_shader_.set_vec2("u_world_max", world_max_);
    field_shader_.set_vec2("u_ambient_H", params_.ambient_E);
    field_shader_.dispatch_2d(static_cast<u32>(resolution_.x), static_cast<u32>(resolution_.y));
    ComputeShader::barrier_image();
}

void ElectrostaticField::bind_field_for_read(u32 unit) const {
    glBindTextureUnit(unit, e_field_tex_);
}

void ElectrostaticField::bind_charge_ssbo(u32 binding) const {
    particle_charge_buf_.bind_base(binding);
}

void ElectrostaticField::init_charges(u32 global_offset, const f32* values, u32 count) {
    if (count == 0 || !values) return;
    particle_charge_buf_.upload(values, count * sizeof(f32), global_offset * sizeof(f32));
}

} // namespace ng
