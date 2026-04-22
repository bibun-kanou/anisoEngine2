#pragma once

#include "core/types.h"
#include "gpu/compute_shader.h"
#include "physics/common/particle_buffer.h"
#include "physics/common/spatial_hash.h"

namespace ng {

class SDFField;
class UniformGrid;
class EulerianFluid;

struct SPHParams {
    f32 rest_density    = 1000.0f;
    f32 gas_constant    = 8.0f;
    f32 viscosity       = 0.1f;
    f32 xsph            = 0.3f;   // XSPH velocity smoothing (0=off, 0.5=strong)
    f32 smoothing_radius = 0.04f;
    f32 particle_mass   = 0.4f;
    vec2 gravity        = vec2(0.0f, -9.81f);
    vec2 bound_min      = vec2(-2.0f, -2.0f);
    vec2 bound_max      = vec2(2.0f, 3.0f);
    bool enable_thermal = false;
    f32  ambient_temp   = 300.0f;
    vec2 heat_source_pos = vec2(0.0f, -1.0f);
    f32  heat_source_radius = 0.5f;
    f32  heat_source_temp   = 800.0f;
    vec2 heat_gun_pos    = vec2(0.0f);
    f32  heat_gun_radius = 0.4f;
    f32  heat_gun_power  = 0.0f;
    f32  particle_cooling_rate = 0.015f;
    i32 vis_mode        = 0;
    bool keep_colors    = false;
    f32  surface_tension = 0.5f;  // CSF surface tension coefficient
    bool immiscible_interfaces = true;
    f32  interface_repulsion = 18.0f;
    f32  interface_tension = 1.15f;
    f32  cross_mix = 0.22f;
    f32  cross_thermal_mix = 0.35f;
    f32  mpm_contact_push = 12.0f;
    f32  mpm_contact_damping = 3.6f;
    f32  mpm_contact_recovery = 0.18f;
    bool codim_enabled  = true;
    f32  codim_threshold = 0.25f;
    u32 highlight_start = 0;
    u32 highlight_end   = 0;
    f32 time            = 0.0f;
};

struct MouseForce {
    vec2 world_pos = vec2(0.0f);
    f32  radius    = 0.3f;
    f32  inner_radius = 0.0f;
    f32  force     = 0.0f; // >0 push, <0 pull, 0 = inactive
    vec2 drag_dir  = vec2(0.0f);
    f32  damping   = 0.0f;
    i32  mode      = 0;    // 0=radial, 1=directional brush, 2=hard-core sweep drag, 3=anchored spring drag
};

class SPHSolver {
public:
    void init();

    void set_params(const SPHParams& params) { params_ = params; }
    const SPHParams& params() const { return params_; }
    void clear_particles() { particle_count_ = 0; }

    // Spawn a rectangular block of particles
    void spawn_block(ParticleBuffer& particles, vec2 min, vec2 max, f32 spacing);

    // Spawn a circle of particles at position
    void spawn_circle(ParticleBuffer& particles, vec2 center, f32 radius, f32 spacing);

    // Spawn particles from a precomputed point set
    void spawn_points(ParticleBuffer& particles, const std::vector<vec2>& positions);

    // Run one SPH step with SDF collision and optional MPM coupling
    void step(ParticleBuffer& particles, SpatialHash& hash, f32 dt,
              const SDFField* sdf = nullptr, const MouseForce& mouse = {},
              const UniformGrid* mpm_grid = nullptr,
              const EulerianFluid* air = nullptr);
    void scatter_contact_heat(ParticleBuffer& particles, const UniformGrid& mpm_grid);

    void begin_spring_drag(ParticleBuffer& particles, vec2 center, f32 radius, f32 falloff_radius);
    void end_spring_drag();
    bool spring_drag_active() const { return spring_drag_active_; }

    u32 particle_count() const { return particle_count_; }
    const GPUBuffer& spring_anchor_buf() const { return spring_anchor_buf_; }
    const GPUBuffer& spring_weight_buf() const { return spring_weight_buf_; }
    GPUBuffer& material_param_buf() { return material_param_buf_; }
    const GPUBuffer& material_param_buf() const { return material_param_buf_; }
    GPUBuffer& thermal_coupling_buf() { return thermal_coupling_buf_; }
    const GPUBuffer& thermal_coupling_buf() const { return thermal_coupling_buf_; }

private:
    void sub_step(ParticleBuffer& particles, SpatialHash& hash, f32 dt,
                  const SDFField* sdf, const MouseForce& mouse,
                  const UniformGrid* mpm_grid, const EulerianFluid* air);

    SPHParams params_;
    u32 particle_count_ = 0;

    ComputeShader density_shader_;
    ComputeShader codim_detect_shader_;
    ComputeShader force_shader_;
    ComputeShader contact_heat_shader_;
    GPUBuffer codim_buf_; // vec4 per SPH particle: tangent, dim_ratio, scale
    GPUBuffer material_param_buf_; // vec4 per SPH particle: gas_constant, viscosity, surface scale, reserved
    GPUBuffer thermal_coupling_buf_; // vec4 per SPH particle: outgas, air heat, cooling, loft
    GPUBuffer spring_anchor_buf_;
    GPUBuffer spring_weight_buf_;
    vec2 spring_origin_ = vec2(0.0f);
    bool spring_drag_active_ = false;
};

} // namespace ng
