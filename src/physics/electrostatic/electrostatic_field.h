#pragma once

#include "core/types.h"
#include "gpu/compute_shader.h"
#include "gpu/buffer.h"

#include <vector>

namespace ng {

class SDFField;
class ParticleBuffer;

// Electrostatic Poisson solver.
//
// Mirrors MagneticField: rasterize charge -> solve ∇²φ = -ρ/ε₀ via Jacobi ->
// E = -∇φ. Particles carry persistent scalar charge (per-particle SSBO);
// Coulomb force F = qE is applied in mpm_g2p.comp.
//
// Reuses the magnetic Jacobi + field shaders for the core Poisson pipeline
// (same math, just different source/ambient). Only the charge rasterization
// and compose passes are electrostatic-specific.
class ElectrostaticField {
public:
    struct Config {
        ivec2 resolution = ivec2(256, 256);
        vec2 world_min = vec2(-3.0f, -2.0f);
        vec2 world_max = vec2(3.0f, 4.0f);
    };

    struct Params {
        bool enabled = false;
        // Debug force-active: runs the solver every frame regardless of
        // other state (for global field overlay without needing toggles).
        bool debug_force_active = false;
        // Scale on rasterized charge before it enters the Poisson source.
        // Bundles (1/ε₀) into solver units so Coulomb forces land in a
        // visible m/s² range at moderate particle charge.
        f32 source_scale = 40.0f;
        // F = qE scale. Tuned so a particle with q~1 in a field |E|~1
        // gets ~5 m/s² — comparable to gravity.
        f32 force_scale = 5.0f;
        i32 jacobi_iterations = 60;
        // Uniform ambient E-field. Earth-analog or a laboratory capacitor
        // bias that drives all charged particles across the domain.
        vec2 ambient_E = vec2(0.0f, 0.0f);
    };

    void init(const Config& config);
    void set_params(const Params& params) { params_ = params; }
    Params& params() { return params_; }
    const Params& params() const { return params_; }

    void step(const SDFField& sdf, ParticleBuffer* particles = nullptr);

    void bind_field_for_read(u32 unit = 12) const;
    void bind_charge_ssbo(u32 binding) const;

    // Seed per-particle charge at the given global particle index range.
    // Called from MPMSolver::spawn_from_positions for charged materials.
    void init_charges(u32 global_offset, const f32* values, u32 count);

    ivec2 resolution() const { return resolution_; }
    vec2 world_min() const { return world_min_; }
    vec2 world_max() const { return world_max_; }
    bool active() const {
        return params_.enabled
            || params_.debug_force_active
            || (params_.ambient_E.x * params_.ambient_E.x +
                params_.ambient_E.y * params_.ambient_E.y) > 1e-8f;
    }

private:
    ivec2 resolution_{0};
    vec2 world_min_{0.0f};
    vec2 world_max_{0.0f};

    Params params_{};

    u32 charge_tex_ = 0;
    u32 source_tex_ = 0;
    u32 phi_tex_ = 0;
    u32 phi2_tex_ = 0;
    u32 e_field_tex_ = 0;

    // Per-particle scalar charge (vec2 .x used so we can reuse the
    // magnetic-buffer-layout mental model; .y reserved for future
    // triboelectric state like surface charge buildup).
    GPUBuffer particle_charge_buf_;
    // Integer fixed-point accumulator for atomic rasterization.
    GPUBuffer particle_charge_accum_buf_;

    ComputeShader rasterize_shader_;
    ComputeShader compose_shader_;
    ComputeShader jacobi_shader_;
    ComputeShader field_shader_;
};

} // namespace ng
