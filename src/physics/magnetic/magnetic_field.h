#pragma once

#include "core/types.h"
#include "gpu/compute_shader.h"
#include "gpu/buffer.h"

#include <vector>

namespace ng {

class SDFField;
class ParticleBuffer;

class MagneticField {
public:
    enum class CursorFieldType : i32 {
        PROBE_POLE = 0,
        BAR_MAGNET = 1,
        WIDE_POLE = 2,
        HORSESHOE = 3
    };

    struct Config {
        ivec2 resolution = ivec2(256, 256);
        vec2 world_min = vec2(-3.0f, -2.0f);
        vec2 world_max = vec2(3.0f, 4.0f);
    };

    struct Params {
        bool enabled = false;
        // Debug-only override: when true, the solver runs every frame
        // regardless of `enabled` or cursor state. Scene magnets also
        // participate (as if `enabled == true`) so all debug views can
        // show the full solved field without the user holding M.
        // This is purely a visualization aid; physics behavior is
        // unchanged whether or not debug is on, because the solved
        // field is read by the same G2P Kelvin force path.
        bool debug_force_active = false;
        // Particle self-rasterized-M scale. Was 24 — but at that magnitude
        // ferrofluid particles' contribution to the Poisson-solved H field
        // dominated any far-field external source, so the dipole-dipole
        // self-attraction at clusters overwhelmed scene-magnet gradients
        // and the pond would compact onto itself even when the cursor /
        // scene magnet was distant. 10 keeps cluster behavior visible
        // (still chains, still spikes) but lets external sources actually
        // compete with self-feedback. Tunable via Solver Strength.
        f32 source_scale = 10.0f;
        // force_scale bundles (μ₀/ρ) conversion from solver |H|² units
        // into particle acceleration. Previously tuned to ~1 for subtle
        // behavior; now raised to 1.6 so a single moderate-strength
        // external source (cursor, scene magnet, or ambient) produces
        // visibly climbing ferrofluid. The external-drive gate in
        // mpm_g2p.comp prevents this higher scale from triggering
        // runaway self-attraction under weak ambient.
        f32 force_scale = 1.6f;
        i32 jacobi_iterations = 72;
        i32 induction_iterations = 3;
        f32 rigid_permanent_scale = 0.18f;
        f32 rigid_soft_scale = 0.10f;
        vec2 cursor_pos = vec2(0.0f);
        vec2 cursor_dir = vec2(0.0f, 1.0f);
        f32 cursor_radius = 0.35f;
        f32 cursor_falloff_radius = 0.65f;
        f32 cursor_strength = 0.0f;
        CursorFieldType cursor_field_type = CursorFieldType::PROBE_POLE;
        // Uniform ambient H field added on top of everything else. Real-world
        // analog is the Earth's ~50 µT ambient magnetic field, which is what
        // gives a puddle of ferrofluid a "default" field to magnetize in,
        // letting it self-organize into Rosensweig spikes and small clumps
        // without any external magnet. Zero by default — toggled and tuned
        // through the "Ambient Field" UI.
        vec2 ambient_H = vec2(0.0f, 0.0f);

        // Per-mechanic debug toggles. Each one disables a specific piece
        // of the magnetic pipeline so the user can A/B which sub-mechanic
        // is producing wrong behavior. All true by default = full physics.
        struct Toggles {
            bool include_scene_sources = true;     // SDF permanent + soft iron rasterization
            bool include_cursor_brush = true;      // M-held brush rasterization
            bool include_particle_demag = true;    // ferro M scattered into drive field
            bool run_induction_passes = true;      // multi-iter soft-iron self-consistency
            bool apply_kelvin_force = true;        // F = (chi/2) grad|H|^2 in mpm_g2p
            bool apply_chain_force = true;         // F = grad(M·H) anisotropic dipole-pair force, post-MPM pass
            bool apply_surface_force = true;       // ferro_surface_force capillary spikes
            bool apply_ext_drive_gate = true;      // ambient/cursor/scene gate (vs always 1)
            bool apply_hf_suppress = true;         // sharpness damping (vs always 1)
        } toggles;
    };

    void init(const Config& config);
    void set_params(const Params& params) { params_ = params; }
    Params& params() { return params_; }
    const Params& params() const { return params_; }

    void step(const SDFField& sdf, ParticleBuffer* particles = nullptr);

    void bind_field_for_read(u32 unit = 4) const;
    void bind_total_field_for_read(u32 unit = 4) const;
    void bind_magnetization_for_read(u32 unit = 4) const;

    // True iff any SDF object in the scene has a nonzero magnetic mode.
    // Used by the Kelvin-force external-drive gate: when Real Magnetics is
    // on but no actual scene magnets exist, we still want the particle
    // self-feedback safety damping active.
    bool has_scene_magnet_sources() const { return scene_magnet_count_ > 0; }
    // Bind the per-particle persistent magnetization buffer to a given
    // SSBO binding. Used by mpm_g2p.comp to read M_prev for the
    // anisotropic dipole-dipole force (chain formation).
    void bind_m_prev_ssbo(u32 binding) const;

    // Apply the anisotropic dipole-dipole pair force F = ∇(M_p · H) to
    // liquid ferro particle velocities — the missing ingredient for real
    // Rosensweig chain patterns. Dedicated post-MPM compute pass with a
    // narrow SSBO set.
    void apply_chain_force(ParticleBuffer& particles, f32 dt);

    vec4 sample_debug(vec2 world_pos);
    vec4 sample_total_debug(vec2 world_pos);
    vec4 sample_magnetization_debug(vec2 world_pos);
    float sample_source_debug(vec2 world_pos);

    ivec2 resolution() const { return resolution_; }
    vec2 world_min() const { return world_min_; }
    vec2 world_max() const { return world_max_; }
    bool active() const {
        return params_.enabled
            || std::abs(params_.cursor_strength) > 1e-4f
            || params_.debug_force_active
            // Ambient field is an independent source — must keep the
            // solver running so it gets rasterized + reaches ferrofluid.
            || (params_.ambient_H.x * params_.ambient_H.x +
                params_.ambient_H.y * params_.ambient_H.y) > 1e-8f;
    }

private:
    struct ObjectMagneticGPU {
        u32 mode = 0u;
        f32 strength = 0.0f;
        f32 susceptibility = 0.0f;
        vec2 dir = vec2(1.0f, 0.0f);
    };

    void upload_object_magnetics(const SDFField& sdf);
    void clear_textures();
    void ensure_debug_cache();

    ivec2 resolution_{0};
    vec2 world_min_{0.0f};
    vec2 world_max_{0.0f};

    Params params_{};

    u32 magnetization_tex_ = 0;
    u32 source_tex_ = 0;
    // Cached after each upload_object_magnetics() — zero when no SDF object
    // has a magnetic mode active.
    u32 scene_magnet_count_ = 0;
    u32 phi_tex_ = 0;
    u32 phi2_tex_ = 0;
    u32 drive_field_tex_ = 0;
    u32 field_tex_ = 0;

    GPUBuffer object_buffer_;
    GPUBuffer particle_magnet_x_buf_;
    GPUBuffer particle_magnet_y_buf_;
    GPUBuffer particle_occ_buf_;
    // Per-particle persistent magnetization M (vec2). Lets each particle
    // retain its magnetic state across frames — enables Preisach-style
    // hysteresis, remanence, slow Curie transitions, and "magnetize by
    // cooling then remove field" permanent-magnet gameplay.
    GPUBuffer particle_m_prev_buf_;
    std::vector<vec4> debug_field_cache_;
    std::vector<vec4> debug_total_field_cache_;
    std::vector<vec2> debug_magnetization_cache_;
    std::vector<float> debug_source_cache_;
    bool debug_cache_dirty_ = true;

    ComputeShader raster_shader_;
    ComputeShader particle_shader_;
    ComputeShader compose_shader_;
    ComputeShader source_shader_;
    ComputeShader jacobi_shader_;
    ComputeShader field_shader_;
    ComputeShader chain_force_shader_;
};

} // namespace ng
