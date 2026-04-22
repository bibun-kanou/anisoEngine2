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
        f32 source_scale = 18.0f;
        // force_scale bundles (μ₀/ρ) conversion from solver |H|² units
        // into particle acceleration. Tuned so that peak Kelvin force
        // sits around 1-5 m/s² in typical scenes — comparable to
        // gravity — which gives visible deformation without saturating
        // the 15 m/s² max_force clamp in mpm_g2p.comp over a wide
        // region (which would turn the Kelvin force into a uniform
        // radial pull and collapse the fluid).
        f32 force_scale = 1.0f;
        i32 jacobi_iterations = 52;
        i32 induction_iterations = 2;
        f32 rigid_permanent_scale = 0.18f;
        f32 rigid_soft_scale = 0.10f;
        vec2 cursor_pos = vec2(0.0f);
        vec2 cursor_dir = vec2(0.0f, 1.0f);
        f32 cursor_radius = 0.35f;
        f32 cursor_falloff_radius = 0.65f;
        f32 cursor_strength = 0.0f;
        CursorFieldType cursor_field_type = CursorFieldType::PROBE_POLE;
    };

    void init(const Config& config);
    void set_params(const Params& params) { params_ = params; }
    Params& params() { return params_; }
    const Params& params() const { return params_; }

    void step(const SDFField& sdf, ParticleBuffer* particles = nullptr);

    void bind_field_for_read(u32 unit = 4) const;
    void bind_total_field_for_read(u32 unit = 4) const;
    void bind_magnetization_for_read(u32 unit = 4) const;

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
            || params_.debug_force_active;
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
    u32 phi_tex_ = 0;
    u32 phi2_tex_ = 0;
    u32 drive_field_tex_ = 0;
    u32 field_tex_ = 0;

    GPUBuffer object_buffer_;
    GPUBuffer particle_magnet_x_buf_;
    GPUBuffer particle_magnet_y_buf_;
    GPUBuffer particle_occ_buf_;
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
};

} // namespace ng
