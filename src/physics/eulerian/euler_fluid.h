#pragma once

#include "core/types.h"
#include "gpu/compute_shader.h"
#include "gpu/buffer.h"

namespace ng {

class ParticleBuffer;
class SDFField;

// Eulerian fluid solver for air/gas medium.
// MAC grid with staggered velocity, cell-centered pressure/temperature/smoke.
// Boussinesq buoyancy for temperature-driven convection.
class EulerianFluid {
public:
    struct Config {
        ivec2 resolution = ivec2(128, 128);
        vec2  world_min  = vec2(-3.0f, -2.0f);
        vec2  world_max  = vec2(3.0f, 4.0f);
        f32   ambient_temp = 300.0f;
        f32   buoyancy_alpha = 2.0f;  // Thermal buoyancy coefficient (strong for visible convection)
        f32   smoke_decay = 0.98f;     // Smoke density decay per second
        f32   air_viscosity = 0.05f;  // Enough drag to stop flow when source removed
        f32   thermal_diffusivity = 0.002f;  // Very slow conduction
        f32   vapor_generation = 0.50f; // How strongly hot material makes steam / vapor
        f32   vapor_pressure = 2.4f;    // Divergence source from fresh vapor creation
        f32   vapor_buoyancy = 0.9f;    // Lift from persistent vapor concentration
        f32   vapor_drag = 0.8f;        // Stronger particle-air coupling in steam pockets
        f32   vapor_decay = 0.75f;      // Condensation / mixing rate
        f32   latent_cooling = 0.30f;   // More vapor means less direct air heating
        f32   combustion_heat_boost = 2.2f; // Extra hot-gas injection from active burning materials
        f32   combustion_hold = 0.55f;   // Hot smoky cells cool a bit more slowly so flames can stay volumetric.
                                          // 0.82 was too high and let blast plumes insulate themselves for 3-5s.
        f32   solid_thermal_diffusivity = 0.004f; // Internal SDF heat spread rate
        f32   solid_heat_capacity = 4.0f;   // Higher = more heat sink / slower temp change
        f32   solid_contact_transfer = 0.012f; // How strongly SDF solids give heat to touching air/objects
        f32   solid_heat_loss = 0.03f;      // Extra non-physical heat disappearance
        f32   cooling_rate = 0.68f;         // Slightly slower cooling so flames can sustain before fading
        bool  physically_based_heat = true; // Conservative diffusion: no numerical heat gain from high k
        bool  bio_enabled = false;          // Gray-Scott reaction-diffusion substrate for bio / automata-like effects
        f32   bio_feed = 0.034f;
        f32   bio_kill = 0.061f;
        f32   bio_diffuse_a = 0.16f;
        f32   bio_diffuse_b = 0.08f;
        f32   bio_seed_strength = 0.55f;    // How strongly particles seed the activator / inhibitor field
        f32   bio_pattern_speed = 1.5f;     // Global rate multiplier for the reaction-diffusion solve
        f32   bio_coupling = 0.85f;         // How strongly MPM bio-ish materials read and respond to the field
        f32   bio_regrowth_rate = 3.0f;     // Global multiplier for living-material regrowth / recovery from the bio field.
                                             // Raised from 1.0: the MPM regrow advance is slow at the old default and
                                             // color restoration was invisible on demo timescales.
        bool  automata_enabled = false;     // Separate continuous automata substrate for colony / self-organization
        f32   automata_birth_lo = 0.278f;
        f32   automata_birth_hi = 0.365f;
        f32   automata_survive_lo = 0.267f;
        f32   automata_survive_hi = 0.445f;
        f32   automata_inner_radius = 1.75f; // In air-grid cells
        f32   automata_outer_radius = 4.25f; // In air-grid cells
        f32   automata_sigmoid = 0.060f;
        f32   automata_seed_strength = 0.70f;
        f32   automata_pattern_speed = 1.0f;
        f32   automata_coupling = 0.90f;
    };

    void init(const Config& config);

    // Step the fluid simulation
    void step(f32 dt, const SDFField* sdf = nullptr);

    // Inject heat + velocity + smoke from particles into the air grid
    void clear_particle_injection_sources();
    void inject_from_particles(const ParticleBuffer& particles,
                                u32 offset, u32 count, f32 dt);
    void update_airtight_from_particles(const ParticleBuffer& particles,
                                        u32 offset, u32 count, f32 dt,
                                        const SDFField* sdf = nullptr);

    // Apply air drag force to particles (writes to velocity buffer)
    void apply_drag_to_particles(ParticleBuffer& particles,
                                  u32 offset, u32 count, f32 dt, f32 drag_coeff);

    // Inject heat at a world position (for heat gun) — GPU compute
    void inject_heat_at(vec2 world_pos, f32 radius, f32 heat_power, f32 dt);

    // Inject velocity + smoke at cursor (blow/wind tool)
    void blow_at(vec2 world_pos, vec2 direction, f32 radius, f32 strength, f32 dt);

    // First-pass compressible blast front: injects a radial impulse ring plus
    // one-frame divergence so explosions feel more like a pressure wave than
    // only hot wind.
    void blast_at(vec2 world_pos, f32 inner_radius, f32 outer_radius,
                  f32 strength, f32 heat, f32 smoke, f32 divergence, f32 dt);

    // Bind temperature texture for visualization
    void bind_for_vis(u32 unit = 2) const;

    f32 dx() const { return dx_; }
    ivec2 resolution() const { return resolution_; }
    vec2 world_min() const { return world_min_; }
    vec2 world_max() const { return world_max_; }
    ivec2 airtight_resolution() const { return airtight_resolution_; }
    f32 airtight_dx() const { return airtight_dx_; }
    Config& config() { return config_; }
    const Config& config() const { return config_; }

    u32 temp_texture() const { return temp_tex_; }
    u32 smoke_texture() const { return smoke_tex_; }
    u32 velocity_texture() const { return vel_tex_; }
    u32 bio_a_texture() const { return bio_a_tex_; }
    u32 bio_b_texture() const { return bio_b_tex_; }
    u32 bio_source_texture() const { return bio_source_tex_; }
    u32 bio_field_texture() const { return bio_field_tex_; }
    u32 bio_support_texture() const { return bio_support_tex_; }
    u32 automata_texture() const { return automata_tex_; }
    f32 bio_field_view_gain() const { return bio_field_view_gain_; }
    f32 automata_view_gain() const { return automata_view_gain_; }
    u32 airtight_pressure_texture() const { return airtight_pressure_tex_; }
    u32 airtight_outside_texture() const { return airtight_outside_tex_; }
    void set_visualization_mode(i32 mode) { visualization_mode_ = mode; }
    void set_particle_visualization_mode(i32 mode) { particle_visualization_mode_ = mode; }

    // SSBO bindings for Eulerian grid (60-69 range)
    static constexpr u32 BIND_VEL_X   = 60; // float[] staggered u
    static constexpr u32 BIND_VEL_Y   = 61; // float[] staggered v
    static constexpr u32 BIND_VEL_X2  = 62; // float[] temp for advection
    static constexpr u32 BIND_VEL_Y2  = 63;
    static constexpr u32 BIND_PRESSURE = 64;
    static constexpr u32 BIND_DIVERGENCE = 65;
    static constexpr u32 BIND_ETEMPERATURE = 66;
    static constexpr u32 BIND_ETEMP2  = 67;
    static constexpr u32 BIND_SMOKE   = 68;
    static constexpr u32 BIND_SMOKE2  = 69;
    static constexpr u32 BIND_VAPOR   = 70;
    static constexpr u32 BIND_VAPOR2  = 71;
    static constexpr u32 BIND_VAPOR_SOURCE = 72;
    static constexpr u32 BIND_AIR_OCCUPANCY = 73;
    static constexpr u32 BIND_AIR_SOURCE = 74;
    static constexpr u32 BIND_AIR_OUTSIDE = 75;
    static constexpr u32 BIND_AIR_OUTSIDE2 = 76;
    static constexpr u32 BIND_AIR_PRESSURE = 77;
    static constexpr u32 BIND_AIR_PRESSURE2 = 78;
    static constexpr u32 BIND_BIO_A = 79;
    static constexpr u32 BIND_BIO_A2 = 80;
    static constexpr u32 BIND_BIO_B = 81;
    static constexpr u32 BIND_BIO_B2 = 82;
    static constexpr u32 BIND_AUTOMATA = 83;
    static constexpr u32 BIND_AUTOMATA2 = 84;
    static constexpr u32 BIND_BIO_SUPPORT_SOURCE = 85;
    static constexpr u32 BIND_BIO_SUPPORT = 86;
    static constexpr u32 BIND_BIO_SUPPORT2 = 87;
    static constexpr u32 BIND_BIO_SOURCE_SEED = 88;
    static constexpr u32 BIND_BIO_SOURCE = 89;
    static constexpr u32 BIND_BIO_SOURCE2 = 90;

    const GPUBuffer& airtight_outside_buf() const { return airtight_outside_buf_; }
    const GPUBuffer& airtight_pressure_buf() const { return airtight_pressure_buf_; }

private:
    Config config_;
    ivec2 resolution_{0};
    vec2  world_min_{0}, world_max_{0};
    f32   dx_ = 0;
    ivec2 airtight_resolution_{0};
    f32   airtight_dx_ = 0;

    // Velocity (staggered MAC): u on vertical faces, v on horizontal faces
    GPUBuffer vel_x_buf_, vel_y_buf_;
    GPUBuffer vel_x2_buf_, vel_y2_buf_; // For advection swap
    GPUBuffer pressure_buf_, divergence_buf_;
    GPUBuffer temp_buf_, temp2_buf_;  // Temperature
    GPUBuffer smoke_buf_, smoke2_buf_; // Smoke density
    GPUBuffer vapor_buf_, vapor2_buf_; // Persistent vapor / steam concentration
    GPUBuffer vapor_source_buf_;       // One-frame expansion source from fresh vapor generation
    GPUBuffer bio_a_buf_;              // Reaction-diffusion nutrient field
    GPUBuffer bio_a2_buf_;             // Scratch for RD solve
    GPUBuffer bio_b_buf_;              // Reaction-diffusion activator field
    GPUBuffer bio_b2_buf_;             // Scratch for RD solve
    GPUBuffer bio_support_source_buf_; // Blob occupancy / support seeded from particles this frame
    GPUBuffer bio_support_buf_;        // Locality envelope for bio/automata
    GPUBuffer bio_support2_buf_;       // Scratch for support blur/decay
    GPUBuffer bio_source_seed_buf_;    // Raw bio particle source before blur / solve
    GPUBuffer bio_source_buf_;         // Blurred bio source cloud that drives RD growth
    GPUBuffer bio_source2_buf_;        // Scratch for bio source solve
    GPUBuffer automata_buf_;           // Continuous automata occupancy / colony field
    GPUBuffer automata2_buf_;          // Scratch for automata solve
    GPUBuffer airtight_occ_buf_;       // int[] shell occupancy / blockage
    GPUBuffer airtight_source_buf_;    // int[] cavity gas source
    GPUBuffer airtight_outside_buf_;   // float[] outside reachability
    GPUBuffer airtight_outside2_buf_;  // float[] scratch
    GPUBuffer airtight_pressure_buf_;  // float[] trapped cavity pressure
    GPUBuffer airtight_pressure2_buf_; // float[] scratch

    u32 temp_tex_ = 0, smoke_tex_ = 0, vel_tex_ = 0;
    u32 bio_a_tex_ = 0, bio_b_tex_ = 0, bio_source_tex_ = 0, bio_field_tex_ = 0;
    u32 bio_support_tex_ = 0;
    u32 automata_tex_ = 0;
    f32 bio_field_view_gain_ = 1.0f;
    f32 automata_view_gain_ = 1.0f;
    i32 visualization_mode_ = 0;
    i32 particle_visualization_mode_ = 0;
    u32 airtight_pressure_tex_ = 0, airtight_outside_tex_ = 0;

    ComputeShader advect_shader_;
    ComputeShader forces_shader_;
    ComputeShader divergence_shader_;
    ComputeShader pressure_shader_;
    ComputeShader project_shader_;
    ComputeShader inject_shader_;
    ComputeShader drag_shader_;
    ComputeShader heat_gun_shader_;
    ComputeShader diffuse_temp_shader_;
    ComputeShader reaction_diffuse_shader_;
    ComputeShader bio_support_shader_;
    ComputeShader bio_source_shader_;
    ComputeShader automata_shader_;
    ComputeShader blow_shader_;
    ComputeShader blast_shader_;
    ComputeShader enforce_bc_shader_;
    ComputeShader airtight_rasterize_shader_;
    ComputeShader airtight_seed_shader_;
    ComputeShader airtight_propagate_shader_;
    ComputeShader airtight_update_shader_;
    ComputeShader airtight_smooth_shader_;

    void bind_all() const;
};

} // namespace ng
