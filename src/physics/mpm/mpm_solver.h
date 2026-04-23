#pragma once

#include "core/types.h"
#include "gpu/compute_shader.h"
#include "gpu/buffer.h"
#include "physics/common/particle_buffer.h"
#include "physics/common/grid.h"
#include <vector>

namespace ng {

class SDFField;
class MagneticField;
class EulerianFluid;

// Material type IDs (must match shader defines)
enum class MPMMaterial : u32 {
    FLUID    = 0,
    ELASTIC  = 1,
    SNOW     = 2,
    ANISO    = 3,
    THERMAL  = 4,
    FRACTURE = 5,
    PHASE    = 6,
    BURNING  = 7,
    EMBER    = 8,
    HARDEN    = 9,
    CERAMIC   = 10,
    COMPOSITE = 11,
    BRITTLE   = 12,
    TOUGH     = 13,
    GLASS     = 14,
    BLOOM     = 15,
    FLAMMABLE_FLUID = 16,
    FOAM      = 17,
    SPLINTER  = 18,
    BREAD     = 19,
    PUFF_CLAY = 20,
    FIRECRACKER = 21,
    GLAZE_CLAY = 22,
    CRUST_DOUGH = 23,
    THERMO_METAL = 24,
    REACTIVE_BURN = 25,
    GLAZE_DRIP = 26,
    STEAM_BUN = 27,
    FILAMENT_GLASS = 28,
    CHEESE_PULL = 29,
    MEMORY_WAX = 30,
    FERRO_FLUID = 31,
    MAILLARD = 32,
    MUSHROOM = 33,
    CRUMB_LOAF = 34,
    BISQUE = 35,
    TEAR_SKIN = 36,
    LAMINATED_PASTRY = 37,
    STONEWARE = 38,
    ORTHO_BEND = 39,
    ORTHO_TEAR = 40,
    VENT_CRUMB = 41,
    VITREOUS_CLAY = 42,
    BLISTER_GLAZE = 43,
    OPEN_CRUMB = 44,
    SINTER_LOCK = 45,
    BINDER_CRUMB = 46,
    CHANNEL_CRUMB = 47,
    BURNOUT_CLAY = 48,
    VENTED_SKIN = 49,
    SPH_WATER = 50,
    SPH_VISCOUS_GOO = 51,
    SPH_LIGHT_OIL = 52,
    SPH_BURNING_OIL = 53,
    SPH_BOILING_WATER = 54,
    SPH_THERMAL_SYRUP = 55,
    SPH_FLASH_FLUID = 56,
    MAG_SOFT_IRON = 57,
    MAGNETIC_RUBBER = 58,
    TOPO_GOO = 59,
    OOBLECK = 60,
    IMPACT_GEL = 61,
    SEALED_CHARGE = 62,
    MORPH_TISSUE = 63,
    ROOT_WEAVE = 64,
    CELL_SHEET = 65,
    ASH_REGROWTH = 66,
    HEAVY_FERRO_FLUID = 67,
    DIAMAGNETIC_FLUID = 68,
    PARA_MIST = 69,
    STICKY_FERRO = 70,
    SUPERCONDUCTOR = 71,
    CURIE_FERRO = 72,
    EDDY_COPPER = 73,
    HARD_MAGNET = 74,
    SAND_GRANULAR = 75,
    PHASE_BRITTLE = 76,
};

vec4 default_thermal_coupling(MPMMaterial material);

struct MPMParams {
    f32  youngs_modulus  = 40000.0f;
    f32  poisson_ratio   = 0.3f;
    f32  fiber_strength  = 3.0f;
    vec2 gravity         = vec2(0.0f, -9.81f);

    // Thermal
    f32  thermal_k       = 0.5f;
    bool enable_thermal  = false;
    bool physically_based_heat = true;
    vec2 heat_source_pos    = vec2(0.0f, -1.0f);
    f32  heat_source_radius = 0.5f;
    f32  heat_source_temp   = 800.0f;

    f32  ambient_temp    = 300.0f;  // From Eulerian air (synced in main loop)

    // Heat gun (set per-frame from main)
    vec2 heat_gun_pos    = vec2(0.0f);
    f32  heat_gun_radius = 0.4f;
    f32  heat_gun_power  = 0.0f;

    // Fracture
    f32  fracture_threshold = 0.02f;
    f32  fracture_rate      = 5.0f;

    // Phase-field melting
    f32  melt_temp    = 500.0f;
    f32  melt_range   = 50.0f;
    f32  latent_heat  = 200.0f;
    f32  particle_cooling_rate = 0.015f;
    bool pseudo_25d_enabled = true;
    f32  pseudo_25d_depth = 0.9f;
    f32  pseudo_25d_shell_support = 0.8f;
    f32  pseudo_25d_enclosure = 0.9f;
    f32  pseudo_25d_cohesion = 0.8f;
    vec2 magnet_pos = vec2(0.0f);
    f32  magnet_radius = 0.35f;
    f32  magnet_falloff_radius = 0.65f;
    f32  magnet_force = 0.0f;
    f32  magnet_spike_strength = 1.0f;
    f32  magnet_chain_rate = 6.0f;
    f32  magnet_spike_freq = 16.0f;

    i32  vis_mode = 0;
    bool keep_colors = false;
    bool multi_scale = false; // Per-material gravity scaling (surreal mode)
    u32  highlight_start = 0;
    u32  highlight_end   = 0;
    f32  time = 0.0f;
};

class MPMSolver {
public:
    void init(UniformGrid& grid);

    void set_params(const MPMParams& p) { params_ = p; }
    MPMParams& params() { return params_; }
    void clear_particles() { particle_count_ = 0; }

    void spawn_block(ParticleBuffer& particles, vec2 min, vec2 max, f32 spacing,
                     MPMMaterial material, f32 initial_temp = 300.0f,
                     vec2 fiber_dir = vec2(1, 0),
                     f32 density_scale = 1.0f,
                     vec4 thermal_coupling = vec4(1.0f));

    void spawn_circle(ParticleBuffer& particles, vec2 center, f32 radius, f32 spacing,
                      MPMMaterial material, f32 initial_temp = 300.0f,
                      vec2 fiber_dir = vec2(1, 0),
                      f32 density_scale = 1.0f,
                      vec4 thermal_coupling = vec4(1.0f));

    void spawn_points(ParticleBuffer& particles, const std::vector<vec2>& positions,
                      const std::vector<f32>& shell_seeds, f32 spacing,
                      MPMMaterial material, f32 initial_temp = 300.0f,
                      vec2 fiber_dir = vec2(1, 0),
                      f32 density_scale = 1.0f,
                      vec4 thermal_coupling = vec4(1.0f));

    void step(ParticleBuffer& particles, UniformGrid& grid, f32 dt,
              const SDFField* sdf = nullptr,
              const MagneticField* magnetic = nullptr,
              const EulerianFluid* air = nullptr,
              vec2 mouse_pos = vec2(0), f32 mouse_radius = 0, f32 mouse_force = 0,
              vec2 mouse_dir = vec2(0), i32 mouse_mode = 0, f32 mouse_inner_radius = 0.0f,
              f32 mouse_damping = 0.0f);

    void begin_spring_drag(ParticleBuffer& particles, vec2 center, f32 radius, f32 falloff_radius);
    void end_spring_drag();
    bool spring_drag_active() const { return spring_drag_active_; }
    void set_kinematic_targets(const std::vector<vec2>& targets,
                               const std::vector<f32>& weights,
                               f32 force, f32 damping);
    void clear_kinematic_targets();
    bool kinematic_targets_active() const { return kinematic_targets_active_; }

    void update_batch_material(ParticleBuffer& particles, u32 global_offset, u32 count,
                               f32 youngs_modulus, f32 poisson_ratio,
                               f32 fiber_strength, f32 temperature, vec2 fiber_dir,
                               f32 outgassing_scale, f32 heat_release_scale,
                               f32 cooling_scale, f32 loft_scale);

    u32 particle_count() const { return particle_count_; }

    // Expose buffers for coupling/vis
    GPUBuffer& damage_buf() { return damage_buf_; }
    GPUBuffer& phase_buf() { return phase_buf_; }
    GPUBuffer& jp_buf() { return jp_buf_; }
    GPUBuffer& mat_params_buf() { return mat_params_buf_; }
    GPUBuffer& thermal_coupling_buf() { return thermal_coupling_buf_; }
    GPUBuffer& spring_anchor_buf() { return spring_anchor_buf_; }
    GPUBuffer& spring_weight_buf() { return spring_weight_buf_; }
    const GPUBuffer& damage_buf() const { return damage_buf_; }
    const GPUBuffer& phase_buf() const { return phase_buf_; }
    const GPUBuffer& jp_buf() const { return jp_buf_; }
    const GPUBuffer& mat_params_buf() const { return mat_params_buf_; }
    const GPUBuffer& thermal_coupling_buf() const { return thermal_coupling_buf_; }
    const GPUBuffer& spring_anchor_buf() const { return spring_anchor_buf_; }
    const GPUBuffer& spring_weight_buf() const { return spring_weight_buf_; }

private:
    void spawn_from_positions(ParticleBuffer& particles, const std::vector<vec2>& positions,
                              const std::vector<f32>& shell_seeds, f32 mass,
                              MPMMaterial material, f32 initial_temp, vec2 fiber_dir,
                              vec4 thermal_coupling);

    void sub_step_mpm(ParticleBuffer& particles, UniformGrid& grid, f32 dt,
                      const SDFField* sdf, const MagneticField* magnetic,
                      const EulerianFluid* air,
                      vec2 mouse_pos, f32 mouse_radius, f32 mouse_force,
                      vec2 mouse_dir, i32 mouse_mode, f32 mouse_inner_radius,
                      f32 mouse_damping);

    MPMParams params_;
    u32 particle_count_ = 0;

    GPUBuffer jp_buf_;
    GPUBuffer fiber_buf_;
    GPUBuffer damage_buf_;   // Fracture: d in [0,1]
    GPUBuffer phase_buf_;    // Phase-field: phi in [0,1]
    GPUBuffer mat_params_buf_; // Per-particle base E / nu / fiber strength
    GPUBuffer thermal_coupling_buf_; // x=outgas scale, y=heat release, z=cooling scale, w=air carry / loft
    GPUBuffer spring_anchor_buf_;
    GPUBuffer spring_weight_buf_;
    vec2 spring_origin_ = vec2(0.0f);
    bool spring_drag_active_ = false;
    bool kinematic_targets_active_ = false;
    f32 kinematic_target_force_ = 0.0f;
    f32 kinematic_target_damping_ = 0.0f;

    ComputeShader clear_shader_;
    ComputeShader p2g_shader_;
    ComputeShader grid_op_shader_;
    ComputeShader impact_contact_shader_;
    ComputeShader g2p_shader_;
    ComputeShader thermal_shader_;
};

} // namespace ng
