#include "core/engine.h"
#include "core/log.h"
#include "core/scenes.h"
#include "core/creation_menu.h"
#include "core/foot_demo.h"
#include "gpu/compute_shader.h"
#include "gpu/buffer.h"
#include "physics/common/particle_buffer.h"
#include "physics/common/spatial_hash.h"
#include "physics/common/grid.h"
#include "physics/sph/sph_solver.h"
#include "physics/mpm/mpm_solver.h"
#include "physics/sdf/sdf_field.h"
#include "physics/magnetic/magnetic_field.h"
#include "physics/eulerian/euler_fluid.h"
#include "render/particle_renderer.h"
#include "render/sdf_renderer.h"
#include "render/metaball_renderer.h"
#include "render/bloom.h"
#include "render/camera.h"
#include "render/pipeline_viewer.h"

#include <glad/gl.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <SDL.h>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_opengl3.h>

// --- Globals ---
static ng::ParticleBuffer g_particles;
static ng::SpatialHash    g_hash;
static ng::SPHSolver      g_sph;
static ng::MPMSolver      g_mpm;
static ng::UniformGrid    g_mpm_grid;
static ng::SDFField       g_sdf;
static ng::MagneticField  g_magnetic;
static ng::EulerianFluid  g_air;
static ng::ParticleRenderer g_particle_renderer;
static ng::SDFRenderer    g_sdf_renderer;
static ng::MetaballRenderer g_metaball;
static ng::BloomRenderer  g_bloom;
static ng::Camera         g_camera;
static ng::Shader         g_preview_shader;
static ng::Shader         g_heat_glow_shader;
static ng::Shader         g_magnetic_field_vis_shader;
static ng::u32            g_preview_vao = 0;

static bool g_paused = false;
static bool g_show_ui = true;
static ng::SceneID g_scene = ng::SceneID::DEFAULT;
static ng::CreationState g_creation;

static int g_color_mode = 0;
static int g_air_vis = 0;
static float g_fire_vis_start_temp = 305.0f;
static float g_fire_vis_temp_range = 400.0f;
static float g_fire_vis_softness = 3.0f;
static int g_sdf_palette = 1; // 0=silver, 1=rose gold, 2=bronze, 3=brass
static int g_euler_res_idx = 1; // 0=256, 1=512, 2=1024, 3=2048
static const int euler_res_values[] = { 256, 512, 1024, 2048 };
static const char* euler_res_names[] = { "256x256", "512x512", "1024x1024", "2048x2048" };
static const char* color_mode_names[] = { "Debug Vis", "Batch Colors" };
static const char* sdf_palette_names[] = { "Silver", "Rose Gold", "Bronze", "Brass" };
static const char* magnetic_debug_view_names[] = {
    "Drive H Arrows", "Drive H Lines", "|Drive H|", "Drive H Iso", "Kelvin Force", "Field Shader",
    "M Arrows", "|M|", "Source Charge"
};

enum class InteractMode { PUSH, PULL, DRAG, SWEEP_DRAG, DROP_BALL, DRAW_WALL, ERASE_WALL, SPRING_DRAG, FOOT_CONTROL, MAGNET, LAUNCHER2 };
static InteractMode g_mode = InteractMode::PUSH;

enum class ProjectilePreset : int {
    STEEL = 0,
    SNOW = 1,
    EXPLOSIVE = 2,
    RUBBER = 3,
    TIME_BOMB = 4,
    REAL_BOMB = 5,
    SIEGE_BOMB = 6,
    DIRECTIONAL_CHARGE = 7,
    CLAYMORE = 8,
    REAL_TIME_BOMB = 9,
    LAYERED_BOMB = 10,
    BROADSIDE_BOMB = 11,
    TRIGGER_CHARGE = 12,
    ROCKET = 13,
    HEAT_CHARGE = 14,
    HESH_CHARGE = 15,
    APHE_SHELL = 16,
    DEMOLITION_CHARGE = 17,
    DEEP_FUSE_BOMB = 18,
    THERMITE_CHARGE = 19,
    TANDEM_HEAT = 20,
    FUEL_AIR_BOMB = 21,
    SMOKE_CANISTER = 22,
    FLASHBANG = 23,
    CLUSTER_CHARGE = 24,
    CATACLYSM_CHARGE = 25,
    CASCADE_CLUSTER = 26,
    CRYO_CHARGE = 27,
    FIRE_SPIRAL_BOMB = 28,
    SOFT_CLAYMORE = 29,
    SOFT_BROADSIDE = 30,
    COOL_SMOKE_POT = 31,
    SOFT_SPIRAL_BOMB = 32,
    MEDIUM_CLAYMORE = 33,
    MEDIUM_BROADSIDE = 34,
    MEDIUM_SMOKE_POT = 35,
    MEDIUM_SPIRAL_BOMB = 36,
    SOFT_HEAT_CHARGE = 37,
    MEDIUM_HEAT_CHARGE = 38,
    ABOVE_MED_CLAYMORE = 39,
    ABOVE_MED_BROADSIDE = 40,
    ABOVE_MED_SMOKE_POT = 41,
    ABOVE_MED_SPIRAL_BOMB = 42,
    ABOVE_MED_HEAT_CHARGE = 43,
    SOFT_LONG_FUSE_BOMB = 44,
    MEDIUM_LONG_FUSE_BOMB = 45,
    ABOVE_MED_LONG_FUSE_BOMB = 46,
    SOFT_ROCKET = 47,
    MEDIUM_ROCKET = 48,
    ABOVE_MED_ROCKET = 49,
    SOFT_DEEP_FUSE_BOMB = 50,
    MEDIUM_DEEP_FUSE_BOMB = 51,
    ABOVE_MED_DEEP_FUSE_BOMB = 52,
    SOFT_EVEN_DEEPER_FUSE_BOMB = 53,
    MEDIUM_EVEN_DEEPER_FUSE_BOMB = 54,
    ABOVE_MED_EVEN_DEEPER_FUSE_BOMB = 55,
    EVEN_DEEPER_FUSE_BOMB = 56,
    // Launcher2 hybrid weapons (composed from existing primitives)
    ROCKET_PAYLOAD = 57,
    ROCKET_SIDE_CLAYMORE = 58,
    CLAYMORE_CLUSTER = 59,
    // Contact-triggered launcher2 weapons (use impact_rupture)
    LATERAL_CONTACT_CHARGE = 60,
    GRAVITY_PENETRATOR = 61,
    CONCUSSION_CHARGE = 62
};

enum class ProjectileDragMode : int {
    NONE = 0,
    AIM = 1,
    CONE = 2
};

struct ParticleSpanRef {
    ng::u32 offset = 0;
    ng::u32 count = 0;

    bool valid() const { return count > 0; }
    ng::u32 end() const { return offset + count; }
};

struct PressureVesselRecord {
    ParticleSpanRef shell;
    ParticleSpanRef fuse;
    ParticleSpanRef core;
    ParticleSpanRef payload;
    ParticleSpanRef trigger;
    ng::f32 gas_mass = 0.0f;
    ng::f32 pressure = 0.0f;
    ng::f32 burst_energy = 0.0f;
    ng::f32 age = 0.0f;
    ng::vec2 preferred_axis = ng::vec2(1.0f, 0.0f);
    ng::f32 axis_bias = 0.0f;
    ng::f32 gas_source_scale = 1.0f;
    ng::f32 rupture_scale = 1.0f;
    ng::f32 burst_scale = 1.0f;
    ng::f32 shell_push_scale = 1.0f;
    ng::f32 core_push_scale = 1.0f;
    ng::f32 leak_scale = 1.0f;
    ng::f32 payload_push_scale = 0.0f;
    ng::f32 payload_cone = 0.55f;
    ng::f32 payload_directionality = 1.0f;
    ng::f32 ignition_delay = 0.0f;
    ng::f32 ignition_window = 0.45f;
    ng::f32 thrust_scale = 0.0f;
    ng::f32 nozzle_open = 0.0f;
    ng::f32 side_blast_scale = 0.0f;
    ng::f32 swirl_blast_scale = 0.0f;
    ng::f32 plume_push_scale = 1.0f;
    ng::f32 plume_heat_scale = 1.0f;
    ng::f32 plume_radius_scale = 1.0f;
    ng::f32 blast_push_scale = 1.0f;
    ng::f32 blast_heat_scale = 1.0f;
    ng::f32 trigger_progress = 0.0f;
    ng::f32 trigger_speed = 0.0f;
    ng::f32 trigger_heat = 980.0f;
    ng::f32 trigger_boost = 0.0f;
    bool auto_arm = true;
    bool ruptured = false;
    ng::f32 rupture_age = -1.0f;
    // Impact-triggered rupture: when true, any of the contact sensors below can
    // fire a rupture regardless of pressure. Used by the contact-triggered bomb
    // family so they only detonate on contact, not mid-flight.
    bool impact_rupture = false;
    // When true AND a sensor fires, preferred_axis is snapped to gravity
    // direction (0, -1) so the blast/payload drives downward through the
    // struck surface regardless of incoming flight angle. Applied at the
    // rupture frame, not the trigger frame, so the fuse delay still ticks.
    bool penetrate_on_impact = false;

    // Sensors. Filled in by the vessel config; only active when impact_rupture
    // is also true.
    //  - Velocity-drop sensor: prev speed > 4 m/s + curr < 45% prev means a
    //    hard, fast collision. Fires in one frame on hard hits. Zero tuning.
    //  - Crack-rate sensor: shell_crack accumulates from both GPU-side stress
    //    (impact crushing of shell particles) and C++-side pressure cracking.
    //    A sudden rise in crack_avg distinguishes "just smashed into something"
    //    from "steady pressure build-up". Rate is per second; cruise pressure
    //    alone gives ~0.3/s, so thresholds around 1.5/s reliably separate them.
    ng::f32 impact_crack_rate_threshold = 0.0f;

    // Delay fuse between any sensor firing and the actual rupture.
    //  - 0.0 s = instant (old behavior).
    //  - 0.02-0.08 s = "contact fuse" — fast but lets a shell hit dig in by one
    //    particle width before the blast goes off.
    //  - 0.1-0.3 s = heavier "delay fuse" — lets a gravity penetrator actually
    //    bury into the surface before going off.
    bool triggered = false;
    ng::f32 trigger_age = 0.0f;
    ng::f32 trigger_to_rupture_delay = 0.0f;

    // Previous-frame cache for rate sensors. Updated at the end of each vessel
    // update so the next frame can compute a derivative.
    ng::f32 prev_shell_speed = 0.0f;
    ng::f32 prev_crack_avg = 0.0f;
};

enum class HoverKind { NONE, BATCH, SDF };
struct HoverSelection {
    HoverKind kind = HoverKind::NONE;
    int batch_index = -1;
    ng::u32 sdf_object_id = 0;
    ng::f32 distance = std::numeric_limits<ng::f32>::max();
};
static HoverSelection g_hover_selection;
static HoverSelection g_pinned_selection;
static bool g_pinned_selection_open = false;
static bool g_selection_mode = false;
static bool g_show_size_hint = false;
static bool g_show_heat_gizmos = false;
static bool g_show_drag_debug = false;
static bool g_show_magnetic_debug = false;
static bool g_show_pipeline = false;
static bool g_show_pipeline_prev = false;
static bool g_show_interaction_window = false;
static bool g_show_environment_window = false;
static bool g_show_backends_window = false;
static bool g_show_appearance_window = false;
static bool g_show_advanced_window = false;
static bool g_show_presets_window = false;
static int g_magnetic_debug_view = 0;
static const ImVec4 kInteractionAccent(0.30f, 0.60f, 0.96f, 0.95f);
static const ImVec4 kEnvironmentAccent(0.22f, 0.74f, 0.86f, 0.95f);
static const ImVec4 kBackendsAccent(0.19f, 0.68f, 0.54f, 0.95f);
static const ImVec4 kAppearanceAccent(0.76f, 0.48f, 0.22f, 0.95f);
static const ImVec4 kAdvancedAccent(0.64f, 0.34f, 0.78f, 0.95f);
static const ImVec4 kPresetsAccent(0.84f, 0.62f, 0.18f, 0.95f);
static const ImVec4 kActionAccent(0.44f, 0.47f, 0.56f, 0.95f);
static const char* magnetic_cursor_field_names[] = {
    "Probe Pole",
    "Bar Magnet",
    "Wide Pole",
    "Horseshoe"
};
static const char* surface_style_names[] = {
    "Liquid",
    "Gel",
    "Clay",
    "Wax",
    "Porcelain",
    "Field Matte",
    "Contour",
    "Soft Fill",
    "Thin Contour",
    "Ink Contour"
};
static const char* projectile_preset_names[] = {
    "Steel",
    "Snow",
    "Explosive",
    "Rubber",
    "Time Bomb",
    "Real Bomb",
    "Siege Bomb",
    "Directional Charge",
    "Claymore",
    "Real Time Bomb",
    "Layered Bomb",
    "Broadside Bomb",
    "Trigger Charge",
    "Rocket",
    "HEAT Charge",
    "HESH Charge",
    "APHE Shell",
    "Demolition Charge",
    "Deep Fuse Bomb",
    "Thermite Charge",
    "Tandem HEAT",
    "Fuel-Air Bomb",
    "Smoke Canister",
    "Flashbang",
    "Cluster Charge",
    "Cataclysm Charge",
    "Cascade Cluster",
    "Cryo Charge",
    "Fire Spiral Bomb",
    "Soft Claymore",
    "Soft Broadside",
    "Cool Smoke Pot",
    "Soft Spiral Bomb",
    "Medium Claymore",
    "Medium Broadside",
    "Medium Smoke Pot",
    "Medium Spiral Bomb",
    "Soft HEAT Charge",
    "Medium HEAT Charge",
    "Above Med Claymore",
    "Above Med Broadside",
    "Above Med Smoke Pot",
    "Above Med Spiral Bomb",
    "Above Med HEAT Charge",
    "Soft Long Fuse Bomb",
    "Medium Long Fuse Bomb",
    "Above Med Long Fuse Bomb",
    "Soft Rocket",
    "Medium Rocket",
    "Above Med Rocket",
    "Soft Deep Fuse Bomb",
    "Medium Deep Fuse Bomb",
    "Above Med Deep Fuse Bomb",
    "Soft Even Deeper Fuse Bomb",
    "Medium Even Deeper Fuse Bomb",
    "Above Med Even Deeper Fuse Bomb",
    "Even Deeper Fuse Bomb"
};
static const char* projectile_shape_names[] = {
    "Circle",
    "Cube",
    "Beam",
    "Triangle"
};

static const char* sph_vis_names[] = { "Default", "Velocity", "Pressure", "Density", "Curl", "Divergence", "Codim 1D/2D", "Temperature" };
static const char* mpm_vis_names[] = {
    "Material",
    "Velocity",
    "Stress",
    "||F||",
    "Damage",
    "Temperature",
    "Jp",
    "Density Proxy",
    "RGB Thermo-Stress",
    "RGB State",
    "Latent SmoothLife [new]",
    "Bio Drive [new]",
    "Automata Drive [new]",
    "Layer Bands [new]"
};
static const char* air_vis_names[] = {
    "Smoke+Fire", "Off", "Temperature", "Smoke Density",
    "Velocity", "Velocity+Smoke", "Curl/Vorticity", "Divergence",
    "Bio Field [new]",
    "Automata Colony [new]",
    "Bio Drive [new]",
    "Automata Drive [new]"
};
static constexpr int kSurfaceStyleCount = 10;
static constexpr int kProjectilePresetCount = 57;
static constexpr int kMpmVisModeCount = 14;

static ng::WindowConfig g_wc;
static ng::f32 g_tool_radius = 0.55f;
static ng::f32 g_tool_force = 55.0f;
static ng::f32 g_drag_force = 72.0f;
static ng::vec2 g_drag_dir = ng::vec2(1.0f, 0.0f);
static ng::f32 g_drag_inner_ratio = 0.45f;
static ng::f32 g_drag_falloff_radius = 0.55f;
static ng::f32 g_spring_force = 72.0f;
static ng::f32 g_spring_damping = 14.0f;
static ng::vec2 g_drag_anchor_world = ng::vec2(0.0f);
static bool g_drag_capture_active = false;
static ng::f32 g_magnet_strength = 8.0f;
static ng::f32 g_blow_radius = 0.55f;
static ng::f32 g_blow_strength = 7.5f;
static ng::f32 g_ball_radius = 0.12f;
static ng::f32 g_ball_weight = 6.0f;
static ng::f32 g_ball_stiffness = 42000.0f;
static ng::f32 g_ball_cone_deg = 0.0f;
static ng::f32 g_ball_launch_gain = 8.0f;
static ng::f32 g_ball_min_launch_speed = 0.25f;
static int g_ball_preset = static_cast<int>(ProjectilePreset::STEEL);
// Launcher2: a focused weapon menu on key 6. Three choices only — time bomb,
// self-propelling rocket, directional claymore. Maps into the same
// ProjectilePreset enum so all the existing fire_projectile machinery (layers,
// fuses, payloads, drag-aim) applies automatically.
enum class Launcher2Preset : int { TIME_BOMB = 0, ROCKET = 1, CLAYMORE = 2 };
static int g_launcher2_preset = static_cast<int>(Launcher2Preset::TIME_BOMB);
static int g_ball_shape = 0;
static bool g_projectile_auto_arm = true;
static ProjectileDragMode g_ball_drag_mode = ProjectileDragMode::NONE;
static bool g_ball_has_aim = false;
static ng::vec2 g_ball_drag_anchor = ng::vec2(0.0f);
static ng::vec2 g_ball_launch_vector = ng::vec2(1.0f, 0.0f);
static ng::f32 g_ball_cone_drag_start_deg = 0.0f;
static ng::u32 g_ball_shot_counter = 0u;
static std::vector<PressureVesselRecord> g_pressure_vessels;
static bool g_mpm_skin_enabled = false;
static int g_sph_surface_style = static_cast<int>(ng::SurfaceStyle::LIQUID);
static int g_mpm_surface_style = static_cast<int>(ng::SurfaceStyle::CLAY);
static float g_mpm_skin_threshold = 0.42f;
static float g_mpm_skin_kernel = 2.15f;

static ng::f32 scene_ambient_temp(ng::SceneID scene) {
    switch (scene) {
    case ng::SceneID::CODIM_THREADS_COLD:
        return 240.0f;
    default:
        return 300.0f;
    }
}

static ng::f32 scene_gravity_y(ng::SceneID scene) {
    switch (scene) {
    case ng::SceneID::ZERO_G_SOFT_LAB:
    case ng::SceneID::EMPTY_ZERO_G:
        return 0.0f;
    case ng::SceneID::ANISO_STRONG_BENCH:
        return -4.6f;
    case ng::SceneID::MORPHOGENESIS_BENCH:
    case ng::SceneID::ROOT_GARDEN_BENCH:
    case ng::SceneID::CELL_COLONY_BENCH:
        return -4.2f;
    default:
        return -9.81f;
    }
}

struct SceneSpaceConfig {
    ng::vec2 sdf_world_min = ng::vec2(-3.0f, -2.0f);
    ng::vec2 sdf_world_max = ng::vec2(3.0f, 4.0f);
    ng::vec2 grid_world_min = ng::vec2(-3.2f, -2.4f);
    ng::vec2 grid_world_max = ng::vec2(3.2f, 4.0f);
    ng::vec2 sph_bound_min = ng::vec2(-2.8f, -1.8f);
    ng::vec2 sph_bound_max = ng::vec2(2.8f, 3.5f);
    ng::vec2 camera_pos = ng::vec2(0.0f, 0.5f);
    ng::f32 camera_zoom = 180.0f;
};

static ng::ivec2 resolution_from_cell_size(ng::vec2 world_min, ng::vec2 world_max, ng::f32 cell_size) {
    ng::vec2 extent = world_max - world_min;
    return ng::ivec2(std::max(8, static_cast<int>(std::ceil(extent.x / glm::max(cell_size, 1e-4f)))),
                     std::max(8, static_cast<int>(std::ceil(extent.y / glm::max(cell_size, 1e-4f)))));
}

static ng::ivec2 resolution_from_max_dim(ng::vec2 world_min, ng::vec2 world_max, int max_dim_res) {
    ng::vec2 extent = world_max - world_min;
    ng::f32 max_extent = glm::max(extent.x, extent.y);
    if (max_extent <= 1e-4f) return ng::ivec2(max_dim_res, max_dim_res);
    ng::f32 scale = static_cast<ng::f32>(max_dim_res) / max_extent;
    return ng::ivec2(std::max(8, static_cast<int>(std::ceil(extent.x * scale))),
                     std::max(8, static_cast<int>(std::ceil(extent.y * scale))));
}

static SceneSpaceConfig scene_space_config(ng::SceneID scene) {
    SceneSpaceConfig cfg;
    switch (scene) {
    case ng::SceneID::OPEN_BLAST_RANGE_XL:
    case ng::SceneID::SPALL_GALLERY_XL:
        cfg.sdf_world_min = ng::vec2(-6.2f, -2.2f);
        cfg.sdf_world_max = ng::vec2(6.2f, 4.4f);
        cfg.grid_world_min = ng::vec2(-6.4f, -2.5f);
        cfg.grid_world_max = ng::vec2(6.4f, 4.6f);
        cfg.sph_bound_min = ng::vec2(-6.0f, -1.9f);
        cfg.sph_bound_max = ng::vec2(6.0f, 3.9f);
        cfg.camera_pos = ng::vec2(0.0f, 0.55f);
        cfg.camera_zoom = 92.0f;
        break;
    case ng::SceneID::BREACH_HALL_XL:
        cfg.sdf_world_min = ng::vec2(-6.1f, -2.3f);
        cfg.sdf_world_max = ng::vec2(6.1f, 4.4f);
        cfg.grid_world_min = ng::vec2(-6.4f, -2.6f);
        cfg.grid_world_max = ng::vec2(6.4f, 4.7f);
        cfg.sph_bound_min = ng::vec2(-5.9f, -2.0f);
        cfg.sph_bound_max = ng::vec2(5.9f, 4.0f);
        cfg.camera_pos = ng::vec2(0.0f, 0.65f);
        cfg.camera_zoom = 90.0f;
        break;
    case ng::SceneID::FOOT_DEMO_BENCH:
        cfg.sdf_world_min = ng::vec2(-6.6f, -2.3f);
        cfg.sdf_world_max = ng::vec2(6.9f, 5.7f);
        cfg.grid_world_min = ng::vec2(-6.8f, -2.5f);
        cfg.grid_world_max = ng::vec2(7.1f, 5.9f);
        cfg.sph_bound_min = ng::vec2(-6.3f, -2.1f);
        cfg.sph_bound_max = ng::vec2(6.6f, 5.2f);
        cfg.camera_pos = ng::vec2(0.2f, 0.6f);
        cfg.camera_zoom = 88.0f;
        break;
    default:
        break;
    }
    return cfg;
}

static const char* magnetic_cursor_field_label(ng::MagneticField::CursorFieldType type) {
    switch (type) {
    case ng::MagneticField::CursorFieldType::PROBE_POLE: return "Probe Pole";
    case ng::MagneticField::CursorFieldType::BAR_MAGNET: return "Bar Magnet";
    case ng::MagneticField::CursorFieldType::WIDE_POLE: return "Wide Pole";
    case ng::MagneticField::CursorFieldType::HORSESHOE: return "Horseshoe";
    default: return "Probe Pole";
    }
}

static void apply_scene_runtime_defaults(ng::SceneID scene) {
    auto& mag = g_magnetic.params();
    auto& air_cfg = g_air.config();
    air_cfg.bio_enabled = false;
    air_cfg.bio_feed = 0.034f;
    air_cfg.bio_kill = 0.061f;
    air_cfg.bio_diffuse_a = 0.16f;
    air_cfg.bio_diffuse_b = 0.08f;
    air_cfg.bio_seed_strength = 0.55f;
    air_cfg.bio_pattern_speed = 1.0f;
    air_cfg.bio_coupling = 0.85f;
    air_cfg.bio_regrowth_rate = 1.0f;
    air_cfg.automata_enabled = false;
    air_cfg.automata_birth_lo = 0.278f;
    air_cfg.automata_birth_hi = 0.365f;
    air_cfg.automata_survive_lo = 0.267f;
    air_cfg.automata_survive_hi = 0.445f;
    air_cfg.automata_inner_radius = 1.75f;
    air_cfg.automata_outer_radius = 4.25f;
    air_cfg.automata_sigmoid = 0.060f;
    air_cfg.automata_seed_strength = 0.70f;
    air_cfg.automata_pattern_speed = 1.0f;
    air_cfg.automata_coupling = 0.90f;
    mag.enabled = false;
    mag.source_scale = 18.0f;
    mag.force_scale = 28.0f;
    mag.jacobi_iterations = 52;
    mag.induction_iterations = 2;
    mag.rigid_permanent_scale = 0.18f;
    mag.rigid_soft_scale = 0.10f;
    mag.cursor_field_type = ng::MagneticField::CursorFieldType::PROBE_POLE;
    if (scene == ng::SceneID::THERMAL_VERIFY_SDF_JUNCTION ||
        scene == ng::SceneID::THERMAL_VERIFY_HOT_BLOCKS ||
        scene == ng::SceneID::THERMAL_VERIFY_CROSS_IGNITION ||
        scene == ng::SceneID::THERMAL_VERIFY_BRIDGE_WITNESS ||
        scene == ng::SceneID::THERMAL_VERIFY_IMPACT_RINGDOWN) {
        air_cfg.solid_thermal_diffusivity = 0.0120f;
        air_cfg.solid_contact_transfer = 0.0080f;
        air_cfg.solid_heat_loss = 0.0450f;
        air_cfg.cooling_rate = 0.74f;
        air_cfg.combustion_hold = 0.72f;
    }
    if (scene == ng::SceneID::THERMAL_VERIFY_CROSS_IGNITION ||
        scene == ng::SceneID::THERMAL_VERIFY_BRIDGE_WITNESS) {
        air_cfg.solid_thermal_diffusivity = 0.0150f;
        air_cfg.solid_contact_transfer = 0.0100f;
        air_cfg.solid_heat_loss = 0.0380f;
    }
    if (scene == ng::SceneID::THERMAL_VERIFY_SDF_JUNCTION) {
        g_show_heat_gizmos = true;
    }
    if (scene == ng::SceneID::MAG_CURSOR_UNIT) {
        g_show_magnetic_debug = true;
    }
    if (scene == ng::SceneID::MAGNETIC_BENCH ||
        scene == ng::SceneID::MAGNETIC_CLIMB_BENCH ||
        scene == ng::SceneID::MAGNETIC_FLOOR_BENCH ||
        scene == ng::SceneID::RIGID_MAGNETIC_FLOOR ||
        scene == ng::SceneID::MAG_PERMANENT_POLE ||
        scene == ng::SceneID::MAG_SOFT_IRON_FIELD ||
        scene == ng::SceneID::MAG_SOFT_IRON_BODY ||
        scene == ng::SceneID::HYBRID_FERRO_SPLASH) {
        mag.enabled = true;
        mag.source_scale = 24.0f;
        mag.force_scale = 20.0f;
        mag.jacobi_iterations = 64;
        mag.induction_iterations = 2;
        if (scene == ng::SceneID::MAG_PERMANENT_POLE ||
            scene == ng::SceneID::MAG_SOFT_IRON_BODY) {
            mag.source_scale = 22.0f;
            mag.force_scale = 14.0f;
            mag.jacobi_iterations = 68;
            mag.induction_iterations = 1;
        }
        if (scene == ng::SceneID::MAG_SOFT_IRON_FIELD) {
            mag.source_scale = 20.0f;
            mag.force_scale = 10.0f;
            mag.jacobi_iterations = 72;
            mag.induction_iterations = 1;
        }
        if (scene == ng::SceneID::HYBRID_FERRO_SPLASH) {
            mag.source_scale = 22.0f;
            mag.force_scale = 14.0f;
            mag.jacobi_iterations = 72;
            mag.induction_iterations = 2;
            mag.cursor_field_type = ng::MagneticField::CursorFieldType::BAR_MAGNET;
        }
        if (scene == ng::SceneID::MAGNETIC_FLOOR_BENCH ||
            scene == ng::SceneID::RIGID_MAGNETIC_FLOOR) {
            mag.source_scale = 34.0f;
            mag.force_scale = 10.0f;
            mag.jacobi_iterations = 76;
            mag.induction_iterations = 2;
        }
        g_show_magnetic_debug = true;
    }

    if (scene == ng::SceneID::REACTIVE_HEARTH ||
        scene == ng::SceneID::SEED_ROASTER ||
        scene == ng::SceneID::OOBLECK_IMPACTOR_BENCH ||
        scene == ng::SceneID::IMPACT_MEMORY_BENCH ||
        scene == ng::SceneID::BIO_REPLICATOR_BENCH ||
        scene == ng::SceneID::MYCELIUM_MORPH_BENCH ||
        scene == ng::SceneID::MORPHOGENESIS_BENCH ||
        scene == ng::SceneID::ROOT_GARDEN_BENCH ||
        scene == ng::SceneID::CELL_COLONY_BENCH ||
        scene == ng::SceneID::AUTOMATA_AIR_COUPLING_BENCH ||
        scene == ng::SceneID::AUTOMATA_FIRE_REGROWTH_BENCH ||
        scene == ng::SceneID::AUTOMATA_MAX_COUPLING_BENCH ||
        scene == ng::SceneID::ASH_REGROWTH_BENCH ||
        scene == ng::SceneID::HYBRID_REGROWTH_WALL) {
        air_cfg.bio_enabled = true;
        air_cfg.bio_pattern_speed = 1.15f;
    }
    if (scene == ng::SceneID::BIO_REPLICATOR_BENCH ||
        scene == ng::SceneID::MYCELIUM_MORPH_BENCH ||
        scene == ng::SceneID::MORPHOGENESIS_BENCH ||
        scene == ng::SceneID::ROOT_GARDEN_BENCH ||
        scene == ng::SceneID::CELL_COLONY_BENCH ||
        scene == ng::SceneID::AUTOMATA_AIR_COUPLING_BENCH ||
        scene == ng::SceneID::AUTOMATA_FIRE_REGROWTH_BENCH ||
        scene == ng::SceneID::AUTOMATA_MAX_COUPLING_BENCH ||
        scene == ng::SceneID::ASH_REGROWTH_BENCH ||
        scene == ng::SceneID::HYBRID_REGROWTH_WALL) {
        air_cfg.automata_enabled = true;
        air_cfg.automata_pattern_speed = 1.05f;
        air_cfg.automata_seed_strength = 0.78f;
        air_cfg.automata_coupling = 1.0f;
    }
    if (scene == ng::SceneID::MORPHOGENESIS_BENCH ||
        scene == ng::SceneID::ROOT_GARDEN_BENCH ||
        scene == ng::SceneID::CELL_COLONY_BENCH ||
        scene == ng::SceneID::AUTOMATA_AIR_COUPLING_BENCH ||
        scene == ng::SceneID::AUTOMATA_FIRE_REGROWTH_BENCH ||
        scene == ng::SceneID::AUTOMATA_MAX_COUPLING_BENCH ||
        scene == ng::SceneID::ASH_REGROWTH_BENCH ||
        scene == ng::SceneID::HYBRID_REGROWTH_WALL) {
        air_cfg.bio_feed = 0.033f;
        air_cfg.bio_kill = 0.056f;
        air_cfg.bio_diffuse_a = 0.19f;
        air_cfg.bio_diffuse_b = 0.105f;
        air_cfg.bio_seed_strength = 0.96f;
        air_cfg.bio_pattern_speed = 1.52f;
        air_cfg.bio_coupling = 1.42f;
        air_cfg.automata_enabled = true;
        air_cfg.automata_birth_lo = 0.276f;
        air_cfg.automata_birth_hi = 0.356f;
        air_cfg.automata_survive_lo = 0.262f;
        air_cfg.automata_survive_hi = 0.452f;
        air_cfg.automata_inner_radius = 1.7f;
        air_cfg.automata_outer_radius = 4.6f;
        air_cfg.automata_sigmoid = 0.052f;
        air_cfg.automata_seed_strength = 1.05f;
        air_cfg.automata_pattern_speed = 1.32f;
        air_cfg.automata_coupling = 1.34f;
    }
    if (scene == ng::SceneID::AUTOMATA_AIR_COUPLING_BENCH ||
        scene == ng::SceneID::AUTOMATA_FIRE_REGROWTH_BENCH ||
        scene == ng::SceneID::AUTOMATA_MAX_COUPLING_BENCH) {
        air_cfg.bio_feed = 0.032f;
        air_cfg.bio_kill = 0.057f;
        air_cfg.bio_diffuse_a = 0.20f;
        air_cfg.bio_diffuse_b = 0.108f;
        air_cfg.bio_seed_strength = 1.18f;
        air_cfg.bio_pattern_speed = 1.78f;
        air_cfg.bio_coupling = 1.78f;
        air_cfg.automata_birth_lo = 0.272f;
        air_cfg.automata_birth_hi = 0.352f;
        air_cfg.automata_survive_lo = 0.258f;
        air_cfg.automata_survive_hi = 0.456f;
        air_cfg.automata_inner_radius = 1.65f;
        air_cfg.automata_outer_radius = 4.90f;
        air_cfg.automata_sigmoid = 0.054f;
        air_cfg.automata_seed_strength = 1.24f;
        air_cfg.automata_pattern_speed = 1.62f;
        air_cfg.automata_coupling = 1.64f;
        g_air_vis = 11;
        g_mpm.params().vis_mode = 12;
    }
    if (scene == ng::SceneID::ASH_REGROWTH_BENCH ||
        scene == ng::SceneID::HYBRID_REGROWTH_WALL) {
        air_cfg.bio_feed = 0.032f;
        air_cfg.bio_kill = 0.056f;
        air_cfg.bio_diffuse_a = 0.20f;
        air_cfg.bio_diffuse_b = 0.108f;
        air_cfg.bio_seed_strength = 1.02f;
        air_cfg.bio_pattern_speed = 1.34f;
        air_cfg.bio_coupling = 1.22f;
        air_cfg.bio_regrowth_rate = (scene == ng::SceneID::HYBRID_REGROWTH_WALL) ? 1.05f : 0.82f;
        air_cfg.automata_birth_lo = 0.272f;
        air_cfg.automata_birth_hi = 0.352f;
        air_cfg.automata_survive_lo = 0.258f;
        air_cfg.automata_survive_hi = 0.456f;
        air_cfg.automata_inner_radius = 1.65f;
        air_cfg.automata_outer_radius = 4.90f;
        air_cfg.automata_sigmoid = 0.054f;
        air_cfg.automata_seed_strength = 1.00f;
        air_cfg.automata_pattern_speed = 1.14f;
        air_cfg.automata_coupling = 0.96f;
        g_air_vis = 10;
        g_mpm.params().vis_mode = 11;
    }
    if (scene == ng::SceneID::AUTOMATA_MAX_COUPLING_BENCH) {
        air_cfg.bio_coupling = 6.0f;
        air_cfg.automata_coupling = 6.0f;
        air_cfg.bio_pattern_speed = 1.90f;
        air_cfg.automata_pattern_speed = 1.85f;
        air_cfg.bio_seed_strength = 1.32f;
        air_cfg.automata_seed_strength = 1.30f;
        g_air_vis = 11;
        g_mpm.params().vis_mode = 12;
    }
    if (scene == ng::SceneID::FOOT_DEMO_BENCH) {
        ng::foot_demo_apply_scene_defaults(g_mpm_skin_enabled, g_mpm_surface_style,
                                           g_mpm_skin_threshold, g_mpm_skin_kernel);
        g_color_mode = 1;
        g_mode = InteractMode::FOOT_CONTROL;
        g_show_interaction_window = true;
        g_show_appearance_window = true;
        g_air_vis = 1;
        g_mpm.params().vis_mode = 13;
    }
}

static void apply_automata_test_preset() {
    auto& air_cfg = g_air.config();
    air_cfg.bio_enabled = true;
    air_cfg.bio_feed = 0.033f;
    air_cfg.bio_kill = 0.057f;
    air_cfg.bio_diffuse_a = 0.19f;
    air_cfg.bio_diffuse_b = 0.102f;
    air_cfg.bio_seed_strength = 1.10f;
    air_cfg.bio_pattern_speed = 1.65f;
    air_cfg.bio_coupling = 1.70f;
    air_cfg.bio_regrowth_rate = 1.40f;

    air_cfg.automata_enabled = true;
    air_cfg.automata_birth_lo = 0.272f;
    air_cfg.automata_birth_hi = 0.352f;
    air_cfg.automata_survive_lo = 0.258f;
    air_cfg.automata_survive_hi = 0.456f;
    air_cfg.automata_inner_radius = 1.65f;
    air_cfg.automata_outer_radius = 4.85f;
    air_cfg.automata_sigmoid = 0.054f;
    air_cfg.automata_seed_strength = 1.22f;
    air_cfg.automata_pattern_speed = 1.55f;
    air_cfg.automata_coupling = 1.55f;

    g_show_advanced_window = true;
    g_show_appearance_window = true;
    g_air_vis = 11;
    g_mpm.params().vis_mode = 12;
}

static bool is_sph_thermal_material(ng::MPMMaterial material) {
    switch (material) {
    case ng::MPMMaterial::SPH_BURNING_OIL:
    case ng::MPMMaterial::SPH_BOILING_WATER:
    case ng::MPMMaterial::SPH_THERMAL_SYRUP:
    case ng::MPMMaterial::SPH_FLASH_FLUID:
        return true;
    default:
        return false;
    }
}

static bool has_sph_thermal_batches() {
    for (const auto& batch : g_creation.batches) {
        if (batch.solver == ng::SpawnSolver::SPH &&
            is_sph_thermal_material(batch.mpm_type) &&
            batch.sph_count > 0) {
            return true;
        }
    }
    return false;
}

static void sync_drag_falloff_radius() {
    if (g_drag_falloff_radius < g_tool_radius) g_drag_falloff_radius = g_tool_radius;
}

static void resize_drag_main_radius(ng::f32 delta) {
    ng::f32 old_radius = g_tool_radius;
    g_tool_radius = glm::clamp(g_tool_radius + delta, 0.15f, 1.25f);
    if (g_drag_falloff_radius < g_tool_radius || std::abs(g_drag_falloff_radius - old_radius) < 1e-4f) {
        g_drag_falloff_radius = g_tool_radius;
    }
}

static void resize_drag_falloff_radius(ng::f32 delta) {
    g_drag_falloff_radius = glm::clamp(g_drag_falloff_radius + delta, g_tool_radius, 1.75f);
}

static void sync_magnet_falloff_radius() {
    auto& mp = g_mpm.params();
    if (mp.magnet_falloff_radius < mp.magnet_radius) mp.magnet_falloff_radius = mp.magnet_radius;
}

static void resize_magnet_main_radius(ng::f32 delta) {
    auto& mp = g_mpm.params();
    ng::f32 old_radius = mp.magnet_radius;
    mp.magnet_radius = glm::clamp(mp.magnet_radius + delta, 0.10f, 1.25f);
    if (mp.magnet_falloff_radius < mp.magnet_radius || std::abs(mp.magnet_falloff_radius - old_radius) < 1e-4f) {
        mp.magnet_falloff_radius = mp.magnet_radius;
    }
}

static void resize_magnet_falloff_radius(ng::f32 delta) {
    auto& mp = g_mpm.params();
    mp.magnet_falloff_radius = glm::clamp(mp.magnet_falloff_radius + delta, mp.magnet_radius, 1.75f);
}

static void clear_spring_drag() {
    if (g_sph.spring_drag_active()) g_sph.end_spring_drag();
    if (g_mpm.spring_drag_active()) g_mpm.end_spring_drag();
    g_drag_capture_active = false;
    g_drag_anchor_world = ng::vec2(0.0f);
}

static ng::vec2 rotate_vec2(ng::vec2 v, ng::f32 angle) {
    ng::f32 c = std::cos(angle);
    ng::f32 s = std::sin(angle);
    return ng::vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

static ng::f32 hash01(ng::u32 n) {
    ng::f32 x = std::sin(static_cast<ng::f32>(n) * 12.9898f + 78.233f) * 43758.5453f;
    return x - std::floor(x);
}

static ng::SpawnShape projectile_spawn_shape() {
    static const ng::SpawnShape kShapes[] = {
        ng::SpawnShape::CIRCLE,
        ng::SpawnShape::RECT,
        ng::SpawnShape::BEAM,
        ng::SpawnShape::TRIANGLE
    };
    ng::i32 idx = glm::clamp(g_ball_shape, 0, 3);
    return kShapes[idx];
}

static ng::f32 projectile_shape_aspect(ng::SpawnShape shape) {
    switch (shape) {
    case ng::SpawnShape::BEAM: return 4.2f;
    case ng::SpawnShape::RECT: return 1.0f;
    case ng::SpawnShape::TRIANGLE: return 1.0f;
    default: return 1.0f;
    }
}

static bool point_in_triangle(ng::vec2 p, ng::vec2 a, ng::vec2 b, ng::vec2 c) {
    ng::vec2 v0 = c - a;
    ng::vec2 v1 = b - a;
    ng::vec2 v2 = p - a;
    ng::f32 dot00 = glm::dot(v0, v0);
    ng::f32 dot01 = glm::dot(v0, v1);
    ng::f32 dot02 = glm::dot(v0, v2);
    ng::f32 dot11 = glm::dot(v1, v1);
    ng::f32 dot12 = glm::dot(v1, v2);
    ng::f32 denom = dot00 * dot11 - dot01 * dot01;
    if (std::abs(denom) < 1e-6f) return false;
    ng::f32 inv = 1.0f / denom;
    ng::f32 u = (dot11 * dot02 - dot01 * dot12) * inv;
    ng::f32 v = (dot00 * dot12 - dot01 * dot02) * inv;
    return u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f;
}

static void build_projectile_points(ng::vec2 center, ng::f32 radius, ng::f32 rotation,
                                    ng::SpawnShape shape, ng::f32 spacing,
                                    std::vector<ng::vec2>& positions,
                                    std::vector<ng::f32>& shell_seeds) {
    positions.clear();
    shell_seeds.clear();

    ng::vec2 half_extents = ng::shape_half_extents(shape, radius, projectile_shape_aspect(shape));
    ng::f32 bound_radius = glm::length(half_extents) + spacing;
    if (shape == ng::SpawnShape::CIRCLE) bound_radius = half_extents.x + spacing;

    ng::vec2 tri_a(0.0f, half_extents.y);
    ng::vec2 tri_b(half_extents.x, -0.55f * half_extents.y);
    ng::vec2 tri_c(-half_extents.x, -0.55f * half_extents.y);

    for (ng::f32 y = center.y - bound_radius; y <= center.y + bound_radius; y += spacing) {
        for (ng::f32 x = center.x - bound_radius; x <= center.x + bound_radius; x += spacing) {
            ng::vec2 world(x, y);
            ng::vec2 local = rotate_vec2(world - center, -rotation);
            bool inside = false;
            switch (shape) {
            case ng::SpawnShape::CIRCLE:
                inside = glm::length(local) <= half_extents.x;
                break;
            case ng::SpawnShape::RECT:
            case ng::SpawnShape::BEAM:
                inside = std::abs(local.x) <= half_extents.x && std::abs(local.y) <= half_extents.y;
                break;
            case ng::SpawnShape::TRIANGLE:
                inside = point_in_triangle(local, tri_a, tri_b, tri_c);
                break;
            default:
                inside = glm::length(local) <= half_extents.x;
                break;
            }
            if (!inside) continue;
            positions.push_back(world);
            shell_seeds.push_back(0.0f);
        }
    }
}

struct ProjectilePresetDesc {
    enum class Kind : int {
        SINGLE = 0,
        TIME_BOMB = 1,
        PRESSURE_VESSEL = 2
    };

    enum class VesselMode : int {
        ROUND = 0,
        HEAVY = 1,
        DIRECTIONAL = 2,
        CLAYMORE = 3,
        TIMED = 4,
        LAYERED = 5,
        BROADSIDE = 6,
        TRIGGER = 7,
        ROCKET = 8,
        HEAT = 9,
        HESH = 10,
        APHE = 11,
        DEMO = 12,
        DEEP_FUSE = 13,
        THERMITE = 14,
        TANDEM = 15,
        FUEL_AIR = 16,
        SMOKE = 17,
        FLASH = 18,
        CLUSTER = 19,
        CATACLYSM = 20,
        CASCADE = 21,
        CRYO = 22,
        SPIRAL = 23,
        EVEN_DEEPER_FUSE = 24,
        // Launcher2 extensions: hybrid weapons composed from the primitives above.
        // ROCKET_PAYLOAD = rocket body that flings claymore-style shrapnel forward on
        //                  rupture (warhead at the end of the flight).
        // ROCKET_SIDE_CLAYMORE = rocket that vents a side claymore burst while
        //                  cruising, plus a forward burst on rupture.
        // CLAYMORE_CLUSTER = claymore base with CLUSTER-style sub-munition payload
        //                  (small FIRECRACKER bomblets) instead of solid beads.
        ROCKET_PAYLOAD = 25,
        ROCKET_SIDE_CLAYMORE = 26,
        CLAYMORE_CLUSTER = 27,
        // Contact-triggered variants. All use impact_rupture.
        // LATERAL_CONTACT     = on impact, fragments fly perpendicular to flight (side blast).
        // PENETRATING_DOWN    = on impact, preferred axis snaps to gravity so the burst
        //                       drives downward through the struck surface.
        // CONCUSSION          = on impact, big air pressure pulse with intentionally
        //                       low heat — a "stun" blast instead of an incendiary.
        LATERAL_CONTACT = 28,
        PENETRATING_DOWN = 29,
        CONCUSSION = 30
    };

    Kind kind;
    VesselMode vessel_mode;
    ng::MPMMaterial material;
    ng::f32 stiffness;
    ng::f32 density_scale;
    ng::f32 initial_temp;
    ng::vec4 thermal_scale;
    const char* summary;
};

static ProjectilePreset active_projectile_preset_id() {
    if (g_mode == InteractMode::LAUNCHER2) {
        // Nine-weapon palette for the key-6 launcher. Rows 4-6 are hybrid
        // compositions (rocket+payload, rocket+side-claymores, claymore+cluster),
        // rows 7-9 are contact-triggered variants that detonate only on impact
        // (lateral burst, gravity-direction penetrator, concussion wave).
        static constexpr ProjectilePreset kLauncher2[9] = {
            ProjectilePreset::REAL_TIME_BOMB,
            ProjectilePreset::ROCKET,
            ProjectilePreset::CLAYMORE,
            ProjectilePreset::ROCKET_PAYLOAD,
            ProjectilePreset::ROCKET_SIDE_CLAYMORE,
            ProjectilePreset::CLAYMORE_CLUSTER,
            ProjectilePreset::LATERAL_CONTACT_CHARGE,
            ProjectilePreset::GRAVITY_PENETRATOR,
            ProjectilePreset::CONCUSSION_CHARGE
        };
        return kLauncher2[glm::clamp(g_launcher2_preset, 0, 8)];
    }
    return static_cast<ProjectilePreset>(
        glm::clamp(g_ball_preset, 0, static_cast<int>(ProjectilePreset::CONCUSSION_CHARGE)));
}

static ProjectilePresetDesc current_projectile_preset() {
    ng::f32 weight_scale = glm::max(g_ball_weight, 0.5f) / 6.0f;
    switch (active_projectile_preset_id()) {
    case ProjectilePreset::SNOW:
        return { ProjectilePresetDesc::Kind::SINGLE, ProjectilePresetDesc::VesselMode::ROUND, ng::MPMMaterial::SNOW, 18000.0f, 2.2f * weight_scale, 268.0f, ng::vec4(1.0f),
                 "Light packed snow shot: low density, breaks and sheds easily." };
    case ProjectilePreset::EXPLOSIVE:
        return { ProjectilePresetDesc::Kind::SINGLE, ProjectilePresetDesc::VesselMode::ROUND, ng::MPMMaterial::FIRECRACKER, 16500.0f, 2.8f * weight_scale, 330.0f, ng::vec4(1.15f, 1.20f, 1.0f, 0.0f),
                 "Reactive charge: relies on high-speed impact heating to ignite after contact." };
    case ProjectilePreset::RUBBER:
        return { ProjectilePresetDesc::Kind::SINGLE, ProjectilePresetDesc::VesselMode::ROUND, ng::MPMMaterial::ELASTIC, 22000.0f, 3.0f * weight_scale, 300.0f, ng::vec4(1.0f),
                 "Rubber slug: springy and good for stress-testing deformation." };
    case ProjectilePreset::TIME_BOMB:
        return { ProjectilePresetDesc::Kind::TIME_BOMB, ProjectilePresetDesc::VesselMode::ROUND, ng::MPMMaterial::STONEWARE, 26000.0f, 3.8f * weight_scale, 300.0f, ng::vec4(0.36f, 0.92f, 0.78f, 0.03f),
                 "Layered shell charge: a hot burn band cooks a gunpowder core inside a hard shell, so even slower shots can explode after a short delay." };
    case ProjectilePreset::REAL_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::ROUND, ng::MPMMaterial::SEALED_CHARGE, 62000.0f, 4.9f * weight_scale, 300.0f, ng::vec4(1.55f, 1.70f, 0.60f, 0.03f),
                 "Reinforced sealed charge: a hotter internal cook path builds pressure behind a firmer shell, so it stays closed longer before venting in a sharper blast." };
    case ProjectilePreset::SIEGE_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::HEAVY, ng::MPMMaterial::SEALED_CHARGE, 86000.0f, 6.8f * weight_scale, 300.0f, ng::vec4(1.72f, 1.95f, 0.62f, 0.02f),
                 "Overbuilt shell with a hotter charge inside. It cooks longer, ruptures later, and dumps a heavier blast once the vessel finally gives way." };
    case ProjectilePreset::DIRECTIONAL_CHARGE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::DIRECTIONAL, ng::MPMMaterial::SEALED_CHARGE, 70000.0f, 4.4f * weight_scale, 300.0f, ng::vec4(1.66f, 1.88f, 0.60f, 0.02f),
                 "Forward-biased shaped charge: one side is seeded weaker, so the vessel vents and throws most of its force along the shot axis instead of equally in all directions." };
    case ProjectilePreset::CLAYMORE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::CLAYMORE, ng::MPMMaterial::SEALED_CHARGE, 76000.0f, 4.8f * weight_scale, 300.0f, ng::vec4(1.80f, 2.00f, 0.60f, 0.02f),
                 "Front-loaded shrapnel mine: a weak forward face plus dense steel beads turns the pressure burst into a short, brutal fragment fan." };
    case ProjectilePreset::REAL_TIME_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::TIMED, ng::MPMMaterial::SEALED_CHARGE, 72000.0f, 4.7f * weight_scale, 300.0f, ng::vec4(1.70f, 1.92f, 0.60f, 0.02f),
                 "Delayed sealed bomb: thick insulated shell, a small conductive cap, and a compact wrapped fuse keep the core colder so the delay feels more like a real bomb." };
    case ProjectilePreset::LAYERED_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::LAYERED, ng::MPMMaterial::SEALED_CHARGE, 88000.0f, 6.2f * weight_scale, 300.0f, ng::vec4(1.88f, 2.05f, 0.62f, 0.02f),
                 "Composite shell bomb: ceramic outside, tougher inner liner, and a hot pressure core. It should hold shape longer and then fail in chunks." };
    case ProjectilePreset::BROADSIDE_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::BROADSIDE, ng::MPMMaterial::SEALED_CHARGE, 76000.0f, 4.9f * weight_scale, 300.0f, ng::vec4(1.74f, 1.94f, 0.60f, 0.02f),
                 "Broadside charge: vents perpendicular to the shot axis, so the blast favors the flanks instead of the nose." };
    case ProjectilePreset::TRIGGER_CHARGE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::TRIGGER, ng::MPMMaterial::SEALED_CHARGE, 69000.0f, 4.6f * weight_scale, 300.0f, ng::vec4(1.68f, 1.90f, 0.60f, 0.02f),
                 "Trigger line bomb: a self-burning train runs along an external line and lights the charge at the far end." };
    case ProjectilePreset::ROCKET:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::ROCKET, ng::MPMMaterial::SEALED_CHARGE, 78000.0f, 4.3f * weight_scale, 300.0f, ng::vec4(1.62f, 1.86f, 0.58f, 0.02f),
                 "Pressure rocket: mostly sealed, but with a hot rear nozzle that vents backward and drives the body forward." };
    case ProjectilePreset::HEAT_CHARGE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::HEAT, ng::MPMMaterial::SEALED_CHARGE, 98000.0f, 4.8f * weight_scale, 300.0f, ng::vec4(1.82f, 1.98f, 0.60f, 0.01f),
                 "HEAT-style shaped charge: thick casing plus a dense forward liner focus the vent into a narrow hot jet instead of a broad blast." };
    case ProjectilePreset::HESH_CHARGE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::HESH, ng::MPMMaterial::SEALED_CHARGE, 94000.0f, 5.0f * weight_scale, 300.0f, ng::vec4(1.76f, 1.94f, 0.64f, 0.02f),
                 "HESH-style charge: favors a broad surface burst and hot side plumes so it behaves more like a squash-head blast than a needle jet." };
    case ProjectilePreset::APHE_SHELL:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::APHE, ng::MPMMaterial::SEALED_CHARGE, 108000.0f, 5.8f * weight_scale, 300.0f, ng::vec4(1.72f, 1.88f, 0.60f, 0.01f),
                 "APHE shell: dense penetrator nose with a delayed internal charge. It should stay intact longer, then vent its energy forward after the casing gives." };
    case ProjectilePreset::DEMOLITION_CHARGE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::DEMO, ng::MPMMaterial::SEALED_CHARGE, 112000.0f, 5.6f * weight_scale, 300.0f, ng::vec4(1.90f, 2.08f, 0.66f, 0.01f),
                 "Demolition charge: extra-firm storage-safe shell around a hotter, slower-cooking core. Good for placing first and lighting later." };
    case ProjectilePreset::DEEP_FUSE_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::DEEP_FUSE, ng::MPMMaterial::SEALED_CHARGE, 124000.0f, 5.4f * weight_scale, 300.0f, ng::vec4(1.72f, 1.88f, 0.70f, 0.01f),
                 "Deep fuse bomb: two insulated layers hide a curled internal fuse, so pre-ignited shots stay round and take a few seconds before the core finally cooks off." };
    case ProjectilePreset::THERMITE_CHARGE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::THERMITE, ng::MPMMaterial::SEALED_CHARGE, 96000.0f, 5.0f * weight_scale, 300.0f, ng::vec4(1.18f, 2.22f, 0.42f, 0.00f),
                 "Thermite charge: not much blast, but once lit it pours sustained heat into the shell and air like a violent incendiary burner." };
    case ProjectilePreset::TANDEM_HEAT:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::TANDEM, ng::MPMMaterial::SEALED_CHARGE, 118000.0f, 5.1f * weight_scale, 300.0f, ng::vec4(1.84f, 2.02f, 0.58f, 0.01f),
                 "Tandem HEAT: a quick precursor path opens the face and the main jet follows, so it should feel like a two-stage shaped charge instead of one single pop." };
    case ProjectilePreset::FUEL_AIR_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::FUEL_AIR, ng::MPMMaterial::SEALED_CHARGE, 72000.0f, 4.9f * weight_scale, 300.0f, ng::vec4(2.18f, 2.30f, 0.58f, 0.06f),
                 "Fuel-air bomb: thin shell, lots of hot gas, and a wider delayed fireball-like pressure plume." };
    case ProjectilePreset::SMOKE_CANISTER:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::SMOKE, ng::MPMMaterial::SEALED_CHARGE, 64000.0f, 4.0f * weight_scale, 300.0f, ng::vec4(0.62f, 0.82f, 0.82f, 0.18f),
                 "Smoke canister: low-heat venting shell that dumps dense smoke and airflow with much less fragment violence." };
    case ProjectilePreset::FLASHBANG:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::FLASH, ng::MPMMaterial::SEALED_CHARGE, 68000.0f, 4.2f * weight_scale, 300.0f, ng::vec4(0.84f, 1.18f, 0.78f, 0.10f),
                 "Flashbang: sharp pressure pulse and bright hot vent, but intentionally weak shell fragments and only a short-lived burn." };
    case ProjectilePreset::CLUSTER_CHARGE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::CLUSTER, ng::MPMMaterial::SEALED_CHARGE, 82000.0f, 4.8f * weight_scale, 300.0f, ng::vec4(1.58f, 1.86f, 0.62f, 0.04f),
                 "Cluster charge: brittle carrier that bursts open and scatters many hot subcharges instead of one single focused blast." };
    case ProjectilePreset::CATACLYSM_CHARGE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::CATACLYSM, ng::MPMMaterial::SEALED_CHARGE, 146000.0f, 6.4f * weight_scale, 300.0f, ng::vec4(2.02f, 2.24f, 0.64f, 0.02f),
                 "Cataclysm charge: demolition-style body with an even heavier two-layer fuse path and a much more violent late rupture." };
    case ProjectilePreset::CASCADE_CLUSTER:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::CASCADE, ng::MPMMaterial::SEALED_CHARGE, 90000.0f, 5.0f * weight_scale, 300.0f, ng::vec4(1.42f, 1.72f, 0.70f, 0.02f),
                 "Cascade cluster: the carrier breaks first, then colder subcharges scatter and only cook off after they have already spread out." };
    case ProjectilePreset::CRYO_CHARGE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::CRYO, ng::MPMMaterial::SEALED_CHARGE, 98000.0f, 4.8f * weight_scale, 248.0f, ng::vec4(0.92f, 1.22f, 0.84f, 0.00f),
                 "Cryo charge: pre-cooled casing with a frozen-looking shell and a slower warm-up before the core finally reaches ignition." };
    case ProjectilePreset::FIRE_SPIRAL_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::SPIRAL, ng::MPMMaterial::SEALED_CHARGE, 88000.0f, 4.7f * weight_scale, 300.0f, ng::vec4(1.78f, 2.02f, 0.60f, 0.04f),
                 "Fire spiral bomb: paired side vents throw hot tangential jets so the blast curls into a rotating flame ribbon instead of a simple puff." };
    case ProjectilePreset::SOFT_CLAYMORE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::CLAYMORE, ng::MPMMaterial::SEALED_CHARGE, 82000.0f, 4.3f * weight_scale, 300.0f, ng::vec4(1.04f, 1.18f, 0.84f, 0.02f),
                 "Soft claymore: a slower, cooler, shorter-range fragment fan with a clearer delay before it opens." };
    case ProjectilePreset::SOFT_BROADSIDE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::BROADSIDE, ng::MPMMaterial::SEALED_CHARGE, 80000.0f, 4.4f * weight_scale, 300.0f, ng::vec4(1.00f, 1.12f, 0.86f, 0.02f),
                 "Soft broadside: side-favoring vent charge tuned to open later and sweep outward without the full-strength room slam." };
    case ProjectilePreset::COOL_SMOKE_POT:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::SMOKE, ng::MPMMaterial::SEALED_CHARGE, 66000.0f, 4.1f * weight_scale, 300.0f, ng::vec4(0.28f, 0.46f, 0.94f, 0.22f),
                 "Cool smoke pot: mostly a delayed venting cloud with only a mild puff of pressure and very little heat." };
    case ProjectilePreset::SOFT_SPIRAL_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::SPIRAL, ng::MPMMaterial::SEALED_CHARGE, 86000.0f, 4.4f * weight_scale, 300.0f, ng::vec4(1.02f, 1.16f, 0.86f, 0.04f),
                 "Soft spiral bomb: tangential venting still curls into a spiral, but with a slower cook-off, less heat, and a gentler blast ring." };
    case ProjectilePreset::MEDIUM_CLAYMORE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::CLAYMORE, ng::MPMMaterial::SEALED_CHARGE, 84000.0f, 4.5f * weight_scale, 300.0f, ng::vec4(1.28f, 1.42f, 0.76f, 0.02f),
                 "Medium claymore: still delayed and cooler than the original, but with a more assertive fragment fan than the soft version." };
    case ProjectilePreset::MEDIUM_BROADSIDE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::BROADSIDE, ng::MPMMaterial::SEALED_CHARGE, 82000.0f, 4.6f * weight_scale, 300.0f, ng::vec4(1.24f, 1.36f, 0.78f, 0.02f),
                 "Medium broadside: side-favoring vent charge with a readable sweep and room push, but still calmer than the original broadside bomb." };
    case ProjectilePreset::MEDIUM_SMOKE_POT:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::SMOKE, ng::MPMMaterial::SEALED_CHARGE, 67000.0f, 4.2f * weight_scale, 300.0f, ng::vec4(0.40f, 0.58f, 0.92f, 0.22f),
                 "Medium smoke pot: delayed venting smoke cloud with a noticeable puff, but still much cooler and less violent than a true blast canister." };
    case ProjectilePreset::MEDIUM_SPIRAL_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::SPIRAL, ng::MPMMaterial::SEALED_CHARGE, 87000.0f, 4.5f * weight_scale, 300.0f, ng::vec4(1.24f, 1.38f, 0.78f, 0.04f),
                 "Medium spiral bomb: swirling side vents with a clearer spiral flame ribbon than the soft version, but still less hot and less forceful than the full bomb." };
    case ProjectilePreset::SOFT_HEAT_CHARGE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::HEAT, ng::MPMMaterial::SEALED_CHARGE, 90000.0f, 4.4f * weight_scale, 300.0f, ng::vec4(1.06f, 1.18f, 0.82f, 0.01f),
                 "Soft HEAT charge: a slower, cooler shaped charge with a narrow focused vent but much less shock and room heating." };
    case ProjectilePreset::MEDIUM_HEAT_CHARGE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::HEAT, ng::MPMMaterial::SEALED_CHARGE, 94000.0f, 4.6f * weight_scale, 300.0f, ng::vec4(1.34f, 1.50f, 0.74f, 0.01f),
                 "Medium HEAT charge: focused jet behavior that still reads clearly, but with less heat and blast than the original HEAT charge." };
    case ProjectilePreset::ABOVE_MED_CLAYMORE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::CLAYMORE, ng::MPMMaterial::SEALED_CHARGE, 85000.0f, 4.7f * weight_scale, 300.0f, ng::vec4(1.56f, 1.72f, 0.68f, 0.02f),
                 "Above med claymore: nearly full-strength fragment fan, but still slightly slower and cooler than the original claymore." };
    case ProjectilePreset::ABOVE_MED_BROADSIDE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::BROADSIDE, ng::MPMMaterial::SEALED_CHARGE, 84000.0f, 4.8f * weight_scale, 300.0f, ng::vec4(1.50f, 1.66f, 0.70f, 0.02f),
                 "Above med broadside: a stronger flank-favoring sweep that sits between the medium preset and the original broadside bomb." };
    case ProjectilePreset::ABOVE_MED_SMOKE_POT:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::SMOKE, ng::MPMMaterial::SEALED_CHARGE, 68000.0f, 4.3f * weight_scale, 300.0f, ng::vec4(0.50f, 0.70f, 0.90f, 0.20f),
                 "Above med smoke pot: thicker smoke and a clearer puff than the medium pot, but still far calmer than the hotter blast canisters." };
    case ProjectilePreset::ABOVE_MED_SPIRAL_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::SPIRAL, ng::MPMMaterial::SEALED_CHARGE, 87500.0f, 4.6f * weight_scale, 300.0f, ng::vec4(1.50f, 1.68f, 0.70f, 0.04f),
                 "Above med spiral bomb: hotter and more forceful than the medium spiral, while still stopping short of the full fire spiral bomb." };
    case ProjectilePreset::ABOVE_MED_HEAT_CHARGE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::HEAT, ng::MPMMaterial::SEALED_CHARGE, 96000.0f, 4.7f * weight_scale, 300.0f, ng::vec4(1.60f, 1.78f, 0.68f, 0.01f),
                 "Above med HEAT charge: strong focused jet behavior with less room heat and shock than the original HEAT charge." };
    case ProjectilePreset::SOFT_LONG_FUSE_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::TIMED, ng::MPMMaterial::SEALED_CHARGE, 90000.0f, 4.4f * weight_scale, 300.0f, ng::vec4(1.00f, 1.16f, 0.84f, 0.02f),
                 "Soft long fuse bomb: a slower wrapped-fuse bomb with a gentle delayed burst and less heat than the original long-fuse design." };
    case ProjectilePreset::MEDIUM_LONG_FUSE_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::TIMED, ng::MPMMaterial::SEALED_CHARGE, 98000.0f, 4.7f * weight_scale, 300.0f, ng::vec4(1.28f, 1.44f, 0.76f, 0.02f),
                 "Medium long fuse bomb: keeps the wrapped-fuse delay, but cooks off with a more readable push than the soft version." };
    case ProjectilePreset::ABOVE_MED_LONG_FUSE_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::TIMED, ng::MPMMaterial::SEALED_CHARGE, 108000.0f, 5.0f * weight_scale, 300.0f, ng::vec4(1.50f, 1.68f, 0.70f, 0.02f),
                 "Above med long fuse bomb: a stronger delayed bomb that sits between the medium variant and the original real time bomb." };
    case ProjectilePreset::SOFT_ROCKET:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::ROCKET, ng::MPMMaterial::SEALED_CHARGE, 86000.0f, 4.1f * weight_scale, 300.0f, ng::vec4(1.02f, 1.18f, 0.84f, 0.02f),
                 "Soft rocket: a gentler pressure rocket with weaker thrust and a cooler plume." };
    case ProjectilePreset::MEDIUM_ROCKET:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::ROCKET, ng::MPMMaterial::SEALED_CHARGE, 92000.0f, 4.3f * weight_scale, 300.0f, ng::vec4(1.28f, 1.44f, 0.76f, 0.02f),
                 "Medium rocket: balanced nozzle venting with a readable push, but less violence than the original rocket." };
    case ProjectilePreset::ABOVE_MED_ROCKET:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::ROCKET, ng::MPMMaterial::SEALED_CHARGE, 100000.0f, 4.5f * weight_scale, 300.0f, ng::vec4(1.46f, 1.64f, 0.70f, 0.02f),
                 "Above med rocket: stronger sustained thrust than the medium rocket while still staying calmer than the original rocket." };
    case ProjectilePreset::SOFT_DEEP_FUSE_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::DEEP_FUSE, ng::MPMMaterial::SEALED_CHARGE, 106000.0f, 4.8f * weight_scale, 300.0f, ng::vec4(1.02f, 1.18f, 0.86f, 0.01f),
                 "Soft deep fuse bomb: same buried fuse idea as the original, but tuned for a cooler and gentler late rupture." };
    case ProjectilePreset::MEDIUM_DEEP_FUSE_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::DEEP_FUSE, ng::MPMMaterial::SEALED_CHARGE, 116000.0f, 5.0f * weight_scale, 300.0f, ng::vec4(1.28f, 1.44f, 0.78f, 0.01f),
                 "Medium deep fuse bomb: buried curled fuse with a calmer core than the original deep fuse bomb, but a stronger finish than the soft version." };
    case ProjectilePreset::ABOVE_MED_DEEP_FUSE_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::DEEP_FUSE, ng::MPMMaterial::SEALED_CHARGE, 120000.0f, 5.2f * weight_scale, 300.0f, ng::vec4(1.48f, 1.66f, 0.74f, 0.01f),
                 "Above med deep fuse bomb: close to the original deep fuse bomb, but still a little less hot and violent on rupture." };
    case ProjectilePreset::SOFT_EVEN_DEEPER_FUSE_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE, ng::MPMMaterial::SEALED_CHARGE, 112000.0f, 4.9f * weight_scale, 300.0f, ng::vec4(1.00f, 1.14f, 0.88f, 0.01f),
                 "Soft even deeper fuse bomb: a double-length buried fuse path for a very delayed, gentle rupture." };
    case ProjectilePreset::MEDIUM_EVEN_DEEPER_FUSE_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE, ng::MPMMaterial::SEALED_CHARGE, 120000.0f, 5.1f * weight_scale, 300.0f, ng::vec4(1.22f, 1.38f, 0.80f, 0.01f),
                 "Medium even deeper fuse bomb: a much longer buried fuse path with a clearer late rupture than the soft variant." };
    case ProjectilePreset::ABOVE_MED_EVEN_DEEPER_FUSE_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE, ng::MPMMaterial::SEALED_CHARGE, 128000.0f, 5.3f * weight_scale, 300.0f, ng::vec4(1.42f, 1.58f, 0.76f, 0.01f),
                 "Above med even deeper fuse bomb: long-delay buried fuse behavior with a stronger rupture that still stops short of the full preset." };
    case ProjectilePreset::EVEN_DEEPER_FUSE_BOMB:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE, ng::MPMMaterial::SEALED_CHARGE, 134000.0f, 5.5f * weight_scale, 300.0f, ng::vec4(1.66f, 1.84f, 0.72f, 0.01f),
                 "Even deeper fuse bomb: an extra-buried multi-wrap fuse path that roughly doubles the original deep fuse delay before the core finally lights in earnest." };
    case ProjectilePreset::ROCKET_PAYLOAD:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::ROCKET_PAYLOAD, ng::MPMMaterial::SEALED_CHARGE, 82000.0f, 4.5f * weight_scale, 300.0f, ng::vec4(1.62f, 1.84f, 0.60f, 0.02f),
                 "Rocket + forward payload: the rocket body propels normally, and when the shell finally ruptures it flings a dense claymore-style shrapnel pack forward along the flight axis." };
    case ProjectilePreset::ROCKET_SIDE_CLAYMORE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::ROCKET_SIDE_CLAYMORE, ng::MPMMaterial::SEALED_CHARGE, 82000.0f, 4.6f * weight_scale, 300.0f, ng::vec4(1.70f, 1.88f, 0.60f, 0.02f),
                 "Rocket with side claymores: a forward-propelling rocket that also vents lateral fragment bursts mid-flight. Good for sweeping a corridor while moving through it." };
    case ProjectilePreset::CLAYMORE_CLUSTER:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::CLAYMORE_CLUSTER, ng::MPMMaterial::SEALED_CHARGE, 78000.0f, 4.6f * weight_scale, 300.0f, ng::vec4(1.72f, 1.92f, 0.62f, 0.03f),
                 "Claymore with cluster payload: instead of solid beads, the claymore front launches a spray of small secondary FIRECRACKER bomblets that each cook off shortly after being flung out." };
    case ProjectilePreset::LATERAL_CONTACT_CHARGE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::LATERAL_CONTACT, ng::MPMMaterial::SEALED_CHARGE, 80000.0f, 4.4f * weight_scale, 300.0f, ng::vec4(1.52f, 1.80f, 0.62f, 0.02f),
                 "Side-burst contact charge: inert in flight, detonates on impact, sprays fragments perpendicular to the flight axis. Good for corridor-clearing hits on walls." };
    case ProjectilePreset::GRAVITY_PENETRATOR:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::PENETRATING_DOWN, ng::MPMMaterial::SEALED_CHARGE, 92000.0f, 5.4f * weight_scale, 300.0f, ng::vec4(1.52f, 1.72f, 0.64f, 0.01f),
                 "Gravity penetrator: inert in flight, on contact the burst + payload snap downward (gravity direction) regardless of incoming angle. For driving shrapnel through a floor or roof you're sitting on top of." };
    case ProjectilePreset::CONCUSSION_CHARGE:
        return { ProjectilePresetDesc::Kind::PRESSURE_VESSEL, ProjectilePresetDesc::VesselMode::CONCUSSION, ng::MPMMaterial::SEALED_CHARGE, 72000.0f, 4.2f * weight_scale, 300.0f, ng::vec4(1.14f, 0.30f, 0.90f, 0.06f),
                 "Concussion charge: high-pressure shockwave with deliberately low heat. Pushes things hard without setting them on fire or leaving a persistent hot zone. Detonates on contact." };
    case ProjectilePreset::STEEL:
    default:
        return { ProjectilePresetDesc::Kind::SINGLE, ProjectilePresetDesc::VesselMode::ROUND, ng::MPMMaterial::THERMO_METAL, g_ball_stiffness, 7.5f * weight_scale, 300.0f, ng::vec4(1.0f),
                 "Steel slug: dense, fast, and able to pick up friction heat on very hard impacts." };
    }
}

static ng::f32 projectile_boundary_metric(ng::vec2 local, ng::SpawnShape shape, ng::vec2 half_extents) {
    const ng::f32 hx = glm::max(half_extents.x, 1e-4f);
    const ng::f32 hy = glm::max(half_extents.y, 1e-4f);
    switch (shape) {
    case ng::SpawnShape::CIRCLE:
        return glm::clamp(glm::length(local) / hx, 0.0f, 1.0f);
    case ng::SpawnShape::RECT:
    case ng::SpawnShape::BEAM:
        return glm::clamp(glm::max(std::abs(local.x) / hx, std::abs(local.y) / hy), 0.0f, 1.0f);
    case ng::SpawnShape::TRIANGLE: {
        ng::vec2 a(0.0f, hy);
        ng::vec2 b(hx, -0.55f * hy);
        ng::vec2 c(-hx, -0.55f * hy);
        ng::f32 denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
        if (std::abs(denom) < 1e-6f) return 1.0f;
        ng::f32 w1 = ((b.y - c.y) * (local.x - c.x) + (c.x - b.x) * (local.y - c.y)) / denom;
        ng::f32 w2 = ((c.y - a.y) * (local.x - c.x) + (a.x - c.x) * (local.y - c.y)) / denom;
        ng::f32 w3 = 1.0f - w1 - w2;
        ng::f32 min_w = glm::clamp(glm::min(w1, glm::min(w2, w3)), 0.0f, 1.0f / 3.0f);
        return 1.0f - min_w / (1.0f / 3.0f);
    }
    default:
        return glm::clamp(glm::length(local) / hx, 0.0f, 1.0f);
    }
}

static void collect_time_bomb_layers(ng::vec2 center, ng::f32 radius, ng::f32 rotation,
                                     ng::SpawnShape shape, ng::f32 spacing,
                                     std::vector<ng::vec2>& core_positions,
                                     std::vector<ng::f32>& core_shell,
                                     std::vector<ng::vec2>& fuse_positions,
                                     std::vector<ng::f32>& fuse_shell,
                                     std::vector<ng::vec2>& shell_positions,
                                     std::vector<ng::f32>& shell_shell) {
    std::vector<ng::vec2> positions;
    std::vector<ng::f32> shell_seeds;
    build_projectile_points(center, radius, rotation, shape, spacing, positions, shell_seeds);

    ng::vec2 half_extents = ng::shape_half_extents(shape, radius, projectile_shape_aspect(shape));
    core_positions.clear();
    core_shell.clear();
    fuse_positions.clear();
    fuse_shell.clear();
    shell_positions.clear();
    shell_shell.clear();

    for (const ng::vec2& world : positions) {
        ng::vec2 local = rotate_vec2(world - center, -rotation);
        ng::f32 metric = projectile_boundary_metric(local, shape, half_extents);
        if (metric >= 0.68f) {
            shell_positions.push_back(world);
            shell_shell.push_back(1.0f);
        } else if (metric >= 0.42f) {
            fuse_positions.push_back(world);
            fuse_shell.push_back(0.72f);
        } else {
            core_positions.push_back(world);
            core_shell.push_back(metric * 0.28f);
        }
    }

    if (core_positions.empty() && !fuse_positions.empty()) {
        core_positions = fuse_positions;
        core_shell = fuse_shell;
        fuse_positions.clear();
        fuse_shell.clear();
    }
}

static void collect_pressure_vessel_layers(ng::vec2 center, ng::f32 radius, ng::f32 rotation,
                                           ng::SpawnShape shape, ng::f32 spacing,
                                           ng::f32 shell_threshold,
                                           ng::f32 core_threshold,
                                           ng::vec2 fuse_half_scale,
                                           ng::vec2 fuse_metric_range,
                                           ng::f32 core_fallback_threshold,
                                           std::vector<ng::vec2>& core_positions,
                                           std::vector<ng::f32>& core_shell,
                                           std::vector<ng::vec2>& fuse_positions,
                                           std::vector<ng::f32>& fuse_shell,
                                           std::vector<ng::vec2>& shell_positions,
                                           std::vector<ng::f32>& shell_shell) {
    std::vector<ng::vec2> positions;
    std::vector<ng::f32> shell_seeds;
    build_projectile_points(center, radius, rotation, shape, spacing, positions, shell_seeds);

    ng::vec2 half_extents = ng::shape_half_extents(shape, radius, projectile_shape_aspect(shape));
    core_positions.clear();
    core_shell.clear();
    fuse_positions.clear();
    fuse_shell.clear();
    shell_positions.clear();
    shell_shell.clear();

    for (const ng::vec2& world : positions) {
        ng::vec2 local = rotate_vec2(world - center, -rotation);
        ng::f32 metric = projectile_boundary_metric(local, shape, half_extents);
        if (metric >= shell_threshold) {
            shell_positions.push_back(world);
            shell_shell.push_back(1.0f);
        } else if (metric <= core_threshold) {
            core_positions.push_back(world);
            core_shell.push_back(metric * 0.12f);
        } else if (metric >= fuse_metric_range.x && metric <= fuse_metric_range.y &&
                   std::abs(local.x) <= half_extents.x * fuse_half_scale.x &&
                   std::abs(local.y) <= half_extents.y * fuse_half_scale.y) {
            fuse_positions.push_back(world);
            fuse_shell.push_back(glm::mix(0.12f, 0.28f, glm::clamp(metric, 0.0f, 1.0f)));
        }
    }

    if (core_positions.empty()) {
        for (const ng::vec2& world : positions) {
            ng::vec2 local = rotate_vec2(world - center, -rotation);
            ng::f32 metric = projectile_boundary_metric(local, shape, half_extents);
            if (metric <= core_fallback_threshold) {
                core_positions.push_back(world);
                core_shell.push_back(metric * 0.14f);
            }
        }
    }
    if (fuse_positions.empty() && !core_positions.empty()) {
        fuse_positions.push_back(core_positions.front());
        fuse_shell.push_back(0.18f);
    }
}

static void collect_real_bomb_layers(ng::vec2 center, ng::f32 radius, ng::f32 rotation,
                                     ng::SpawnShape shape, ng::f32 spacing,
                                     std::vector<ng::vec2>& core_positions,
                                     std::vector<ng::f32>& core_shell,
                                     std::vector<ng::vec2>& fuse_positions,
                                     std::vector<ng::f32>& fuse_shell,
                                     std::vector<ng::vec2>& shell_positions,
                                     std::vector<ng::f32>& shell_shell) {
    collect_pressure_vessel_layers(center, radius, rotation, shape, spacing,
                                   0.70f, 0.12f,
                                   ng::vec2(0.18f, 0.20f), ng::vec2(0.14f, 0.38f), 0.18f,
                                   core_positions, core_shell, fuse_positions, fuse_shell,
                                   shell_positions, shell_shell);
}

static void collect_heavy_bomb_layers(ng::vec2 center, ng::f32 radius, ng::f32 rotation,
                                      ng::SpawnShape shape, ng::f32 spacing,
                                      std::vector<ng::vec2>& core_positions,
                                      std::vector<ng::f32>& core_shell,
                                      std::vector<ng::vec2>& fuse_positions,
                                      std::vector<ng::f32>& fuse_shell,
                                      std::vector<ng::vec2>& shell_positions,
                                      std::vector<ng::f32>& shell_shell) {
    collect_pressure_vessel_layers(center, radius, rotation, shape, spacing,
                                   0.58f, 0.08f,
                                   ng::vec2(0.14f, 0.18f), ng::vec2(0.16f, 0.34f), 0.14f,
                                   core_positions, core_shell, fuse_positions, fuse_shell,
                                   shell_positions, shell_shell);
}

static void collect_layered_bomb_layers(ng::vec2 center, ng::f32 radius, ng::f32 rotation,
                                        ng::SpawnShape shape, ng::f32 spacing,
                                        std::vector<ng::vec2>& core_positions,
                                        std::vector<ng::f32>& core_shell,
                                        std::vector<ng::vec2>& fuse_positions,
                                        std::vector<ng::f32>& fuse_shell,
                                        std::vector<ng::vec2>& shell_positions,
                                        std::vector<ng::f32>& shell_shell,
                                        std::vector<ng::vec2>& armor_positions,
                                        std::vector<ng::f32>& armor_shell) {
    std::vector<ng::vec2> positions;
    std::vector<ng::f32> shell_seeds;
    build_projectile_points(center, radius, rotation, shape, spacing, positions, shell_seeds);

    ng::vec2 half_extents = ng::shape_half_extents(shape, radius, projectile_shape_aspect(shape));
    core_positions.clear();
    core_shell.clear();
    fuse_positions.clear();
    fuse_shell.clear();
    shell_positions.clear();
    shell_shell.clear();
    armor_positions.clear();
    armor_shell.clear();

    for (const ng::vec2& world : positions) {
        ng::vec2 local = rotate_vec2(world - center, -rotation);
        ng::f32 metric = projectile_boundary_metric(local, shape, half_extents);
        if (metric >= 0.82f) {
            shell_positions.push_back(world);
            shell_shell.push_back(1.0f);
        } else if (metric >= 0.60f) {
            armor_positions.push_back(world);
            armor_shell.push_back(glm::mix(0.38f, 0.82f, metric));
        } else if (metric >= 0.28f) {
            fuse_positions.push_back(world);
            fuse_shell.push_back(glm::mix(0.18f, 0.40f, metric));
        } else {
            core_positions.push_back(world);
            core_shell.push_back(metric * 0.10f);
        }
    }

    if (core_positions.empty()) {
        for (const ng::vec2& world : positions) {
            ng::vec2 local = rotate_vec2(world - center, -rotation);
            ng::f32 metric = projectile_boundary_metric(local, shape, half_extents);
            if (metric <= 0.20f) {
                core_positions.push_back(world);
                core_shell.push_back(metric * 0.12f);
            }
        }
    }
    if (fuse_positions.empty() && !armor_positions.empty()) {
        fuse_positions.push_back(armor_positions.front());
        fuse_shell.push_back(0.24f);
    }
}

static void collect_claymore_payload(ng::vec2 center, ng::f32 radius, ng::f32 rotation,
                                     ng::SpawnShape shape, ng::f32 spacing,
                                     std::vector<ng::vec2>& payload_positions,
                                     std::vector<ng::f32>& payload_shell) {
    std::vector<ng::vec2> positions;
    std::vector<ng::f32> shell_seeds;
    build_projectile_points(center, radius, rotation, shape, spacing, positions, shell_seeds);
    ng::vec2 half_extents = ng::shape_half_extents(shape, radius, projectile_shape_aspect(shape));

    payload_positions.clear();
    payload_shell.clear();

    for (const ng::vec2& world : positions) {
        ng::vec2 local = rotate_vec2(world - center, -rotation);
        ng::f32 metric = projectile_boundary_metric(local, shape, half_extents);
        ng::f32 front = local.x / glm::max(half_extents.x, 1e-4f);
        if (front < 0.18f || front > 0.82f) continue;
        if (std::abs(local.y) > half_extents.y * 0.46f) continue;
        if (metric < 0.20f || metric > 0.70f) continue;
        payload_positions.push_back(world);
        payload_shell.push_back(glm::clamp(metric * 0.16f, 0.02f, 0.22f));
    }
}

static void collect_payload_cloud_points(ng::vec2 center, ng::f32 radius, ng::f32 rotation,
                                         ng::SpawnShape shape, ng::f32 spacing,
                                         std::vector<ng::vec2>& payload_positions,
                                         std::vector<ng::f32>& payload_shell) {
    std::vector<ng::vec2> positions;
    std::vector<ng::f32> shell_seeds;
    build_projectile_points(center, radius, rotation, shape, spacing, positions, shell_seeds);
    ng::vec2 half_extents = ng::shape_half_extents(shape, radius, projectile_shape_aspect(shape));

    payload_positions.clear();
    payload_shell.clear();

    for (const ng::vec2& world : positions) {
        ng::vec2 local = rotate_vec2(world - center, -rotation);
        ng::f32 metric = projectile_boundary_metric(local, shape, half_extents);
        if (metric < 0.24f || metric > 0.62f) continue;
        payload_positions.push_back(world);
        payload_shell.push_back(glm::clamp(0.10f + metric * 0.10f, 0.10f, 0.20f));
    }
}

static void collect_trigger_wrap_points(ng::vec2 center,
                                        ng::f32 radius,
                                        ng::f32 rotation,
                                        ng::SpawnShape shape,
                                        ng::f32 spacing,
                                        ng::f32 start_angle_deg,
                                        ng::f32 end_angle_deg,
                                        std::vector<ng::vec2>& trigger_positions,
                                        std::vector<ng::f32>& trigger_shell) {
    trigger_positions.clear();
    trigger_shell.clear();

    std::vector<ng::vec2> positions;
    std::vector<ng::f32> shell_seeds;
    build_projectile_points(center, radius, rotation, shape, spacing, positions, shell_seeds);
    ng::vec2 half_extents = ng::shape_half_extents(shape, radius, projectile_shape_aspect(shape));

    ng::f32 start_angle = glm::radians(start_angle_deg);
    ng::f32 end_angle = glm::radians(end_angle_deg);
    const ng::f32 two_pi = 6.28318530718f;
    while (start_angle < 0.0f) start_angle += two_pi;
    while (end_angle < start_angle) end_angle += two_pi;
    const ng::f32 span = glm::max(end_angle - start_angle, 1e-4f);

    for (const ng::vec2& world : positions) {
        ng::vec2 local = rotate_vec2(world - center, -rotation);
        ng::f32 metric = projectile_boundary_metric(local, shape, half_extents);
        if (metric < 0.74f || metric > 0.96f) continue;

        ng::f32 angle = std::atan2(local.y, local.x);
        if (angle < 0.0f) angle += two_pi;
        while (angle < start_angle) angle += two_pi;
        if (angle < start_angle || angle > end_angle) continue;

        ng::f32 t = glm::clamp((angle - start_angle) / span, 0.0f, 1.0f);
        trigger_positions.push_back(world);
        trigger_shell.push_back(t);
    }
}

static void append_trigger_wrap_segment(ng::vec2 center,
                                        ng::f32 radius,
                                        ng::f32 rotation,
                                        ng::SpawnShape shape,
                                        ng::f32 spacing,
                                        ng::f32 start_angle_deg,
                                        ng::f32 end_angle_deg,
                                        ng::f32 t0,
                                        ng::f32 t1,
                                        std::vector<ng::vec2>& trigger_positions,
                                        std::vector<ng::f32>& trigger_shell) {
    std::vector<ng::vec2> segment_positions;
    std::vector<ng::f32> segment_shell;
    collect_trigger_wrap_points(center, radius, rotation, shape, spacing,
                                start_angle_deg, end_angle_deg,
                                segment_positions, segment_shell);
    for (size_t i = 0; i < segment_positions.size(); ++i) {
        trigger_positions.push_back(segment_positions[i]);
        trigger_shell.push_back(glm::mix(t0, t1, segment_shell[i]));
    }
}

static void collect_deep_fuse_trigger_points(ng::vec2 center,
                                             ng::f32 radius,
                                             ng::f32 rotation,
                                             ng::SpawnShape shape,
                                             ng::f32 spacing,
                                             std::vector<ng::vec2>& trigger_positions,
                                             std::vector<ng::f32>& trigger_shell) {
    trigger_positions.clear();
    trigger_shell.clear();
    append_trigger_wrap_segment(center, radius * 0.92f, rotation, shape, spacing,
                                132.0f, 448.0f, 0.00f, 0.34f,
                                trigger_positions, trigger_shell);
    append_trigger_wrap_segment(center, radius * 0.74f, rotation, shape, spacing,
                                156.0f, 516.0f, 0.34f, 0.69f,
                                trigger_positions, trigger_shell);
    append_trigger_wrap_segment(center, radius * 0.56f, rotation, shape, spacing,
                                188.0f, 584.0f, 0.69f, 1.00f,
                                trigger_positions, trigger_shell);
}

static void collect_even_deeper_fuse_trigger_points(ng::vec2 center,
                                                    ng::f32 radius,
                                                    ng::f32 rotation,
                                                    ng::SpawnShape shape,
                                                    ng::f32 spacing,
                                                    std::vector<ng::vec2>& trigger_positions,
                                                    std::vector<ng::f32>& trigger_shell) {
    trigger_positions.clear();
    trigger_shell.clear();
    append_trigger_wrap_segment(center, radius * 0.96f, rotation, shape, spacing,
                                136.0f, 476.0f, 0.00f, 0.20f,
                                trigger_positions, trigger_shell);
    append_trigger_wrap_segment(center, radius * 0.82f, rotation, shape, spacing,
                                162.0f, 556.0f, 0.20f, 0.44f,
                                trigger_positions, trigger_shell);
    append_trigger_wrap_segment(center, radius * 0.68f, rotation, shape, spacing,
                                188.0f, 636.0f, 0.44f, 0.68f,
                                trigger_positions, trigger_shell);
    append_trigger_wrap_segment(center, radius * 0.54f, rotation, shape, spacing,
                                216.0f, 712.0f, 0.68f, 0.88f,
                                trigger_positions, trigger_shell);
    append_trigger_wrap_segment(center, radius * 0.42f, rotation, shape, spacing,
                                246.0f, 780.0f, 0.88f, 1.00f,
                                trigger_positions, trigger_shell);
}

static void collect_shell_cap_points(ng::vec2 center,
                                     ng::f32 radius,
                                     ng::f32 rotation,
                                     ng::SpawnShape shape,
                                     ng::f32 spacing,
                                     std::vector<ng::vec2>& cap_positions,
                                     std::vector<ng::f32>& cap_shell) {
    std::vector<ng::vec2> positions;
    std::vector<ng::f32> shell_seeds;
    build_projectile_points(center, radius, rotation, shape, spacing, positions, shell_seeds);
    ng::vec2 half_extents = ng::shape_half_extents(shape, radius, projectile_shape_aspect(shape));

    cap_positions.clear();
    cap_shell.clear();
    for (const ng::vec2& world : positions) {
        ng::vec2 local = rotate_vec2(world - center, -rotation);
        ng::f32 metric = projectile_boundary_metric(local, shape, half_extents);
        ng::f32 front = local.x / glm::max(half_extents.x, 1e-4f);
        if (front < 0.54f) continue;
        if (std::abs(local.y) > half_extents.y * 0.24f) continue;
        if (metric < 0.72f || metric > 0.98f) continue;
        cap_positions.push_back(world);
        cap_shell.push_back(glm::clamp(0.84f + metric * 0.12f, 0.84f, 0.98f));
    }
}

static ParticleSpanRef merge_contiguous_spans(const ParticleSpanRef& a, const ParticleSpanRef& b) {
    if (!a.valid()) return b;
    if (!b.valid()) return a;
    if (a.end() != b.offset) return a;
    return { a.offset, a.count + b.count };
}

static void seed_shell_crack_bias(const ParticleSpanRef& span,
                                  const std::vector<ng::vec2>& shell_positions,
                                  ng::vec2 center,
                                  ng::vec2 axis,
                                  ng::f32 front_start,
                                  ng::f32 crack_min,
                                  ng::f32 crack_max) {
    if (!span.valid() || shell_positions.empty()) return;
    const auto& mpm_range = g_particles.range(ng::SolverType::MPM);
    if (span.offset < mpm_range.offset || span.end() > (mpm_range.offset + mpm_range.count)) return;

    ng::vec2 axis_n = glm::length(axis) > 1e-5f ? glm::normalize(axis) : ng::vec2(1.0f, 0.0f);
    std::vector<ng::f32> cracks(span.count, 0.0f);
    for (ng::u32 i = 0; i < span.count && i < shell_positions.size(); ++i) {
        ng::vec2 radial = shell_positions[i] - center;
        ng::f32 r = glm::length(radial);
        if (r < 1e-5f) continue;
        ng::f32 alignment = glm::dot(radial / r, axis_n);
        if (alignment <= front_start) continue;
        ng::f32 t = glm::clamp((alignment - front_start) / glm::max(1.0f - front_start, 1e-4f), 0.0f, 1.0f);
        cracks[i] = glm::mix(crack_min, crack_max, t * t);
    }

    ng::u32 local_offset = span.offset - mpm_range.offset;
    g_mpm.jp_buf().upload(cracks.data(), span.count * sizeof(ng::f32), local_offset * sizeof(ng::f32));
}

static void seed_shell_crack_bias_symmetric(const ParticleSpanRef& span,
                                            const std::vector<ng::vec2>& shell_positions,
                                            ng::vec2 center,
                                            ng::vec2 axis,
                                            ng::f32 side_start,
                                            ng::f32 crack_min,
                                            ng::f32 crack_max) {
    if (!span.valid() || shell_positions.empty()) return;
    const auto& mpm_range = g_particles.range(ng::SolverType::MPM);
    if (span.offset < mpm_range.offset || span.end() > (mpm_range.offset + mpm_range.count)) return;

    ng::vec2 axis_n = glm::length(axis) > 1e-5f ? glm::normalize(axis) : ng::vec2(1.0f, 0.0f);
    std::vector<ng::f32> cracks(span.count, 0.0f);
    for (ng::u32 i = 0; i < span.count && i < shell_positions.size(); ++i) {
        ng::vec2 radial = shell_positions[i] - center;
        ng::f32 r = glm::length(radial);
        if (r < 1e-5f) continue;
        ng::f32 alignment = std::abs(glm::dot(radial / r, axis_n));
        if (alignment <= side_start) continue;
        ng::f32 t = glm::clamp((alignment - side_start) / glm::max(1.0f - side_start, 1e-4f), 0.0f, 1.0f);
        cracks[i] = glm::mix(crack_min, crack_max, t * t);
    }

    ng::u32 local_offset = span.offset - mpm_range.offset;
    g_mpm.jp_buf().upload(cracks.data(), span.count * sizeof(ng::f32), local_offset * sizeof(ng::f32));
}

static ng::vec2 current_projectile_vector() {
    ng::f32 fallback_len = glm::max(g_ball_radius * 2.25f,
                                    g_ball_min_launch_speed / glm::max(g_ball_launch_gain, 0.1f));
    ng::vec2 aim = g_ball_has_aim ? g_ball_launch_vector
                                  : ng::vec2(fallback_len, 0.0f);
    if (glm::length(aim) < 0.05f) {
        aim = ng::vec2(fallback_len, 0.0f);
    }
    return aim;
}

static bool pressure_vessel_span_alive(const ParticleSpanRef& span) {
    const auto& mpm_range = g_particles.range(ng::SolverType::MPM);
    return span.count > 0 &&
           span.offset >= mpm_range.offset &&
           span.end() <= (mpm_range.offset + mpm_range.count);
}

static void update_pressure_vessels(ng::f32 dt) {
    if (g_pressure_vessels.empty()) return;

    const auto& mpm_range = g_particles.range(ng::SolverType::MPM);
    if (mpm_range.count == 0) {
        g_pressure_vessels.clear();
        return;
    }

    std::vector<PressureVesselRecord> next;
    next.reserve(g_pressure_vessels.size());

    for (PressureVesselRecord vessel : g_pressure_vessels) {
        if (!pressure_vessel_span_alive(vessel.shell) || !pressure_vessel_span_alive(vessel.core)) {
            continue;
        }

        auto download_vec2_particles = [&](const ParticleSpanRef& span, std::vector<ng::vec2>& out) {
            out.resize(span.count);
            g_particles.positions().download(out.data(), span.count * sizeof(ng::vec2), span.offset * sizeof(ng::vec2));
        };
        auto download_vec2_vel = [&](const ParticleSpanRef& span, std::vector<ng::vec2>& out) {
            out.resize(span.count);
            g_particles.velocities().download(out.data(), span.count * sizeof(ng::vec2), span.offset * sizeof(ng::vec2));
        };
        auto upload_vec2_vel = [&](const ParticleSpanRef& span, const std::vector<ng::vec2>& data) {
            if (span.count == 0 || data.empty()) return;
            g_particles.upload_velocities(span.offset, data.data(), span.count);
        };
        auto download_particle_temps = [&](const ParticleSpanRef& span, std::vector<ng::f32>& out) {
            out.resize(span.count);
            g_particles.temperatures().download(out.data(), span.count * sizeof(ng::f32), span.offset * sizeof(ng::f32));
        };
        auto upload_particle_temps = [&](const ParticleSpanRef& span, const std::vector<ng::f32>& data) {
            if (span.count == 0 || data.empty()) return;
            g_particles.upload_temperatures(span.offset, data.data(), span.count);
        };
        auto download_mpm_scalar = [&](const ng::GPUBuffer& buf, const ParticleSpanRef& span, std::vector<ng::f32>& out) {
            out.resize(span.count);
            ng::u32 local_offset = span.offset - mpm_range.offset;
            buf.download(out.data(), span.count * sizeof(ng::f32), local_offset * sizeof(ng::f32));
        };
        auto download_mpm_vec4 = [&](const ng::GPUBuffer& buf, const ParticleSpanRef& span, std::vector<ng::vec4>& out) {
            out.resize(span.count);
            ng::u32 local_offset = span.offset - mpm_range.offset;
            buf.download(out.data(), span.count * sizeof(ng::vec4), local_offset * sizeof(ng::vec4));
        };
        auto upload_mpm_scalar = [&](ng::GPUBuffer& buf, const ParticleSpanRef& span, const std::vector<ng::f32>& data) {
            if (span.count == 0 || data.empty()) return;
            ng::u32 local_offset = span.offset - mpm_range.offset;
            buf.upload(data.data(), span.count * sizeof(ng::f32), local_offset * sizeof(ng::f32));
        };

        std::vector<ng::vec2> shell_pos, shell_vel, core_pos, core_vel;
        std::vector<ng::f32> shell_temp, shell_crack, core_temp, core_damage, core_phase;
        download_vec2_particles(vessel.shell, shell_pos);
        download_vec2_vel(vessel.shell, shell_vel);
        download_particle_temps(vessel.shell, shell_temp);
        download_mpm_scalar(g_mpm.jp_buf(), vessel.shell, shell_crack);

        download_vec2_particles(vessel.core, core_pos);
        download_vec2_vel(vessel.core, core_vel);
        download_particle_temps(vessel.core, core_temp);
        download_mpm_scalar(g_mpm.damage_buf(), vessel.core, core_damage);
        download_mpm_scalar(g_mpm.phase_buf(), vessel.core, core_phase);

        std::vector<ng::vec2> fuse_pos, fuse_vel;
        std::vector<ng::f32> fuse_temp, fuse_damage;
        if (pressure_vessel_span_alive(vessel.fuse)) {
            download_vec2_particles(vessel.fuse, fuse_pos);
            download_vec2_vel(vessel.fuse, fuse_vel);
            download_particle_temps(vessel.fuse, fuse_temp);
            download_mpm_scalar(g_mpm.damage_buf(), vessel.fuse, fuse_damage);
        }

        std::vector<ng::vec2> payload_pos, payload_vel;
        if (pressure_vessel_span_alive(vessel.payload)) {
            download_vec2_particles(vessel.payload, payload_pos);
            download_vec2_vel(vessel.payload, payload_vel);
        }
        std::vector<ng::vec2> trigger_pos, trigger_vel;
        std::vector<ng::f32> trigger_temp;
        std::vector<ng::vec4> trigger_params;
        if (pressure_vessel_span_alive(vessel.trigger)) {
            download_vec2_particles(vessel.trigger, trigger_pos);
            download_vec2_vel(vessel.trigger, trigger_vel);
            download_particle_temps(vessel.trigger, trigger_temp);
            download_mpm_vec4(g_mpm.mat_params_buf(), vessel.trigger, trigger_params);
        }

        ng::vec2 center(0.0f);
        for (const ng::vec2& p : shell_pos) center += p;
        center /= glm::max<ng::f32>(1.0f, static_cast<ng::f32>(shell_pos.size()));

        ng::f32 radius_avg = 0.0f;
        ng::f32 radius_max = 0.05f;
        ng::f32 crack_avg = 0.0f;
        ng::vec2 breach_dir(0.0f);
        ng::f32 breach_w = 0.0f;
        for (ng::u32 i = 0; i < vessel.shell.count; ++i) {
            ng::vec2 radial = shell_pos[i] - center;
            ng::f32 r = glm::length(radial);
            radius_avg += r;
            radius_max = glm::max(radius_max, r);
            ng::f32 crack = glm::clamp(shell_crack[i], 0.0f, 1.0f);
            crack_avg += crack;
            if (r > 1e-5f) {
                ng::f32 w = crack * crack + 0.01f;
                breach_dir += (radial / r) * w;
                breach_w += w;
            }
        }
        radius_avg /= glm::max<ng::f32>(1.0f, static_cast<ng::f32>(shell_pos.size()));
        crack_avg /= glm::max<ng::f32>(1.0f, static_cast<ng::f32>(shell_crack.size()));
        if (breach_w > 1e-5f && glm::length(breach_dir) > 1e-5f) {
            breach_dir = glm::normalize(breach_dir);
        } else {
            breach_dir = ng::vec2(0.0f, 1.0f);
        }
        ng::vec2 preferred_axis = glm::length(vessel.preferred_axis) > 1e-5f ? glm::normalize(vessel.preferred_axis)
                                                                              : breach_dir;
        if (vessel.axis_bias > 1e-4f) {
            breach_dir = glm::normalize(glm::mix(breach_dir, preferred_axis, glm::clamp(vessel.axis_bias, 0.0f, 1.0f)));
        }
        ng::vec2 side_axis(-preferred_axis.y, preferred_axis.x);

        auto avg_scalar = [](const std::vector<ng::f32>& data) -> ng::f32 {
            if (data.empty()) return 0.0f;
            ng::f32 sum = 0.0f;
            for (ng::f32 v : data) sum += v;
            return sum / static_cast<ng::f32>(data.size());
        };

        ng::f32 shell_temp_avg = avg_scalar(shell_temp);
        ng::f32 core_temp_avg = avg_scalar(core_temp);
        ng::f32 core_cooked = avg_scalar(core_damage);
        ng::f32 core_gas = avg_scalar(core_phase);
        ng::f32 fuse_temp_avg = avg_scalar(fuse_temp);
        ng::f32 fuse_burn = avg_scalar(fuse_damage);

        ng::f32 shell_hot = glm::clamp((shell_temp_avg - 420.0f) / 520.0f, 0.0f, 1.0f);
        ng::f32 heat_drive = glm::clamp((glm::max(core_temp_avg, fuse_temp_avg) - 360.0f) / 420.0f, 0.0f, 1.0f);
        ng::f32 flame_drive = glm::clamp(glm::max(core_cooked, fuse_burn * 1.15f), 0.0f, 1.0f);
        ng::f32 shell_integrity = glm::clamp(1.0f - glm::smoothstep(0.24f, 0.78f, crack_avg), 0.0f, 1.0f);
        ng::f32 vent_open = glm::clamp(glm::smoothstep(0.38f, 0.82f, crack_avg), 0.0f, 1.0f);
        ng::f32 ignition_gate = vessel.auto_arm ? 1.0f : 0.0f;
        ng::f32 ignite_ramp = vessel.auto_arm ? 1.0f : 0.0f;
        ng::f32 trigger_gate = 0.0f;
        if (!trigger_pos.empty()) {
            ng::f32 front_heat = 0.0f;
            ng::f32 start_heat = 0.0f;
            for (ng::u32 i = 0; i < trigger_pos.size(); ++i) {
                ng::f32 burn_t = trigger_params.empty() ? 0.0f : glm::clamp(trigger_params[i].w, 0.0f, 1.0f);
                ng::f32 hot = glm::smoothstep(360.0f, 520.0f, trigger_temp[i]);
                if (burn_t <= 0.08f) start_heat = glm::max(start_heat, hot);
                if (std::abs(burn_t - vessel.trigger_progress) <= 0.12f) front_heat = glm::max(front_heat, hot);
            }
            if (vessel.auto_arm) {
                vessel.trigger_progress = glm::min(vessel.trigger_progress + dt * glm::max(vessel.trigger_speed, 0.0f), 1.25f);
            } else if (start_heat > 0.08f || vessel.trigger_progress > 0.0f) {
                ng::f32 drive = glm::max(start_heat, front_heat);
                vessel.trigger_progress = glm::min(vessel.trigger_progress +
                                                   dt * glm::max(vessel.trigger_speed, 0.0f) * glm::max(drive, 0.18f),
                                                   1.25f);
            }
            for (ng::u32 i = 0; i < trigger_pos.size(); ++i) {
                ng::f32 burn_t = trigger_params.empty() ? 0.0f : glm::clamp(trigger_params[i].w, 0.0f, 1.0f);
                if (burn_t <= vessel.trigger_progress) {
                    trigger_temp[i] = glm::max(trigger_temp[i], vessel.trigger_heat);
                }
            }
            trigger_gate = glm::smoothstep(0.92f, 1.02f, vessel.trigger_progress);
            ignite_ramp = glm::max(ignite_ramp, trigger_gate);
            ignition_gate = vessel.auto_arm ? glm::mix(0.05f, 1.0f, trigger_gate) : trigger_gate;
            if (trigger_gate > 0.0f) {
                vessel.gas_mass += vessel.trigger_boost * trigger_gate * dt;
                for (ng::f32& temp : fuse_temp) temp = glm::max(temp, 720.0f + trigger_gate * 320.0f);
            }
        } else if (vessel.ignition_delay > 0.0f && vessel.ignition_delay < 90.0f) {
            ng::f32 time_drive = vessel.age + fuse_burn * 0.55f + heat_drive * 0.25f;
            ng::f32 ramp = glm::smoothstep(vessel.ignition_delay,
                                           vessel.ignition_delay + glm::max(vessel.ignition_window, 0.02f),
                                           time_drive);
            ignite_ramp = glm::max(ignite_ramp, ramp);
            ignition_gate = vessel.auto_arm ? glm::mix(0.06f, 1.0f, ramp) : ramp;
        }
        if (!vessel.auto_arm) {
            ng::f32 manual_heat_gate = glm::clamp((glm::max(core_temp_avg, fuse_temp_avg) - 390.0f) / 150.0f, 0.0f, 1.0f);
            ignite_ramp = glm::max(ignite_ramp, glm::max(manual_heat_gate, flame_drive));
            ignition_gate = glm::max(ignition_gate, glm::max(manual_heat_gate, flame_drive));
        }
        ng::f32 nozzle_gate = 0.0f;
        if (vessel.nozzle_open > 1e-4f) {
            nozzle_gate = glm::smoothstep(0.10f, 0.42f, glm::max(fuse_burn, heat_drive));
        }
        ng::f32 effective_vent = glm::clamp(vent_open + vessel.nozzle_open * nozzle_gate, 0.0f, 1.5f);

        ng::f32 gas_source = (0.18f + core_gas * 1.85f + flame_drive * 1.10f) *
                             (0.35f + heat_drive * 1.65f) *
                             vessel.gas_source_scale * ignition_gate;
        vessel.gas_mass += gas_source * dt;
        vessel.gas_mass = glm::max(vessel.gas_mass, 0.0f);

        ng::f32 volume = glm::max(3.1415926f * radius_avg * radius_avg, 0.035f);
        ng::f32 confinement = 0.85f + shell_integrity * 2.30f - shell_hot * 0.35f;
        ng::f32 target_pressure = vessel.gas_mass * confinement * (1.0f + heat_drive * 1.8f + flame_drive * 1.2f) / volume;
        vessel.pressure += (target_pressure - vessel.pressure) * glm::min(dt * (2.8f - glm::min(effective_vent, 1.0f) * 1.2f), 1.0f);

        ng::f32 leak = effective_vent * (0.45f + vessel.pressure * 0.16f) * vessel.leak_scale * dt;
        vessel.gas_mass = glm::max(vessel.gas_mass - leak, 0.0f);
        vessel.pressure *= glm::max(1.0f - (0.05f + effective_vent * 1.35f * vessel.leak_scale) * dt, 0.0f);

        ng::f32 rupture_threshold = (3.4f + shell_integrity * 2.8f - shell_hot * 0.8f) * vessel.rupture_scale;
        bool just_ruptured = false;

        // ------------------------------------------------------------------
        // Contact sensors + delay fuse. Reference map of bomb "blocks" used
        // here: B3 = crack-rate sensor, B4 = velocity-drop sensor, B5 = delay
        // fuse. Any sensor flips vessel.triggered, then the delay fuse ticks
        // up to trigger_to_rupture_delay before the actual rupture fires.
        // ------------------------------------------------------------------
        ng::f32 curr_shell_speed = 0.0f;
        if (!shell_vel.empty()) {
            for (const ng::vec2& v : shell_vel) curr_shell_speed += glm::length(v);
            curr_shell_speed /= static_cast<ng::f32>(shell_vel.size());
        }
        // B4: sharp velocity drop. Only valid after the vessel was actually
        // flying (prev > 4 m/s) to avoid false positives from spawn frames.
        bool vel_sensor = (vessel.prev_shell_speed > 4.0f &&
                           curr_shell_speed < 0.45f * vessel.prev_shell_speed);
        // B3: crack-rate sensor. Skip the first frame (prev_crack_avg not
        // initialized yet) to avoid a false spike from the crack seed bias.
        ng::f32 crack_rate = 0.0f;
        if (vessel.age > 0.02f) {
            crack_rate = (crack_avg - vessel.prev_crack_avg) / glm::max(dt, 1e-4f);
        }
        bool crack_sensor = (vessel.impact_crack_rate_threshold > 1e-4f &&
                             crack_rate > vessel.impact_crack_rate_threshold);

        if (vessel.impact_rupture && !vessel.ruptured && !vessel.triggered &&
            (vel_sensor || crack_sensor)) {
            vessel.triggered = true;
            vessel.trigger_age = 0.0f;
        }
        vessel.prev_shell_speed = curr_shell_speed;
        vessel.prev_crack_avg = crack_avg;

        // B5: delay fuse. Ticks once triggered. When it reaches the configured
        // delay, the vessel actually ruptures. Axis-snap for PENETRATING_DOWN
        // happens here (at rupture frame) so the fuse delay is spent falling
        // along the flight axis — which is what lets a penetrator bury a bit
        // before the downward burst fires.
        if (vessel.triggered && !vessel.ruptured) {
            vessel.trigger_age += dt;
            if (vessel.trigger_age >= vessel.trigger_to_rupture_delay) {
                if (vessel.penetrate_on_impact) {
                    vessel.preferred_axis = ng::vec2(0.0f, -1.0f);
                    preferred_axis = ng::vec2(0.0f, -1.0f);
                    breach_dir = ng::vec2(0.0f, -1.0f);
                    side_axis = ng::vec2(1.0f, 0.0f);
                }
                vessel.ruptured = true;
                vessel.rupture_age = 0.0f;
                just_ruptured = true;
                vessel.burst_energy += (3.6f + vessel.pressure * 0.8f) *
                                       glm::max(vessel.burst_scale, 0.50f);
            }
        }

        if (!vessel.ruptured && vessel.pressure > rupture_threshold) {
            vessel.ruptured = true;
            vessel.rupture_age = 0.0f;
            just_ruptured = true;
            vessel.burst_energy += (3.0f + (vessel.pressure - rupture_threshold) * 1.8f) * vessel.burst_scale;
        }
        if (vessel.rupture_age >= 0.0f) {
            vessel.rupture_age += dt;
        }
        vessel.burst_energy *= std::exp(-4.8f * dt);

        ng::f32 shell_push = (vessel.pressure * (1.8f + effective_vent * 4.6f) + vessel.burst_energy * 18.0f) * vessel.shell_push_scale * dt;
        ng::f32 crack_drive = (vessel.pressure * (0.07f + 0.24f * (1.0f - shell_integrity)) +
                               vessel.burst_energy * 0.85f + shell_hot * 0.05f) * dt;
        for (ng::u32 i = 0; i < vessel.shell.count; ++i) {
            ng::vec2 radial = shell_pos[i] - center;
            ng::f32 r = glm::length(radial);
            ng::vec2 dir = (r > 1e-5f) ? (radial / r) : breach_dir;
            ng::f32 weakness = glm::clamp(shell_crack[i], 0.0f, 1.0f);
            ng::f32 forward = glm::dot(dir, breach_dir);
            ng::f32 directional_gain = glm::mix(1.0f,
                                                (1.0f + glm::max(forward, 0.0f) * 1.55f) *
                                                (1.0f - glm::max(-forward, 0.0f) * 0.60f),
                                                glm::clamp(vessel.axis_bias, 0.0f, 1.0f));
            shell_vel[i] += dir * shell_push * (0.72f + weakness * 1.85f) * directional_gain;
            shell_crack[i] = glm::clamp(shell_crack[i] + crack_drive * (0.55f + weakness * 0.95f), 0.0f, 1.0f);
        }

        ng::f32 core_push = (vessel.pressure * (0.70f + effective_vent * 1.30f) + vessel.burst_energy * 9.0f) * vessel.core_push_scale * dt;
        ng::f32 forced_temp_gate = glm::smoothstep(0.10f, 0.45f, glm::max(ignite_ramp, flame_drive));
        for (ng::u32 i = 0; i < vessel.core.count; ++i) {
            ng::vec2 radial = core_pos[i] - center;
            ng::f32 r = glm::length(radial);
            ng::vec2 dir = (r > 1e-5f) ? (radial / r) : breach_dir;
            ng::f32 forward = glm::dot(dir, breach_dir);
            ng::f32 directional_gain = glm::mix(1.0f, 0.48f + glm::max(forward, 0.0f) * 1.95f, glm::clamp(vessel.axis_bias, 0.0f, 1.0f));
            core_vel[i] += dir * core_push * directional_gain;
            if (forced_temp_gate > 0.0f) {
                core_temp[i] = glm::max(core_temp[i], 340.0f + forced_temp_gate * (280.0f + vessel.pressure * 28.0f));
            }
        }
        for (ng::u32 i = 0; i < fuse_vel.size(); ++i) {
            ng::vec2 radial = fuse_pos[i] - center;
            ng::f32 r = glm::length(radial);
            ng::vec2 dir = (r > 1e-5f) ? (radial / r) : breach_dir;
            fuse_vel[i] += dir * core_push * 0.85f;
            if (forced_temp_gate > 0.0f) {
                fuse_temp[i] = glm::max(fuse_temp[i], 340.0f + forced_temp_gate * 420.0f);
            }
        }
        if (!payload_vel.empty()) {
            ng::f32 payload_push = (vessel.pressure * (1.2f + effective_vent * 2.1f) + vessel.burst_energy * 26.0f) *
                                   vessel.payload_push_scale * dt;
            for (ng::u32 i = 0; i < payload_vel.size(); ++i) {
                ng::vec2 radial = payload_pos[i] - center;
                ng::f32 r = glm::length(radial);
                ng::vec2 dir = (r > 1e-5f) ? (radial / r) : breach_dir;
                ng::f32 alignment = glm::dot(dir, breach_dir);
                ng::f32 focus = glm::smoothstep(vessel.payload_cone, 1.0f, alignment);
                ng::vec2 primary_dir = glm::normalize(glm::mix(dir, breach_dir, glm::clamp(vessel.payload_directionality, 0.0f, 1.0f)));
                ng::f32 primary_focus = glm::mix(1.0f, focus, glm::clamp(vessel.payload_directionality, 0.0f, 1.0f));
                payload_vel[i] += primary_dir * payload_push * (0.40f + primary_focus * 2.40f);
                payload_vel[i] += dir * payload_push * (0.12f + (1.0f - vessel.payload_directionality) * 0.72f + focus * 0.24f);
            }
        }
        if (vessel.side_blast_scale > 1e-4f) {
            ng::f32 side_push = (vessel.pressure * (0.95f + effective_vent * 1.55f) + vessel.burst_energy * 14.0f) *
                                vessel.side_blast_scale * dt;
            for (ng::u32 i = 0; i < vessel.shell.count; ++i) {
                ng::vec2 radial = shell_pos[i] - center;
                ng::f32 r = glm::length(radial);
                ng::vec2 dir = (r > 1e-5f) ? (radial / r) : side_axis;
                ng::f32 side_align = std::abs(glm::dot(dir, side_axis));
                ng::f32 side_sign = glm::dot(dir, side_axis) >= 0.0f ? 1.0f : -1.0f;
                shell_vel[i] += side_axis * (side_sign * side_push * (0.18f + side_align * 1.40f));
                shell_crack[i] = glm::clamp(shell_crack[i] + crack_drive * side_align * 0.58f, 0.0f, 1.0f);
            }
            for (ng::u32 i = 0; i < vessel.core.count; ++i) {
                ng::vec2 radial = core_pos[i] - center;
                ng::f32 r = glm::length(radial);
                ng::vec2 dir = (r > 1e-5f) ? (radial / r) : side_axis;
                ng::f32 side_align = std::abs(glm::dot(dir, side_axis));
                ng::f32 side_sign = glm::dot(dir, side_axis) >= 0.0f ? 1.0f : -1.0f;
                core_vel[i] += side_axis * (side_sign * side_push * (0.24f + side_align * 1.10f));
            }
        }
        if (vessel.swirl_blast_scale > 1e-4f) {
            ng::f32 swirl_push = (vessel.pressure * (0.85f + effective_vent * 1.45f) + vessel.burst_energy * 13.0f) *
                                 vessel.swirl_blast_scale * dt;
            for (ng::u32 i = 0; i < vessel.shell.count; ++i) {
                ng::vec2 radial = shell_pos[i] - center;
                ng::f32 r = glm::length(radial);
                ng::vec2 dir = (r > 1e-5f) ? (radial / r) : side_axis;
                ng::f32 side_sign = glm::dot(dir, side_axis) >= 0.0f ? 1.0f : -1.0f;
                ng::vec2 tangent = preferred_axis * side_sign;
                ng::f32 side_align = std::abs(glm::dot(dir, side_axis));
                shell_vel[i] += tangent * swirl_push * (0.22f + side_align * 1.35f);
                shell_crack[i] = glm::clamp(shell_crack[i] + crack_drive * side_align * 0.44f, 0.0f, 1.0f);
            }
            for (ng::u32 i = 0; i < vessel.core.count; ++i) {
                ng::vec2 radial = core_pos[i] - center;
                ng::f32 r = glm::length(radial);
                ng::vec2 dir = (r > 1e-5f) ? (radial / r) : side_axis;
                ng::f32 side_sign = glm::dot(dir, side_axis) >= 0.0f ? 1.0f : -1.0f;
                ng::vec2 tangent = preferred_axis * side_sign;
                ng::f32 side_align = std::abs(glm::dot(dir, side_axis));
                core_vel[i] += tangent * swirl_push * (0.26f + side_align * 1.18f);
            }
        }
        if (vessel.thrust_scale > 1e-4f && vessel.nozzle_open > 1e-4f) {
            ng::vec2 nozzle_dir = -preferred_axis;
            ng::f32 thrust = vessel.pressure * vessel.nozzle_open * vessel.thrust_scale *
                             (0.24f + nozzle_gate * 0.96f) * dt;
            for (ng::vec2& vel : shell_vel) vel += preferred_axis * thrust * 0.72f;
            for (ng::vec2& vel : core_vel) vel += preferred_axis * thrust * 1.10f;
            for (ng::vec2& vel : fuse_vel) vel += preferred_axis * thrust * 1.18f;
            vessel.gas_mass = glm::max(vessel.gas_mass - (0.12f + nozzle_gate * 0.34f) * vessel.nozzle_open * dt, 0.0f);
            if (!shell_crack.empty()) {
                for (ng::u32 i = 0; i < shell_crack.size(); ++i) {
                    ng::vec2 radial = shell_pos[i] - center;
                    ng::f32 r = glm::length(radial);
                    if (r < 1e-5f) continue;
                    ng::f32 rear = glm::dot(radial / r, nozzle_dir);
                    if (rear > 0.55f) {
                        shell_crack[i] = glm::clamp(shell_crack[i] + dt * 0.08f * vessel.nozzle_open, 0.0f, 1.0f);
                    }
                }
            }
        }

        upload_vec2_vel(vessel.shell, shell_vel);
        upload_mpm_scalar(g_mpm.jp_buf(), vessel.shell, shell_crack);
        upload_vec2_vel(vessel.core, core_vel);
        upload_particle_temps(vessel.core, core_temp);
        if (!fuse_vel.empty()) {
            upload_vec2_vel(vessel.fuse, fuse_vel);
            upload_particle_temps(vessel.fuse, fuse_temp);
        }
        if (!payload_vel.empty()) {
            upload_vec2_vel(vessel.payload, payload_vel);
        }
        if (!trigger_vel.empty()) {
            upload_vec2_vel(vessel.trigger, trigger_vel);
            upload_particle_temps(vessel.trigger, trigger_temp);
        }

        if (vessel.rupture_age >= 0.0f) {
            ng::f32 wave_life = glm::clamp(0.24f + vessel.burst_scale * 0.10f +
                                           vessel.plume_radius_scale * 0.05f,
                                           0.22f, 0.56f);
            ng::f32 wave_phase = 1.0f - glm::clamp(vessel.rupture_age / wave_life, 0.0f, 1.0f);
            if (wave_phase > 0.02f && (just_ruptured || vessel.burst_energy > 0.02f || vessel.pressure > 0.10f)) {
                ng::f32 front_speed = 3.8f + vessel.plume_push_scale * 1.45f +
                                      vessel.burst_scale * 0.55f;
                ng::f32 front_center = glm::max(radius_avg * 0.55f,
                                                radius_avg + vessel.rupture_age * front_speed);
                ng::f32 ring_half = glm::clamp(0.08f + radius_avg * 0.34f +
                                               vessel.plume_radius_scale * 0.04f,
                                               0.10f, 0.42f);
                ng::f32 inner_radius = glm::max(0.0f, front_center - ring_half);
                ng::f32 outer_radius = glm::clamp(front_center + ring_half,
                                                  radius_avg * 0.90f, 2.15f);
                ng::f32 burst_pulse = 1.0f + (just_ruptured ? 0.35f : 0.0f);
                ng::f32 blast_strength =
                    (vessel.burst_energy * 62.0f + vessel.pressure * 9.0f + effective_vent * 8.0f) *
                    vessel.plume_push_scale * vessel.blast_push_scale * burst_pulse * (0.62f + 0.38f * wave_phase);
                ng::f32 blast_heat =
                    (110.0f + vessel.burst_energy * 420.0f + vessel.pressure * 38.0f) *
                    vessel.plume_heat_scale * vessel.blast_heat_scale * (0.50f + 0.50f * wave_phase);
                ng::f32 blast_smoke =
                    (0.22f + vessel.burst_energy * 0.95f + effective_vent * 0.28f) *
                    glm::max(vessel.plume_radius_scale, 0.7f);
                ng::f32 blast_divergence =
                    (0.12f + vessel.burst_energy * 1.65f + vessel.pressure * 0.16f) *
                    glm::max(vessel.plume_push_scale, 0.40f) * burst_pulse *
                    (0.58f + 0.42f * wave_phase);
                g_air.blast_at(center, inner_radius, outer_radius, blast_strength,
                               blast_heat, blast_smoke, blast_divergence, dt);
            }
        }

        if (effective_vent > 0.04f || vessel.burst_energy > 0.02f) {
            ng::vec2 vent_pos = center + breach_dir * glm::max(radius_avg, 0.06f);
            ng::f32 plume_strength = (vessel.pressure * (0.8f + effective_vent * 2.4f) + vessel.burst_energy * 16.0f * vessel.burst_scale) *
                                     vessel.plume_push_scale;
            ng::f32 plume_radius = glm::clamp((radius_avg * 0.42f + 0.08f) * vessel.plume_radius_scale, 0.10f, 0.72f);
            g_air.blow_at(vent_pos, breach_dir, plume_radius, plume_strength, dt);
            g_air.inject_heat_at(vent_pos, plume_radius * 0.92f,
                                 (680.0f + vessel.pressure * 160.0f + vessel.burst_energy * 420.0f * vessel.burst_scale) * vessel.plume_heat_scale,
                                 dt);
        }
        if (vessel.side_blast_scale > 1e-4f && (effective_vent > 0.06f || vessel.burst_energy > 0.03f)) {
            ng::f32 plume_radius = glm::clamp((radius_avg * 0.34f + 0.07f) * vessel.plume_radius_scale, 0.09f, 0.58f);
            ng::f32 side_strength = (vessel.pressure * (0.75f + effective_vent * 1.85f) +
                                     vessel.burst_energy * 12.0f) * vessel.side_blast_scale * vessel.plume_push_scale;
            for (int sign : { -1, 1 }) {
                ng::vec2 dir = side_axis * static_cast<ng::f32>(sign);
                ng::vec2 vent_pos = center + dir * glm::max(radius_avg, 0.06f);
                g_air.blow_at(vent_pos, dir, plume_radius, side_strength, dt);
                g_air.inject_heat_at(vent_pos, plume_radius * 0.88f,
                                     (620.0f + vessel.pressure * 120.0f + vessel.burst_energy * 260.0f) * vessel.plume_heat_scale,
                                     dt);
            }
        }
        if (vessel.swirl_blast_scale > 1e-4f && (effective_vent > 0.06f || vessel.burst_energy > 0.03f)) {
            ng::f32 plume_radius = glm::clamp((radius_avg * 0.32f + 0.06f) * vessel.plume_radius_scale, 0.08f, 0.54f);
            ng::f32 swirl_strength = (vessel.pressure * (0.82f + effective_vent * 1.70f) +
                                      vessel.burst_energy * 11.0f) * vessel.swirl_blast_scale * vessel.plume_push_scale;
            for (int sign : { -1, 1 }) {
                ng::vec2 vent_side = side_axis * static_cast<ng::f32>(sign);
                ng::vec2 dir = preferred_axis * static_cast<ng::f32>(sign);
                ng::vec2 vent_pos = center + vent_side * glm::max(radius_avg, 0.06f);
                g_air.blow_at(vent_pos, dir, plume_radius, swirl_strength, dt);
                g_air.inject_heat_at(vent_pos, plume_radius * 0.86f,
                                     (700.0f + vessel.pressure * 130.0f + vessel.burst_energy * 280.0f) * vessel.plume_heat_scale,
                                     dt);
            }
        }
        if (vessel.thrust_scale > 1e-4f && vessel.nozzle_open > 1e-4f && (nozzle_gate > 0.02f || vessel.pressure > 0.08f)) {
            ng::vec2 nozzle_dir = -preferred_axis;
            ng::vec2 vent_pos = center + nozzle_dir * glm::max(radius_avg, 0.06f);
            ng::f32 plume_radius = glm::clamp((radius_avg * 0.30f + 0.06f) * vessel.plume_radius_scale, 0.08f, 0.42f);
            ng::f32 thrust_strength = vessel.pressure * (1.25f + nozzle_gate * 2.80f) * vessel.thrust_scale * vessel.plume_push_scale;
            g_air.blow_at(vent_pos, nozzle_dir, plume_radius, thrust_strength, dt);
            g_air.inject_heat_at(vent_pos, plume_radius * 0.86f,
                                 (760.0f + vessel.pressure * 140.0f + nozzle_gate * 180.0f) * vessel.plume_heat_scale,
                                 dt);
        }

        vessel.age += dt;
        if (vessel.age < 8.0f || vessel.pressure > 0.08f || vessel.gas_mass > 0.04f || vessel.burst_energy > 0.02f) {
            next.push_back(vessel);
        }
    }

    g_pressure_vessels = std::move(next);
}

static void fire_projectile(ng::vec2 origin) {
    const ProjectilePreset preset_id = active_projectile_preset_id();
    ng::vec2 aim = current_projectile_vector();
    ng::f32 aim_len = glm::length(aim);
    if (aim_len < 0.05f) {
        aim = ng::vec2(0.0f, 1.0f);
        aim_len = 1.0f;
    }

    ng::f32 base_speed = glm::clamp(aim_len * g_ball_launch_gain, g_ball_min_launch_speed, 42.0f);
    ng::f32 cone_rad = glm::radians(glm::clamp(g_ball_cone_deg, 0.0f, 65.0f));
    ng::f32 cone_jitter = ((hash01(++g_ball_shot_counter) * 2.0f) - 1.0f) * cone_rad;
    ng::vec2 launch_dir = rotate_vec2(glm::normalize(aim), cone_jitter);
    ng::vec2 launch_vel = launch_dir * base_speed;

    ng::f32 spacing = g_mpm_grid.dx() * 0.5f;
    const ProjectilePresetDesc preset = current_projectile_preset();
    const ng::SpawnShape shape = projectile_spawn_shape();
    ng::u32 before = g_particles.range(ng::SolverType::MPM).count;
    ng::u32 global_offset = g_particles.range(ng::SolverType::MPM).offset + before;

    ng::f32 old_E = g_mpm.params().youngs_modulus;
    if (preset.kind == ProjectilePresetDesc::Kind::TIME_BOMB ||
        preset.kind == ProjectilePresetDesc::Kind::PRESSURE_VESSEL) {
        const ng::f32 launch_angle = std::atan2(launch_dir.y, launch_dir.x);
        std::vector<ng::vec2> core_positions;
        std::vector<ng::f32> core_shell;
        std::vector<ng::vec2> fuse_positions;
        std::vector<ng::f32> fuse_shell;
        std::vector<ng::vec2> shell_positions;
        std::vector<ng::f32> shell_shell;
        std::vector<ng::vec2> armor_positions;
        std::vector<ng::f32> armor_shell;
        std::vector<ng::vec2> cap_positions;
        std::vector<ng::f32> cap_shell;
        std::vector<ng::vec2> payload_positions;
        std::vector<ng::f32> payload_shell;
        std::vector<ng::vec2> trigger_positions;
        std::vector<ng::f32> trigger_shell;
        if (preset.kind == ProjectilePresetDesc::Kind::PRESSURE_VESSEL &&
            preset.vessel_mode == ProjectilePresetDesc::VesselMode::HEAVY) {
            collect_heavy_bomb_layers(origin, g_ball_radius, launch_angle, shape, spacing,
                                      core_positions, core_shell, fuse_positions, fuse_shell, shell_positions, shell_shell);
        } else if (preset.kind == ProjectilePresetDesc::Kind::PRESSURE_VESSEL &&
                   (preset.vessel_mode == ProjectilePresetDesc::VesselMode::LAYERED ||
                    preset.vessel_mode == ProjectilePresetDesc::VesselMode::TIMED ||
                    preset.vessel_mode == ProjectilePresetDesc::VesselMode::HEAT ||
                    preset.vessel_mode == ProjectilePresetDesc::VesselMode::HESH ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::APHE ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEMO ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEEP_FUSE ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::THERMITE ||
                    preset.vessel_mode == ProjectilePresetDesc::VesselMode::TANDEM ||
                    preset.vessel_mode == ProjectilePresetDesc::VesselMode::FUEL_AIR ||
                    preset.vessel_mode == ProjectilePresetDesc::VesselMode::CLUSTER ||
                    preset.vessel_mode == ProjectilePresetDesc::VesselMode::CATACLYSM ||
                    preset.vessel_mode == ProjectilePresetDesc::VesselMode::CASCADE ||
                    preset.vessel_mode == ProjectilePresetDesc::VesselMode::CRYO ||
                    preset.vessel_mode == ProjectilePresetDesc::VesselMode::SPIRAL ||
                    preset.vessel_mode == ProjectilePresetDesc::VesselMode::ROCKET_PAYLOAD ||
                    preset.vessel_mode == ProjectilePresetDesc::VesselMode::ROCKET_SIDE_CLAYMORE ||
                    preset.vessel_mode == ProjectilePresetDesc::VesselMode::CLAYMORE_CLUSTER ||
                    preset.vessel_mode == ProjectilePresetDesc::VesselMode::LATERAL_CONTACT ||
                    preset.vessel_mode == ProjectilePresetDesc::VesselMode::PENETRATING_DOWN ||
                    preset.vessel_mode == ProjectilePresetDesc::VesselMode::CONCUSSION)) {
            collect_layered_bomb_layers(origin, g_ball_radius, launch_angle, shape, spacing,
                                        core_positions, core_shell, fuse_positions, fuse_shell,
                                        shell_positions, shell_shell, armor_positions, armor_shell);
        } else {
            collect_real_bomb_layers(origin, g_ball_radius, launch_angle, shape, spacing,
                                     core_positions, core_shell, fuse_positions, fuse_shell, shell_positions, shell_shell);
        }
        if (preset.kind == ProjectilePresetDesc::Kind::TIME_BOMB) {
            collect_time_bomb_layers(origin, g_ball_radius, launch_angle, shape, spacing,
                                     core_positions, core_shell, fuse_positions, fuse_shell, shell_positions, shell_shell);
        } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::CLAYMORE ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::HEAT ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::APHE ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::TANDEM ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::ROCKET_PAYLOAD ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::ROCKET_SIDE_CLAYMORE ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::LATERAL_CONTACT ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::PENETRATING_DOWN) {
            // Claymore-style shrapnel pack. Direction is decided later by the vessel
            // setup (forward / lateral / gravity-snap).
            collect_claymore_payload(origin, g_ball_radius, launch_angle, shape, spacing,
                                     payload_positions, payload_shell);
        } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::CLUSTER ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::CLAYMORE_CLUSTER) {
            // CLAYMORE_CLUSTER: same cluster bomblet spray as CLUSTER, but the base
            // vessel is tuned like a claymore (forward-biased burst). Wired here so
            // the payload material is FIRECRACKER (secondary cook-off) not steel.
            collect_payload_cloud_points(origin, g_ball_radius, launch_angle, shape, spacing,
                                         payload_positions, payload_shell);
        } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::CASCADE) {
            collect_payload_cloud_points(origin, g_ball_radius, launch_angle, shape, spacing,
                                         payload_positions, payload_shell);
        } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::TIMED) {
            collect_trigger_wrap_points(origin, g_ball_radius * 1.06f, launch_angle, shape, spacing,
                                        138.0f, 388.0f,
                                        trigger_positions, trigger_shell);
            collect_shell_cap_points(origin, g_ball_radius * 1.04f, launch_angle, shape, spacing,
                                     cap_positions, cap_shell);
            for (ng::f32& s : core_shell) s = glm::clamp(0.24f + s * 0.60f, 0.24f, 0.48f);
            for (ng::f32& s : fuse_shell) s = glm::clamp(0.58f + s * 0.24f, 0.58f, 0.82f);
            for (ng::f32& s : armor_shell) s = glm::clamp(0.72f + s * 0.16f, 0.72f, 0.92f);
        } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEEP_FUSE ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::CATACLYSM ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::CRYO) {
            if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE) {
                collect_even_deeper_fuse_trigger_points(origin, g_ball_radius, launch_angle, shape, spacing,
                                                        trigger_positions, trigger_shell);
            } else {
                collect_deep_fuse_trigger_points(origin, g_ball_radius, launch_angle, shape, spacing,
                                                 trigger_positions, trigger_shell);
            }
            collect_shell_cap_points(origin, g_ball_radius * 1.03f, launch_angle, shape, spacing,
                                     cap_positions, cap_shell);
            if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE) {
                for (ng::f32& s : core_shell) s = glm::clamp(0.16f + s * 0.48f, 0.16f, 0.38f);
                for (ng::f32& s : fuse_shell) s = glm::clamp(0.70f + s * 0.16f, 0.70f, 0.88f);
                for (ng::f32& s : armor_shell) s = glm::clamp(0.80f + s * 0.10f, 0.80f, 0.95f);
            } else {
                for (ng::f32& s : core_shell) s = glm::clamp(0.20f + s * 0.54f, 0.20f, 0.42f);
                for (ng::f32& s : fuse_shell) s = glm::clamp(0.66f + s * 0.18f, 0.66f, 0.86f);
                for (ng::f32& s : armor_shell) s = glm::clamp(0.78f + s * 0.12f, 0.78f, 0.94f);
            }
        } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::TANDEM) {
            collect_trigger_wrap_points(origin, g_ball_radius * 1.00f, launch_angle, shape, spacing,
                                        144.0f, 324.0f,
                                        trigger_positions, trigger_shell);
            collect_shell_cap_points(origin, g_ball_radius * 1.02f, launch_angle, shape, spacing,
                                     cap_positions, cap_shell);
            for (ng::f32& s : core_shell) s = glm::clamp(0.18f + s * 0.58f, 0.18f, 0.44f);
            for (ng::f32& s : fuse_shell) s = glm::clamp(0.56f + s * 0.22f, 0.56f, 0.82f);
            for (ng::f32& s : armor_shell) s = glm::clamp(0.70f + s * 0.18f, 0.70f, 0.90f);
        } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::HEAT ||
                   preset.vessel_mode == ProjectilePresetDesc::VesselMode::APHE) {
            collect_shell_cap_points(origin, g_ball_radius * 1.02f, launch_angle, shape, spacing,
                                     cap_positions, cap_shell);
        } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::TRIGGER) {
            collect_trigger_wrap_points(origin, g_ball_radius * 1.08f, launch_angle, shape, spacing,
                                        112.0f, 430.0f,
                                        trigger_positions, trigger_shell);
        }

        auto spawn_layer = [&](const std::vector<ng::vec2>& positions,
                               const std::vector<ng::f32>& shell_seeds,
                               ng::MPMMaterial material,
                               ng::f32 stiffness,
                               ng::f32 initial_temp,
                               ng::f32 density_scale,
                               ng::vec4 thermal_scale) -> ParticleSpanRef {
            if (positions.empty()) return {};
            ng::u32 local_before = g_particles.range(ng::SolverType::MPM).count;
            g_mpm.params().youngs_modulus = stiffness;
            g_mpm.spawn_points(g_particles, positions, shell_seeds, spacing,
                               material, initial_temp, ng::vec2(1.0f, 0.0f),
                               density_scale, thermal_scale);
            ng::u32 local_after = g_particles.range(ng::SolverType::MPM).count;
            return { g_particles.range(ng::SolverType::MPM).offset + local_before, local_after - local_before };
        };

        if (preset.kind == ProjectilePresetDesc::Kind::TIME_BOMB) {
            ng::f32 legacy_fuse_temp = g_projectile_auto_arm ? 520.0f : 300.0f;
            ng::f32 legacy_core_temp = g_projectile_auto_arm ? 335.0f : 300.0f;
            (void)spawn_layer(shell_positions, shell_shell,
                              ng::MPMMaterial::STONEWARE, 26000.0f, 300.0f,
                              4.6f * glm::max(g_ball_weight, 0.5f) / 6.0f,
                              ng::vec4(0.34f, 0.92f, 0.78f, 0.03f));
            (void)spawn_layer(fuse_positions, fuse_shell,
                              ng::MPMMaterial::BURNING, 14000.0f, legacy_fuse_temp,
                              1.8f * glm::max(g_ball_weight, 0.5f) / 6.0f,
                              ng::vec4(1.05f, 1.30f, 0.44f, 0.10f));
            (void)spawn_layer(core_positions, core_shell,
                              ng::MPMMaterial::FIRECRACKER, 11000.0f, legacy_core_temp,
                              1.3f * glm::max(g_ball_weight, 0.5f) / 6.0f,
                              ng::vec4(1.42f, 1.55f, 0.72f, 0.04f));
        } else {
            ng::f32 shell_stiffness = 94000.0f;
            ng::f32 shell_density = 5.4f * glm::max(g_ball_weight, 0.5f) / 6.0f;
            ng::f32 fuse_stiffness = 24000.0f;
            ng::f32 fuse_density = 0.90f * glm::max(g_ball_weight, 0.5f) / 6.0f;
            ng::f32 core_stiffness = 17000.0f;
            ng::f32 core_density = 1.25f * glm::max(g_ball_weight, 0.5f) / 6.0f;
            ng::MPMMaterial fuse_material = ng::MPMMaterial::BURNING;
            ng::f32 shell_initial_temp = 300.0f;
            ng::f32 fuse_initial_temp = 760.0f;
            ng::f32 core_initial_temp = 360.0f;
            ng::vec4 shell_thermal(0.02f, 0.22f, 1.02f, 0.00f);
            ng::vec4 fuse_thermal(1.10f, 1.34f, 0.46f, 0.04f);
            ng::vec4 core_thermal(3.40f, 2.10f, 0.58f, 0.00f);
            ng::MPMMaterial shell_material = ng::MPMMaterial::STONEWARE;
            ng::MPMMaterial payload_material = ng::MPMMaterial::THERMO_METAL;
            ng::f32 payload_stiffness = 118000.0f;
            ng::f32 payload_density = 5.8f * glm::max(g_ball_weight, 0.5f) / 6.0f;
            ng::f32 payload_initial_temp = 300.0f;
            ng::vec4 payload_thermal(0.12f, 0.18f, 1.02f, 0.00f);
            ng::f32 idle_shell_guard = 1.0f;

            PressureVesselRecord vessel{};
            vessel.gas_mass = 0.25f;
            vessel.preferred_axis = glm::normalize(launch_dir);
            vessel.auto_arm = g_projectile_auto_arm;

            if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::HEAVY) {
                shell_stiffness = 132000.0f;
                shell_density *= 1.38f;
                core_stiffness = 22000.0f;
                core_density *= 1.55f;
                core_thermal = ng::vec4(4.10f, 2.55f, 0.62f, 0.00f);
                vessel.gas_mass = 0.34f;
                vessel.gas_source_scale = 1.42f;
                vessel.rupture_scale = 1.34f;
                vessel.burst_scale = 1.55f;
                vessel.shell_push_scale = 1.24f;
                vessel.core_push_scale = 1.20f;
                vessel.leak_scale = 0.78f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::DIRECTIONAL) {
                shell_stiffness = 98000.0f;
                shell_density *= 0.96f;
                core_stiffness = 19500.0f;
                core_density *= 1.18f;
                core_thermal = ng::vec4(3.75f, 2.30f, 0.58f, 0.00f);
                vessel.gas_mass = 0.22f;
                vessel.axis_bias = 0.84f;
                vessel.gas_source_scale = 1.08f;
                vessel.rupture_scale = 0.92f;
                vessel.burst_scale = 1.20f;
                vessel.shell_push_scale = 1.02f;
                vessel.core_push_scale = 1.28f;
                vessel.leak_scale = 1.08f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::CLAYMORE) {
                shell_stiffness = 126000.0f;
                shell_density *= 1.05f;
                core_stiffness = 20500.0f;
                core_density *= 1.24f;
                core_thermal = ng::vec4(4.05f, 2.48f, 0.62f, 0.00f);
                vessel.gas_mass = 0.24f;
                vessel.axis_bias = 0.92f;
                vessel.gas_source_scale = 1.18f;
                vessel.rupture_scale = 0.88f;
                vessel.burst_scale = 1.34f;
                vessel.shell_push_scale = 0.92f;
                vessel.core_push_scale = 1.40f;
                vessel.leak_scale = 1.12f;
                vessel.payload_push_scale = 2.35f;
                vessel.payload_cone = 0.68f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::TIMED) {
                shell_stiffness = 116000.0f;
                shell_density *= 1.28f;
                fuse_stiffness = 46000.0f;
                fuse_density *= 1.38f;
                core_stiffness = 21400.0f;
                core_density *= 1.30f;
                fuse_material = ng::MPMMaterial::STONEWARE;
                shell_initial_temp = 300.0f;
                fuse_initial_temp = 300.0f;
                core_initial_temp = 296.0f;
                shell_thermal = ng::vec4(0.00f, 0.08f, 1.10f, 0.00f);
                fuse_thermal = ng::vec4(0.00f, 0.10f, 1.08f, 0.00f);
                core_thermal = ng::vec4(2.45f, 1.72f, 0.72f, 0.00f);
                vessel.gas_mass = 0.08f;
                vessel.gas_source_scale = 0.94f;
                vessel.rupture_scale = 1.32f;
                vessel.burst_scale = 1.26f;
                vessel.shell_push_scale = 1.08f;
                vessel.core_push_scale = 1.10f;
                vessel.leak_scale = 0.76f;
                vessel.trigger_speed = 0.24f;
                vessel.trigger_heat = 1080.0f;
                vessel.trigger_boost = 1.42f;
                vessel.ignition_delay = 99.0f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::LAYERED) {
                shell_stiffness = 128000.0f;
                shell_density *= 1.20f;
                fuse_stiffness = 28000.0f;
                fuse_density *= 1.18f;
                core_stiffness = 21000.0f;
                core_density *= 1.46f;
                core_thermal = ng::vec4(4.30f, 2.56f, 0.64f, 0.00f);
                vessel.gas_mass = 0.32f;
                vessel.gas_source_scale = 1.38f;
                vessel.rupture_scale = 1.32f;
                vessel.burst_scale = 1.42f;
                vessel.shell_push_scale = 1.18f;
                vessel.core_push_scale = 1.18f;
                vessel.leak_scale = 0.74f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::BROADSIDE) {
                shell_stiffness = 108000.0f;
                shell_density *= 1.06f;
                core_stiffness = 19200.0f;
                core_density *= 1.22f;
                core_thermal = ng::vec4(3.95f, 2.34f, 0.60f, 0.00f);
                vessel.gas_mass = 0.24f;
                vessel.gas_source_scale = 1.15f;
                vessel.rupture_scale = 1.02f;
                vessel.burst_scale = 1.26f;
                vessel.shell_push_scale = 1.06f;
                vessel.core_push_scale = 1.12f;
                vessel.leak_scale = 0.96f;
                vessel.side_blast_scale = 1.45f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::TRIGGER) {
                shell_stiffness = 102000.0f;
                shell_density *= 1.04f;
                fuse_stiffness = 21000.0f;
                fuse_density *= 0.92f;
                core_stiffness = 18600.0f;
                core_density *= 1.20f;
                fuse_initial_temp = 300.0f;
                core_initial_temp = 300.0f;
                shell_thermal = ng::vec4(0.00f, 0.12f, 1.06f, 0.00f);
                fuse_thermal = ng::vec4(0.00f, 0.14f, 1.02f, 0.00f);
                core_thermal = ng::vec4(3.40f, 2.04f, 0.68f, 0.00f);
                vessel.gas_mass = 0.10f;
                vessel.gas_source_scale = 1.12f;
                vessel.rupture_scale = 1.08f;
                vessel.burst_scale = 1.22f;
                vessel.shell_push_scale = 1.05f;
                vessel.core_push_scale = 1.12f;
                vessel.leak_scale = 0.88f;
                vessel.trigger_speed = 0.95f;
                vessel.trigger_heat = 1020.0f;
                vessel.trigger_boost = 1.30f;
                vessel.ignition_delay = 99.0f;
                fuse_material = ng::MPMMaterial::STONEWARE;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::ROCKET) {
                // Rocket tuning: biased strongly toward "propel, don't rupture".
                // The previous values let gas build faster than the nozzle could
                // vent, so pressure reached rupture_threshold within a fraction of
                // a second and the rocket exploded instead of flying. Now:
                //  - gas_source_scale is halved so gas generation matches nozzle
                //    leak at steady state rather than outrunning it
                //  - rupture_scale ~doubled so the pressure required to rupture
                //    is far above the steady-state working pressure
                //  - leak_scale + nozzle_open raised so gas exits faster
                //  - thrust_scale raised so the reduced working pressure still
                //    produces strong forward thrust
                //  - burst_scale reduced so if the rocket ever does rupture late
                //    in flight, the burst is a polite pop, not a fireball
                shell_stiffness = 118000.0f;
                shell_density *= 0.96f;
                fuse_stiffness = 17000.0f;
                fuse_density *= 0.82f;
                core_stiffness = 21400.0f;
                core_density *= 1.12f;
                fuse_initial_temp = 560.0f;
                core_initial_temp = 340.0f;
                core_thermal = ng::vec4(2.20f, 1.70f, 0.58f, 0.00f);
                vessel.gas_mass = 0.08f;
                vessel.axis_bias = 1.0f;
                vessel.gas_source_scale = 0.55f;
                vessel.rupture_scale = 3.40f;
                vessel.burst_scale = 0.22f;
                vessel.shell_push_scale = 0.62f;
                vessel.core_push_scale = 0.82f;
                vessel.leak_scale = 2.20f;
                vessel.nozzle_open = 0.48f;
                vessel.thrust_scale = 4.20f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::HEAT) {
                shell_stiffness = 152000.0f;
                shell_density *= 1.12f;
                fuse_stiffness = 30000.0f;
                fuse_density *= 1.02f;
                core_stiffness = 23800.0f;
                core_density *= 1.32f;
                shell_thermal = ng::vec4(0.00f, 0.10f, 1.14f, 0.00f);
                fuse_thermal = ng::vec4(0.34f, 0.78f, 0.78f, 0.00f);
                core_thermal = ng::vec4(4.00f, 2.32f, 0.60f, 0.00f);
                vessel.gas_mass = 0.18f;
                vessel.axis_bias = 1.0f;
                vessel.gas_source_scale = 1.02f;
                vessel.rupture_scale = 1.26f;
                vessel.burst_scale = 1.08f;
                vessel.shell_push_scale = 0.78f;
                vessel.core_push_scale = 1.26f;
                vessel.leak_scale = 0.60f;
                vessel.payload_push_scale = 4.40f;
                vessel.payload_cone = 0.84f;
                payload_stiffness = 168000.0f;
                payload_density = 7.6f * glm::max(g_ball_weight, 0.5f) / 6.0f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::HESH) {
                shell_stiffness = 138000.0f;
                shell_density *= 1.16f;
                fuse_stiffness = 26000.0f;
                fuse_density *= 1.04f;
                core_stiffness = 20800.0f;
                core_density *= 1.28f;
                shell_thermal = ng::vec4(0.00f, 0.10f, 1.12f, 0.00f);
                fuse_thermal = ng::vec4(0.42f, 0.92f, 0.74f, 0.01f);
                core_thermal = ng::vec4(4.05f, 2.30f, 0.62f, 0.00f);
                vessel.gas_mass = 0.26f;
                vessel.gas_source_scale = 1.18f;
                vessel.rupture_scale = 1.06f;
                vessel.burst_scale = 1.52f;
                vessel.shell_push_scale = 1.04f;
                vessel.core_push_scale = 1.14f;
                vessel.leak_scale = 0.76f;
                vessel.side_blast_scale = 2.10f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::APHE) {
                shell_stiffness = 164000.0f;
                shell_density *= 1.28f;
                fuse_stiffness = 32000.0f;
                fuse_density *= 1.10f;
                core_stiffness = 23400.0f;
                core_density *= 1.42f;
                shell_thermal = ng::vec4(0.00f, 0.08f, 1.16f, 0.00f);
                fuse_thermal = ng::vec4(0.22f, 0.66f, 0.82f, 0.00f);
                core_thermal = ng::vec4(3.86f, 2.20f, 0.62f, 0.00f);
                vessel.gas_mass = 0.16f;
                vessel.axis_bias = 0.96f;
                vessel.gas_source_scale = 0.96f;
                vessel.rupture_scale = 1.54f;
                vessel.burst_scale = 1.04f;
                vessel.shell_push_scale = 0.70f;
                vessel.core_push_scale = 1.20f;
                vessel.leak_scale = 0.58f;
                vessel.payload_push_scale = 2.55f;
                vessel.payload_cone = 0.78f;
                payload_stiffness = 176000.0f;
                payload_density = 8.2f * glm::max(g_ball_weight, 0.5f) / 6.0f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEMO) {
                shell_stiffness = 176000.0f;
                shell_density *= 1.34f;
                fuse_stiffness = 36000.0f;
                fuse_density *= 1.18f;
                core_stiffness = 22600.0f;
                core_density *= 1.52f;
                shell_initial_temp = 300.0f;
                fuse_initial_temp = 300.0f;
                core_initial_temp = 298.0f;
                shell_thermal = ng::vec4(0.00f, 0.08f, 1.18f, 0.00f);
                fuse_thermal = ng::vec4(0.00f, 0.10f, 1.10f, 0.00f);
                core_thermal = ng::vec4(3.72f, 2.06f, 0.68f, 0.00f);
                vessel.gas_mass = 0.18f;
                vessel.gas_source_scale = 1.34f;
                vessel.rupture_scale = 1.78f;
                vessel.burst_scale = 1.78f;
                vessel.shell_push_scale = 1.22f;
                vessel.core_push_scale = 1.24f;
                vessel.leak_scale = 0.54f;
                vessel.ignition_delay = 0.20f;
                vessel.ignition_window = 0.42f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEEP_FUSE) {
                shell_stiffness = 168000.0f;
                shell_density *= 1.32f;
                fuse_stiffness = 72000.0f;
                fuse_density *= 1.28f;
                core_stiffness = 21800.0f;
                core_density *= 1.36f;
                fuse_material = ng::MPMMaterial::STONEWARE;
                shell_initial_temp = 300.0f;
                fuse_initial_temp = 300.0f;
                core_initial_temp = 296.0f;
                shell_thermal = ng::vec4(0.00f, 0.08f, 1.18f, 0.00f);
                fuse_thermal = ng::vec4(0.00f, 0.06f, 1.24f, 0.00f);
                core_thermal = ng::vec4(3.08f, 1.96f, 0.76f, 0.00f);
                vessel.gas_mass = 0.06f;
                vessel.gas_source_scale = 1.06f;
                vessel.rupture_scale = 1.68f;
                vessel.burst_scale = 1.48f;
                vessel.shell_push_scale = 1.06f;
                vessel.core_push_scale = 1.10f;
                vessel.leak_scale = 0.52f;
                vessel.trigger_speed = 0.19f;
                vessel.trigger_heat = 1120.0f;
                vessel.trigger_boost = 1.58f;
                vessel.ignition_delay = 99.0f;
                vessel.ignition_window = 0.50f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE) {
                shell_stiffness = 174000.0f;
                shell_density *= 1.36f;
                fuse_stiffness = 82000.0f;
                fuse_density *= 1.34f;
                core_stiffness = 21400.0f;
                core_density *= 1.40f;
                fuse_material = ng::MPMMaterial::STONEWARE;
                shell_initial_temp = 300.0f;
                fuse_initial_temp = 300.0f;
                core_initial_temp = 296.0f;
                shell_thermal = ng::vec4(0.00f, 0.06f, 1.20f, 0.00f);
                fuse_thermal = ng::vec4(0.00f, 0.04f, 1.28f, 0.00f);
                core_thermal = ng::vec4(2.90f, 1.84f, 0.80f, 0.00f);
                vessel.gas_mass = 0.05f;
                vessel.gas_source_scale = 0.98f;
                vessel.rupture_scale = 1.86f;
                vessel.burst_scale = 1.54f;
                vessel.shell_push_scale = 1.08f;
                vessel.core_push_scale = 1.12f;
                vessel.leak_scale = 0.44f;
                vessel.trigger_speed = 0.095f;
                vessel.trigger_heat = 1140.0f;
                vessel.trigger_boost = 1.70f;
                vessel.ignition_delay = 99.0f;
                vessel.ignition_window = 0.62f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::THERMITE) {
                shell_stiffness = 142000.0f;
                shell_density *= 1.18f;
                fuse_stiffness = 26000.0f;
                fuse_density *= 0.96f;
                core_stiffness = 18800.0f;
                core_density *= 1.24f;
                shell_thermal = ng::vec4(0.02f, 0.14f, 1.10f, 0.00f);
                fuse_thermal = ng::vec4(0.28f, 0.98f, 0.64f, 0.00f);
                core_thermal = ng::vec4(0.34f, 2.80f, 0.34f, 0.00f);
                vessel.gas_mass = 0.04f;
                vessel.gas_source_scale = 0.28f;
                vessel.rupture_scale = 2.10f;
                vessel.burst_scale = 0.14f;
                vessel.shell_push_scale = 0.22f;
                vessel.core_push_scale = 0.34f;
                vessel.leak_scale = 0.26f;
                vessel.plume_push_scale = 0.22f;
                vessel.plume_heat_scale = 2.90f;
                vessel.plume_radius_scale = 0.86f;
                vessel.ignition_delay = 0.55f;
                vessel.ignition_window = 0.75f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::TANDEM) {
                shell_stiffness = 158000.0f;
                shell_density *= 1.20f;
                fuse_stiffness = 52000.0f;
                fuse_density *= 1.08f;
                core_stiffness = 23800.0f;
                core_density *= 1.34f;
                fuse_material = ng::MPMMaterial::STONEWARE;
                shell_thermal = ng::vec4(0.00f, 0.08f, 1.16f, 0.00f);
                fuse_thermal = ng::vec4(0.00f, 0.08f, 1.10f, 0.00f);
                core_thermal = ng::vec4(4.12f, 2.30f, 0.58f, 0.00f);
                vessel.gas_mass = 0.15f;
                vessel.axis_bias = 1.0f;
                vessel.gas_source_scale = 1.06f;
                vessel.rupture_scale = 1.44f;
                vessel.burst_scale = 1.18f;
                vessel.shell_push_scale = 0.68f;
                vessel.core_push_scale = 1.28f;
                vessel.leak_scale = 0.50f;
                vessel.payload_push_scale = 5.20f;
                vessel.payload_cone = 0.90f;
                vessel.trigger_speed = 0.44f;
                vessel.trigger_heat = 1160.0f;
                vessel.trigger_boost = 0.96f;
                vessel.ignition_delay = 99.0f;
                vessel.plume_push_scale = 1.08f;
                vessel.plume_heat_scale = 1.18f;
                payload_stiffness = 182000.0f;
                payload_density = 8.6f * glm::max(g_ball_weight, 0.5f) / 6.0f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::FUEL_AIR) {
                shell_stiffness = 82000.0f;
                shell_density *= 0.92f;
                fuse_stiffness = 18000.0f;
                fuse_density *= 0.84f;
                core_stiffness = 16600.0f;
                core_density *= 1.12f;
                shell_thermal = ng::vec4(0.06f, 0.16f, 0.98f, 0.03f);
                fuse_thermal = ng::vec4(0.42f, 1.02f, 0.62f, 0.08f);
                core_thermal = ng::vec4(4.50f, 2.60f, 0.54f, 0.10f);
                vessel.gas_mass = 0.34f;
                vessel.gas_source_scale = 1.82f;
                vessel.rupture_scale = 0.84f;
                vessel.burst_scale = 2.18f;
                vessel.shell_push_scale = 0.62f;
                vessel.core_push_scale = 0.88f;
                vessel.leak_scale = 1.58f;
                vessel.side_blast_scale = 1.84f;
                vessel.plume_push_scale = 2.40f;
                vessel.plume_heat_scale = 2.10f;
                vessel.plume_radius_scale = 1.62f;
                vessel.ignition_delay = 0.34f;
                vessel.ignition_window = 0.36f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::SMOKE) {
                shell_stiffness = 56000.0f;
                shell_density *= 0.94f;
                fuse_stiffness = 14000.0f;
                fuse_density *= 0.82f;
                core_stiffness = 12000.0f;
                core_density *= 0.98f;
                shell_thermal = ng::vec4(0.04f, 0.16f, 0.94f, 0.10f);
                fuse_thermal = ng::vec4(0.18f, 0.56f, 0.80f, 0.18f);
                core_thermal = ng::vec4(0.60f, 0.88f, 0.86f, 0.24f);
                vessel.gas_mass = 0.26f;
                vessel.gas_source_scale = 0.92f;
                vessel.rupture_scale = 0.72f;
                vessel.burst_scale = 0.18f;
                vessel.shell_push_scale = 0.20f;
                vessel.core_push_scale = 0.28f;
                vessel.leak_scale = 2.30f;
                vessel.side_blast_scale = 1.44f;
                vessel.plume_push_scale = 2.70f;
                vessel.plume_heat_scale = 0.16f;
                vessel.plume_radius_scale = 1.80f;
                vessel.ignition_delay = 0.14f;
                vessel.ignition_window = 0.18f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::FLASH) {
                shell_stiffness = 72000.0f;
                shell_density *= 0.98f;
                fuse_stiffness = 18000.0f;
                fuse_density *= 0.86f;
                core_stiffness = 14600.0f;
                core_density *= 1.02f;
                shell_thermal = ng::vec4(0.04f, 0.20f, 0.96f, 0.06f);
                fuse_thermal = ng::vec4(0.34f, 0.86f, 0.74f, 0.10f);
                core_thermal = ng::vec4(1.00f, 1.22f, 0.76f, 0.10f);
                vessel.gas_mass = 0.20f;
                vessel.gas_source_scale = 1.06f;
                vessel.rupture_scale = 0.88f;
                vessel.burst_scale = 0.74f;
                vessel.shell_push_scale = 0.24f;
                vessel.core_push_scale = 0.36f;
                vessel.leak_scale = 1.72f;
                vessel.side_blast_scale = 2.20f;
                vessel.plume_push_scale = 3.10f;
                vessel.plume_heat_scale = 0.52f;
                vessel.plume_radius_scale = 1.38f;
                vessel.ignition_delay = 0.10f;
                vessel.ignition_window = 0.14f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::CLUSTER) {
                shell_stiffness = 76000.0f;
                shell_density *= 0.98f;
                fuse_stiffness = 18000.0f;
                fuse_density *= 0.86f;
                core_stiffness = 16800.0f;
                core_density *= 1.10f;
                shell_thermal = ng::vec4(0.02f, 0.12f, 1.02f, 0.04f);
                fuse_thermal = ng::vec4(0.54f, 1.08f, 0.70f, 0.06f);
                core_thermal = ng::vec4(3.10f, 1.94f, 0.62f, 0.04f);
                vessel.gas_mass = 0.14f;
                vessel.gas_source_scale = 0.94f;
                vessel.rupture_scale = 0.80f;
                vessel.burst_scale = 1.04f;
                vessel.shell_push_scale = 0.50f;
                vessel.core_push_scale = 0.62f;
                vessel.leak_scale = 1.10f;
                vessel.payload_push_scale = 2.90f;
                vessel.payload_cone = 0.04f;
                vessel.payload_directionality = 0.0f;
                vessel.plume_push_scale = 1.24f;
                vessel.plume_heat_scale = 0.92f;
                vessel.plume_radius_scale = 1.12f;
                payload_material = ng::MPMMaterial::FIRECRACKER;
                payload_stiffness = 12000.0f;
                payload_density = 1.8f * glm::max(g_ball_weight, 0.5f) / 6.0f;
                payload_initial_temp = g_projectile_auto_arm ? 332.0f : 300.0f;
                payload_thermal = ng::vec4(1.28f, 1.44f, 0.76f, 0.06f);
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::CATACLYSM) {
                shell_stiffness = 196000.0f;
                shell_density *= 1.48f;
                fuse_stiffness = 86000.0f;
                fuse_density *= 1.42f;
                core_stiffness = 25200.0f;
                core_density *= 1.86f;
                fuse_material = ng::MPMMaterial::STONEWARE;
                shell_thermal = ng::vec4(0.00f, 0.06f, 1.24f, 0.00f);
                fuse_thermal = ng::vec4(0.00f, 0.04f, 1.30f, 0.00f);
                core_thermal = ng::vec4(4.65f, 2.62f, 0.60f, 0.00f);
                vessel.gas_mass = 0.26f;
                vessel.gas_source_scale = 1.66f;
                vessel.rupture_scale = 2.16f;
                vessel.burst_scale = 2.70f;
                vessel.shell_push_scale = 1.42f;
                vessel.core_push_scale = 1.46f;
                vessel.leak_scale = 0.38f;
                vessel.trigger_speed = 0.15f;
                vessel.trigger_heat = 1180.0f;
                vessel.trigger_boost = 1.94f;
                vessel.ignition_delay = 99.0f;
                vessel.plume_push_scale = 1.82f;
                vessel.plume_heat_scale = 1.94f;
                vessel.plume_radius_scale = 1.38f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::CASCADE) {
                shell_stiffness = 90000.0f;
                shell_density *= 1.04f;
                fuse_stiffness = 22000.0f;
                fuse_density *= 0.94f;
                core_stiffness = 17200.0f;
                core_density *= 1.14f;
                shell_thermal = ng::vec4(0.02f, 0.10f, 1.04f, 0.02f);
                fuse_thermal = ng::vec4(0.22f, 0.66f, 0.90f, 0.02f);
                core_thermal = ng::vec4(3.20f, 1.88f, 0.66f, 0.02f);
                vessel.gas_mass = 0.16f;
                vessel.gas_source_scale = 0.88f;
                vessel.rupture_scale = 0.88f;
                vessel.burst_scale = 1.10f;
                vessel.shell_push_scale = 0.54f;
                vessel.core_push_scale = 0.68f;
                vessel.leak_scale = 0.94f;
                vessel.payload_push_scale = 3.10f;
                vessel.payload_cone = 0.08f;
                vessel.payload_directionality = 0.0f;
                vessel.plume_push_scale = 1.20f;
                vessel.plume_heat_scale = 0.84f;
                vessel.plume_radius_scale = 1.14f;
                vessel.ignition_delay = 0.24f;
                vessel.ignition_window = 0.26f;
                payload_material = ng::MPMMaterial::FIRECRACKER;
                payload_stiffness = 14000.0f;
                payload_density = 1.9f * glm::max(g_ball_weight, 0.5f) / 6.0f;
                payload_initial_temp = g_projectile_auto_arm ? 248.0f : 232.0f;
                payload_thermal = ng::vec4(0.24f, 0.58f, 0.96f, 0.01f);
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::CRYO) {
                shell_material = ng::MPMMaterial::SNOW;
                shell_stiffness = 98000.0f;
                shell_density *= 1.10f;
                fuse_stiffness = 46000.0f;
                fuse_density *= 1.08f;
                core_stiffness = 19800.0f;
                core_density *= 1.24f;
                fuse_material = ng::MPMMaterial::STONEWARE;
                shell_initial_temp = 242.0f;
                fuse_initial_temp = 246.0f;
                core_initial_temp = 248.0f;
                shell_thermal = ng::vec4(0.00f, 0.06f, 1.28f, 0.00f);
                fuse_thermal = ng::vec4(0.00f, 0.06f, 1.20f, 0.00f);
                core_thermal = ng::vec4(2.24f, 1.72f, 0.86f, 0.00f);
                vessel.gas_mass = 0.08f;
                vessel.gas_source_scale = 0.82f;
                vessel.rupture_scale = 1.52f;
                vessel.burst_scale = 1.16f;
                vessel.shell_push_scale = 0.82f;
                vessel.core_push_scale = 0.96f;
                vessel.leak_scale = 0.62f;
                vessel.trigger_speed = 0.22f;
                vessel.trigger_heat = 1080.0f;
                vessel.trigger_boost = 1.20f;
                vessel.ignition_delay = 99.0f;
                vessel.plume_push_scale = 0.90f;
                vessel.plume_heat_scale = 0.98f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::SPIRAL) {
                shell_stiffness = 104000.0f;
                shell_density *= 1.08f;
                fuse_stiffness = 22000.0f;
                fuse_density *= 0.96f;
                core_stiffness = 19000.0f;
                core_density *= 1.20f;
                shell_thermal = ng::vec4(0.02f, 0.16f, 1.04f, 0.02f);
                fuse_thermal = ng::vec4(0.34f, 0.88f, 0.78f, 0.04f);
                core_thermal = ng::vec4(3.82f, 2.14f, 0.60f, 0.08f);
                vessel.gas_mass = 0.20f;
                vessel.gas_source_scale = 1.10f;
                vessel.rupture_scale = 0.96f;
                vessel.burst_scale = 1.18f;
                vessel.shell_push_scale = 0.58f;
                vessel.core_push_scale = 0.78f;
                vessel.leak_scale = 1.22f;
                vessel.swirl_blast_scale = 2.40f;
                vessel.plume_push_scale = 1.62f;
                vessel.plume_heat_scale = 1.84f;
                vessel.plume_radius_scale = 1.18f;
                vessel.ignition_delay = 0.18f;
                vessel.ignition_window = 0.18f;
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::ROCKET_PAYLOAD) {
                // Building-block recipe:
                //   B1 Armor Shell (stoneware)  +  B6 Propellant  +  B7 Containment
                //   +  B3 crack-rate sensor  +  B4 velocity-drop sensor
                //   +  B5 short delay fuse  +  B9 forward shrapnel
                // Containment (rupture_scale 9.9) means pressure alone cannot
                // rupture this bomb. The sensors are the only path to rupture.
                shell_stiffness = 124000.0f;
                shell_density *= 0.98f;
                fuse_stiffness = 17000.0f;
                fuse_density *= 0.82f;
                core_stiffness = 21400.0f;
                core_density *= 1.10f;
                fuse_initial_temp = 490.0f;
                core_initial_temp = 285.0f;
                core_thermal = ng::vec4(0.70f, 1.30f, 0.68f, 0.00f);
                vessel.gas_mass = 0.06f;
                vessel.axis_bias = 1.0f;
                vessel.gas_source_scale = 0.35f;
                vessel.rupture_scale = 9.90f;   // B7: sealed against pressure-rupture
                vessel.burst_scale = 0.48f;
                vessel.shell_push_scale = 0.62f;
                vessel.core_push_scale = 0.82f;
                vessel.leak_scale = 2.40f;
                vessel.nozzle_open = 0.50f;    // B6: propellant nozzle
                vessel.thrust_scale = 5.20f;
                vessel.payload_push_scale = 5.50f;
                vessel.payload_cone = 0.65f;
                vessel.payload_directionality = 1.0f;
                vessel.impact_rupture = true;                 // enable sensors
                vessel.impact_crack_rate_threshold = 1.5f;    // B3: ~1.5 crack/s
                vessel.trigger_to_rupture_delay = 0.04f;      // B5: 40 ms fuse
                payload_material = ng::MPMMaterial::THERMO_METAL;
                payload_stiffness = 140000.0f;
                payload_density = 6.2f * glm::max(g_ball_weight, 0.5f) / 6.0f;
                payload_initial_temp = 300.0f;
                payload_thermal = ng::vec4(0.14f, 0.22f, 1.00f, 0.00f);
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::ROCKET_SIDE_CLAYMORE) {
                // Rocket that vents a lateral fragment burst mid-flight via side_blast
                // plus a mild forward payload on final rupture. Takes longer to rupture
                // than ROCKET_PAYLOAD since most of its energy escapes sideways.
                shell_stiffness = 112000.0f;
                shell_density *= 0.98f;
                fuse_stiffness = 18000.0f;
                fuse_density *= 0.84f;
                core_stiffness = 21800.0f;
                core_density *= 1.10f;
                fuse_initial_temp = 580.0f;
                core_initial_temp = 340.0f;
                core_thermal = ng::vec4(2.45f, 1.78f, 0.58f, 0.00f);
                vessel.gas_mass = 0.10f;
                vessel.axis_bias = 0.75f;
                vessel.gas_source_scale = 0.72f;
                vessel.rupture_scale = 2.80f;
                vessel.burst_scale = 0.44f;
                vessel.shell_push_scale = 0.68f;
                vessel.core_push_scale = 0.80f;
                vessel.leak_scale = 1.80f;
                vessel.nozzle_open = 0.40f;
                vessel.thrust_scale = 3.60f;
                vessel.side_blast_scale = 1.70f;
                vessel.payload_push_scale = 2.60f;
                vessel.payload_cone = 0.28f;
                vessel.payload_directionality = 0.40f;
                payload_material = ng::MPMMaterial::THERMO_METAL;
                payload_stiffness = 132000.0f;
                payload_density = 5.4f * glm::max(g_ball_weight, 0.5f) / 6.0f;
                payload_initial_temp = 300.0f;
                payload_thermal = ng::vec4(0.14f, 0.22f, 1.00f, 0.00f);
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::CLAYMORE_CLUSTER) {
                // Claymore body with CLUSTER-style FIRECRACKER bomblets. The base
                // stays forward-biased (claymore shell config), but the payload is
                // small reactive charges that each cook off after being ejected.
                shell_stiffness = 126000.0f;
                shell_density *= 1.02f;
                fuse_stiffness = 24000.0f;
                fuse_density *= 0.90f;
                core_stiffness = 19400.0f;
                core_density *= 1.18f;
                core_thermal = ng::vec4(3.20f, 1.94f, 0.60f, 0.03f);
                vessel.gas_mass = 0.14f;
                vessel.axis_bias = 0.95f;
                vessel.gas_source_scale = 1.00f;
                vessel.rupture_scale = 1.10f;
                vessel.burst_scale = 0.96f;
                vessel.shell_push_scale = 0.72f;
                vessel.core_push_scale = 0.92f;
                vessel.leak_scale = 0.70f;
                vessel.payload_push_scale = 3.40f;
                vessel.payload_cone = 0.62f;
                vessel.payload_directionality = 0.85f;
                vessel.plume_push_scale = 1.08f;
                vessel.plume_heat_scale = 1.12f;
                payload_material = ng::MPMMaterial::FIRECRACKER;
                payload_stiffness = 14000.0f;
                payload_density = 1.9f * glm::max(g_ball_weight, 0.5f) / 6.0f;
                payload_initial_temp = g_projectile_auto_arm ? 332.0f : 300.0f;
                payload_thermal = ng::vec4(1.28f, 1.44f, 0.76f, 0.06f);
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::LATERAL_CONTACT) {
                // Recipe: B1 shell + B7 containment + B3/B4 sensors + B5 short
                // fuse + B9 radial shrapnel + strong side_blast. Body is inert in
                // flight; on contact, fragments + shell push fling sideways.
                shell_stiffness = 116000.0f;
                shell_density *= 0.98f;
                fuse_stiffness = 22000.0f;
                fuse_density *= 0.88f;
                core_stiffness = 19800.0f;
                core_density *= 1.10f;
                fuse_initial_temp = 470.0f;
                core_initial_temp = 290.0f;
                core_thermal = ng::vec4(0.60f, 1.20f, 0.72f, 0.02f);
                vessel.gas_mass = 0.06f;
                vessel.axis_bias = 1.0f;
                vessel.gas_source_scale = 0.30f;
                vessel.rupture_scale = 9.90f;          // B7: sealed
                vessel.burst_scale = 0.70f;
                vessel.shell_push_scale = 0.40f;
                vessel.core_push_scale = 0.60f;
                vessel.leak_scale = 1.80f;
                vessel.side_blast_scale = 3.40f;       // dominant side push
                vessel.payload_push_scale = 3.60f;
                vessel.payload_cone = 0.02f;           // near-zero → radial
                vessel.payload_directionality = 0.0f;
                vessel.plume_push_scale = 1.20f;
                vessel.plume_heat_scale = 0.70f;
                vessel.impact_rupture = true;
                vessel.impact_crack_rate_threshold = 1.3f;  // B3: slightly looser
                vessel.trigger_to_rupture_delay = 0.03f;    // B5: snappy
                payload_material = ng::MPMMaterial::THERMO_METAL;
                payload_stiffness = 128000.0f;
                payload_density = 5.2f * glm::max(g_ball_weight, 0.5f) / 6.0f;
                payload_initial_temp = 300.0f;
                payload_thermal = ng::vec4(0.14f, 0.22f, 1.00f, 0.00f);
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::PENETRATING_DOWN) {
                // Recipe: dense B1 armor body + B7 containment + B3/B4 sensors +
                // B5 *long* fuse (0.12 s — the penetrator buries itself during
                // this delay) + gravity-axis-snap + B9 downward shrapnel.
                shell_stiffness = 148000.0f;
                shell_density *= 1.14f;
                fuse_stiffness = 24000.0f;
                fuse_density *= 0.96f;
                core_stiffness = 22000.0f;
                core_density *= 1.24f;
                fuse_initial_temp = 500.0f;
                core_initial_temp = 290.0f;
                core_thermal = ng::vec4(1.10f, 1.40f, 0.68f, 0.00f);
                vessel.gas_mass = 0.06f;
                vessel.axis_bias = 1.0f;
                vessel.gas_source_scale = 0.34f;
                vessel.rupture_scale = 9.90f;           // B7: sealed
                vessel.burst_scale = 0.80f;
                vessel.shell_push_scale = 0.72f;
                vessel.core_push_scale = 1.05f;
                vessel.leak_scale = 1.70f;
                vessel.payload_push_scale = 5.80f;
                vessel.payload_cone = 0.76f;
                vessel.payload_directionality = 1.0f;
                vessel.plume_push_scale = 1.40f;
                vessel.plume_heat_scale = 0.90f;
                vessel.impact_rupture = true;
                vessel.impact_crack_rate_threshold = 1.8f;  // B3: firmer impact needed
                vessel.trigger_to_rupture_delay = 0.12f;    // B5: long dig-in delay
                vessel.penetrate_on_impact = true;
                payload_material = ng::MPMMaterial::THERMO_METAL;
                payload_stiffness = 162000.0f;
                payload_density = 7.4f * glm::max(g_ball_weight, 0.5f) / 6.0f;
                payload_initial_temp = 300.0f;
                payload_thermal = ng::vec4(0.12f, 0.20f, 1.02f, 0.00f);
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::CONCUSSION) {
                // Recipe: B1 shell + B7 containment + B3/B4 sensors + B5 very
                // short fuse + B8 heat-light blast character. No payload.
                shell_stiffness = 104000.0f;
                shell_density *= 1.02f;
                fuse_stiffness = 20000.0f;
                fuse_density *= 0.88f;
                core_stiffness = 18800.0f;
                core_density *= 1.06f;
                fuse_initial_temp = 440.0f;
                core_initial_temp = 290.0f;
                core_thermal = ng::vec4(0.14f, 0.36f, 1.02f, 0.02f);
                shell_thermal = ng::vec4(0.00f, 0.06f, 1.12f, 0.00f);
                fuse_thermal = ng::vec4(0.04f, 0.22f, 0.98f, 0.00f);
                vessel.gas_mass = 0.12f;
                vessel.axis_bias = 0.4f;
                vessel.gas_source_scale = 0.70f;
                vessel.rupture_scale = 9.90f;          // B7: sealed
                vessel.burst_scale = 1.60f;
                vessel.shell_push_scale = 1.10f;
                vessel.core_push_scale = 0.60f;
                vessel.leak_scale = 1.40f;
                vessel.plume_push_scale = 2.60f;       // B8: push-heavy
                vessel.plume_heat_scale = 0.18f;       //     heat-light
                vessel.plume_radius_scale = 1.45f;
                vessel.blast_push_scale = 2.80f;
                vessel.blast_heat_scale = 0.14f;
                vessel.impact_rupture = true;
                vessel.impact_crack_rate_threshold = 1.3f;  // B3
                vessel.trigger_to_rupture_delay = 0.02f;    // B5: instant pop
            }

            switch (preset_id) {
            case ProjectilePreset::SOFT_CLAYMORE:
                shell_stiffness *= 1.06f;
                shell_density *= 0.96f;
                core_stiffness *= 0.84f;
                core_density *= 0.86f;
                core_thermal = ng::vec4(1.14f, 1.24f, 0.88f, 0.00f);
                payload_stiffness *= 0.86f;
                payload_density *= 0.72f;
                vessel.gas_mass = 0.07f;
                vessel.gas_source_scale = 0.46f;
                vessel.rupture_scale = 1.22f;
                vessel.burst_scale = 0.34f;
                vessel.shell_push_scale = 0.48f;
                vessel.core_push_scale = 0.54f;
                vessel.leak_scale = 0.94f;
                vessel.payload_push_scale *= 0.38f;
                vessel.plume_push_scale = 0.44f;
                vessel.plume_heat_scale = 0.22f;
                vessel.plume_radius_scale = 0.96f;
                vessel.blast_push_scale = 0.28f;
                vessel.blast_heat_scale = 0.14f;
                vessel.ignition_delay = 0.46f;
                vessel.ignition_window = 0.26f;
                break;
            case ProjectilePreset::SOFT_BROADSIDE:
                shell_stiffness *= 1.04f;
                shell_density *= 0.98f;
                core_stiffness *= 0.88f;
                core_density *= 0.88f;
                core_thermal = ng::vec4(1.08f, 1.18f, 0.90f, 0.00f);
                vessel.gas_mass = 0.08f;
                vessel.gas_source_scale = 0.42f;
                vessel.rupture_scale = 1.18f;
                vessel.burst_scale = 0.30f;
                vessel.shell_push_scale = 0.42f;
                vessel.core_push_scale = 0.50f;
                vessel.leak_scale = 0.96f;
                vessel.side_blast_scale *= 0.52f;
                vessel.plume_push_scale = 0.46f;
                vessel.plume_heat_scale = 0.18f;
                vessel.plume_radius_scale = 1.02f;
                vessel.blast_push_scale = 0.22f;
                vessel.blast_heat_scale = 0.12f;
                vessel.ignition_delay = 0.44f;
                vessel.ignition_window = 0.24f;
                break;
            case ProjectilePreset::COOL_SMOKE_POT:
                shell_stiffness *= 0.92f;
                shell_density *= 0.94f;
                core_stiffness *= 0.78f;
                core_density *= 0.82f;
                shell_thermal = ng::vec4(0.01f, 0.08f, 1.02f, 0.16f);
                fuse_thermal = ng::vec4(0.10f, 0.28f, 0.94f, 0.30f);
                core_thermal = ng::vec4(0.22f, 0.32f, 0.96f, 0.34f);
                vessel.gas_mass = 0.06f;
                vessel.gas_source_scale = 0.36f;
                vessel.rupture_scale = 0.92f;
                vessel.burst_scale = 0.04f;
                vessel.shell_push_scale = 0.08f;
                vessel.core_push_scale = 0.14f;
                vessel.leak_scale = 2.80f;
                vessel.side_blast_scale = 0.18f;
                vessel.plume_push_scale = 0.24f;
                vessel.plume_heat_scale = 0.02f;
                vessel.plume_radius_scale = 1.34f;
                vessel.blast_push_scale = 0.04f;
                vessel.blast_heat_scale = 0.02f;
                vessel.ignition_delay = 0.34f;
                vessel.ignition_window = 0.16f;
                break;
            case ProjectilePreset::SOFT_SPIRAL_BOMB:
                shell_stiffness *= 1.02f;
                shell_density *= 0.98f;
                core_stiffness *= 0.86f;
                core_density *= 0.88f;
                core_thermal = ng::vec4(1.10f, 1.20f, 0.88f, 0.06f);
                vessel.gas_mass = 0.08f;
                vessel.gas_source_scale = 0.44f;
                vessel.rupture_scale = 1.14f;
                vessel.burst_scale = 0.26f;
                vessel.shell_push_scale = 0.34f;
                vessel.core_push_scale = 0.44f;
                vessel.leak_scale = 1.08f;
                vessel.swirl_blast_scale *= 0.40f;
                vessel.plume_push_scale = 0.38f;
                vessel.plume_heat_scale = 0.20f;
                vessel.plume_radius_scale = 1.08f;
                vessel.blast_push_scale = 0.18f;
                vessel.blast_heat_scale = 0.10f;
                vessel.ignition_delay = 0.42f;
                vessel.ignition_window = 0.20f;
                break;
            case ProjectilePreset::MEDIUM_CLAYMORE:
                shell_stiffness *= 1.04f;
                shell_density *= 0.98f;
                core_stiffness *= 0.90f;
                core_density *= 0.90f;
                core_thermal = ng::vec4(1.36f, 1.50f, 0.82f, 0.00f);
                payload_stiffness *= 0.92f;
                payload_density *= 0.82f;
                vessel.gas_mass = 0.10f;
                vessel.gas_source_scale = 0.64f;
                vessel.rupture_scale = 1.16f;
                vessel.burst_scale = 0.56f;
                vessel.shell_push_scale = 0.64f;
                vessel.core_push_scale = 0.72f;
                vessel.leak_scale = 0.90f;
                vessel.payload_push_scale *= 0.60f;
                vessel.plume_push_scale = 0.68f;
                vessel.plume_heat_scale = 0.38f;
                vessel.plume_radius_scale = 0.98f;
                vessel.blast_push_scale = 0.48f;
                vessel.blast_heat_scale = 0.28f;
                vessel.ignition_delay = 0.34f;
                vessel.ignition_window = 0.22f;
                break;
            case ProjectilePreset::MEDIUM_BROADSIDE:
                shell_stiffness *= 1.03f;
                shell_density *= 0.99f;
                core_stiffness *= 0.92f;
                core_density *= 0.92f;
                core_thermal = ng::vec4(1.28f, 1.40f, 0.84f, 0.00f);
                vessel.gas_mass = 0.11f;
                vessel.gas_source_scale = 0.60f;
                vessel.rupture_scale = 1.14f;
                vessel.burst_scale = 0.52f;
                vessel.shell_push_scale = 0.58f;
                vessel.core_push_scale = 0.66f;
                vessel.leak_scale = 0.92f;
                vessel.side_blast_scale *= 0.72f;
                vessel.plume_push_scale = 0.72f;
                vessel.plume_heat_scale = 0.30f;
                vessel.plume_radius_scale = 1.04f;
                vessel.blast_push_scale = 0.42f;
                vessel.blast_heat_scale = 0.24f;
                vessel.ignition_delay = 0.32f;
                vessel.ignition_window = 0.20f;
                break;
            case ProjectilePreset::MEDIUM_SMOKE_POT:
                shell_stiffness *= 0.96f;
                shell_density *= 0.96f;
                core_stiffness *= 0.84f;
                core_density *= 0.86f;
                shell_thermal = ng::vec4(0.02f, 0.10f, 1.00f, 0.18f);
                fuse_thermal = ng::vec4(0.16f, 0.34f, 0.92f, 0.32f);
                core_thermal = ng::vec4(0.34f, 0.42f, 0.94f, 0.38f);
                vessel.gas_mass = 0.08f;
                vessel.gas_source_scale = 0.46f;
                vessel.rupture_scale = 0.96f;
                vessel.burst_scale = 0.08f;
                vessel.shell_push_scale = 0.12f;
                vessel.core_push_scale = 0.20f;
                vessel.leak_scale = 2.40f;
                vessel.side_blast_scale = 0.26f;
                vessel.plume_push_scale = 0.36f;
                vessel.plume_heat_scale = 0.04f;
                vessel.plume_radius_scale = 1.30f;
                vessel.blast_push_scale = 0.08f;
                vessel.blast_heat_scale = 0.03f;
                vessel.ignition_delay = 0.28f;
                vessel.ignition_window = 0.14f;
                break;
            case ProjectilePreset::MEDIUM_SPIRAL_BOMB:
                shell_stiffness *= 1.01f;
                shell_density *= 0.99f;
                core_stiffness *= 0.90f;
                core_density *= 0.92f;
                core_thermal = ng::vec4(1.30f, 1.42f, 0.82f, 0.08f);
                vessel.gas_mass = 0.11f;
                vessel.gas_source_scale = 0.62f;
                vessel.rupture_scale = 1.10f;
                vessel.burst_scale = 0.48f;
                vessel.shell_push_scale = 0.46f;
                vessel.core_push_scale = 0.58f;
                vessel.leak_scale = 1.12f;
                vessel.swirl_blast_scale *= 0.62f;
                vessel.plume_push_scale = 0.62f;
                vessel.plume_heat_scale = 0.36f;
                vessel.plume_radius_scale = 1.10f;
                vessel.blast_push_scale = 0.30f;
                vessel.blast_heat_scale = 0.16f;
                vessel.ignition_delay = 0.34f;
                vessel.ignition_window = 0.18f;
                break;
            case ProjectilePreset::SOFT_HEAT_CHARGE:
                shell_stiffness *= 1.06f;
                shell_density *= 0.98f;
                core_stiffness *= 0.88f;
                core_density *= 0.90f;
                core_thermal = ng::vec4(1.18f, 1.30f, 0.84f, 0.00f);
                payload_stiffness *= 0.88f;
                payload_density *= 0.78f;
                vessel.gas_mass = 0.08f;
                vessel.gas_source_scale = 0.44f;
                vessel.rupture_scale = 1.24f;
                vessel.burst_scale = 0.32f;
                vessel.shell_push_scale = 0.26f;
                vessel.core_push_scale = 0.56f;
                vessel.leak_scale = 0.76f;
                vessel.payload_push_scale *= 0.34f;
                vessel.plume_push_scale = 0.42f;
                vessel.plume_heat_scale = 0.22f;
                vessel.plume_radius_scale = 0.94f;
                vessel.blast_push_scale = 0.14f;
                vessel.blast_heat_scale = 0.08f;
                vessel.ignition_delay = 0.40f;
                vessel.ignition_window = 0.18f;
                break;
            case ProjectilePreset::MEDIUM_HEAT_CHARGE:
                shell_stiffness *= 1.04f;
                shell_density *= 0.99f;
                core_stiffness *= 0.92f;
                core_density *= 0.94f;
                core_thermal = ng::vec4(1.42f, 1.58f, 0.78f, 0.00f);
                payload_stiffness *= 0.94f;
                payload_density *= 0.88f;
                vessel.gas_mass = 0.11f;
                vessel.gas_source_scale = 0.60f;
                vessel.rupture_scale = 1.20f;
                vessel.burst_scale = 0.52f;
                vessel.shell_push_scale = 0.34f;
                vessel.core_push_scale = 0.72f;
                vessel.leak_scale = 0.72f;
                vessel.payload_push_scale *= 0.56f;
                vessel.plume_push_scale = 0.58f;
                vessel.plume_heat_scale = 0.34f;
                vessel.plume_radius_scale = 0.96f;
                vessel.blast_push_scale = 0.22f;
                vessel.blast_heat_scale = 0.14f;
                vessel.ignition_delay = 0.30f;
                vessel.ignition_window = 0.16f;
                break;
            case ProjectilePreset::ABOVE_MED_CLAYMORE:
                shell_stiffness *= 1.02f;
                shell_density *= 1.00f;
                core_stiffness *= 0.96f;
                core_density *= 0.96f;
                core_thermal = ng::vec4(1.58f, 1.72f, 0.74f, 0.00f);
                payload_stiffness *= 0.96f;
                payload_density *= 0.90f;
                vessel.gas_mass = 0.13f;
                vessel.gas_source_scale = 0.80f;
                vessel.rupture_scale = 1.10f;
                vessel.burst_scale = 0.78f;
                vessel.shell_push_scale = 0.80f;
                vessel.core_push_scale = 0.88f;
                vessel.leak_scale = 0.86f;
                vessel.payload_push_scale *= 0.78f;
                vessel.plume_push_scale = 0.88f;
                vessel.plume_heat_scale = 0.52f;
                vessel.plume_radius_scale = 0.98f;
                vessel.blast_push_scale = 0.62f;
                vessel.blast_heat_scale = 0.36f;
                vessel.ignition_delay = 0.26f;
                vessel.ignition_window = 0.18f;
                break;
            case ProjectilePreset::ABOVE_MED_BROADSIDE:
                shell_stiffness *= 1.02f;
                shell_density *= 1.00f;
                core_stiffness *= 0.96f;
                core_density *= 0.96f;
                core_thermal = ng::vec4(1.52f, 1.66f, 0.76f, 0.00f);
                vessel.gas_mass = 0.13f;
                vessel.gas_source_scale = 0.78f;
                vessel.rupture_scale = 1.08f;
                vessel.burst_scale = 0.74f;
                vessel.shell_push_scale = 0.74f;
                vessel.core_push_scale = 0.82f;
                vessel.leak_scale = 0.88f;
                vessel.side_blast_scale *= 0.88f;
                vessel.plume_push_scale = 0.84f;
                vessel.plume_heat_scale = 0.46f;
                vessel.plume_radius_scale = 1.06f;
                vessel.blast_push_scale = 0.56f;
                vessel.blast_heat_scale = 0.32f;
                vessel.ignition_delay = 0.24f;
                vessel.ignition_window = 0.18f;
                break;
            case ProjectilePreset::ABOVE_MED_SMOKE_POT:
                shell_stiffness *= 0.98f;
                shell_density *= 0.98f;
                core_stiffness *= 0.88f;
                core_density *= 0.90f;
                shell_thermal = ng::vec4(0.02f, 0.10f, 1.00f, 0.18f);
                fuse_thermal = ng::vec4(0.20f, 0.40f, 0.90f, 0.30f);
                core_thermal = ng::vec4(0.46f, 0.58f, 0.92f, 0.34f);
                vessel.gas_mass = 0.10f;
                vessel.gas_source_scale = 0.60f;
                vessel.rupture_scale = 0.98f;
                vessel.burst_scale = 0.12f;
                vessel.shell_push_scale = 0.16f;
                vessel.core_push_scale = 0.24f;
                vessel.leak_scale = 2.05f;
                vessel.side_blast_scale = 0.38f;
                vessel.plume_push_scale = 0.54f;
                vessel.plume_heat_scale = 0.06f;
                vessel.plume_radius_scale = 1.24f;
                vessel.blast_push_scale = 0.12f;
                vessel.blast_heat_scale = 0.04f;
                vessel.ignition_delay = 0.24f;
                vessel.ignition_window = 0.12f;
                break;
            case ProjectilePreset::ABOVE_MED_SPIRAL_BOMB:
                shell_stiffness *= 1.00f;
                shell_density *= 1.00f;
                core_stiffness *= 0.94f;
                core_density *= 0.96f;
                core_thermal = ng::vec4(1.56f, 1.70f, 0.76f, 0.08f);
                vessel.gas_mass = 0.14f;
                vessel.gas_source_scale = 0.76f;
                vessel.rupture_scale = 1.04f;
                vessel.burst_scale = 0.70f;
                vessel.shell_push_scale = 0.56f;
                vessel.core_push_scale = 0.66f;
                vessel.leak_scale = 1.16f;
                vessel.swirl_blast_scale *= 0.82f;
                vessel.plume_push_scale = 0.90f;
                vessel.plume_heat_scale = 0.56f;
                vessel.plume_radius_scale = 1.12f;
                vessel.blast_push_scale = 0.44f;
                vessel.blast_heat_scale = 0.24f;
                vessel.ignition_delay = 0.26f;
                vessel.ignition_window = 0.16f;
                break;
            case ProjectilePreset::ABOVE_MED_HEAT_CHARGE:
                shell_stiffness *= 1.02f;
                shell_density *= 1.00f;
                core_stiffness *= 0.96f;
                core_density *= 0.98f;
                core_thermal = ng::vec4(1.64f, 1.82f, 0.72f, 0.00f);
                payload_stiffness *= 0.98f;
                payload_density *= 0.94f;
                vessel.gas_mass = 0.14f;
                vessel.gas_source_scale = 0.76f;
                vessel.rupture_scale = 1.12f;
                vessel.burst_scale = 0.74f;
                vessel.shell_push_scale = 0.46f;
                vessel.core_push_scale = 0.92f;
                vessel.leak_scale = 0.66f;
                vessel.payload_push_scale *= 0.76f;
                vessel.plume_push_scale = 0.78f;
                vessel.plume_heat_scale = 0.48f;
                vessel.plume_radius_scale = 0.98f;
                vessel.blast_push_scale = 0.30f;
                vessel.blast_heat_scale = 0.18f;
                vessel.ignition_delay = 0.24f;
                vessel.ignition_window = 0.14f;
                break;
            case ProjectilePreset::SOFT_LONG_FUSE_BOMB:
                shell_stiffness *= 1.08f;
                shell_density *= 0.98f;
                core_stiffness *= 0.86f;
                core_density *= 0.88f;
                core_thermal = ng::vec4(1.02f, 1.18f, 0.88f, 0.00f);
                vessel.gas_mass = 0.05f;
                vessel.gas_source_scale = 0.38f;
                vessel.rupture_scale = 1.34f;
                vessel.burst_scale = 0.26f;
                vessel.shell_push_scale = 0.34f;
                vessel.core_push_scale = 0.40f;
                vessel.leak_scale = 0.70f;
                vessel.trigger_speed = 0.20f;
                vessel.trigger_heat = 1040.0f;
                vessel.trigger_boost = 1.12f;
                vessel.plume_push_scale = 0.28f;
                vessel.plume_heat_scale = 0.12f;
                vessel.plume_radius_scale = 0.90f;
                vessel.blast_push_scale = 0.14f;
                vessel.blast_heat_scale = 0.08f;
                vessel.ignition_window = 0.58f;
                break;
            case ProjectilePreset::MEDIUM_LONG_FUSE_BOMB:
                shell_stiffness *= 1.06f;
                shell_density *= 1.00f;
                core_stiffness *= 0.92f;
                core_density *= 0.94f;
                core_thermal = ng::vec4(1.28f, 1.44f, 0.80f, 0.00f);
                vessel.gas_mass = 0.07f;
                vessel.gas_source_scale = 0.60f;
                vessel.rupture_scale = 1.38f;
                vessel.burst_scale = 0.56f;
                vessel.shell_push_scale = 0.66f;
                vessel.core_push_scale = 0.72f;
                vessel.leak_scale = 0.64f;
                vessel.trigger_speed = 0.22f;
                vessel.trigger_heat = 1080.0f;
                vessel.trigger_boost = 1.30f;
                vessel.plume_push_scale = 0.52f;
                vessel.plume_heat_scale = 0.24f;
                vessel.plume_radius_scale = 0.94f;
                vessel.blast_push_scale = 0.30f;
                vessel.blast_heat_scale = 0.14f;
                vessel.ignition_window = 0.60f;
                break;
            case ProjectilePreset::ABOVE_MED_LONG_FUSE_BOMB:
                shell_stiffness *= 1.04f;
                shell_density *= 1.02f;
                core_stiffness *= 0.96f;
                core_density *= 0.98f;
                core_thermal = ng::vec4(1.52f, 1.68f, 0.76f, 0.00f);
                vessel.gas_mass = 0.09f;
                vessel.gas_source_scale = 0.78f;
                vessel.rupture_scale = 1.44f;
                vessel.burst_scale = 0.92f;
                vessel.shell_push_scale = 0.86f;
                vessel.core_push_scale = 0.92f;
                vessel.leak_scale = 0.60f;
                vessel.trigger_speed = 0.23f;
                vessel.trigger_heat = 1100.0f;
                vessel.trigger_boost = 1.42f;
                vessel.plume_push_scale = 0.72f;
                vessel.plume_heat_scale = 0.36f;
                vessel.plume_radius_scale = 0.98f;
                vessel.blast_push_scale = 0.48f;
                vessel.blast_heat_scale = 0.22f;
                vessel.ignition_window = 0.62f;
                break;
            case ProjectilePreset::SOFT_ROCKET:
                shell_stiffness *= 0.96f;
                shell_density *= 0.92f;
                core_stiffness *= 0.86f;
                core_density *= 0.88f;
                fuse_initial_temp = g_projectile_auto_arm ? 540.0f : 300.0f;
                core_initial_temp = g_projectile_auto_arm ? 338.0f : 300.0f;
                core_thermal = ng::vec4(1.06f, 1.22f, 0.84f, 0.00f);
                vessel.gas_mass = 0.08f;
                vessel.gas_source_scale = 0.62f;
                vessel.rupture_scale = 2.10f;
                vessel.burst_scale = 0.18f;
                vessel.shell_push_scale = 0.30f;
                vessel.core_push_scale = 0.42f;
                vessel.leak_scale = 1.60f;
                vessel.nozzle_open = 0.28f;
                vessel.thrust_scale = 1.60f;
                vessel.plume_push_scale = 0.34f;
                vessel.plume_heat_scale = 0.18f;
                vessel.plume_radius_scale = 0.92f;
                vessel.blast_push_scale = 0.08f;
                vessel.blast_heat_scale = 0.04f;
                break;
            case ProjectilePreset::MEDIUM_ROCKET:
                shell_stiffness *= 0.98f;
                shell_density *= 0.96f;
                core_stiffness *= 0.92f;
                core_density *= 0.94f;
                fuse_initial_temp = g_projectile_auto_arm ? 580.0f : 300.0f;
                core_initial_temp = g_projectile_auto_arm ? 344.0f : 300.0f;
                core_thermal = ng::vec4(1.30f, 1.48f, 0.78f, 0.00f);
                vessel.gas_mass = 0.10f;
                vessel.gas_source_scale = 0.82f;
                vessel.rupture_scale = 1.98f;
                vessel.burst_scale = 0.24f;
                vessel.shell_push_scale = 0.42f;
                vessel.core_push_scale = 0.56f;
                vessel.leak_scale = 1.54f;
                vessel.nozzle_open = 0.31f;
                vessel.thrust_scale = 2.40f;
                vessel.plume_push_scale = 0.52f;
                vessel.plume_heat_scale = 0.28f;
                vessel.plume_radius_scale = 0.96f;
                vessel.blast_push_scale = 0.12f;
                vessel.blast_heat_scale = 0.06f;
                break;
            case ProjectilePreset::ABOVE_MED_ROCKET:
                shell_stiffness *= 1.00f;
                shell_density *= 0.98f;
                core_stiffness *= 0.96f;
                core_density *= 0.98f;
                fuse_initial_temp = g_projectile_auto_arm ? 600.0f : 300.0f;
                core_initial_temp = g_projectile_auto_arm ? 348.0f : 300.0f;
                core_thermal = ng::vec4(1.50f, 1.70f, 0.72f, 0.00f);
                vessel.gas_mass = 0.12f;
                vessel.gas_source_scale = 0.96f;
                vessel.rupture_scale = 1.92f;
                vessel.burst_scale = 0.30f;
                vessel.shell_push_scale = 0.52f;
                vessel.core_push_scale = 0.68f;
                vessel.leak_scale = 1.50f;
                vessel.nozzle_open = 0.33f;
                vessel.thrust_scale = 2.95f;
                vessel.plume_push_scale = 0.70f;
                vessel.plume_heat_scale = 0.40f;
                vessel.plume_radius_scale = 1.00f;
                vessel.blast_push_scale = 0.16f;
                vessel.blast_heat_scale = 0.08f;
                break;
            case ProjectilePreset::SOFT_DEEP_FUSE_BOMB:
                shell_stiffness *= 0.98f;
                shell_density *= 0.96f;
                core_stiffness *= 0.84f;
                core_density *= 0.86f;
                core_thermal = ng::vec4(1.06f, 1.20f, 0.88f, 0.00f);
                vessel.gas_mass = 0.04f;
                vessel.gas_source_scale = 0.58f;
                vessel.rupture_scale = 1.74f;
                vessel.burst_scale = 0.42f;
                vessel.shell_push_scale = 0.40f;
                vessel.core_push_scale = 0.48f;
                vessel.leak_scale = 0.48f;
                vessel.trigger_speed = 0.17f;
                vessel.trigger_heat = 1100.0f;
                vessel.trigger_boost = 1.18f;
                vessel.plume_push_scale = 0.34f;
                vessel.plume_heat_scale = 0.16f;
                vessel.plume_radius_scale = 0.92f;
                vessel.blast_push_scale = 0.16f;
                vessel.blast_heat_scale = 0.08f;
                vessel.ignition_window = 0.56f;
                break;
            case ProjectilePreset::MEDIUM_DEEP_FUSE_BOMB:
                shell_stiffness *= 1.00f;
                shell_density *= 0.98f;
                core_stiffness *= 0.90f;
                core_density *= 0.92f;
                core_thermal = ng::vec4(1.30f, 1.46f, 0.80f, 0.00f);
                vessel.gas_mass = 0.05f;
                vessel.gas_source_scale = 0.80f;
                vessel.rupture_scale = 1.78f;
                vessel.burst_scale = 0.78f;
                vessel.shell_push_scale = 0.68f;
                vessel.core_push_scale = 0.76f;
                vessel.leak_scale = 0.50f;
                vessel.trigger_speed = 0.18f;
                vessel.trigger_heat = 1110.0f;
                vessel.trigger_boost = 1.34f;
                vessel.plume_push_scale = 0.50f;
                vessel.plume_heat_scale = 0.24f;
                vessel.plume_radius_scale = 0.94f;
                vessel.blast_push_scale = 0.28f;
                vessel.blast_heat_scale = 0.12f;
                vessel.ignition_window = 0.58f;
                break;
            case ProjectilePreset::ABOVE_MED_DEEP_FUSE_BOMB:
                shell_stiffness *= 1.01f;
                shell_density *= 1.00f;
                core_stiffness *= 0.95f;
                core_density *= 0.96f;
                core_thermal = ng::vec4(1.50f, 1.66f, 0.76f, 0.00f);
                vessel.gas_mass = 0.055f;
                vessel.gas_source_scale = 0.94f;
                vessel.rupture_scale = 1.72f;
                vessel.burst_scale = 1.12f;
                vessel.shell_push_scale = 0.88f;
                vessel.core_push_scale = 0.94f;
                vessel.leak_scale = 0.50f;
                vessel.trigger_speed = 0.185f;
                vessel.trigger_heat = 1115.0f;
                vessel.trigger_boost = 1.46f;
                vessel.plume_push_scale = 0.62f;
                vessel.plume_heat_scale = 0.32f;
                vessel.plume_radius_scale = 0.96f;
                vessel.blast_push_scale = 0.38f;
                vessel.blast_heat_scale = 0.18f;
                vessel.ignition_window = 0.60f;
                break;
            case ProjectilePreset::SOFT_EVEN_DEEPER_FUSE_BOMB:
                shell_stiffness *= 0.98f;
                shell_density *= 0.96f;
                core_stiffness *= 0.84f;
                core_density *= 0.86f;
                core_thermal = ng::vec4(1.02f, 1.16f, 0.90f, 0.00f);
                vessel.gas_mass = 0.035f;
                vessel.gas_source_scale = 0.50f;
                vessel.rupture_scale = 1.88f;
                vessel.burst_scale = 0.36f;
                vessel.shell_push_scale = 0.30f;
                vessel.core_push_scale = 0.38f;
                vessel.leak_scale = 0.38f;
                vessel.trigger_speed = 0.088f;
                vessel.trigger_heat = 1120.0f;
                vessel.trigger_boost = 1.08f;
                vessel.plume_push_scale = 0.26f;
                vessel.plume_heat_scale = 0.12f;
                vessel.plume_radius_scale = 0.90f;
                vessel.blast_push_scale = 0.12f;
                vessel.blast_heat_scale = 0.06f;
                vessel.ignition_window = 0.66f;
                break;
            case ProjectilePreset::MEDIUM_EVEN_DEEPER_FUSE_BOMB:
                shell_stiffness *= 1.00f;
                shell_density *= 0.98f;
                core_stiffness *= 0.90f;
                core_density *= 0.92f;
                core_thermal = ng::vec4(1.24f, 1.40f, 0.82f, 0.00f);
                vessel.gas_mass = 0.045f;
                vessel.gas_source_scale = 0.70f;
                vessel.rupture_scale = 1.92f;
                vessel.burst_scale = 0.70f;
                vessel.shell_push_scale = 0.54f;
                vessel.core_push_scale = 0.60f;
                vessel.leak_scale = 0.40f;
                vessel.trigger_speed = 0.092f;
                vessel.trigger_heat = 1130.0f;
                vessel.trigger_boost = 1.28f;
                vessel.plume_push_scale = 0.44f;
                vessel.plume_heat_scale = 0.20f;
                vessel.plume_radius_scale = 0.92f;
                vessel.blast_push_scale = 0.24f;
                vessel.blast_heat_scale = 0.12f;
                vessel.ignition_window = 0.68f;
                break;
            case ProjectilePreset::ABOVE_MED_EVEN_DEEPER_FUSE_BOMB:
                shell_stiffness *= 1.01f;
                shell_density *= 1.00f;
                core_stiffness *= 0.95f;
                core_density *= 0.96f;
                core_thermal = ng::vec4(1.44f, 1.60f, 0.78f, 0.00f);
                vessel.gas_mass = 0.05f;
                vessel.gas_source_scale = 0.86f;
                vessel.rupture_scale = 1.90f;
                vessel.burst_scale = 1.02f;
                vessel.shell_push_scale = 0.72f;
                vessel.core_push_scale = 0.80f;
                vessel.leak_scale = 0.42f;
                vessel.trigger_speed = 0.094f;
                vessel.trigger_heat = 1140.0f;
                vessel.trigger_boost = 1.46f;
                vessel.plume_push_scale = 0.60f;
                vessel.plume_heat_scale = 0.28f;
                vessel.plume_radius_scale = 0.94f;
                vessel.blast_push_scale = 0.36f;
                vessel.blast_heat_scale = 0.16f;
                vessel.ignition_window = 0.70f;
                break;
            case ProjectilePreset::EVEN_DEEPER_FUSE_BOMB:
                vessel.gas_mass = 0.055f;
                vessel.gas_source_scale = 1.02f;
                vessel.rupture_scale = 1.88f;
                vessel.burst_scale = 1.34f;
                vessel.shell_push_scale = 0.92f;
                vessel.core_push_scale = 0.98f;
                vessel.leak_scale = 0.44f;
                vessel.trigger_speed = 0.095f;
                vessel.trigger_heat = 1140.0f;
                vessel.trigger_boost = 1.68f;
                vessel.plume_push_scale = 0.78f;
                vessel.plume_heat_scale = 0.38f;
                vessel.plume_radius_scale = 0.96f;
                vessel.blast_push_scale = 0.52f;
                vessel.blast_heat_scale = 0.24f;
                vessel.ignition_window = 0.72f;
                break;
            default:
                break;
            }

            if (!g_projectile_auto_arm) {
                shell_initial_temp = glm::min(shell_initial_temp, 300.0f);
                fuse_initial_temp = glm::min(fuse_initial_temp, 300.0f);
                core_initial_temp = glm::min(core_initial_temp, 296.0f);
                vessel.gas_mass = 0.0f;
                idle_shell_guard = 1.18f;
                shell_stiffness *= idle_shell_guard;
                shell_density *= 1.08f;
                vessel.rupture_scale *= 1.10f;
                vessel.leak_scale *= 0.80f;
                vessel.shell_push_scale *= 0.92f;
            }

            std::vector<ng::vec2> shell_positions_combined = shell_positions;
            ParticleSpanRef shell_span = spawn_layer(shell_positions, shell_shell,
                                                     shell_material, shell_stiffness, shell_initial_temp,
                                                     shell_density, shell_thermal);
            if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::LAYERED ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::TIMED ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::HEAT ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::HESH ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::APHE ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEMO ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEEP_FUSE ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::THERMITE ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::TANDEM ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::FUEL_AIR ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::CLUSTER ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::CATACLYSM ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::CASCADE ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::CRYO ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::SPIRAL) {
                ParticleSpanRef armor_span = spawn_layer(armor_positions, armor_shell,
                                                         (preset.vessel_mode == ProjectilePresetDesc::VesselMode::TIMED ||
                                                          preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEEP_FUSE ||
                                                          preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE ||
                                                          preset.vessel_mode == ProjectilePresetDesc::VesselMode::CATACLYSM ||
                                                          preset.vessel_mode == ProjectilePresetDesc::VesselMode::CRYO)
                                                             ? ng::MPMMaterial::STONEWARE
                                                             : ng::MPMMaterial::THERMO_METAL,
                                                         ((preset.vessel_mode == ProjectilePresetDesc::VesselMode::TIMED ||
                                                           preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEEP_FUSE ||
                                                           preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE ||
                                                           preset.vessel_mode == ProjectilePresetDesc::VesselMode::CATACLYSM ||
                                                           preset.vessel_mode == ProjectilePresetDesc::VesselMode::CRYO) ? 138000.0f : 152000.0f) * idle_shell_guard,
                                                         300.0f,
                                                         shell_density * ((preset.vessel_mode == ProjectilePresetDesc::VesselMode::TIMED ||
                                                                           preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEEP_FUSE ||
                                                                           preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE ||
                                                                           preset.vessel_mode == ProjectilePresetDesc::VesselMode::CATACLYSM ||
                                                                           preset.vessel_mode == ProjectilePresetDesc::VesselMode::CRYO) ? 0.96f : 1.05f),
                                                         preset.vessel_mode == ProjectilePresetDesc::VesselMode::TIMED
                                                             ? ng::vec4(0.00f, 0.08f, 1.12f, 0.00f)
                                                             : (preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEEP_FUSE ||
                                                                preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE ||
                                                                preset.vessel_mode == ProjectilePresetDesc::VesselMode::CATACLYSM ||
                                                                preset.vessel_mode == ProjectilePresetDesc::VesselMode::CRYO)
                                                                   ? ng::vec4(0.00f, 0.06f, 1.20f, 0.00f)
                                                                   : ng::vec4(0.08f, 0.18f, 1.18f, 0.00f));
                shell_span = merge_contiguous_spans(shell_span, armor_span);
                shell_positions_combined.insert(shell_positions_combined.end(), armor_positions.begin(), armor_positions.end());
            }
            if ((preset.vessel_mode == ProjectilePresetDesc::VesselMode::TIMED ||
                 preset.vessel_mode == ProjectilePresetDesc::VesselMode::HEAT ||
                 preset.vessel_mode == ProjectilePresetDesc::VesselMode::APHE ||
                 preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEEP_FUSE ||
                 preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE ||
                 preset.vessel_mode == ProjectilePresetDesc::VesselMode::TANDEM ||
                 preset.vessel_mode == ProjectilePresetDesc::VesselMode::CATACLYSM ||
                 preset.vessel_mode == ProjectilePresetDesc::VesselMode::CRYO) && !cap_positions.empty()) {
                ParticleSpanRef cap_span = spawn_layer(cap_positions, cap_shell,
                                                       ng::MPMMaterial::THERMO_METAL,
                                                       (preset.vessel_mode == ProjectilePresetDesc::VesselMode::HEAT ? 166000.0f :
                                                        preset.vessel_mode == ProjectilePresetDesc::VesselMode::APHE ? 174000.0f :
                                                        preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEEP_FUSE ? 154000.0f :
                                                        preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE ? 150000.0f :
                                                        preset.vessel_mode == ProjectilePresetDesc::VesselMode::TANDEM ? 168000.0f :
                                                        preset.vessel_mode == ProjectilePresetDesc::VesselMode::CATACLYSM ? 176000.0f :
                                                        preset.vessel_mode == ProjectilePresetDesc::VesselMode::CRYO ? 146000.0f :
                                                        132000.0f) * idle_shell_guard,
                                                       300.0f,
                                                       shell_density * (preset.vessel_mode == ProjectilePresetDesc::VesselMode::TIMED ? 0.48f :
                                                                        (preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEEP_FUSE ||
                                                                         preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE ||
                                                                         preset.vessel_mode == ProjectilePresetDesc::VesselMode::CATACLYSM ||
                                                                         preset.vessel_mode == ProjectilePresetDesc::VesselMode::CRYO) ? 0.42f : 0.62f),
                                                       preset.vessel_mode == ProjectilePresetDesc::VesselMode::TIMED
                                                           ? ng::vec4(0.02f, 0.12f, 0.98f, 0.00f)
                                                           : (preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEEP_FUSE ||
                                                              preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE ||
                                                              preset.vessel_mode == ProjectilePresetDesc::VesselMode::CATACLYSM ||
                                                              preset.vessel_mode == ProjectilePresetDesc::VesselMode::CRYO)
                                                                 ? ng::vec4(0.00f, 0.10f, 1.10f, 0.00f)
                                                                 : ng::vec4(0.02f, 0.14f, 1.06f, 0.00f));
                shell_span = merge_contiguous_spans(shell_span, cap_span);
                shell_positions_combined.insert(shell_positions_combined.end(), cap_positions.begin(), cap_positions.end());
            }
            ParticleSpanRef fuse_span = spawn_layer(fuse_positions, fuse_shell,
                                                    fuse_material, fuse_stiffness, fuse_initial_temp,
                                                    fuse_density, fuse_thermal);
            ParticleSpanRef core_span = spawn_layer(core_positions, core_shell,
                                                    ng::MPMMaterial::SEALED_CHARGE, core_stiffness, core_initial_temp,
                                                    core_density, core_thermal);
            ParticleSpanRef payload_span{};
            ParticleSpanRef trigger_span{};
            if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::CLAYMORE ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::HEAT ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::APHE ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::TANDEM ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::CLUSTER ||
                preset.vessel_mode == ProjectilePresetDesc::VesselMode::CASCADE) {
                payload_span = spawn_layer(payload_positions, payload_shell,
                                           payload_material,
                                           payload_stiffness,
                                           payload_initial_temp,
                                           payload_density,
                                           payload_thermal);
            } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::TIMED ||
                       preset.vessel_mode == ProjectilePresetDesc::VesselMode::TRIGGER ||
                       preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEEP_FUSE ||
                       preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE ||
                       preset.vessel_mode == ProjectilePresetDesc::VesselMode::TANDEM ||
                       preset.vessel_mode == ProjectilePresetDesc::VesselMode::CATACLYSM ||
                       preset.vessel_mode == ProjectilePresetDesc::VesselMode::CRYO) {
                trigger_span = spawn_layer(trigger_positions, trigger_shell,
                                           ng::MPMMaterial::BURNING,
                                           preset.vessel_mode == ProjectilePresetDesc::VesselMode::TIMED ? 52000.0f :
                                           preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEEP_FUSE ? 46000.0f :
                                           preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE ? 43000.0f :
                                           preset.vessel_mode == ProjectilePresetDesc::VesselMode::TANDEM ? 38000.0f :
                                           preset.vessel_mode == ProjectilePresetDesc::VesselMode::CATACLYSM ? 56000.0f :
                                           preset.vessel_mode == ProjectilePresetDesc::VesselMode::CRYO ? 42000.0f :
                                           26000.0f,
                                           g_projectile_auto_arm ? 305.0f : 300.0f,
                                           (preset.vessel_mode == ProjectilePresetDesc::VesselMode::TIMED ? 1.02f :
                                            preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEEP_FUSE ? 0.66f :
                                            preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE ? 0.60f :
                                            preset.vessel_mode == ProjectilePresetDesc::VesselMode::TANDEM ? 0.72f :
                                            preset.vessel_mode == ProjectilePresetDesc::VesselMode::CATACLYSM ? 0.74f :
                                            preset.vessel_mode == ProjectilePresetDesc::VesselMode::CRYO ? 0.62f :
                                            0.78f) *
                                           glm::max(g_ball_weight, 0.5f) / 6.0f,
                                           preset.vessel_mode == ProjectilePresetDesc::VesselMode::DEEP_FUSE
                                               ? ng::vec4(0.54f, 0.98f, 0.22f, 0.02f)
                                               : preset.vessel_mode == ProjectilePresetDesc::VesselMode::EVEN_DEEPER_FUSE
                                                     ? ng::vec4(0.48f, 0.92f, 0.20f, 0.01f)
                                               : preset.vessel_mode == ProjectilePresetDesc::VesselMode::TANDEM
                                                     ? ng::vec4(0.76f, 1.12f, 0.24f, 0.03f)
                                               : preset.vessel_mode == ProjectilePresetDesc::VesselMode::CATACLYSM
                                                     ? ng::vec4(0.66f, 1.16f, 0.20f, 0.02f)
                                               : preset.vessel_mode == ProjectilePresetDesc::VesselMode::CRYO
                                                     ? ng::vec4(0.46f, 0.86f, 0.20f, 0.01f)
                                               : ng::vec4(0.80f, 1.18f, 0.24f, 0.04f));
            }
            if (shell_span.valid() && core_span.valid()) {
                vessel.shell = shell_span;
                vessel.fuse = fuse_span;
                vessel.core = core_span;
                vessel.payload = payload_span;
                vessel.trigger = trigger_span;
                g_pressure_vessels.push_back(vessel);
                if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::DIRECTIONAL) {
                    seed_shell_crack_bias(shell_span, shell_positions_combined, origin, launch_dir, 0.56f, 0.12f, 0.42f);
                } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::CLAYMORE) {
                    seed_shell_crack_bias(shell_span, shell_positions_combined, origin, launch_dir, 0.40f, 0.22f, 0.60f);
                } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::BROADSIDE) {
                    ng::vec2 side_axis(-launch_dir.y, launch_dir.x);
                    seed_shell_crack_bias_symmetric(shell_span, shell_positions_combined, origin, side_axis, 0.46f, 0.10f, 0.44f);
                } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::HEAT) {
                    seed_shell_crack_bias(shell_span, shell_positions_combined, origin, launch_dir, 0.74f, 0.04f, 0.18f);
                } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::HESH) {
                    ng::vec2 side_axis(-launch_dir.y, launch_dir.x);
                    seed_shell_crack_bias_symmetric(shell_span, shell_positions_combined, origin, side_axis, 0.62f, 0.12f, 0.32f);
                } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::APHE) {
                    seed_shell_crack_bias(shell_span, shell_positions_combined, origin, launch_dir, 0.68f, 0.05f, 0.20f);
                } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::TANDEM) {
                    seed_shell_crack_bias(shell_span, shell_positions_combined, origin, launch_dir, 0.78f, 0.04f, 0.14f);
                } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::SPIRAL) {
                    ng::vec2 side_axis(-launch_dir.y, launch_dir.x);
                    seed_shell_crack_bias_symmetric(shell_span, shell_positions_combined, origin, side_axis, 0.58f, 0.08f, 0.24f);
                } else if (preset.vessel_mode == ProjectilePresetDesc::VesselMode::ROCKET) {
                    seed_shell_crack_bias(shell_span, shell_positions_combined, origin, -launch_dir, 0.66f, 0.06f, 0.16f);
                }
            }
        }
    } else {
        g_mpm.params().youngs_modulus = preset.stiffness;
        if (shape == ng::SpawnShape::CIRCLE) {
            g_mpm.spawn_circle(g_particles, origin, g_ball_radius, spacing,
                               preset.material, preset.initial_temp, ng::vec2(1.0f, 0.0f),
                               preset.density_scale, preset.thermal_scale);
        } else if (shape == ng::SpawnShape::RECT) {
            ng::vec2 half_extents = ng::shape_half_extents(shape, g_ball_radius, projectile_shape_aspect(shape));
            g_mpm.spawn_block(g_particles, origin - half_extents, origin + half_extents, spacing,
                              preset.material, preset.initial_temp, ng::vec2(1.0f, 0.0f),
                              preset.density_scale, preset.thermal_scale);
        } else {
            std::vector<ng::vec2> positions;
            std::vector<ng::f32> shell_seeds;
            build_projectile_points(origin, g_ball_radius, std::atan2(launch_dir.y, launch_dir.x),
                                    shape, spacing, positions, shell_seeds);
            g_mpm.spawn_points(g_particles, positions, shell_seeds, spacing,
                               preset.material, preset.initial_temp, ng::vec2(1.0f, 0.0f),
                               preset.density_scale, preset.thermal_scale);
        }
    }
    g_mpm.params().youngs_modulus = old_E;

    ng::u32 after = g_particles.range(ng::SolverType::MPM).count;
    ng::u32 spawned = after - before;
    if (spawned == 0) return;

    std::vector<ng::vec2> launch_velocities(spawned, launch_vel);
    g_particles.upload_velocities(global_offset, launch_velocities.data(), spawned);
}

// Kelvin body force on a ferrofluid particle: F = (chi/2) * grad|H|^2.
// This is the exact same equation as the MPM G2P code path — here we
// just sample the solved |H|^2 field on CPU and finite-difference it
// for debug visualization. Returns zero when there's no solved field.
static ng::vec2 magnetic_kelvin_force(ng::vec2 pos) {
    if (!g_magnetic.active()) return ng::vec2(0.0f);
    // Step size matches roughly the magnetic grid cell (~0.023 m) but
    // is kept finite-difference-friendly. Sampling is via the debug
    // cache (GPU→CPU download) so this runs at overlay framerate only.
    const ng::f32 step = 0.025f;
    ng::f32 h2_l = g_magnetic.sample_debug(pos - ng::vec2(step, 0.0f)).w;
    ng::f32 h2_r = g_magnetic.sample_debug(pos + ng::vec2(step, 0.0f)).w;
    ng::f32 h2_b = g_magnetic.sample_debug(pos - ng::vec2(0.0f, step)).w;
    ng::f32 h2_t = g_magnetic.sample_debug(pos + ng::vec2(0.0f, step)).w;
    ng::vec2 grad_h2(
        (h2_r - h2_l) / (2.0f * step),
        (h2_t - h2_b) / (2.0f * step));
    const ng::f32 chi = 0.95f; // ferrofluid susceptibility (matches G2P)
    ng::f32 force_scale = g_magnetic.params().force_scale;
    return 0.5f * chi * force_scale * grad_h2;
}

static void draw_magnetic_field_overlay(InteractMode active_mode) {
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const ng::MPMParams& mp = g_mpm.params();
    const bool real_active = g_magnetic.active();

    if (real_active) {
        ImVec2 screen = ImGui::GetIO().DisplaySize;
        const auto& mp2 = g_magnetic.params();
        const bool scene_on = mp2.enabled;
        const bool cursor_on = std::abs(mp2.cursor_strength) > 1e-4f;
        const bool forced_on = mp2.debug_force_active;
        const char* mode_text =
            (forced_on && scene_on && cursor_on) ? "debug-force + scene + cursor" :
            (forced_on && cursor_on)             ? "debug-force + cursor" :
            (forced_on)                          ? "debug-force (scene)" :
            (scene_on && cursor_on)              ? "scene + cursor" :
            (scene_on)                           ? "scene magnets"  :
                                                   "cursor only";
        char banner[200];
        snprintf(banner, sizeof(banner), "Magnet active: %s | inner %.2f / falloff %.2f | strength %.2f",
                 mode_text, mp.magnet_radius, mp.magnet_falloff_radius, g_magnet_strength);
        ImVec2 text_size = ImGui::CalcTextSize(banner);
        ImVec2 pos((screen.x - text_size.x) * 0.5f - 10.0f, 10.0f);
        ImVec2 end(pos.x + text_size.x + 20.0f, pos.y + text_size.y + 10.0f);
        dl->AddRectFilled(pos, end, IM_COL32(18, 24, 36, 190), 8.0f);
        dl->AddRect(pos, end, IM_COL32(110, 190, 255, 210), 8.0f, 0, 1.3f);
        dl->AddText(ImVec2(pos.x + 10.0f, pos.y + 5.0f), IM_COL32(210, 240, 255, 235), banner);
    }

    if (!g_show_magnetic_debug) return;

    auto sample_field = [&](ng::vec2 pos) -> ng::vec4 {
        if (!real_active) return ng::vec4(0.0f);
        if (g_magnetic_debug_view == 4) {
            // Kelvin body force F = (chi/2) * grad|H|^2 on ferrofluid.
            ng::vec2 f = magnetic_kelvin_force(pos);
            ng::f32 mag = glm::length(f);
            return ng::vec4(f.x, f.y, mag, mag * mag);
        }
        return g_magnetic.sample_debug(pos);
    };
    auto sample_magnetization = [&](ng::vec2 pos) -> ng::vec4 {
        if (!real_active) return ng::vec4(0.0f);
        return g_magnetic.sample_magnetization_debug(pos);
    };
    auto sample_total_field = [&](ng::vec2 pos) -> ng::vec4 {
        if (!real_active) return ng::vec4(0.0f);
        return g_magnetic.sample_total_debug(pos);
    };
    auto sample_source = [&](ng::vec2 pos) -> ng::f32 {
        if (!real_active) return 0.0f;
        return g_magnetic.sample_source_debug(pos);
    };
    auto visible_world_min = g_camera.screen_to_world(ng::vec2(0.0f, ImGui::GetIO().DisplaySize.y));
    auto visible_world_max = g_camera.screen_to_world(ng::vec2(ImGui::GetIO().DisplaySize.x, 0.0f));
    ng::vec2 world_min = visible_world_min;
    ng::vec2 world_max = visible_world_max;
    if (real_active) {
        world_min = glm::max(world_min, g_magnetic.world_min());
        world_max = glm::min(world_max, g_magnetic.world_max());
    }

    float view_max_mag = 1e-3f;
    float view_max_total_mag = 1e-3f;
    float view_max_energy = 1e-3f;
    float view_max_total_energy = 1e-3f;
    float view_max_mag_M = 1e-3f;
    float view_max_source = 1e-3f;
    float view_max_delta_H = 1e-3f;
    if (g_magnetic_debug_view != 5) {
        const int stat_nx = 48;
        const int stat_ny = 34;
        for (int iy = 0; iy < stat_ny; ++iy) {
            for (int ix = 0; ix < stat_nx; ++ix) {
                ng::f32 tx = static_cast<ng::f32>(ix) / static_cast<ng::f32>(stat_nx - 1);
                ng::f32 ty = static_cast<ng::f32>(iy) / static_cast<ng::f32>(stat_ny - 1);
                ng::vec2 p = glm::mix(world_min, world_max, ng::vec2(tx, ty));
                ng::vec4 sample = sample_field(p);
                ng::vec4 total_sample = sample_total_field(p);
                ng::vec4 mag_sample = sample_magnetization(p);
                ng::f32 source_sample = std::abs(sample_source(p));
                view_max_mag = std::max(view_max_mag, sample.z);
                view_max_total_mag = std::max(view_max_total_mag, total_sample.z);
                view_max_energy = std::max(view_max_energy, sample.w);
                view_max_total_energy = std::max(view_max_total_energy, total_sample.w);
                view_max_mag_M = std::max(view_max_mag_M, mag_sample.z);
                view_max_source = std::max(view_max_source, source_sample);
                view_max_delta_H = std::max(view_max_delta_H, glm::length(ng::vec2(total_sample.x - sample.x, total_sample.y - sample.y)));
            }
        }
    }

    auto pack_color = [](ng::vec3 rgb, ng::f32 alpha) -> ImU32 {
        return IM_COL32(static_cast<int>(glm::clamp(rgb.r, 0.0f, 1.0f) * 255.0f),
                        static_cast<int>(glm::clamp(rgb.g, 0.0f, 1.0f) * 255.0f),
                        static_cast<int>(glm::clamp(rgb.b, 0.0f, 1.0f) * 255.0f),
                        static_cast<int>(glm::clamp(alpha, 0.0f, 1.0f) * 255.0f));
    };
    auto field_color_dir = [&](ng::vec2 dir, ng::f32 mag, ng::f32 alpha_scale = 1.0f) -> ImU32 {
        (void)dir;
        ng::f32 norm = glm::clamp(mag / view_max_mag, 0.0f, 1.0f);
        ng::f32 t = std::pow(norm, 0.72f);
        ng::vec3 cold(0.20f, 0.42f, 1.0f);
        ng::vec3 hot(1.0f, 0.20f, 0.16f);
        ng::vec3 rgb = glm::mix(cold, hot, t);
        return pack_color(rgb, (0.26f + 0.74f * std::pow(norm, 0.68f)) * alpha_scale);
    };
    auto scalar_color = [&](ng::f32 value_norm, ng::f32 alpha_scale = 1.0f) -> ImU32 {
        ng::f32 t = glm::clamp(value_norm, 0.0f, 1.0f);
        ng::vec3 cold(0.12f, 0.26f, 0.92f);
        ng::vec3 hot(1.0f, 0.18f, 0.16f);
        ng::vec3 rgb = glm::mix(cold, hot, std::pow(t, 0.78f));
        return pack_color(rgb, (0.15f + 0.78f * std::pow(t, 0.70f)) * alpha_scale);
    };
    auto signed_scalar_color = [&](ng::f32 signed_value, ng::f32 max_abs, ng::f32 alpha_scale = 1.0f) -> ImU32 {
        ng::f32 t = glm::clamp(std::abs(signed_value) / std::max(max_abs, 1e-5f), 0.0f, 1.0f);
        ng::vec3 neg(0.16f, 0.34f, 1.0f);
        ng::vec3 pos(1.0f, 0.22f, 0.18f);
        ng::vec3 rgb = (signed_value >= 0.0f) ? pos : neg;
        rgb = glm::mix(rgb * 0.25f, rgb, std::pow(t, 0.72f));
        return pack_color(rgb, (0.14f + 0.82f * std::pow(t, 0.72f)) * alpha_scale);
    };
    auto draw_cursor_rings = [&]() {
        ng::vec2 center = real_active ? g_magnetic.params().cursor_pos : mp.magnet_pos;
        ng::f32 inner = real_active ? g_magnetic.params().cursor_radius : mp.magnet_radius;
        ng::f32 outer = real_active ? g_magnetic.params().cursor_falloff_radius : mp.magnet_falloff_radius;
        ng::vec2 c = g_camera.world_to_screen(center);
        dl->AddCircle(ImVec2(c.x, c.y), inner * g_camera.zoom_level(), IM_COL32(210, 245, 255, 220), 64, 1.8f);
        if (outer > inner + 1e-4f) {
            dl->AddCircle(ImVec2(c.x, c.y), outer * g_camera.zoom_level(), IM_COL32(130, 205, 255, 150), 64, 1.3f);
        }
    };

    if (g_magnetic_debug_view == 0) {
        const int nx = 24;
        const int ny = 18;
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                ng::f32 tx = static_cast<ng::f32>(ix) / static_cast<ng::f32>(nx - 1);
                ng::f32 ty = static_cast<ng::f32>(iy) / static_cast<ng::f32>(ny - 1);
                ng::vec2 pos = glm::mix(world_min, world_max, ng::vec2(tx, ty));
                ng::vec4 sample = sample_field(pos);
                ng::vec2 field(sample.x, sample.y);
                ng::f32 mag = glm::length(field);
                if (mag <= 1e-4f) continue;
                ng::vec2 dir = field / mag;
                ng::f32 norm = glm::clamp(sample.z / view_max_mag, 0.0f, 1.0f);
                ng::f32 world_len = glm::clamp(0.04f + 0.16f * std::pow(norm, 0.62f), 0.04f, 0.18f);
                ng::vec2 end = pos + dir * world_len;
                ng::vec2 s0 = g_camera.world_to_screen(pos);
                ng::vec2 s1 = g_camera.world_to_screen(end);
                ImU32 col = field_color_dir(dir, sample.z, 1.0f);
                dl->AddLine(ImVec2(s0.x, s0.y), ImVec2(s1.x, s1.y), col, 1.7f);
                ng::vec2 perp(-dir.y, dir.x);
                ng::vec2 ah0 = end - dir * (world_len * 0.28f) + perp * (world_len * 0.12f);
                ng::vec2 ah1 = end - dir * (world_len * 0.28f) - perp * (world_len * 0.12f);
                ng::vec2 sh0 = g_camera.world_to_screen(ah0);
                ng::vec2 sh1 = g_camera.world_to_screen(ah1);
                dl->AddLine(ImVec2(s1.x, s1.y), ImVec2(sh0.x, sh0.y), col, 1.4f);
                dl->AddLine(ImVec2(s1.x, s1.y), ImVec2(sh1.x, sh1.y), col, 1.4f);
            }
        }
    } else if (g_magnetic_debug_view == 1) {
        const int sx = 18;
        const int sy = 12;
        const ng::f32 step = std::max(std::min((world_max.x - world_min.x) / 90.0f, (world_max.y - world_min.y) / 70.0f), 0.025f);
        for (int iy = 0; iy < sy; ++iy) {
            for (int ix = 0; ix < sx; ++ix) {
                ng::f32 tx = (static_cast<ng::f32>(ix) + 0.5f) / static_cast<ng::f32>(sx);
                ng::f32 ty = (static_cast<ng::f32>(iy) + 0.5f) / static_cast<ng::f32>(sy);
                ng::vec2 seed = glm::mix(world_min, world_max, ng::vec2(tx, ty));
                ng::vec4 seed_sample = sample_field(seed);
                if (seed_sample.z <= view_max_mag * 0.04f) continue;

                std::vector<ImVec2> points;
                points.reserve(96);
                auto trace_one = [&](ng::vec2 p0, ng::f32 sign) {
                    ng::vec2 p = p0;
                    for (int i = 0; i < 44; ++i) {
                        ng::vec4 sample = sample_field(p);
                        ng::vec2 field(sample.x, sample.y);
                        ng::f32 mag = glm::length(field);
                        if (mag <= 1e-4f) break;
                        field /= mag;
                        p += field * (step * sign);
                        if (p.x < world_min.x || p.x > world_max.x || p.y < world_min.y || p.y > world_max.y) break;
                        ng::vec2 sp = g_camera.world_to_screen(p);
                        if (sign < 0.0f) points.insert(points.begin(), ImVec2(sp.x, sp.y));
                        else points.push_back(ImVec2(sp.x, sp.y));
                    }
                };
                ng::vec2 ss = g_camera.world_to_screen(seed);
                points.push_back(ImVec2(ss.x, ss.y));
                trace_one(seed, -1.0f);
                trace_one(seed, 1.0f);
                if (points.size() >= 3) {
                    ng::vec2 seed_dir = glm::normalize(ng::vec2(seed_sample.x, seed_sample.y));
                    dl->AddPolyline(points.data(), static_cast<int>(points.size()), field_color_dir(seed_dir, seed_sample.z, 0.92f), 0, 1.9f);
                }
            }
        }
    } else if (g_magnetic_debug_view == 2) {
        const int nx = 84;
        const int ny = 60;
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                ng::f32 tx0 = static_cast<ng::f32>(ix) / static_cast<ng::f32>(nx);
                ng::f32 ty0 = static_cast<ng::f32>(iy) / static_cast<ng::f32>(ny);
                ng::f32 tx1 = static_cast<ng::f32>(ix + 1) / static_cast<ng::f32>(nx);
                ng::f32 ty1 = static_cast<ng::f32>(iy + 1) / static_cast<ng::f32>(ny);
                ng::vec2 center = glm::mix(world_min, world_max, ng::vec2((tx0 + tx1) * 0.5f, (ty0 + ty1) * 0.5f));
                ng::vec4 sample = sample_field(center);
                if (sample.z <= 1e-4f) continue;
                ng::vec2 a = g_camera.world_to_screen(glm::mix(world_min, world_max, ng::vec2(tx0, ty0)));
                ng::vec2 b = g_camera.world_to_screen(glm::mix(world_min, world_max, ng::vec2(tx1, ty1)));
                ng::f32 norm = std::pow(glm::clamp(sample.z / view_max_mag, 0.0f, 1.0f), 0.55f);
                dl->AddRectFilled(ImVec2(a.x, b.y), ImVec2(b.x, a.y), scalar_color(norm, 0.62f));
            }
        }
    } else if (g_magnetic_debug_view == 3) {
        const int nx = 88;
        const int ny = 62;
        std::vector<ng::f32> values(nx * ny, 0.0f);
        auto sample_scalar = [&](int ix, int iy) -> ng::f32& { return values[iy * nx + ix]; };
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                ng::f32 tx = static_cast<ng::f32>(ix) / static_cast<ng::f32>(nx - 1);
                ng::f32 ty = static_cast<ng::f32>(iy) / static_cast<ng::f32>(ny - 1);
                sample_scalar(ix, iy) = sample_field(glm::mix(world_min, world_max, ng::vec2(tx, ty))).w;
            }
        }
        ng::f32 iso_levels[] = {
            view_max_energy * 0.07f,
            view_max_energy * 0.16f,
            view_max_energy * 0.30f,
            view_max_energy * 0.50f,
            view_max_energy * 0.72f
        };
        auto interp = [&](ng::vec2 a, ng::vec2 b, ng::f32 va, ng::f32 vb, ng::f32 iso) -> ng::vec2 {
            ng::f32 denom = vb - va;
            ng::f32 t = (std::abs(denom) <= 1e-6f) ? 0.5f : glm::clamp((iso - va) / denom, 0.0f, 1.0f);
            return glm::mix(a, b, t);
        };
        for (ng::f32 iso : iso_levels) {
            for (int iy = 0; iy < ny - 1; ++iy) {
                for (int ix = 0; ix < nx - 1; ++ix) {
                    ng::f32 tx0 = static_cast<ng::f32>(ix) / static_cast<ng::f32>(nx - 1);
                    ng::f32 ty0 = static_cast<ng::f32>(iy) / static_cast<ng::f32>(ny - 1);
                    ng::f32 tx1 = static_cast<ng::f32>(ix + 1) / static_cast<ng::f32>(nx - 1);
                    ng::f32 ty1 = static_cast<ng::f32>(iy + 1) / static_cast<ng::f32>(ny - 1);
                    ng::vec2 p00 = glm::mix(world_min, world_max, ng::vec2(tx0, ty0));
                    ng::vec2 p10 = glm::mix(world_min, world_max, ng::vec2(tx1, ty0));
                    ng::vec2 p11 = glm::mix(world_min, world_max, ng::vec2(tx1, ty1));
                    ng::vec2 p01 = glm::mix(world_min, world_max, ng::vec2(tx0, ty1));
                    ng::f32 v00 = sample_scalar(ix, iy);
                    ng::f32 v10 = sample_scalar(ix + 1, iy);
                    ng::f32 v11 = sample_scalar(ix + 1, iy + 1);
                    ng::f32 v01 = sample_scalar(ix, iy + 1);

                    std::vector<ng::vec2> pts;
                    auto edge_if = [&](ng::f32 va, ng::f32 vb, ng::vec2 a, ng::vec2 b) {
                        if ((va < iso) != (vb < iso)) pts.push_back(interp(a, b, va, vb, iso));
                    };
                    edge_if(v00, v10, p00, p10);
                    edge_if(v10, v11, p10, p11);
                    edge_if(v11, v01, p11, p01);
                    edge_if(v01, v00, p01, p00);
                    if (pts.size() == 2) {
                        ng::vec2 a = g_camera.world_to_screen(pts[0]);
                        ng::vec2 b = g_camera.world_to_screen(pts[1]);
                        ng::f32 norm = std::pow(glm::clamp(iso / view_max_energy, 0.0f, 1.0f), 0.55f);
                        dl->AddLine(ImVec2(a.x, a.y), ImVec2(b.x, b.y), scalar_color(norm, 0.88f), 1.8f);
                    } else if (pts.size() == 4) {
                        ng::vec2 a0 = g_camera.world_to_screen(pts[0]);
                        ng::vec2 a1 = g_camera.world_to_screen(pts[1]);
                        ng::vec2 b0 = g_camera.world_to_screen(pts[2]);
                        ng::vec2 b1 = g_camera.world_to_screen(pts[3]);
                        ng::f32 norm = std::pow(glm::clamp(iso / view_max_energy, 0.0f, 1.0f), 0.55f);
                        ImU32 col = scalar_color(norm, 0.74f);
                        dl->AddLine(ImVec2(a0.x, a0.y), ImVec2(a1.x, a1.y), col, 1.6f);
                        dl->AddLine(ImVec2(b0.x, b0.y), ImVec2(b1.x, b1.y), col, 1.6f);
                    }
                }
            }
        }
    } else if (g_magnetic_debug_view == 4) {
        const int nx = 24;
        const int ny = 18;
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                ng::f32 tx = static_cast<ng::f32>(ix) / static_cast<ng::f32>(nx - 1);
                ng::f32 ty = static_cast<ng::f32>(iy) / static_cast<ng::f32>(ny - 1);
                ng::vec2 pos = glm::mix(world_min, world_max, ng::vec2(tx, ty));
                ng::vec4 sample = sample_field(pos);
                ng::vec2 field(sample.x, sample.y);
                ng::f32 mag = glm::length(field);
                if (mag <= 1e-4f) continue;
                ng::vec2 dir = field / mag;
                ng::f32 norm = glm::clamp(sample.z / view_max_mag, 0.0f, 1.0f);
                ng::f32 world_len = glm::clamp(0.05f + 0.15f * std::pow(norm, 0.58f), 0.05f, 0.18f);
                ng::vec2 end = pos + dir * world_len;
                ng::vec2 s0 = g_camera.world_to_screen(pos);
                ng::vec2 s1 = g_camera.world_to_screen(end);
                ImU32 col = field_color_dir(dir, sample.z * 1.3f, 1.0f);
                dl->AddLine(ImVec2(s0.x, s0.y), ImVec2(s1.x, s1.y), col, 1.8f);

                ng::vec2 perp(-dir.y, dir.x);
                ng::vec2 ah0 = end - dir * (world_len * 0.24f) + perp * (world_len * 0.11f);
                ng::vec2 ah1 = end - dir * (world_len * 0.24f) - perp * (world_len * 0.11f);
                ng::vec2 sh0 = g_camera.world_to_screen(ah0);
                ng::vec2 sh1 = g_camera.world_to_screen(ah1);
                dl->AddLine(ImVec2(s1.x, s1.y), ImVec2(sh0.x, sh0.y), col, 1.4f);
                dl->AddLine(ImVec2(s1.x, s1.y), ImVec2(sh1.x, sh1.y), col, 1.4f);

                ng::vec2 hub = pos + perp * (0.022f * std::sin((pos.x - mp.magnet_pos.x) * 9.0f));
                ng::vec2 sh = g_camera.world_to_screen(hub);
                dl->AddCircleFilled(ImVec2(sh.x, sh.y), 1.5f, IM_COL32(185, 235, 255, 170), 10);
            }
        }
    } else if (g_magnetic_debug_view == 6) {
        const int nx = 26;
        const int ny = 20;
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                ng::f32 tx = static_cast<ng::f32>(ix) / static_cast<ng::f32>(nx - 1);
                ng::f32 ty = static_cast<ng::f32>(iy) / static_cast<ng::f32>(ny - 1);
                ng::vec2 pos = glm::mix(world_min, world_max, ng::vec2(tx, ty));
                ng::vec4 sample = sample_magnetization(pos);
                ng::vec2 m(sample.x, sample.y);
                ng::f32 mag = glm::length(m);
                if (mag <= 1e-5f) continue;
                ng::vec2 dir = m / mag;
                ng::f32 norm = glm::clamp(sample.z / view_max_mag_M, 0.0f, 1.0f);
                ng::f32 world_len = glm::clamp(0.035f + 0.12f * std::pow(norm, 0.68f), 0.035f, 0.14f);
                ng::vec2 end = pos + dir * world_len;
                ng::vec2 s0 = g_camera.world_to_screen(pos);
                ng::vec2 s1 = g_camera.world_to_screen(end);
                ImU32 col = field_color_dir(dir, sample.z, 1.0f);
                dl->AddLine(ImVec2(s0.x, s0.y), ImVec2(s1.x, s1.y), col, 1.8f);
                ng::vec2 perp(-dir.y, dir.x);
                ng::vec2 ah0 = end - dir * (world_len * 0.26f) + perp * (world_len * 0.11f);
                ng::vec2 ah1 = end - dir * (world_len * 0.26f) - perp * (world_len * 0.11f);
                ng::vec2 sh0 = g_camera.world_to_screen(ah0);
                ng::vec2 sh1 = g_camera.world_to_screen(ah1);
                dl->AddLine(ImVec2(s1.x, s1.y), ImVec2(sh0.x, sh0.y), col, 1.4f);
                dl->AddLine(ImVec2(s1.x, s1.y), ImVec2(sh1.x, sh1.y), col, 1.4f);
            }
        }
    } else if (g_magnetic_debug_view == 7) {
        const int nx = 84;
        const int ny = 60;
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                ng::f32 tx0 = static_cast<ng::f32>(ix) / static_cast<ng::f32>(nx);
                ng::f32 ty0 = static_cast<ng::f32>(iy) / static_cast<ng::f32>(ny);
                ng::f32 tx1 = static_cast<ng::f32>(ix + 1) / static_cast<ng::f32>(nx);
                ng::f32 ty1 = static_cast<ng::f32>(iy + 1) / static_cast<ng::f32>(ny);
                ng::vec2 center = glm::mix(world_min, world_max, ng::vec2((tx0 + tx1) * 0.5f, (ty0 + ty1) * 0.5f));
                ng::vec4 sample = sample_magnetization(center);
                if (sample.z <= 1e-5f) continue;
                ng::vec2 a = g_camera.world_to_screen(glm::mix(world_min, world_max, ng::vec2(tx0, ty0)));
                ng::vec2 b = g_camera.world_to_screen(glm::mix(world_min, world_max, ng::vec2(tx1, ty1)));
                ng::f32 norm = std::pow(glm::clamp(sample.z / view_max_mag_M, 0.0f, 1.0f), 0.58f);
                dl->AddRectFilled(ImVec2(a.x, b.y), ImVec2(b.x, a.y), scalar_color(norm, 0.64f));
            }
        }
    } else if (g_magnetic_debug_view == 8) {
        const int nx = 84;
        const int ny = 60;
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                ng::f32 tx0 = static_cast<ng::f32>(ix) / static_cast<ng::f32>(nx);
                ng::f32 ty0 = static_cast<ng::f32>(iy) / static_cast<ng::f32>(ny);
                ng::f32 tx1 = static_cast<ng::f32>(ix + 1) / static_cast<ng::f32>(nx);
                ng::f32 ty1 = static_cast<ng::f32>(iy + 1) / static_cast<ng::f32>(ny);
                ng::vec2 center = glm::mix(world_min, world_max, ng::vec2((tx0 + tx1) * 0.5f, (ty0 + ty1) * 0.5f));
                ng::f32 sample = sample_source(center);
                if (std::abs(sample) <= 1e-6f) continue;
                ng::vec2 a = g_camera.world_to_screen(glm::mix(world_min, world_max, ng::vec2(tx0, ty0)));
                ng::vec2 b = g_camera.world_to_screen(glm::mix(world_min, world_max, ng::vec2(tx1, ty1)));
                dl->AddRectFilled(ImVec2(a.x, b.y), ImVec2(b.x, a.y), signed_scalar_color(sample, view_max_source, 0.70f));
            }
        }
    }

    const char* debug_label = nullptr;
    if (g_magnetic_debug_view == 4) debug_label = "Magnetic Debug: Kelvin force (drive field)";
    else if (g_magnetic_debug_view == 5) debug_label = "Magnetic Debug: drive field shader";
    else if (g_magnetic_debug_view == 6) debug_label = "Magnetic Debug: magnetization vectors";
    else if (g_magnetic_debug_view == 7) debug_label = "Magnetic Debug: |M|";
    else if (g_magnetic_debug_view == 8) debug_label = "Magnetic Debug: source charge";
    else debug_label = "Magnetic Debug: drive field";
    dl->AddText(ImVec2(18.0f, 64.0f), IM_COL32(190, 235, 255, 220), debug_label);
    ng::vec2 probe_world = real_active ? g_magnetic.params().cursor_pos : mp.magnet_pos;
    ng::vec4 probe = sample_field(probe_world);
    ng::vec4 probe_total = sample_total_field(probe_world);
    ng::vec4 probe_M = sample_magnetization(probe_world);
    ng::f32 probe_source = sample_source(probe_world);
    char probe_text[220];
    snprintf(probe_text, sizeof(probe_text), "Probe | drive H=(%.2f, %.2f) | |H|=%.2f | total H=(%.2f, %.2f) | |Ht|=%.2f",
             probe.x, probe.y, probe.z, probe_total.x, probe_total.y, probe_total.z);
    dl->AddText(ImVec2(18.0f, 84.0f), IM_COL32(190, 235, 255, 210), probe_text);
    char probe_text2[260];
    snprintf(probe_text2, sizeof(probe_text2), "M=(%.2f, %.2f) | |M|=%.2f | source=%.2f | max drive |H|=%.2f | max total |H|=%.2f | max dH=%.2f | max |M|=%.2f",
             probe_M.x, probe_M.y, probe_M.z, probe_source, view_max_mag, view_max_total_mag, view_max_delta_H, view_max_mag_M);
    dl->AddText(ImVec2(18.0f, 104.0f), IM_COL32(190, 235, 255, 210), probe_text2);
    draw_cursor_rings();
}

static const char* persistent_mode_label(InteractMode mode) {
    switch (mode) {
    case InteractMode::PUSH: return "1 Pull / Push";
    case InteractMode::SPRING_DRAG: return "2 Spring Drag";
    case InteractMode::DRAG: return "3 Telekinesis";
    case InteractMode::SWEEP_DRAG: return "4 Sweep Drag";
    case InteractMode::DROP_BALL: return "5 Projectile Launcher";
    case InteractMode::LAUNCHER2: return "6 Weapon Launcher";
    case InteractMode::DRAW_WALL: return "7 Draw Wall";
    case InteractMode::ERASE_WALL: return "8 Erase Wall";
    case InteractMode::FOOT_CONTROL: return "9 Foot Control";
    case InteractMode::PULL: return "Pull";
    case InteractMode::MAGNET: return "Magnet";
    default: return "Mode";
    }
}

static void draw_active_tooltip(InteractMode active_mode,
                                bool heat_active,
                                bool cool_active,
                                bool magnet_active,
                                bool lmb_active,
                                bool rmb_active) {
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const char* mode_label = persistent_mode_label(active_mode);
    const char* action_label = nullptr;
    if (magnet_active) {
        static char magnet_text[96];
        snprintf(magnet_text, sizeof(magnet_text), "Active: Magnet (%s)",
                 magnetic_cursor_field_label(g_magnetic.params().cursor_field_type));
        action_label = magnet_text;
    } else if (heat_active && cool_active) {
        action_label = "Active: Heat + Cool";
    } else if (heat_active) {
        action_label = "Active: Heating";
    } else if (cool_active) {
        action_label = "Active: Cooling";
    } else if (active_mode == InteractMode::PUSH) {
        if (lmb_active) action_label = "Active: Pull";
        else if (rmb_active) action_label = "Active: Push";
        else action_label = "LMB Pull | RMB Push";
    } else if (active_mode == InteractMode::SPRING_DRAG) {
        action_label = lmb_active ? "Active: Spring Drag" : "LMB drag: shape-preserving patch";
    } else if (active_mode == InteractMode::DRAG) {
        action_label = lmb_active ? "Active: Telekinesis" : "LMB drag: translate captured patch";
    } else if (active_mode == InteractMode::SWEEP_DRAG) {
        action_label = lmb_active ? "Active: Sweep Drag" : "LMB drag: directional brush";
    } else if (active_mode == InteractMode::DROP_BALL) {
        if (g_ball_drag_mode == ProjectileDragMode::AIM) action_label = "Active: Aim Launcher";
        else if (g_ball_drag_mode == ProjectileDragMode::CONE) action_label = "Active: Tune Cone";
        else if (lmb_active) action_label = "Active: Fire Projectile";
        else action_label = "RMB drag: aim | Alt+RMB: cone | LMB: fire";
    } else if (active_mode == InteractMode::FOOT_CONTROL && ng::foot_demo_active()) {
        static char foot_text[128];
        snprintf(foot_text, sizeof(foot_text), "Active: Foot Control (%s)", ng::foot_demo_focus_name());
        action_label = foot_text;
    } else {
        action_label = "Ready";
    }

    char text[196];
    snprintf(text, sizeof(text), "Mode: %s | %s | Hold G/H/M for heat, cool, magnet",
             mode_label, action_label);
    ImVec2 size = ImGui::CalcTextSize(text);
    ImVec2 pos(12.0f, ImGui::GetIO().DisplaySize.y - size.y - 20.0f);
    ImVec2 end(pos.x + size.x + 18.0f, pos.y + size.y + 10.0f);
    dl->AddRectFilled(pos, end, IM_COL32(18, 22, 34, 190), 8.0f);
    dl->AddRect(pos, end, IM_COL32(150, 205, 255, 175), 8.0f, 0, 1.2f);
    dl->AddText(ImVec2(pos.x + 9.0f, pos.y + 5.0f), IM_COL32(230, 238, 250, 235), text);
}

static void render_magnetic_field_shader_overlay(ng::f32 time) {
    if (!g_show_magnetic_debug || g_magnetic_debug_view != 5) return;

    const bool real_active = g_magnetic.active();
    const ng::MPMParams& mp = g_mpm.params();
    ng::vec2 visible_world_min = g_camera.screen_to_world(ng::vec2(0.0f, static_cast<ng::f32>(g_wc.height)));
    ng::vec2 visible_world_max = g_camera.screen_to_world(ng::vec2(static_cast<ng::f32>(g_wc.width), 0.0f));

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    g_magnetic_field_vis_shader.bind();
    g_magnetic_field_vis_shader.set_float("u_time", time);
    g_magnetic_field_vis_shader.set_vec2("u_visible_world_min", visible_world_min);
    g_magnetic_field_vis_shader.set_vec2("u_visible_world_max", visible_world_max);
    g_magnetic_field_vis_shader.set_vec2("u_field_world_min", g_magnetic.world_min());
    g_magnetic_field_vis_shader.set_vec2("u_field_world_max", g_magnetic.world_max());
    g_magnetic_field_vis_shader.set_int("u_use_real_field", real_active ? 1 : 0);
    g_magnetic_field_vis_shader.set_vec2("u_brush_pos", mp.magnet_pos);
    g_magnetic_field_vis_shader.set_float("u_brush_inner_radius", mp.magnet_radius);
    g_magnetic_field_vis_shader.set_float("u_brush_outer_radius", mp.magnet_falloff_radius);
    g_magnetic_field_vis_shader.set_float("u_brush_strength", std::abs(mp.magnet_force));
    g_magnetic_field_vis_shader.set_float("u_brush_spike_strength", mp.magnet_spike_strength);
    g_magnetic_field_vis_shader.set_float("u_brush_spike_freq", mp.magnet_spike_freq);
    g_magnetic_field_vis_shader.set_float("u_overlay_alpha", 0.72f);
    if (real_active) {
        g_magnetic.bind_field_for_read(7);
        g_magnetic_field_vis_shader.set_int("u_field_tex", 7);
    } else {
        g_magnetic_field_vis_shader.set_int("u_field_tex", 7);
    }

    glBindVertexArray(g_preview_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    g_magnetic_field_vis_shader.unbind();
}

static void reload_scene();
static void clear_non_sdf_objects();

static void push_panel_style(ImVec4 accent) {
    ImVec4 title_bg(accent.x * 0.38f, accent.y * 0.38f, accent.z * 0.38f, 0.88f);
    ImVec4 title_active(glm::min(accent.x * 0.72f + 0.10f, 1.0f),
                        glm::min(accent.y * 0.72f + 0.10f, 1.0f),
                        glm::min(accent.z * 0.72f + 0.10f, 1.0f),
                        0.95f);
    ImVec4 title_collapsed(title_bg.x * 0.92f, title_bg.y * 0.92f, title_bg.z * 0.92f, 0.78f);
    ImVec4 window_bg(0.07f + accent.x * 0.07f,
                     0.08f + accent.y * 0.07f,
                     0.10f + accent.z * 0.07f,
                     0.95f);
    ImVec4 border(glm::min(accent.x * 0.75f + 0.10f, 1.0f),
                  glm::min(accent.y * 0.75f + 0.10f, 1.0f),
                  glm::min(accent.z * 0.75f + 0.10f, 1.0f),
                  0.86f);
    ImVec4 header(accent.x * 0.26f, accent.y * 0.26f, accent.z * 0.26f, 0.74f);
    ImVec4 header_hover(accent.x * 0.38f, accent.y * 0.38f, accent.z * 0.38f, 0.84f);
    ImVec4 header_active(accent.x * 0.46f, accent.y * 0.46f, accent.z * 0.46f, 0.92f);
    ImGui::PushStyleColor(ImGuiCol_TitleBg, title_bg);
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, title_active);
    ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed, title_collapsed);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, window_bg);
    ImGui::PushStyleColor(ImGuiCol_Border, border);
    ImGui::PushStyleColor(ImGuiCol_Header, header);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, header_hover);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, header_active);
}

static void pop_panel_style() {
    ImGui::PopStyleColor(8);
}

static ImVec2 next_panel_position(ImVec2 size, ng::f32& cursor_x, ng::f32& cursor_y,
                                  ng::f32& row_height, ng::f32 start_x, ng::f32 max_x) {
    const ng::f32 gap = 10.0f;
    if (cursor_x > start_x && (cursor_x + size.x) > max_x) {
        cursor_x = start_x;
        cursor_y += row_height + gap;
        row_height = 0.0f;
    }
    ImVec2 pos(cursor_x, cursor_y);
    cursor_x += size.x + gap;
    row_height = glm::max(row_height, size.y);
    return pos;
}

static void draw_window_toggle_button(const char* label, bool* open, ImVec4 accent) {
    ImVec4 on_color = accent;
    ImVec4 off_color(accent.x * 0.30f, accent.y * 0.30f, accent.z * 0.30f, 0.85f);
    ImVec4 hover_color = *open
        ? ImVec4(glm::min(accent.x + 0.10f, 1.0f), glm::min(accent.y + 0.10f, 1.0f), glm::min(accent.z + 0.10f, 1.0f), 1.0f)
        : ImVec4(glm::min(off_color.x + 0.08f, 1.0f), glm::min(off_color.y + 0.08f, 1.0f), glm::min(off_color.z + 0.08f, 1.0f), 0.95f);
    ImGui::PushStyleColor(ImGuiCol_Button, *open ? on_color : off_color);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hover_color);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, hover_color);
    ImGui::PushStyleColor(ImGuiCol_Text, *open ? ImVec4(0.98f, 0.99f, 1.0f, 0.98f) : ImVec4(0.82f, 0.88f, 0.96f, 0.92f));
    if (ImGui::Button(label)) *open = !*open;
    ImGui::PopStyleColor(4);
}

static void draw_ui_top_bar(ng::f32 frame_ms) {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + 10.0f, viewport->WorkPos.y + 10.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x - 20.0f, 38.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.95f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 6.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.0f, 4.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6.0f, 0.0f));
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoScrollWithMouse;
    if (!ImGui::Begin("Window Toggles", nullptr, flags)) {
        ImGui::End();
        ImGui::PopStyleVar(3);
        return;
    }

    ng::f32 fps = frame_ms > 1e-4f ? (1000.0f / frame_ms) : 0.0f;
    ng::f32 row_y = ImGui::GetCursorPosY();

    draw_window_toggle_button("Interaction", &g_show_interaction_window, kInteractionAccent);
    ImGui::SameLine();
    draw_window_toggle_button("Environment", &g_show_environment_window, kEnvironmentAccent);
    ImGui::SameLine();
    draw_window_toggle_button("Backends", &g_show_backends_window, kBackendsAccent);
    ImGui::SameLine();
    draw_window_toggle_button("Appearance", &g_show_appearance_window, kAppearanceAccent);
    ImGui::SameLine();
    draw_window_toggle_button("Advanced", &g_show_advanced_window, kAdvancedAccent);
    ImGui::SameLine();
    draw_window_toggle_button("Presets", &g_show_presets_window, kPresetsAccent);
    ImGui::SameLine();
    bool reset_proxy = false;
    draw_window_toggle_button("Reset", &reset_proxy, kActionAccent);
    if (ImGui::IsItemClicked()) reload_scene();
    ImGui::SameLine();
    bool resume_proxy = g_paused;
    draw_window_toggle_button(g_paused ? "Resume" : "Pause", &resume_proxy, g_paused ? ImVec4(0.20f, 0.70f, 0.42f, 0.95f)
                                                                                     : ImVec4(0.82f, 0.46f, 0.22f, 0.95f));
    if (ImGui::IsItemClicked()) g_paused = !g_paused;
    ImGui::SameLine();
    bool create_proxy = g_creation.active;
    draw_window_toggle_button(g_creation.active ? "Close Create" : "Create", &create_proxy, ImVec4(0.72f, 0.58f, 0.20f, 0.95f));
    if (ImGui::IsItemClicked()) g_creation.active = !g_creation.active;
    ImGui::SameLine();
    bool hide_proxy = false;
    draw_window_toggle_button("Hide UI", &hide_proxy, ImVec4(0.40f, 0.44f, 0.52f, 0.92f));
    if (ImGui::IsItemClicked()) g_show_ui = false;

    const char* run_label = g_paused ? "PAUSED" : "RUNNING";
    const char* mode_label = g_creation.active ? "CREATE MODE" : "PLAY MODE";
    char fps_text[64];
    snprintf(fps_text, sizeof(fps_text), "%.1f FPS", fps);
    char counts_text[96];
    snprintf(counts_text, sizeof(counts_text), "SPH:%u  MPM:%u", g_sph.particle_count(), g_mpm.particle_count());

    ImVec2 fps_size = ImGui::CalcTextSize(fps_text);
    ImVec2 counts_size = ImGui::CalcTextSize(counts_text);
    ImVec2 mode_size = ImGui::CalcTextSize(mode_label);
    ImVec2 run_size = ImGui::CalcTextSize(run_label);
    ng::f32 right_margin = ImGui::GetStyle().WindowPadding.x + 8.0f;
    ng::f32 spacing = 14.0f;
    ng::f32 right_block_w = run_size.x + spacing + mode_size.x + spacing + counts_size.x + spacing + fps_size.x;
    ng::f32 start_x = glm::max(ImGui::GetCursorPosX(), ImGui::GetWindowWidth() - right_block_w - right_margin);
    ImGui::SetCursorPos(ImVec2(start_x, row_y + 4.0f));
    ImGui::PushStyleColor(ImGuiCol_Text, g_paused ? ImVec4(0.96f, 0.54f, 0.30f, 0.98f)
                                                  : ImVec4(0.42f, 0.92f, 0.60f, 0.98f));
    ImGui::TextUnformatted(run_label);
    ImGui::PopStyleColor();
    ImGui::SameLine(0.0f, spacing);
    ImGui::PushStyleColor(ImGuiCol_Text, g_creation.active ? ImVec4(0.98f, 0.82f, 0.38f, 0.98f)
                                                           : ImVec4(0.76f, 0.84f, 0.96f, 0.95f));
    ImGui::TextUnformatted(mode_label);
    ImGui::PopStyleColor();
    ImGui::SameLine(0.0f, spacing);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.74f, 0.88f, 0.98f, 0.94f));
    ImGui::TextUnformatted(counts_text);
    ImGui::PopStyleColor();
    ImGui::SameLine(0.0f, spacing);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.97f, 1.0f, 0.98f));
    ImGui::TextUnformatted(fps_text);
    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleVar(3);
}

static void draw_environment_window(ImVec2 pos, ImVec2 size, ImVec4 accent) {
    if (!g_show_environment_window) return;

    push_panel_style(accent);
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    if (!ImGui::Begin("Environment", &g_show_environment_window)) {
        ImGui::End();
        pop_panel_style();
        return;
    }

    ng::SPHParams& sp = const_cast<ng::SPHParams&>(g_sph.params());
    ng::MPMParams& mp = g_mpm.params();
    auto& ac = g_air.config();
    SceneSpaceConfig cfg = scene_space_config(g_scene);
    ng::vec2 world_size = cfg.sdf_world_max - cfg.sdf_world_min;

    if (ImGui::CollapsingHeader("Scene & Session", ImGuiTreeNodeFlags_DefaultOpen)) {
        int s = static_cast<int>(g_scene);
        if (ImGui::Combo("Scene Preset", &s, ng::scene_names, ng::SCENE_COUNT))
            g_scene = static_cast<ng::SceneID>(s);
        if (ImGui::Button("Load Scene")) reload_scene();
        ImGui::SameLine();
        if (ImGui::Button("Clear Non-SDF")) clear_non_sdf_objects();
        ImGui::TextDisabled("F1-F12 still jump through the main authored scenes. Use the combo for the full catalog.");
        ImGui::TextDisabled("Scene world: %.1f x %.1f m | Camera zoom %.0f", world_size.x, world_size.y, cfg.camera_zoom);
    }

    if (ImGui::CollapsingHeader("World & Atmosphere", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::SliderFloat("Gravity Y", &sp.gravity.y, -20.0f, 20.0f, "%.2f")) {
            mp.gravity.y = sp.gravity.y;
        } else {
            mp.gravity.y = sp.gravity.y;
        }
        ImGui::Checkbox("Multi-Scale Gravity", &mp.multi_scale);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Each material feels gravity scaled to its intended physical size. Turn this off for a uniform world scale.");
        }
        ImGui::SliderFloat("Ambient Temp", &ac.ambient_temp, 50.0f, 800.0f, "%.0f K");
        ImGui::SliderFloat("Buoyancy", &ac.buoyancy_alpha, 0.0f, 2.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How strongly hot gas rises through the current scene.");
        ImGui::Checkbox("Show Heat Gizmos", &g_show_heat_gizmos);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Turns the authored scene heat source visualization on or off. When hidden, the authored heat sources are also disabled.");
        }
    }

    if (ImGui::CollapsingHeader("Assist Brushes", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Wind Radius", &g_blow_radius, 0.15f, 1.50f, "%.2f");
        ImGui::SliderFloat("Wind Strength", &g_blow_strength, 0.50f, 20.0f, "%.1f");
        ImGui::TextDisabled("Hold B + LMB to blow air. Hold G/H/M for heat, cooling, and the temporary magnet brush.");
    }

    ImGui::End();
    pop_panel_style();
}

static void draw_presets_window(ImVec2 pos, ImVec2 size, ImVec4 accent) {
    if (!g_show_presets_window) return;

    push_panel_style(accent);
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    if (!ImGui::Begin("Presets", &g_show_presets_window)) {
        ImGui::End();
        pop_panel_style();
        return;
    }

    if (ImGui::CollapsingHeader("Runtime Presets", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Automata Test")) {
            apply_automata_test_preset();
        }
        ImGui::TextDisabled("Turns on Bio + Continuous Automata, switches Air View to Automata Drive, and switches MPM View to Automata Drive so the steering field is readable immediately.");
    }

    if (ImGui::CollapsingHeader("Scene Shortcuts", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("SDF Junction")) {
            g_scene = ng::SceneID::THERMAL_VERIFY_SDF_JUNCTION;
            reload_scene();
        }
        ImGui::SameLine();
        if (ImGui::Button("Hot Blocks")) {
            g_scene = ng::SceneID::THERMAL_VERIFY_HOT_BLOCKS;
            reload_scene();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cross Ignition")) {
            g_scene = ng::SceneID::THERMAL_VERIFY_CROSS_IGNITION;
            reload_scene();
        }
        if (ImGui::Button("Bridge Witness")) {
            g_scene = ng::SceneID::THERMAL_VERIFY_BRIDGE_WITNESS;
            reload_scene();
        }
        ImGui::SameLine();
        if (ImGui::Button("Impact Ringdown")) {
            g_scene = ng::SceneID::THERMAL_VERIFY_IMPACT_RINGDOWN;
            reload_scene();
        }
        ImGui::TextDisabled("Thermal verify pack: explicit pass/fail benches for cooldown, cross-ignition, conductive bridges, and post-impact hotspot decay.");

        if (ImGui::Button("Morphogenesis Bench")) {
            g_scene = ng::SceneID::MORPHOGENESIS_BENCH;
            reload_scene();
        }
        ImGui::SameLine();
        if (ImGui::Button("Root Garden")) {
            g_scene = ng::SceneID::ROOT_GARDEN_BENCH;
            reload_scene();
        }
        if (ImGui::Button("Cell Colony")) {
            g_scene = ng::SceneID::CELL_COLONY_BENCH;
            reload_scene();
        }
        ImGui::SameLine();
        if (ImGui::Button("Fire Regrowth")) {
            g_scene = ng::SceneID::AUTOMATA_FIRE_REGROWTH_BENCH;
            reload_scene();
        }
        if (ImGui::Button("Air Coupling")) {
            g_scene = ng::SceneID::AUTOMATA_AIR_COUPLING_BENCH;
            reload_scene();
        }
        ImGui::SameLine();
        if (ImGui::Button("Max Coupling")) {
            g_scene = ng::SceneID::AUTOMATA_MAX_COUPLING_BENCH;
            reload_scene();
        }
        ImGui::SameLine();
        if (ImGui::Button("Ash Regrowth")) {
            g_scene = ng::SceneID::ASH_REGROWTH_BENCH;
            reload_scene();
        }
        ImGui::SameLine();
        if (ImGui::Button("Foot Demo")) {
            g_scene = ng::SceneID::FOOT_DEMO_BENCH;
            reload_scene();
        }
        if (ImGui::Button("Hybrid Regrowth")) {
            g_scene = ng::SceneID::HYBRID_REGROWTH_WALL;
            reload_scene();
        }
        ImGui::SameLine();
        if (ImGui::Button("Hybrid Kiln")) {
            g_scene = ng::SceneID::HYBRID_KILN_PROCESS;
            reload_scene();
        }
        ImGui::SameLine();
        if (ImGui::Button("Soft HEAT Range")) {
            g_scene = ng::SceneID::HYBRID_SOFT_HEAT_RANGE;
            reload_scene();
        }
        if (ImGui::Button("Pressure Pottery")) {
            g_scene = ng::SceneID::HYBRID_PRESSURE_POTTERY;
            reload_scene();
        }
        ImGui::SameLine();
        if (ImGui::Button("Ferro Splash")) {
            g_scene = ng::SceneID::HYBRID_FERRO_SPLASH;
            reload_scene();
        }
        ImGui::SameLine();
        if (ImGui::Button("Oobleck Armor")) {
            g_scene = ng::SceneID::HYBRID_OOBLECK_ARMOR;
            reload_scene();
        }
        ImGui::TextDisabled("These benches are tuned to make the air field and particle response easier to read before manual tweaking.");
    }

    ImGui::End();
    pop_panel_style();
}

static void draw_interaction_window(ImVec2 pos, ImVec2 size, ImVec4 accent) {
    if (!g_show_interaction_window) return;

    push_panel_style(accent);
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    if (!ImGui::Begin("Interaction", &g_show_interaction_window)) {
        ImGui::End();
        pop_panel_style();
        return;
    }

    ng::MPMParams& mp = g_mpm.params();

    ImGui::Text("Selection: %s", g_selection_mode ? "ON (X)" : "OFF (X)");
    ImGui::Text("Current Mode: %s", persistent_mode_label(g_mode));
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Active Tool", ImGuiTreeNodeFlags_DefaultOpen)) {
        static const char* tnames[] = {
            "1 Pull / Push",
            "2 Spring Drag",
            "3 Telekinesis",
            "4 Sweep Drag",
            "5 Projectile Launcher",
            "6 Weapon Launcher",
            "7 Draw Wall",
            "8 Erase Wall",
            "9 Foot Control"
        };
        static const InteractMode tool_modes[] = {
            InteractMode::PUSH,
            InteractMode::SPRING_DRAG,
            InteractMode::DRAG,
            InteractMode::SWEEP_DRAG,
            InteractMode::DROP_BALL,
            InteractMode::LAUNCHER2,
            InteractMode::DRAW_WALL,
            InteractMode::ERASE_WALL,
            InteractMode::FOOT_CONTROL
        };
        const int kToolCount = 9;
        int tool = 0;
        for (int i = 0; i < kToolCount; ++i) {
            if (g_mode == tool_modes[i]) { tool = i; break; }
        }
        if (ImGui::Combo("Tool Mode", &tool, tnames, kToolCount))
            g_mode = tool_modes[tool];

        auto draw_main_radius_slider = [&]() {
            ng::f32 old_tool_radius = g_tool_radius;
            ImGui::SliderFloat("Inner Radius", &g_tool_radius, 0.15f, 1.25f, "%.2f");
            if (std::abs(g_tool_radius - old_tool_radius) > 1e-4f) {
                if (g_mode == InteractMode::PUSH || g_mode == InteractMode::DRAG || g_mode == InteractMode::SPRING_DRAG) {
                    if (g_drag_falloff_radius < g_tool_radius || std::abs(g_drag_falloff_radius - old_tool_radius) < 1e-4f) {
                        g_drag_falloff_radius = g_tool_radius;
                    }
                } else {
                    sync_drag_falloff_radius();
                }
            }
        };

        switch (g_mode) {
        case InteractMode::PUSH:
            draw_main_radius_slider();
            ImGui::SliderFloat("Outer Falloff", &g_drag_falloff_radius, g_tool_radius, 1.75f, "%.2f");
            sync_drag_falloff_radius();
            ImGui::SliderFloat("Push / Pull Strength", &g_tool_force, 10.0f, 180.0f, "%.0f");
            ImGui::TextDisabled("Mode 1 keeps the force radial: LMB contracts toward the inner ring, RMB expels outward.");
            break;
        case InteractMode::SPRING_DRAG:
            draw_main_radius_slider();
            ImGui::SliderFloat("Spring Falloff", &g_drag_falloff_radius, g_tool_radius, 1.75f, "%.2f");
            sync_drag_falloff_radius();
            ImGui::SliderFloat("Spring Force", &g_spring_force, 20.0f, 220.0f, "%.0f");
            ImGui::SliderFloat("Spring Damping", &g_spring_damping, 2.0f, 48.0f, "%.1f");
            ImGui::TextDisabled("Mode 2 captures a patch, then drags a shadow copy so the selection translates instead of collapsing.");
            break;
        case InteractMode::DRAG:
            draw_main_radius_slider();
            ImGui::SliderFloat("Telekinesis Falloff", &g_drag_falloff_radius, g_tool_radius, 1.75f, "%.2f");
            sync_drag_falloff_radius();
            ImGui::SliderFloat("Telekinesis Force", &g_drag_force, 10.0f, 220.0f, "%.0f");
            ImGui::TextDisabled("Mode 3 captures a patch and translates it toward the cursor as a moving region.");
            break;
        case InteractMode::SWEEP_DRAG:
            draw_main_radius_slider();
            ImGui::SliderFloat("Sweep Inner Ratio", &g_drag_inner_ratio, 0.10f, 0.90f, "%.2f");
            ImGui::SliderFloat("Sweep Strength", &g_drag_force, 10.0f, 220.0f, "%.0f");
            ImGui::TextDisabled("Mode 4 drags along the cursor motion direction. Inner circle is full pull, outer ring fades to zero.");
            break;
        case InteractMode::DROP_BALL: {
            ImGui::Combo("Projectile Preset", &g_ball_preset, projectile_preset_names, kProjectilePresetCount);
            ProjectilePresetDesc ui_preset = current_projectile_preset();
            if (ui_preset.kind != ProjectilePresetDesc::Kind::SINGLE) {
                ImGui::Checkbox("Auto-Arm on Launch", &g_projectile_auto_arm);
                ImGui::TextDisabled(g_projectile_auto_arm
                                        ? "On: the fuse path starts itself after launch."
                                        : "Off: bombs spawn inert; place them, then heat them to arm.");
            }
            ImGui::Combo("Projectile Shape", &g_ball_shape, projectile_shape_names, 4);
            ImGui::SliderFloat("Ball Radius", &g_ball_radius, 0.05f, 0.45f, "%.2f");
            ImGui::SliderFloat("Ball Weight", &g_ball_weight, 0.5f, 16.0f, "%.1f");
            ImGui::SliderFloat("Ball Stiffness", &g_ball_stiffness, 10000.0f, 60000.0f, "%.0f");
            ImGui::SliderFloat("Min Launch Speed", &g_ball_min_launch_speed, 0.02f, 8.0f, "%.2f");
            ImGui::SliderFloat("Launch Gain", &g_ball_launch_gain, 2.0f, 18.0f, "%.1f");
            ImGui::SliderFloat("Cone Angle", &g_ball_cone_deg, 0.0f, 65.0f, "%.1f deg");

            ng::f32 current_speed = glm::clamp(glm::length(current_projectile_vector()) * glm::max(g_ball_launch_gain, 0.1f),
                                               g_ball_min_launch_speed, 42.0f);
            ng::f32 guide_speed = 0.50f * glm::max(g_ball_launch_gain, 0.1f);
            ImGui::Text("Current Launch Speed: %.2f", current_speed);
            ImGui::TextDisabled("Guide saturates near %.2f speed. Beyond that the beam shifts blue -> yellow -> red.", guide_speed);
            ImGui::TextDisabled("%s", ui_preset.summary);
            ImGui::TextDisabled("Mode 5: RMB drag aims, Alt+RMB adjusts cone width, and LMB fires from the cursor.");
            break;
        }
        case InteractMode::LAUNCHER2: {
            static const char* launcher2_names[] = {
                "Time Bomb",
                "Rocket",
                "Claymore",
                "Rocket + Payload",
                "Rocket + Side Claymores",
                "Claymore + Cluster",
                "Side Burst (Contact)",
                "Gravity Penetrator (Contact)",
                "Concussion Charge (Contact)"
            };
            ImGui::Combo("Weapon", &g_launcher2_preset, launcher2_names, 9);
            ProjectilePresetDesc ui_preset = current_projectile_preset();
            if (ui_preset.kind != ProjectilePresetDesc::Kind::SINGLE) {
                ImGui::Checkbox("Auto-Arm on Launch", &g_projectile_auto_arm);
                ImGui::TextDisabled(g_projectile_auto_arm
                                        ? "On: the fuse path starts itself after launch."
                                        : "Off: weapons spawn inert; place them, then heat them to arm.");
            }
            ImGui::Combo("Projectile Shape", &g_ball_shape, projectile_shape_names, 4);
            ImGui::SliderFloat("Ball Radius", &g_ball_radius, 0.05f, 0.45f, "%.2f");
            ImGui::SliderFloat("Ball Weight", &g_ball_weight, 0.5f, 16.0f, "%.1f");
            ImGui::SliderFloat("Min Launch Speed", &g_ball_min_launch_speed, 0.02f, 8.0f, "%.2f");
            ImGui::SliderFloat("Launch Gain", &g_ball_launch_gain, 2.0f, 18.0f, "%.1f");
            ImGui::SliderFloat("Cone Angle", &g_ball_cone_deg, 0.0f, 65.0f, "%.1f deg");

            ng::f32 current_speed = glm::clamp(glm::length(current_projectile_vector()) * glm::max(g_ball_launch_gain, 0.1f),
                                               g_ball_min_launch_speed, 42.0f);
            ImGui::Text("Current Launch Speed: %.2f", current_speed);
            ImGui::TextDisabled("%s", ui_preset.summary);
            ImGui::TextDisabled("Mode 6: RMB drag aims, Alt+RMB adjusts cone width, and LMB fires.");
            break;
        }
        case InteractMode::DRAW_WALL:
            ImGui::TextDisabled("Mode 7 draws temporary SDF walls directly into the scene.");
            break;
        case InteractMode::ERASE_WALL:
            ImGui::TextDisabled("Mode 8 erases temporary SDF walls under the cursor.");
            break;
        case InteractMode::FOOT_CONTROL: {
            static const char* focus_names[] = {
                "Ankle / Heel Root",
                "Heel Pad",
                "Ball of Foot",
                "All Toes",
                "Big Toe",
                "Toe 2",
                "Toe 3",
                "Toe 4",
                "Pinky Toe"
            };
            int focus = static_cast<int>(ng::foot_demo_focus());
            if (ImGui::Combo("Foot Focus", &focus, focus_names, 9)) {
                ng::set_foot_demo_focus(static_cast<ng::FootControlFocus>(focus));
            }
            ImGui::TextWrapped("%s", ng::foot_demo_mode_hint());
            if (!ng::foot_demo_active()) {
                ImGui::TextDisabled("Load Foot Demo Bench to activate the anatomical calf + foot rig.");
            }
            break;
        }
        default:
            break;
        }
    }

    if (ImGui::CollapsingHeader("Transient Tools", ImGuiTreeNodeFlags_DefaultOpen)) {
        ng::f32 old_magnet_radius = mp.magnet_radius;
        ImGui::SliderFloat("Magnet Inner", &mp.magnet_radius, 0.10f, 1.25f, "%.2f");
        if (std::abs(mp.magnet_radius - old_magnet_radius) > 1e-4f) {
            if (mp.magnet_falloff_radius < mp.magnet_radius || std::abs(mp.magnet_falloff_radius - old_magnet_radius) < 1e-4f) {
                mp.magnet_falloff_radius = mp.magnet_radius;
            }
        }
        ImGui::SliderFloat("Magnet Falloff", &mp.magnet_falloff_radius, mp.magnet_radius, 1.75f, "%.2f");
        sync_magnet_falloff_radius();
        ImGui::SliderFloat("Magnet Strength", &g_magnet_strength, 0.1f, 20.0f, "%.2f");
        ImGui::TextDisabled("Hold G/H/M for heat, cooling, and the temporary magnet brush. Ctrl+wheel adjusts the inner radius, Alt+wheel adjusts falloff.");
    }

    if (ImGui::CollapsingHeader("Mode Help", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextColored(ImVec4(1.0f, 0.92f, 0.58f, 1.0f), "SPACE");
        ImGui::SameLine();
        ImGui::TextUnformatted("toggles the creation menu.");
        ImGui::TextColored(ImVec4(0.72f, 0.90f, 1.0f, 1.0f), "X");
        ImGui::SameLine();
        ImGui::TextUnformatted("toggles selection mode.");
        ImGui::TextColored(ImVec4(0.72f, 0.90f, 1.0f, 1.0f), "Mouse wheel / MMB");
        ImGui::SameLine();
        ImGui::TextUnformatted("zoom and pan the camera.");
        if (g_mode == InteractMode::PUSH) {
            ImGui::BulletText("LMB pulls inward and RMB pushes outward.");
        } else if (g_mode == InteractMode::SPRING_DRAG) {
            ImGui::BulletText("LMB captures a region, then drags its shadow copy.");
        } else if (g_mode == InteractMode::DRAG) {
            ImGui::BulletText("LMB captures a region and translates it toward the cursor.");
        } else if (g_mode == InteractMode::SWEEP_DRAG) {
            ImGui::BulletText("LMB sweeps in the cursor motion direction with a soft falloff.");
        } else if (g_mode == InteractMode::DROP_BALL) {
            ImGui::BulletText("RMB drags the shot vector, Alt+RMB widens the cone, and LMB fires.");
        } else if (g_mode == InteractMode::LAUNCHER2) {
            ImGui::BulletText("Three-weapon launcher: time bomb, rocket, claymore.");
            ImGui::BulletText("RMB drags the shot vector, Alt+RMB widens the cone, LMB fires.");
        } else if (g_mode == InteractMode::FOOT_CONTROL) {
            ImGui::BulletText("LMB drags the focused anatomy target.");
            ImGui::BulletText("Hold RMB to pin the ankle while you curl or place the forefoot and toes.");
            ImGui::BulletText("[ / ] cycle focus. Wheel adds fine curl. Z/S curl and straighten. C/B contract and extend.");
        } else if (g_mode == InteractMode::DRAW_WALL) {
            ImGui::BulletText("LMB paints temporary SDF walls.");
        } else if (g_mode == InteractMode::ERASE_WALL) {
            ImGui::BulletText("LMB erases temporary SDF walls.");
        }
    }

    ImGui::End();
    pop_panel_style();
}

static void draw_backends_window(ImVec2 pos, ImVec2 size, ImVec4 accent) {
    if (!g_show_backends_window) return;

    push_panel_style(accent);
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    if (!ImGui::Begin("Backends", &g_show_backends_window)) {
        ImGui::End();
        pop_panel_style();
        return;
    }

    ng::SPHParams& sp = const_cast<ng::SPHParams&>(g_sph.params());
    ng::MPMParams& mp = g_mpm.params();
    auto& ac = g_air.config();

    if (ImGui::CollapsingHeader("SPH", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Gas Constant", &sp.gas_constant, 1.0f, 50.0f);
        ImGui::SliderFloat("Viscosity", &sp.viscosity, 0.0f, 1.0f, "%.3f");
        ImGui::SliderFloat("XSPH", &sp.xsph, 0.0f, 0.5f);
        ImGui::SliderFloat("Surface Tension", &sp.surface_tension, 0.0f, 3.0f, "%.2f");
        ImGui::Checkbox("Immiscible Interfaces", &sp.immiscible_interfaces);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Adds unlike-liquid interface tension and reduced cross-mixing so oil/water layers stay sharper.");
        }
        if (sp.immiscible_interfaces) {
            ImGui::SliderFloat("Interface Repulsion", &sp.interface_repulsion, 0.0f, 48.0f, "%.1f");
            ImGui::SliderFloat("Interface Tension", &sp.interface_tension, 0.0f, 3.0f, "%.2f");
            ImGui::SliderFloat("Cross Mix", &sp.cross_mix, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Cross Thermal", &sp.cross_thermal_mix, 0.0f, 1.0f, "%.2f");
        }
        ImGui::SliderFloat("MPM Contact Push", &sp.mpm_contact_push, 0.0f, 36.0f, "%.1f");
        ImGui::SliderFloat("MPM Contact Damp", &sp.mpm_contact_damping, 0.0f, 10.0f, "%.2f");
        ImGui::SliderFloat("MPM Contact Recover", &sp.mpm_contact_recovery, 0.0f, 0.45f, "%.2f");
        ImGui::Checkbox("Codim SPH", &sp.codim_enabled);
        if (sp.codim_enabled)
            ImGui::SliderFloat("Codim Threshold", &sp.codim_threshold, 0.05f, 0.5f);
    }

    if (ImGui::CollapsingHeader("MPM", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Young's Modulus", &mp.youngs_modulus, 500.0f, 50000.0f, "%.0f");
        ImGui::SliderFloat("Poisson", &mp.poisson_ratio, 0.05f, 0.45f);
        ImGui::SliderFloat("Fiber Strength", &mp.fiber_strength, 0.0f, 10.0f, "%.1f");
        ImGui::SliderFloat("Fracture Threshold", &mp.fracture_threshold, 0.005f, 0.1f, "%.3f");
        ImGui::SliderFloat("Fracture Rate", &mp.fracture_rate, 0.1f, 20.0f);
        ImGui::SliderFloat("Melt Temp", &mp.melt_temp, 200.0f, 1000.0f, "%.0f");
        ImGui::SliderFloat("Melt Range", &mp.melt_range, 10.0f, 200.0f, "%.0f");
        ImGui::SliderFloat("Latent Heat", &mp.latent_heat, 0.0f, 500.0f, "%.0f");
        ImGui::SliderFloat("Material Cooling", &mp.particle_cooling_rate, 0.0f, 0.08f, "%.3f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Global cooling to ambient for thermal MPM materials. Lower values keep things burning longer.");
        ImGui::Checkbox("Thermal", &mp.enable_thermal);
        if (mp.enable_thermal) ImGui::SliderFloat("Heat Source Temp", &mp.heat_source_temp, 0.0f, 1500.0f, "%.0f");
    }

    if (ImGui::CollapsingHeader("Gas", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Air Viscosity", &ac.air_viscosity, 0.0f, 0.1f, "%.3f");
        ImGui::SliderFloat("Smoke Decay", &ac.smoke_decay, 0.9f, 1.0f, "%.3f");
        ImGui::SliderFloat("Heat Conduction", &ac.thermal_diffusivity, 0.0f, 0.05f, "%.4f");
        ImGui::SliderFloat("Heat Loss Rate", &ac.cooling_rate, 0.0f, 2.0f, "%.2f");
        ImGui::Checkbox("Physically Based Heat", &ac.physically_based_heat);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Uses conservative diffusion coefficients so stronger conduction does not numerically create extra heat.");
        ImGui::Combo("Grid Resolution", &g_euler_res_idx, euler_res_names, 4);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Press R to apply the new resolution.");
        ImGui::TextDisabled("Current: %dx%d (%.4f m/cell)", g_air.resolution().x, g_air.resolution().y, g_air.dx());
    }

    if (ImGui::CollapsingHeader("SDF", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Heat Transfer", &ac.solid_thermal_diffusivity, 0.0f, 0.05f, "%.4f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How fast heat spreads inside rigid SDF solids.");
        ImGui::SliderFloat("Heat Sink", &ac.solid_heat_capacity, 0.25f, 12.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Higher values hold more heat and change temperature more slowly.");
        ImGui::SliderFloat("Dissipation", &ac.solid_contact_transfer, 0.0f, 0.05f, "%.4f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How strongly hot SDF solids give heat to nearby air and touching materials.");
        ImGui::SliderFloat("Heat Leak", &ac.solid_heat_loss, 0.0f, 0.2f, "%.4f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Extra non-physical heat disappearance inside SDF solids.");
    }

    ImGui::End();
    pop_panel_style();
}

static void draw_appearance_window(ImVec2 pos, ImVec2 size, ImVec4 accent) {
    if (!g_show_appearance_window) return;

    push_panel_style(accent);
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    if (!ImGui::Begin("Appearance", &g_show_appearance_window)) {
        ImGui::End();
        pop_panel_style();
        return;
    }

    if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
        ng::SPHParams& sp = const_cast<ng::SPHParams&>(g_sph.params());
        ng::MPMParams& mp = g_mpm.params();
        ImGui::Combo("Color Mode", &g_color_mode, color_mode_names, 2);
        if (g_color_mode == 0) ImGui::TextDisabled("V/N cycles SPH/MPM debug views.");
        else ImGui::TextDisabled("Batch Colors shows each placed batch with its authored palette.");

        ImGui::Combo("SPH View", &sp.vis_mode, sph_vis_names, 8);
        if (ImGui::IsItemHovered()) {
            const char* sph_vis_tips[] = {
                "Blue water coloring with depth gradient",
                "Speed magnitude (viridis colormap)",
                "Pressure: blue=low, red=high (cool-warm)",
                "Density ratio rho/rho0 (viridis)",
                "Vorticity/curl: blue=CW, red=CCW",
                "Divergence: blue=converging, red=diverging",
                "Codim detection: red=1D thread, blue=2D bulk",
                "Temperature: blue=cool, orange=hot"
            };
            ImGui::SetTooltip("%s", sph_vis_tips[sp.vis_mode]);
        }

        ImGui::Combo("MPM View", &mp.vis_mode, mpm_vis_names, kMpmVisModeCount);
        if (ImGui::IsItemHovered()) {
            const char* mpm_vis_tips[] = {
                "Color by material type (elastic=green, snow=white, etc)",
                "Speed magnitude (viridis colormap)",
                "Stress deviation ||F-R||: how far from equilibrium",
                "Deformation gradient norm ||F||: total deformation",
                "Damage/burn/crack fraction (black=intact, red=damaged)",
                "Temperature (blue=cold, yellow=warm, red=hot)",
                "Plastic deformation Jp (snow hardening history)",
                "Compression / density proxy from J = det(F). Hot colors mean denser or more compressed regions.",
                "Packed RGB view: R=temperature, G=stress, B=damage.",
                "Packed RGB state: R=damage/hardening, G=phase or memory, B=plasticity/compression.",
                "Latent SmoothLife field sampled at each particle. This is the scalar colony occupancy itself, not the force. Bright rims mark active colony fronts and higher occupancy.",
                "Bio drive debug: R=rightward pull, G=upward pull, B=total force from the reaction-diffusion scout/activator field plus its support frontier. Mid gray means little or no steering.",
                "Automata drive debug: R=rightward pull, G=upward pull, B=total force from the continuous SmoothLife-style colony field plus its support frontier. Mid gray means little or no steering.",
                "Layer bands view: uses the authored particle colors directly, so layered feet or layered props can show core, pad, shell, and other constructed bands without switching to Batch Colors."
            };
            ImGui::SetTooltip("%s", mpm_vis_tips[mp.vis_mode]);
        }

        ImGui::Combo("Air View", &g_air_vis, air_vis_names, 12);
        ImGui::SliderFloat("Fire Start Temp", &g_fire_vis_start_temp, 260.0f, 700.0f, "%.0f K");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Visualization threshold for the fire overlay. Lower values make dimmer hot gas show up as flame.");
        ImGui::SliderFloat("Fire Temp Range", &g_fire_vis_temp_range, 80.0f, 900.0f, "%.0f K");
        ImGui::SliderFloat("Fire Softness", &g_fire_vis_softness, 0.5f, 8.0f, "%.2f");
        if (g_air_vis == 8) {
            ImGui::TextDisabled("Bio field view shows the composite scout + activator field on the air grid. The scout component can lead into open air before the blob follows.");
        } else if (g_air_vis == 9) {
            ImGui::TextDisabled("Automata colony view shows the continuous SmoothLife-style occupancy field on the air grid.");
        } else if (g_air_vis == 10) {
            ImGui::TextDisabled("Bio drive view: R=rightward pull, G=upward pull, B=strength. The more saturated the blue channel, the stronger the scout/edge field wants nearby particles to move.");
        } else if (g_air_vis == 11) {
            ImGui::TextDisabled("Automata drive view: R=rightward pull, G=upward pull, B=strength. Use this with Automata Test to verify the air colony is steering instead of only coloring.");
        }
    }

    if (ImGui::CollapsingHeader("Render / Appearance", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Combo("Metal Palette", &g_sdf_palette, sdf_palette_names, 4);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Visual tint for rigid SDF solids.");
        ImGui::Checkbox("SPH Surface", &g_metaball.enabled);
        ImGui::Checkbox("MPM Skin [new]", &g_mpm_skin_enabled);
        ImGui::Checkbox("Keep Points Under Skin", &g_metaball.keep_particles);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Draw the old particle view under the skin pass for debugging or mixed looks.");
        }
        if (g_metaball.enabled) {
            ImGui::Combo("SPH Style", &g_sph_surface_style, surface_style_names, kSurfaceStyleCount);
            ImGui::SliderFloat("SPH Threshold", &g_metaball.threshold, 0.05f, 3.0f);
            ImGui::SliderFloat("SPH Kernel", &g_metaball.kernel_scale, 1.0f, 6.0f);
        }
        if (g_mpm_skin_enabled) {
            ImGui::Combo("MPM Skin Style", &g_mpm_surface_style, surface_style_names, kSurfaceStyleCount);
            ImGui::SliderFloat("MPM Threshold", &g_mpm_skin_threshold, 0.05f, 2.0f);
            ImGui::SliderFloat("MPM Kernel", &g_mpm_skin_kernel, 1.0f, 5.0f);
        }
        if (g_metaball.enabled || g_mpm_skin_enabled) {
            ImGui::SliderFloat("Skin Softness", &g_metaball.edge_softness, 0.4f, 4.0f, "%.2f");
            ImGui::SliderFloat("Skin Gloss", &g_metaball.gloss, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Skin Rim", &g_metaball.rim, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Skin Opacity", &g_metaball.opacity, 0.20f, 1.0f, "%.2f");
            ImGui::TextDisabled("Field Matte / Contour / Soft Fill / Thin Contour smooth the interior while keeping a cleaner silhouette.");
        }
    }

    if (ImGui::CollapsingHeader("Overlays / Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Drag Debug (arrows)", &g_show_drag_debug);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show per-particle drag arrows in spring-drag mode. May be slow with many particles.");
        ImGui::Checkbox("Magnetic Debug", &g_show_magnetic_debug);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show the magnetic field overlay. Scene-driven H views are the ones magnetic MPMs actually read.");
        if (g_show_magnetic_debug) {
            auto& mag_cfg = g_magnetic.params();
            ImGui::Combo("Magnetic View", &g_magnetic_debug_view, magnetic_debug_view_names, 9);
            ImGui::Checkbox("Force Solver Always", &mag_cfg.debug_force_active);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Runs the magnetic solver every frame so scene-wide fields stay visible without holding M.");
        }
        ImGui::Checkbox("Pipeline Viewer", &g_show_pipeline);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Interactive diagram of all physics passes and their couplings.");
    }

    ImGui::End();
    pop_panel_style();
}

static void draw_advanced_window(ImVec2 pos, ImVec2 size, ImVec4 accent) {
    if (!g_show_advanced_window) return;

    push_panel_style(accent);
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    if (!ImGui::Begin("Advanced", &g_show_advanced_window)) {
        ImGui::End();
        pop_panel_style();
        return;
    }

    ng::MPMParams& mp = g_mpm.params();
    auto& ac = g_air.config();

    if (ImGui::CollapsingHeader("Bake / Thickness [new]", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("2.5D Bake/Kiln Support", &mp.pseudo_25d_enabled);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Adds pseudo-thickness, shell support, closed-cell retention, and split resistance so porous bake and pottery behaviors read more like thick 3D bodies.");
        }
        if (mp.pseudo_25d_enabled) {
            ImGui::SliderFloat("2.5D Thickness", &mp.pseudo_25d_depth, 0.0f, 2.0f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Higher values retain puff and heat more like a thicker hidden body.");
            ImGui::SliderFloat("Shell Support", &mp.pseudo_25d_shell_support, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("Closed-Cell Hold", &mp.pseudo_25d_enclosure, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("Split Resistance", &mp.pseudo_25d_cohesion, 0.0f, 2.0f, "%.2f");
        }
    }

    if (ImGui::CollapsingHeader("Combustion / Vapor [new]", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Steam Generation", &ac.vapor_generation, 0.0f, 3.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How strongly hot materials emit vapor into the air grid.");
        ImGui::SliderFloat("Steam Pressure", &ac.vapor_pressure, 0.0f, 12.0f, "%.2f");
        ImGui::SliderFloat("Steam Buoyancy", &ac.vapor_buoyancy, 0.0f, 4.0f, "%.2f");
        ImGui::SliderFloat("Steam Drag", &ac.vapor_drag, 0.0f, 4.0f, "%.2f");
        ImGui::SliderFloat("Steam Decay", &ac.vapor_decay, 0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Latent Cooling", &ac.latent_cooling, 0.0f, 1.5f, "%.2f");
        ImGui::Separator();
        ImGui::SliderFloat("Combustion Boost", &ac.combustion_heat_boost, 0.0f, 3.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Extra hot-gas injection from actively burning materials so fire can grow into a visible flame pocket.");
        ImGui::SliderFloat("Flame Hold", &ac.combustion_hold, 0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How much hot smoky combustion cells resist cooling so flames stay volumetric longer.");
    }

    if (ImGui::CollapsingHeader("Bio / Automata [new][experimental]", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable Bio Field", &ac.bio_enabled);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Runs a Gray-Scott-style reaction-diffusion field on the air grid. Bio/cooking materials seed it, and selected MPM materials read it back as growth, healing, pore formation, and budding cues.");
        }
        ImGui::SliderFloat("Pattern Speed", &ac.bio_pattern_speed, 0.1f, 8.0f, "%.2f");
        ImGui::SliderFloat("Feed", &ac.bio_feed, 0.010f, 0.090f, "%.3f");
        ImGui::SliderFloat("Kill", &ac.bio_kill, 0.030f, 0.090f, "%.3f");
        ImGui::SliderFloat("Diffuse A", &ac.bio_diffuse_a, 0.02f, 0.30f, "%.3f");
        ImGui::SliderFloat("Diffuse B", &ac.bio_diffuse_b, 0.01f, 0.20f, "%.3f");
        ImGui::SliderFloat("Particle Seeding", &ac.bio_seed_strength, 0.0f, 4.0f, "%.2f");
        ImGui::SliderFloat("Physics Coupling", &ac.bio_coupling, 0.0f, 6.0f, "%.2f");
        ImGui::SliderFloat("Regrowth Rate [new]", &ac.bio_regrowth_rate, 0.0f, 16.0f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Global multiplier for living-material recovery. The upper end is intentionally extreme and now ramps nonlinearly, so high values should make ash-regrowth and related bio materials recover and overgrow much faster.");
        }
        ImGui::TextDisabled("Good starting range for self-duplicating spots: feed ~0.03-0.04, kill ~0.055-0.065. Higher speed/coupling ranges are now intentionally aggressive.");

        ImGui::Separator();
        ImGui::Checkbox("Enable Continuous Automata", &ac.automata_enabled);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Adds a separate continuous automata substrate inspired by SmoothLife. This one is meant for colony growth, budding sheets, and self-organizing tissue patterns instead of just texture-like chemical spots.");
        }
        ImGui::SliderFloat("Automata Speed", &ac.automata_pattern_speed, 0.1f, 8.0f, "%.2f");
        ImGui::SliderFloat("Birth Low", &ac.automata_birth_lo, 0.05f, 0.60f, "%.3f");
        ImGui::SliderFloat("Birth High", &ac.automata_birth_hi, 0.10f, 0.75f, "%.3f");
        ImGui::SliderFloat("Survive Low", &ac.automata_survive_lo, 0.05f, 0.60f, "%.3f");
        ImGui::SliderFloat("Survive High", &ac.automata_survive_hi, 0.10f, 0.80f, "%.3f");
        ImGui::SliderFloat("Inner Radius", &ac.automata_inner_radius, 0.75f, 6.0f, "%.2f cells");
        ImGui::SliderFloat("Outer Radius", &ac.automata_outer_radius, 1.25f, 9.0f, "%.2f cells");
        if (ac.automata_outer_radius < ac.automata_inner_radius + 0.25f) {
            ac.automata_outer_radius = ac.automata_inner_radius + 0.25f;
        }
        ImGui::SliderFloat("Smoothness", &ac.automata_sigmoid, 0.010f, 0.180f, "%.3f");
        ImGui::SliderFloat("Automata Seeding", &ac.automata_seed_strength, 0.0f, 4.0f, "%.2f");
        ImGui::SliderFloat("Automata Coupling", &ac.automata_coupling, 0.0f, 6.0f, "%.2f");
        ImGui::TextDisabled("Good starting benches: Morphogenesis, Root Garden, and Cell Colony with Air View = Automata Colony. The top half of the range is intentionally strong.");
    }

    if (ImGui::CollapsingHeader("Magnetics [new][experimental]", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto& mag_cfg = g_magnetic.params();
        ImGui::Checkbox("Real Magnetics", &mag_cfg.enabled);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Rasterizes SDF permanent magnets, iterates induced soft-iron response, solves a 2D magnetic potential, then feeds the solved field into magnetic MPM materials.");
        }
        int cursor_field_idx = static_cast<int>(mag_cfg.cursor_field_type);
        if (ImGui::Combo("Cursor Field", &cursor_field_idx, magnetic_cursor_field_names, 4)) {
            mag_cfg.cursor_field_type = static_cast<ng::MagneticField::CursorFieldType>(cursor_field_idx);
        }
        ImGui::SliderFloat("Source Scale", &mag_cfg.source_scale, 0.5f, 40.0f, "%.2f");
        ImGui::SliderFloat("Force Scale", &mag_cfg.force_scale, 0.1f, 60.0f, "%.2f");
        ImGui::SliderInt("Field Iterations", &mag_cfg.jacobi_iterations, 8, 96);
        ImGui::SliderInt("Induction Passes", &mag_cfg.induction_iterations, 0, 5);
        ImGui::SliderFloat("Rigid Perm Scale", &mag_cfg.rigid_permanent_scale, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Rigid Soft Scale", &mag_cfg.rigid_soft_scale, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled("Hold M for the temporary brush field. Probe Pole, Bar Magnet, Wide Pole, and Horseshoe stay available as quick cursor presets.");
    }

    if (ImGui::CollapsingHeader("Airtight / Pressure [new]", ImGuiTreeNodeFlags_DefaultOpen)) {
        ng::f32 peak_pressure = 0.0f;
        ng::f32 peak_burst = 0.0f;
        for (const PressureVesselRecord& vessel : g_pressure_vessels) {
            peak_pressure = glm::max(peak_pressure, vessel.pressure);
            peak_burst = glm::max(peak_burst, vessel.burst_energy);
        }

        ImGui::Text("Active Pressure Vessels: %d", static_cast<int>(g_pressure_vessels.size()));
        ImGui::Text("Peak Vessel Pressure: %.2f", peak_pressure);
        ImGui::Text("Peak Burst Energy: %.2f", peak_burst);
        ImGui::Text("Airtight Grid: %dx%d (%.4f m/cell)",
                    g_air.airtight_resolution().x,
                    g_air.airtight_resolution().y,
                    g_air.airtight_dx());
        ImGui::TextWrapped("Current milestone path: MPM shell particles still write into the airtight cavity grid, while explicit shell/core pressure vessels add stronger internal gas build-up, directed venting, and hot blast coupling for bomb-style projectiles.");
    }

    ImGui::End();
    pop_panel_style();
}

static void init_systems() {
    SceneSpaceConfig scene_cfg = scene_space_config(g_scene);
    static constexpr ng::f32 kSdfCellSize = 6.0f / 512.0f;
    static constexpr ng::f32 kMagneticCellSize = 6.0f / 256.0f;
    static constexpr ng::f32 kGridCellSize = 6.4f / 192.0f;

    ng::ParticleBuffer::Config pb;
    pb.max_particles = 500000; pb.sph_capacity = 300000;
    pb.mpm_capacity = 100000; pb.spring_capacity = 50000; pb.arcane_capacity = 50000;
    g_particles.init(pb);

    ng::SpatialHash::Config sh;
    sh.table_size = 262144; sh.cell_size = 0.08f;
    sh.world_min = scene_cfg.grid_world_min - ng::vec2(2.0f);
    sh.world_max = scene_cfg.grid_world_max + ng::vec2(2.0f);
    g_hash.init(sh);

    ng::SDFField::Config sc;
    sc.world_min = scene_cfg.sdf_world_min;
    sc.world_max = scene_cfg.sdf_world_max;
    sc.resolution = resolution_from_cell_size(sc.world_min, sc.world_max, kSdfCellSize);
    g_sdf.init(sc);

    ng::MagneticField::Config mc;
    mc.resolution = resolution_from_cell_size(sc.world_min, sc.world_max, kMagneticCellSize);
    mc.world_min = sc.world_min;
    mc.world_max = sc.world_max;
    g_magnetic.init(mc);

    ng::UniformGrid::Config gc;
    gc.world_min = scene_cfg.grid_world_min;
    gc.world_max = scene_cfg.grid_world_max;
    gc.resolution = resolution_from_cell_size(gc.world_min, gc.world_max, kGridCellSize);
    g_mpm_grid.init(gc);

    ng::SPHParams sp;
    sp.smoothing_radius = 0.04f;
    ng::f32 spacing = sp.smoothing_radius * 0.5f;
    sp.rest_density = 1000.0f; sp.particle_mass = sp.rest_density * spacing * spacing;
    sp.gas_constant = 8.0f; sp.viscosity = 0.1f; sp.xsph = 0.3f;
    sp.gravity = ng::vec2(0.0f, scene_gravity_y(g_scene));
    sp.bound_min = scene_cfg.sph_bound_min;
    sp.bound_max = scene_cfg.sph_bound_max;
    g_sph.set_params(sp);
    g_sph.init();

    ng::MPMParams mp;
    mp.youngs_modulus = 40000.0f; mp.poisson_ratio = 0.3f;
    mp.gravity = ng::vec2(0.0f, scene_gravity_y(g_scene));
    g_mpm.set_params(mp);
    g_mpm.init(g_mpm_grid);

    g_particle_renderer.init();
    g_sdf_renderer.init();
    ng::EulerianFluid::Config ac;
    int eres = euler_res_values[g_euler_res_idx];
    ac.resolution = resolution_from_max_dim(scene_cfg.sdf_world_min, scene_cfg.sdf_world_max, eres);
    ac.world_min = scene_cfg.sdf_world_min;
    ac.world_max = scene_cfg.sdf_world_max;
    ac.ambient_temp = scene_ambient_temp(g_scene); ac.buoyancy_alpha = 0.5f;
    g_air.init(ac);

    g_metaball.init(g_wc.width, g_wc.height);
    g_bloom.init(g_wc.width, g_wc.height);
    g_preview_shader.load("shaders/render/preview.vert", "shaders/render/preview.frag");
    g_heat_glow_shader.load("shaders/render/particle_draw.vert", "shaders/render/heat_glow.frag");
    g_magnetic_field_vis_shader.load("shaders/render/fullscreen.vert", "shaders/render/magnetic_field_vis.frag");
    glCreateVertexArrays(1, &g_preview_vao);

    g_camera.set_position(scene_cfg.camera_pos);
    g_camera.set_zoom(scene_cfg.camera_zoom);
}

static void reload_scene() {
    clear_spring_drag();
    ng::clear_foot_demo();
    g_pressure_vessels.clear();
    g_particles.destroy();
    init_systems();
    ng::load_scene(g_scene, g_particles, g_sph, g_mpm, g_mpm_grid, g_sdf, &g_creation);
    apply_scene_runtime_defaults(g_scene);
    g_hover_selection = {};
    g_pinned_selection = {};
    g_pinned_selection_open = false;
    g_camera.set_viewport(g_wc.width, g_wc.height);
}

static void clear_non_sdf_objects() {
    clear_spring_drag();
    ng::clear_foot_demo();
    g_pressure_vessels.clear();
    g_particles.range(ng::SolverType::SPH).count = 0;
    g_particles.range(ng::SolverType::MPM).count = 0;
    g_sph.clear_particles();
    g_mpm.clear_particles();
    g_creation.batches.clear();
    g_creation.highlighted_batch = -1;
    g_creation.batch_counter = 0;
    g_hover_selection = {};
    g_pinned_selection = {};
    g_pinned_selection_open = false;
    auto air_cfg = g_air.config();
    g_air.init(air_cfg);
}

// Draw creation menu UI and return whether it consumed input
static void draw_creation_menu() {
    if (!g_creation.active) return;

    const auto& presets = ng::get_presets();
    auto select_first_matching_preset = [&](int quick_tab) {
        for (int i = 0; i < static_cast<int>(presets.size()); ++i) {
            if (quick_tab == static_cast<int>(ng::PresetQuickTab::ALL) ||
                static_cast<int>(presets[i].quick_tab) == quick_tab) {
                g_creation.select_preset(i);
                return;
            }
        }
    };

    ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 350, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(340, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Create Object", &g_creation.active,
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);

    ImGui::TextDisabled("Right-click to place | Scroll to resize");

    auto wrapped_small_button = [](const char* label, bool active, const ImVec4& active_color) {
        if (active) ImGui::PushStyleColor(ImGuiCol_Button, active_color);
        bool clicked = ImGui::SmallButton(label);
        if (active) ImGui::PopStyleColor();
        return clicked;
    };
    auto maybe_same_line_for_next = [](const char* next_label) {
        const ImGuiStyle& style = ImGui::GetStyle();
        float next_width = ImGui::CalcTextSize(next_label).x + style.FramePadding.x * 2.0f;
        float next_x2 = ImGui::GetItemRectMax().x + style.ItemSpacing.x + next_width;
        float content_max_x = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
        if (next_x2 <= content_max_x) ImGui::SameLine();
    };

    // --- Quick tabs for newer heat-focused content ---
    for (int q = 0; q < ng::QUICK_TAB_COUNT; q++) {
        bool active = (g_creation.quick_tab == q);
        if (wrapped_small_button(ng::quick_tab_names[q], active, ImVec4(0.75f, 0.48f, 0.22f, 1.0f))) {
            g_creation.quick_tab = q;
            if (q != static_cast<int>(ng::PresetQuickTab::ALL)) {
                select_first_matching_preset(q);
            }
        }
        if (q + 1 < ng::QUICK_TAB_COUNT) maybe_same_line_for_next(ng::quick_tab_names[q + 1]);
    }
    ImGui::Separator();

    // --- Category tabs ---
    if (g_creation.quick_tab == static_cast<int>(ng::PresetQuickTab::ALL)) {
        for (int c = 0; c < ng::CATEGORY_COUNT; c++) {
            bool active = (g_creation.category == c);
            if (wrapped_small_button(ng::category_names[c], active, ImVec4(0.3f, 0.5f, 0.8f, 1.0f))) {
                g_creation.category = c;
            }
            if (c + 1 < ng::CATEGORY_COUNT) maybe_same_line_for_next(ng::category_names[c + 1]);
        }
    } else {
        ImGui::TextDisabled("Showing curated new materials. Switch back to All for the full category list.");
    }
    ImGui::Separator();

    // --- Preset list for current category ---
    ImGui::BeginChild("PresetList", ImVec2(0, 150), true);
    for (int i = 0; i < static_cast<int>(presets.size()); i++) {
        const auto& p = presets[i];
        if (g_creation.quick_tab == static_cast<int>(ng::PresetQuickTab::ALL)) {
            if (static_cast<int>(p.category) != g_creation.category) continue;
        } else {
            if (static_cast<int>(p.quick_tab) != g_creation.quick_tab) continue;
        }

        bool selected = (g_creation.preset_index == i);
        ImVec4 col(p.color.r, p.color.g, p.color.b, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, col);
        if (ImGui::Selectable(p.name, selected)) {
            g_creation.select_preset(i);
        }
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered() && p.description) {
            ImGui::SetTooltip("%s", p.description);
        }
    }
    ImGui::EndChild();

    // --- Selected preset info ---
    auto& cp = g_creation.custom;
    ImGui::TextColored(ImVec4(cp.color.r, cp.color.g, cp.color.b, 1), "%s", cp.name);
    if (cp.description) ImGui::TextWrapped("%s", cp.description);
    ImGui::TextDisabled("Try size: %.2f", cp.recommended_size);
    ImGui::TextWrapped("%s", cp.recommended_note);
    ImGui::TextDisabled("Hold Q for a dim recommended-size guide.");

    // --- Techniques used ---
    {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.8f, 0.5f, 1.0f));
        if (cp.solver == ng::SpawnSolver::SPH) {
            ImGui::TextWrapped("SPH: WCSPH pressure, XSPH smoothing, CCD ray-march collision");
            ImGui::TextWrapped("+ Surface tension (CSF), Codimensional adaptive kernels");
            switch (cp.mpm_type) {
            case ng::MPMMaterial::SPH_BURNING_OIL:
                ImGui::TextWrapped("+ Thermal ignition, self-heating oil burn, smoke/vapor emission [experimental]"); break;
            case ng::MPMMaterial::SPH_BOILING_WATER:
                ImGui::TextWrapped("+ Heat diffusion, boil state, steam venting, and bubbling lift [experimental]"); break;
            case ng::MPMMaterial::SPH_THERMAL_SYRUP:
                ImGui::TextWrapped("+ Heat-thinning and cool-thickening syrup flow [experimental]"); break;
            case ng::MPMMaterial::SPH_FLASH_FLUID:
                ImGui::TextWrapped("+ Low-boiling flash vaporization, unstable loft, and rapid evaporation [experimental]"); break;
            default:
                break;
            }
        } else {
            ImGui::TextWrapped("MPM: MLS-MPM (APIC), Fixed corotated constitutive model");
            switch (cp.mpm_type) {
            case ng::MPMMaterial::FLUID:
                ImGui::TextWrapped("+ Volumetric EOS (Kirchhoff), isotropic F reset"); break;
            case ng::MPMMaterial::ELASTIC:
                ImGui::TextWrapped("+ Neo-Hookean elasticity, SVD polar decomposition"); break;
            case ng::MPMMaterial::SNOW:
                ImGui::TextWrapped("+ Stomakhin plasticity, singular value clamping, hardening"); break;
            case ng::MPMMaterial::ANISO:
                ImGui::TextWrapped("+ Anisotropic fiber stress, directional reinforcement"); break;
            case ng::MPMMaterial::THERMAL:
                ImGui::TextWrapped("+ Temperature-dependent E, thermal softening"); break;
            case ng::MPMMaterial::FRACTURE:
                ImGui::TextWrapped("+ Damage accumulation, stress degradation, debris conversion"); break;
            case ng::MPMMaterial::PHASE:
                ImGui::TextWrapped("+ Phase-field (phi), glass transition, latent heat, thermal expansion"); break;
            case ng::MPMMaterial::BURNING:
                ImGui::TextWrapped("+ Combustion model, exothermic self-heating, buoyancy, grid heat transfer"); break;
            case ng::MPMMaterial::EMBER:
                ImGui::TextWrapped("+ Slow-cooling sparks, sustained buoyancy, intense exothermic reaction"); break;
            case ng::MPMMaterial::HARDEN:
                ImGui::TextWrapped("+ Irreversible curing, F annealing, E ramp-up (4x)"); break;
            case ng::MPMMaterial::CERAMIC:
                ImGui::TextWrapped("+ Curing + brittle fracture, 40%% residual stiffness in chunks"); break;
            case ng::MPMMaterial::COMPOSITE:
                ImGui::TextWrapped("+ Fiber + curing + directional fracture along grain"); break;
            case ng::MPMMaterial::BRITTLE:
                ImGui::TextWrapped("+ Compression-resistant brittle solid, tensile/shear fracture only"); break;
            case ng::MPMMaterial::TOUGH:
                ImGui::TextWrapped("+ Tough fracture solid, higher crack threshold, chunk retention"); break;
            case ng::MPMMaterial::GLASS:
                ImGui::TextWrapped("+ Reversible glass transition, hot viscous flow, cool re-hardening"); break;
            case ng::MPMMaterial::BLOOM:
                ImGui::TextWrapped("+ Heat curing + internal expansion + petal-like bursting fracture"); break;
            case ng::MPMMaterial::FLAMMABLE_FLUID:
                ImGui::TextWrapped("+ Fluid EOS + ignition + exothermic burning pool + smoke"); break;
            case ng::MPMMaterial::FOAM:
                ImGui::TextWrapped("+ Super-firm solid that softens and grows porous when heated"); break;
            case ng::MPMMaterial::SPLINTER:
                ImGui::TextWrapped("+ Heat-cures stiff, builds self-stress, then splinters along fiber"); break;
            case ng::MPMMaterial::BREAD:
                ImGui::TextWrapped("+ Heat-generated gas + soft expansion + partial setting + tear-open rupture"); break;
            case ng::MPMMaterial::PUFF_CLAY:
                ImGui::TextWrapped("+ Hot curing + trapped gas swelling + crumbly pressure fracture"); break;
            case ng::MPMMaterial::FIRECRACKER:
                ImGui::TextWrapped("+ Heated shell + fast gas pressure + burst-like rupture [experimental]"); break;
            case ng::MPMMaterial::GLAZE_CLAY:
                ImGui::TextWrapped("+ Shell/core kiln curing + glaze skin + mismatch cracking [experimental]"); break;
            case ng::MPMMaterial::CRUST_DOUGH:
                ImGui::TextWrapped("+ Core gas growth + drying crust + loaf tear vents [experimental]"); break;
            case ng::MPMMaterial::THERMO_METAL:
                ImGui::TextWrapped("+ Thermoelastic expansion stress + hot anneal + cool hardening [experimental]"); break;
            case ng::MPMMaterial::REACTIVE_BURN:
                ImGui::TextWrapped("+ Char shell + pyro gas + smoky blister burst [experimental]"); break;
            case ng::MPMMaterial::SEALED_CHARGE:
                ImGui::TextWrapped("+ Airtight shell/core confinement + delayed rupture + hotter vent-to-air blast [experimental]"); break;
            case ng::MPMMaterial::GLAZE_DRIP:
                ImGui::TextWrapped("+ Shell/core pottery + runny glaze skin + ceramic core retention [experimental]"); break;
            case ng::MPMMaterial::STEAM_BUN:
                ImGui::TextWrapped("+ Steam-lifted shell/core dough + springy crumb + softer skin venting [experimental]"); break;
            case ng::MPMMaterial::FILAMENT_GLASS:
                ImGui::TextWrapped("+ First-pass codim-style hot glass + stretch-aligned thread memory + filament pull [experimental]"); break;
            case ng::MPMMaterial::CHEESE_PULL:
                ImGui::TextWrapped("+ First-pass codim-style viscoelastic food melt + extensional cohesion + pull strands [experimental]"); break;
            case ng::MPMMaterial::MAG_SOFT_IRON:
                ImGui::TextWrapped("+ Real magnetic field sampling + soft ferromagnetic pull toward stronger |H|^2 regions [experimental]"); break;
            case ng::MPMMaterial::MAGNETIC_RUBBER:
                ImGui::TextWrapped("+ Real magnetic field sampling + compliant magnetizable body for bend-and-drift tests [experimental]"); break;
            case ng::MPMMaterial::VENT_CRUMB:
                ImGui::TextWrapped("+ Shell-first drying + retained core moisture + larger vent channels + tunnel crumb [experimental]"); break;
            case ng::MPMMaterial::OPEN_CRUMB:
                ImGui::TextWrapped("+ Bubble coalescence + larger retained pore chambers + stronger baked scaffold [experimental]"); break;
            case ng::MPMMaterial::VITREOUS_CLAY:
                ImGui::TextWrapped("+ Stronger vitrification + denser shrink + tighter kiln-fired cracking [experimental]"); break;
            case ng::MPMMaterial::BLISTER_GLAZE:
                ImGui::TextWrapped("+ Glossy glaze shell + trapped volatile blisters + vent pits over a ceramic core [experimental]"); break;
            case ng::MPMMaterial::SINTER_LOCK:
                ImGui::TextWrapped("+ Shell-first sintering + stronger shrink-set lock + fired shape retention [experimental]"); break;
            default: break;
            }
            ImGui::TextWrapped("+ SDF collision (grid + particle), int atomic P2G scatter");
        }
        ImGui::PopStyleColor();
    }

    // --- Customize panel ---
    if (ImGui::CollapsingHeader("Customize", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (cp.solver == ng::SpawnSolver::SPH) {
            ImGui::SliderFloat("Gas Constant", &cp.gas_constant, 1, 30, "%.1f");
            ImGui::SliderFloat("Viscosity", &cp.viscosity, 0, 1, "%.3f");
        } else {
            ImGui::SliderFloat("Stiffness (E)", &cp.youngs_modulus, 500, 50000, "%.0f");
            ImGui::SliderFloat("Poisson", &cp.poisson_ratio, 0.05f, 0.45f, "%.2f");
            ImGui::SliderFloat("Init Temp (K)", &cp.initial_temp, 0, 1000, "%.0f");

            if (cp.fiber_strength > 0 || cp.mpm_type == ng::MPMMaterial::ANISO ||
                cp.mpm_type == ng::MPMMaterial::FRACTURE || cp.mpm_type == ng::MPMMaterial::COMPOSITE ||
                cp.mpm_type == ng::MPMMaterial::BRITTLE || cp.mpm_type == ng::MPMMaterial::TOUGH ||
                cp.mpm_type == ng::MPMMaterial::FILAMENT_GLASS || cp.mpm_type == ng::MPMMaterial::CHEESE_PULL) {
                ImGui::SliderFloat("Fiber Strength", &cp.fiber_strength, 0, 8, "%.1f");
                ImGui::SliderFloat("Fiber Angle", &g_creation.fiber_angle, -180, 180, "%.0f deg");
                cp.fiber_dir = ng::vec2(std::cos(g_creation.fiber_angle * 3.14159265f / 180.0f),
                                        std::sin(g_creation.fiber_angle * 3.14159265f / 180.0f));
            }
        }
        ImGui::SliderFloat("Gravity Scale", &cp.physical_scale, 0.01f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Physical size: 0.05=5cm, 0.1=10cm, 1.0=full scale.\nSmaller = less gravity effect, holds shape better.");
        if (ImGui::Button("Reset to Default")) {
            g_creation.select_preset(g_creation.preset_index);
        }
    }

    // --- Shape & Size ---
    ImGui::Separator();
    int shape_idx = static_cast<int>(g_creation.shape);
    ImGui::SetNextItemWidth(180.0f);
    if (ImGui::Combo("Shape", &shape_idx, ng::spawn_shape_names, ng::SHAPE_COUNT)) {
        g_creation.shape = static_cast<ng::SpawnShape>(shape_idx);
    }
    ImGui::SliderFloat("Size", &g_creation.size, 0.05f, 1.0f, "%.2f");
    if (ng::shape_uses_aspect(g_creation.shape)) {
        if (g_creation.shape == ng::SpawnShape::BEAM) {
            ImGui::SliderFloat("Length Ratio", &g_creation.aspect, 1.0f, 8.0f, "%.1f");
        } else {
            ImGui::SliderFloat("Aspect", &g_creation.aspect, 0.2f, 5.0f, "%.1f");
        }
    }
    if (ng::shape_uses_rotation(g_creation.shape)) {
        ImGui::SliderFloat("Shape Angle", &g_creation.shape_angle, -180.0f, 180.0f, "%.0f deg");
    }
    if (ng::shape_is_shell(g_creation.shape)) {
        ImGui::TextDisabled("Shell thickness is inferred from size so thin rings/frames stay readable.");
    }

    // --- Batch list ---
    if (!g_creation.batches.empty()) {
        ImGui::Separator();
        ImGui::Text("Objects (%zu)", g_creation.batches.size());
        ImGui::BeginChild("BatchList", ImVec2(0, 80), true);
        for (int i = 0; i < static_cast<int>(g_creation.batches.size()); i++) {
            auto& b = g_creation.batches[i];
            ImGui::PushID(i);
            ImVec4 col(b.color.r, b.color.g, b.color.b, 1.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, col);
            bool highlighted = (g_creation.highlighted_batch == i);
            ng::u32 cnt = b.sph_count > 0 ? b.sph_count : b.mpm_count;
            char label[192];
            snprintf(label, sizeof(label), "#%d %s%s (%u)", i, b.label.c_str(),
                     b.scene_authored ? " [scene]" : "", cnt);
            if (ImGui::Selectable(label, highlighted))
                g_creation.highlighted_batch = highlighted ? -1 : i;
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s\n%s", b.description.c_str(), b.techniques.c_str());
            }
            ImGui::PopStyleColor();
            ImGui::PopID();
        }
        ImGui::EndChild();
    }

    ImGui::End();
}

static void render_preview(ng::f32 time) {
    if (!g_creation.active) return;

    const auto& preset = ng::get_presets()[g_creation.preset_index];
    ng::vec2 sz = ng::shape_half_extents(g_creation.shape, g_creation.size, g_creation.aspect);
    ng::f32 rotation = g_creation.shape_angle * 3.14159265f / 180.0f;

    g_preview_shader.bind();
    g_preview_shader.set_mat4("u_view_proj", g_camera.view_proj());
    g_preview_shader.set_vec2("u_center", g_creation.preview_pos);
    g_preview_shader.set_float("u_time", time);
    g_preview_shader.set_float("u_rotation", rotation);

    auto draw_outline = [&](ng::SpawnShape shape, ng::vec2 size, ng::vec4 color) {
        g_preview_shader.set_vec2("u_size", size);
        g_preview_shader.set_int("u_shape", static_cast<ng::i32>(shape));
        g_preview_shader.set_vec4("u_color", color);
        glDrawArrays(GL_LINE_LOOP, 0, 64);
    };

    glBindVertexArray(g_preview_vao);
    glLineWidth(2.0f);
    draw_outline(ng::base_preview_shape(g_creation.shape), sz, preset.color);

    if (ng::shape_is_shell(g_creation.shape)) {
        ng::vec2 inner = sz;
        ng::f32 thickness = ng::shape_shell_thickness(g_creation.shape, sz);
        if (g_creation.shape == ng::SpawnShape::SHELL_CIRCLE) {
            inner = ng::vec2(glm::max(sz.x - thickness, sz.x * 0.35f));
        } else {
            inner = glm::max(sz - ng::vec2(thickness), sz * 0.32f);
        }
        draw_outline(ng::base_preview_shape(g_creation.shape), inner,
                     ng::vec4(preset.color.r, preset.color.g, preset.color.b, 0.82f));
    }

    if (g_show_size_hint) {
        ng::vec2 hint_size = ng::shape_half_extents(g_creation.shape, g_creation.custom.recommended_size, g_creation.aspect);
        draw_outline(ng::base_preview_shape(g_creation.shape), hint_size, ng::vec4(0.95f, 0.92f, 0.78f, 0.28f));
        if (ng::shape_is_shell(g_creation.shape)) {
            ng::vec2 hint_inner = hint_size;
            ng::f32 thickness = ng::shape_shell_thickness(g_creation.shape, hint_size);
            if (g_creation.shape == ng::SpawnShape::SHELL_CIRCLE) {
                hint_inner = ng::vec2(glm::max(hint_size.x - thickness, hint_size.x * 0.35f));
            } else {
                hint_inner = glm::max(hint_size - ng::vec2(thickness), hint_size * 0.32f);
            }
            draw_outline(ng::base_preview_shape(g_creation.shape), hint_inner,
                         ng::vec4(0.95f, 0.92f, 0.78f, 0.20f));
        }
    }

    glBindVertexArray(0);
    g_preview_shader.unbind();
}

static void render_tool_indicator(ng::vec2 mouse_world, ng::f32 time, InteractMode active_mode,
                                  bool lmb_active, bool rmb_active) {
    if (g_creation.active || g_selection_mode) return;
    if (active_mode != InteractMode::PUSH &&
        active_mode != InteractMode::DRAG &&
        active_mode != InteractMode::SWEEP_DRAG &&
        active_mode != InteractMode::SPRING_DRAG &&
        active_mode != InteractMode::DROP_BALL &&
        active_mode != InteractMode::LAUNCHER2 &&
        active_mode != InteractMode::FOOT_CONTROL) return;

    auto draw_drag_debug = [&](ng::SolverType solver_type,
                               const ng::GPUBuffer& anchor_buf,
                               const ng::GPUBuffer& weight_buf,
                               const ng::vec4& point_color,
                               const ng::vec4& beam_color) {
        const auto& range = g_particles.range(solver_type);
        if (range.count == 0) return;

        std::vector<ng::vec2> positions(range.count);
        std::vector<ng::vec2> anchors(range.count);
        std::vector<ng::f32> weights(range.count);
        g_particles.positions().download(
            positions.data(), range.count * sizeof(ng::vec2), range.offset * sizeof(ng::vec2));
        anchor_buf.download(anchors.data(), range.count * sizeof(ng::vec2));
        weight_buf.download(weights.data(), range.count * sizeof(ng::f32));

        ng::u32 selected_count = 0;
        for (ng::u32 i = 0; i < range.count; ++i) {
            if (weights[i] > 0.001f) ++selected_count;
        }
        if (selected_count == 0) return;

        constexpr ng::u32 kMaxDebugLines = 768u;
        ng::u32 step = (selected_count > kMaxDebugLines)
            ? ((selected_count + kMaxDebugLines - 1u) / kMaxDebugLines)
            : 1u;
        ng::u32 selected_seen = 0;

        for (ng::u32 i = 0; i < range.count; ++i) {
            if (weights[i] <= 0.001f) continue;
            if ((selected_seen++ % step) != 0u) continue;

            const ng::vec2 p = positions[i];
            ng::vec2 target = p;
            if (active_mode == InteractMode::SPRING_DRAG) {
                target = mouse_world + anchors[i];
            } else {
                target = p + (mouse_world - g_drag_anchor_world);
            }

            ng::vec2 delta = target - p;
            ng::f32 len = glm::length(delta);
            if (len < 0.0025f) continue;

            ng::f32 angle = std::atan2(delta.y, delta.x);
            ng::vec2 mid = (p + target) * 0.5f;

            g_preview_shader.set_int("u_shape", static_cast<ng::i32>(ng::SpawnShape::BEAM));
            g_preview_shader.set_float("u_rotation", angle);
            g_preview_shader.set_vec2("u_center", mid);
            g_preview_shader.set_vec2("u_size", ng::vec2(len * 0.5f, 0.0035f));
            ng::vec4 beam_tint = beam_color;
            beam_tint.a *= glm::clamp(weights[i], 0.12f, 1.0f);
            g_preview_shader.set_vec4("u_color", beam_tint);
            glDrawArrays(GL_LINE_LOOP, 0, 64);

            g_preview_shader.set_int("u_shape", static_cast<ng::i32>(ng::SpawnShape::CIRCLE));
            g_preview_shader.set_float("u_rotation", 0.0f);
            g_preview_shader.set_vec2("u_center", p);
            g_preview_shader.set_vec2("u_size", ng::vec2(0.012f));
            ng::vec4 point_tint = point_color;
            point_tint.a *= glm::clamp(weights[i], 0.18f, 1.0f);
            g_preview_shader.set_vec4("u_color", point_tint);
            glDrawArrays(GL_LINE_LOOP, 0, 64);
        }
    };

    g_preview_shader.bind();
    g_preview_shader.set_mat4("u_view_proj", g_camera.view_proj());
    g_preview_shader.set_vec2("u_center", mouse_world);
    g_preview_shader.set_int("u_shape", 0);
    g_preview_shader.set_float("u_time", time);
    g_preview_shader.set_float("u_rotation", 0.0f);

    glBindVertexArray(g_preview_vao);
    glLineWidth(2.0f);

    if (active_mode == InteractMode::PUSH ||
        active_mode == InteractMode::DRAG ||
        active_mode == InteractMode::SPRING_DRAG) {
        sync_drag_falloff_radius();
        g_preview_shader.set_vec2("u_size", ng::vec2(g_drag_falloff_radius));
        g_preview_shader.set_vec4("u_color",
            (active_mode == InteractMode::PUSH)
                ? ng::vec4(0.90f, 0.84f, 0.58f, 0.24f)
                : (active_mode == InteractMode::SPRING_DRAG)
                ? ng::vec4(0.62f, 0.98f, 0.82f, 0.26f)
                : ng::vec4(0.42f, 0.86f, 1.0f, 0.24f));
        glDrawArrays(GL_LINE_LOOP, 0, 64);
    }

    if (active_mode == InteractMode::FOOT_CONTROL && ng::foot_demo_active()) {
        g_preview_shader.set_vec2("u_center", ng::foot_demo_focus_point());
        g_preview_shader.set_vec2("u_size", ng::vec2(ng::foot_demo_focus_radius()));
        g_preview_shader.set_vec4("u_color", ng::vec4(0.96f, 0.82f, 0.58f, 0.78f));
        glDrawArrays(GL_LINE_LOOP, 0, 64);
        g_preview_shader.set_vec2("u_size", ng::vec2(ng::foot_demo_focus_radius() * 0.52f));
        g_preview_shader.set_vec4("u_color", ng::vec4(0.98f, 0.95f, 0.78f, 0.42f));
        glDrawArrays(GL_LINE_LOOP, 0, 64);
    }

    if (active_mode == InteractMode::PUSH ||
        active_mode == InteractMode::DRAG ||
        active_mode == InteractMode::SPRING_DRAG) {
        g_preview_shader.set_vec2("u_size", ng::vec2(g_tool_radius));
        g_preview_shader.set_vec4("u_color",
            (active_mode == InteractMode::PUSH)
                ? ng::vec4(0.96f, 0.88f, 0.54f, 0.72f)
                : (active_mode == InteractMode::SPRING_DRAG)
                ? ng::vec4(0.62f, 0.98f, 0.82f, 0.68f)
                : ng::vec4(0.42f, 0.86f, 1.0f, 0.55f));
        glDrawArrays(GL_LINE_LOOP, 0, 64);
    }

    if (active_mode == InteractMode::PUSH) {
        bool pulling = lmb_active && !rmb_active;
        bool pushing = rmb_active && !lmb_active;
        ng::vec4 guide_color = pulling ? ng::vec4(0.60f, 0.92f, 1.0f, 0.80f)
                                       : pushing ? ng::vec4(1.0f, 0.62f, 0.44f, 0.82f)
                                                 : ng::vec4(0.96f, 0.88f, 0.54f, 0.42f);
        const int spokes = 8;
        ng::f32 inner = g_tool_radius;
        ng::f32 outer = g_drag_falloff_radius;
        for (int i = 0; i < spokes; ++i) {
            ng::f32 a = (6.2831853f * static_cast<ng::f32>(i)) / static_cast<ng::f32>(spokes);
            ng::vec2 dir(std::cos(a), std::sin(a));
            ng::vec2 a0 = mouse_world + dir * (pulling ? outer : inner);
            ng::vec2 a1 = mouse_world + dir * (pulling ? inner : outer);
            ng::vec2 mid = (a0 + a1) * 0.5f;
            ng::f32 len = glm::length(a1 - a0);
            g_preview_shader.set_int("u_shape", static_cast<ng::i32>(ng::SpawnShape::BEAM));
            g_preview_shader.set_float("u_rotation", a);
            g_preview_shader.set_vec2("u_center", mid);
            g_preview_shader.set_vec2("u_size", ng::vec2(len * 0.5f, (pulling || pushing) ? 0.012f : 0.007f));
            g_preview_shader.set_vec4("u_color", guide_color);
            glDrawArrays(GL_LINE_LOOP, 0, 64);
        }
        g_preview_shader.set_int("u_shape", 0);
        g_preview_shader.set_float("u_rotation", 0.0f);
        g_preview_shader.set_vec2("u_center", mouse_world);
    }

    if (active_mode == InteractMode::SWEEP_DRAG) {
        ng::f32 inner_r = g_tool_radius * g_drag_inner_ratio;
        g_preview_shader.set_vec2("u_size", ng::vec2(inner_r));
        g_preview_shader.set_vec4("u_color",
            ng::vec4(0.95f, 0.98f, 1.0f, 0.82f));
        glDrawArrays(GL_LINE_LOOP, 0, 64);
    }

    if (active_mode == InteractMode::DROP_BALL || active_mode == InteractMode::LAUNCHER2) {
        ng::vec2 origin = mouse_world;
        if (g_ball_drag_mode != ProjectileDragMode::NONE) {
            origin = g_ball_drag_anchor;
        }
        ng::vec2 aim = current_projectile_vector();
        if (g_ball_drag_mode == ProjectileDragMode::AIM) {
            aim = mouse_world - g_ball_drag_anchor;
        }
        ng::f32 raw_aim_len = glm::length(aim);
        if (raw_aim_len < 0.05f) {
            aim = ng::vec2(1.0f, 0.0f);
            raw_aim_len = 0.8f;
        } else {
            aim /= raw_aim_len;
        }
        constexpr ng::f32 kPreviewAimCap = 0.50f;
        ng::f32 aim_len = glm::min(raw_aim_len, kPreviewAimCap);
        ng::f32 actual_speed = glm::clamp(raw_aim_len * glm::max(g_ball_launch_gain, 0.1f), g_ball_min_launch_speed, 42.0f);
        ng::f32 capped_speed = kPreviewAimCap * glm::max(g_ball_launch_gain, 0.1f);
        ng::f32 overflow_t = glm::clamp((actual_speed - capped_speed) / glm::max(42.0f - capped_speed, 1.0f), 0.0f, 1.0f);
        ng::vec4 overflow_color;
        if (overflow_t <= 0.0f) {
            overflow_color = ng::vec4(0.72f, 0.90f, 1.0f, 0.62f);
        } else if (overflow_t < 0.5f) {
            overflow_color = glm::mix(ng::vec4(0.30f, 0.62f, 1.0f, 0.78f),
                                      ng::vec4(1.0f, 0.92f, 0.34f, 0.82f),
                                      overflow_t * 2.0f);
        } else {
            overflow_color = glm::mix(ng::vec4(1.0f, 0.92f, 0.34f, 0.82f),
                                      ng::vec4(1.0f, 0.28f, 0.18f, 0.88f),
                                      (overflow_t - 0.5f) * 2.0f);
        }
        ng::vec2 target = origin + aim * aim_len;

        ng::f32 base_angle = std::atan2(aim.y, aim.x);
        ng::f32 cone_rad = glm::radians(glm::clamp(g_ball_cone_deg, 0.0f, 65.0f));
        ng::vec4 origin_color = (g_ball_drag_mode == ProjectileDragMode::AIM)
            ? ng::vec4(1.0f, 0.88f, 0.58f, 0.88f)
            : (g_ball_drag_mode == ProjectileDragMode::CONE)
            ? ng::vec4(0.96f, 0.72f, 1.0f, 0.88f)
            : ng::vec4(0.82f, 0.92f, 1.0f, 0.84f);
        ng::vec4 beam_color = lmb_active
            ? glm::mix(overflow_color, ng::vec4(1.0f, 0.68f, 0.34f, 0.92f), 0.40f)
            : overflow_color;

        g_preview_shader.set_int("u_shape", static_cast<ng::i32>(projectile_spawn_shape()));
        g_preview_shader.set_float("u_rotation", base_angle);
        g_preview_shader.set_vec2("u_center", origin);
        g_preview_shader.set_vec2("u_size", ng::shape_half_extents(projectile_spawn_shape(), g_ball_radius, projectile_shape_aspect(projectile_spawn_shape())));
        g_preview_shader.set_vec4("u_color", ng::vec4(origin_color.r, origin_color.g, origin_color.b, 0.76f));
        glDrawArrays(GL_LINE_LOOP, 0, 64);

        g_preview_shader.set_int("u_shape", static_cast<ng::i32>(ng::SpawnShape::BEAM));
        g_preview_shader.set_float("u_rotation", base_angle);
        g_preview_shader.set_vec2("u_center", (origin + target) * 0.5f);
        g_preview_shader.set_vec2("u_size", ng::vec2(aim_len * 0.5f, 0.012f));
        g_preview_shader.set_vec4("u_color", beam_color);
        glDrawArrays(GL_LINE_LOOP, 0, 64);

        if (cone_rad > 1e-4f) {
            ng::vec2 left = origin + rotate_vec2(aim, cone_rad) * aim_len;
            ng::vec2 right = origin + rotate_vec2(aim, -cone_rad) * aim_len;
            ng::f32 left_angle = std::atan2(left.y - origin.y, left.x - origin.x);
            ng::f32 right_angle = std::atan2(right.y - origin.y, right.x - origin.x);

            g_preview_shader.set_vec2("u_center", (origin + left) * 0.5f);
            g_preview_shader.set_float("u_rotation", left_angle);
            g_preview_shader.set_vec2("u_size", ng::vec2(glm::length(left - origin) * 0.5f, 0.006f));
            g_preview_shader.set_vec4("u_color", ng::vec4(beam_color.r, beam_color.g, beam_color.b, 0.42f));
            glDrawArrays(GL_LINE_LOOP, 0, 64);

            g_preview_shader.set_vec2("u_center", (origin + right) * 0.5f);
            g_preview_shader.set_float("u_rotation", right_angle);
            g_preview_shader.set_vec2("u_size", ng::vec2(glm::length(right - origin) * 0.5f, 0.006f));
            g_preview_shader.set_vec4("u_color", ng::vec4(beam_color.r, beam_color.g, beam_color.b, 0.42f));
            glDrawArrays(GL_LINE_LOOP, 0, 64);
        }

        g_preview_shader.set_int("u_shape", static_cast<ng::i32>(ng::SpawnShape::CIRCLE));
        g_preview_shader.set_float("u_rotation", 0.0f);
        g_preview_shader.set_vec2("u_center", target);
        g_preview_shader.set_vec2("u_size", ng::vec2(0.03f));
        g_preview_shader.set_vec4("u_color", ng::vec4(beam_color.r, beam_color.g, beam_color.b, 0.95f));
        glDrawArrays(GL_LINE_LOOP, 0, 64);
    }

    if (g_drag_capture_active &&
        (active_mode == InteractMode::DRAG || active_mode == InteractMode::SPRING_DRAG)) {
        ng::vec2 drag_delta = mouse_world - g_drag_anchor_world;
        ng::f32 drag_len = glm::length(drag_delta);

        if (active_mode == InteractMode::SPRING_DRAG) {
            // Real Drag: shadow copy circle at cursor (SAME size as grab)
            g_preview_shader.set_vec2("u_center", mouse_world);
            g_preview_shader.set_int("u_shape", 0);
            g_preview_shader.set_float("u_rotation", 0.0f);
            g_preview_shader.set_vec2("u_size", ng::vec2(g_drag_falloff_radius));
            g_preview_shader.set_vec4("u_color", ng::vec4(0.68f, 1.0f, 0.86f, 0.22f));
            glDrawArrays(GL_LINE_LOOP, 0, 64);
            g_preview_shader.set_vec2("u_size", ng::vec2(g_tool_radius));
            g_preview_shader.set_vec4("u_color", ng::vec4(0.68f, 1.0f, 0.86f, 0.5f));
            glDrawArrays(GL_LINE_LOOP, 0, 64);

            // Original grab circle (same size, dimmer)
            if (drag_len > 0.01f) {
                g_preview_shader.set_vec2("u_center", g_drag_anchor_world);
                g_preview_shader.set_vec2("u_size", ng::vec2(g_drag_falloff_radius));
                g_preview_shader.set_vec4("u_color", ng::vec4(0.5f, 0.8f, 0.65f, 0.14f));
                glDrawArrays(GL_LINE_LOOP, 0, 64);
                g_preview_shader.set_vec2("u_size", ng::vec2(g_tool_radius));
                g_preview_shader.set_vec4("u_color", ng::vec4(0.5f, 0.8f, 0.65f, 0.25f));
                glDrawArrays(GL_LINE_LOOP, 0, 64);
            }
        } else {
            // Telekinesis: full-size circle at anchor + arrow to cursor
            ng::f32 drag_angle = (drag_len > 1e-5f) ? std::atan2(drag_delta.y, drag_delta.x) : 0.0f;
            ng::vec4 guide_color(0.56f, 0.92f, 1.0f, 0.78f);

            // FULL SIZE circle at anchor (NOT small!)
            g_preview_shader.set_vec2("u_center", g_drag_anchor_world);
            g_preview_shader.set_int("u_shape", 0);
            g_preview_shader.set_float("u_rotation", 0.0f);
            g_preview_shader.set_vec2("u_size", ng::vec2(g_drag_falloff_radius));
            g_preview_shader.set_vec4("u_color", ng::vec4(guide_color.r, guide_color.g, guide_color.b, 0.18f));
            glDrawArrays(GL_LINE_LOOP, 0, 64);
            g_preview_shader.set_vec2("u_size", ng::vec2(g_tool_radius));
            g_preview_shader.set_vec4("u_color", ng::vec4(guide_color.r, guide_color.g, guide_color.b, 0.35f));
            glDrawArrays(GL_LINE_LOOP, 0, 64);

            if (drag_len > 1e-4f) {
                // Arrow beam from anchor to cursor
                g_preview_shader.set_vec2("u_center", g_drag_anchor_world + drag_delta * 0.5f);
                g_preview_shader.set_int("u_shape", static_cast<ng::i32>(ng::SpawnShape::BEAM));
                g_preview_shader.set_float("u_rotation", drag_angle);
                g_preview_shader.set_vec2("u_size", ng::vec2(drag_len * 0.5f, 0.02f));
                g_preview_shader.set_vec4("u_color", ng::vec4(guide_color.r, guide_color.g, guide_color.b, 0.45f));
                glDrawArrays(GL_LINE_LOOP, 0, 64);

                // Arrowhead at cursor
                g_preview_shader.set_vec2("u_center", mouse_world);
                g_preview_shader.set_int("u_shape", static_cast<ng::i32>(ng::SpawnShape::TRIANGLE));
                g_preview_shader.set_float("u_rotation", drag_angle - 1.57079632f);
                g_preview_shader.set_vec2("u_size", ng::vec2(0.05f));
                g_preview_shader.set_vec4("u_color", guide_color);
                glDrawArrays(GL_LINE_LOOP, 0, 64);
            }
        }

        if (g_show_drag_debug) {
            if (g_sph.spring_drag_active()) {
                draw_drag_debug(ng::SolverType::SPH, g_sph.spring_anchor_buf(), g_sph.spring_weight_buf(),
                                ng::vec4(0.54f, 0.86f, 1.0f, 0.82f),
                                ng::vec4(0.42f, 0.86f, 1.0f, 0.35f));
            }
            if (g_mpm.spring_drag_active()) {
                draw_drag_debug(ng::SolverType::MPM, g_mpm.spring_anchor_buf(), g_mpm.spring_weight_buf(),
                                ng::vec4(0.52f, 1.0f, 0.70f, 0.86f),
                                ng::vec4(0.40f, 1.0f, 0.60f, 0.32f));
            }
        }
    }

    glBindVertexArray(0);
    g_preview_shader.unbind();
}

static ng::f32 sdf_object_distance(const ng::SDFField::ObjectRecord& obj, ng::vec2 p) {
    switch (obj.type) {
    case ng::SDFField::PRIM_BOX: {
        ng::vec2 q = glm::abs(p - obj.a) - obj.b;
        ng::vec2 mq = glm::max(q, ng::vec2(0.0f));
        return glm::length(mq) + std::min(std::max(q.x, q.y), 0.0f);
    }
    case ng::SDFField::PRIM_CIRCLE:
        return std::abs(glm::length(p - obj.a) - obj.radius_or_thickness);
    case ng::SDFField::PRIM_SEGMENT: {
        ng::vec2 pa = p - obj.a;
        ng::vec2 ba = obj.b - obj.a;
        ng::f32 h = glm::clamp(glm::dot(pa, ba) / std::max(glm::dot(ba, ba), 1e-6f), 0.0f, 1.0f);
        return std::abs(glm::length(pa - ba * h) - obj.radius_or_thickness);
    }
    default:
        return std::numeric_limits<ng::f32>::max();
    }
}

static void update_hover_selection(ng::vec2 mouse_world) {
    g_hover_selection = {};
    if (!g_selection_mode || g_creation.active) return;

    ng::f32 pick_radius = glm::clamp(18.0f / g_camera.zoom_level(), 0.05f, 0.24f);
    ng::f32 best_distance = pick_radius;

    auto test_batch = [&](const std::vector<ng::vec2>& positions, ng::u32 base_offset,
                          int batch_index, ng::u32 offset, ng::u32 count) {
        if (count == 0 || offset < base_offset) return;
        ng::u32 local_begin = offset - base_offset;
        if (local_begin >= positions.size()) return;
        ng::u32 local_end = std::min<ng::u32>(local_begin + count, static_cast<ng::u32>(positions.size()));
        for (ng::u32 i = local_begin; i < local_end; ++i) {
            ng::f32 d = glm::length(positions[i] - mouse_world);
            if (d < best_distance) {
                best_distance = d;
                g_hover_selection.kind = HoverKind::BATCH;
                g_hover_selection.batch_index = batch_index;
                g_hover_selection.sdf_object_id = 0;
                g_hover_selection.distance = d;
            }
        }
    };

    auto sph_range = g_particles.range(ng::SolverType::SPH);
    auto mpm_range = g_particles.range(ng::SolverType::MPM);
    std::vector<ng::vec2> sph_positions;
    std::vector<ng::vec2> mpm_positions;
    if (sph_range.count > 0) {
        sph_positions.resize(sph_range.count);
        g_particles.positions().download(
            sph_positions.data(),
            sph_range.count * sizeof(ng::vec2),
            sph_range.offset * sizeof(ng::vec2));
    }
    if (mpm_range.count > 0) {
        mpm_positions.resize(mpm_range.count);
        g_particles.positions().download(
            mpm_positions.data(),
            mpm_range.count * sizeof(ng::vec2),
            mpm_range.offset * sizeof(ng::vec2));
    }

    for (int i = 0; i < static_cast<int>(g_creation.batches.size()); ++i) {
        const auto& batch = g_creation.batches[i];
        if (batch.sph_count > 0) {
            test_batch(sph_positions, sph_range.offset, i, batch.sph_offset, batch.sph_count);
        }
        if (batch.mpm_count > 0) {
            test_batch(mpm_positions, mpm_range.offset, i, batch.mpm_offset, batch.mpm_count);
        }
    }

    for (const auto& obj : g_sdf.objects()) {
        ng::f32 d = sdf_object_distance(obj, mouse_world);
        if (d < best_distance) {
            best_distance = d;
            g_hover_selection.kind = HoverKind::SDF;
            g_hover_selection.batch_index = -1;
            g_hover_selection.sdf_object_id = obj.id;
            g_hover_selection.distance = d;
        }
    }
}

static void refresh_batch_summary(ng::BatchRecord& batch) {
    batch.techniques = ng::technique_summary(batch.solver, batch.mpm_type);
    batch.properties = ng::property_summary(batch.solver, batch.mpm_type,
                                            batch.youngs_modulus, batch.poisson_ratio,
                                            batch.temperature, batch.fiber_dir);
    if (batch.solver == ng::SpawnSolver::MPM) {
        char thermal_buf[160];
        snprintf(thermal_buf, sizeof(thermal_buf),
                 " | outgas %.2f | air heat %.2f | cool %.2f | loft %.2f",
                 batch.outgassing_scale, batch.heat_release_scale, batch.cooling_scale,
                 batch.loft_scale);
        batch.properties += thermal_buf;
    }
}

static int sdf_palette_combo_from_code(ng::u32 code) {
    switch (code) {
    case 1u: return 1; // Silver
    case 2u: return 2; // Rose Gold
    case 3u: return 3; // Bronze
    case 4u: return 4; // Brass
    default: return 0; // Global
    }
}

static ng::u32 sdf_palette_code_from_combo(int combo) {
    switch (combo) {
    case 1: return 1u;
    case 2: return 2u;
    case 3: return 3u;
    case 4: return 4u;
    default: return 0u;
    }
}

static void draw_selection_tooltip() {
    if (!g_selection_mode || g_hover_selection.kind == HoverKind::NONE) return;

    ImGui::BeginTooltip();
    if (g_hover_selection.kind == HoverKind::BATCH &&
        g_hover_selection.batch_index >= 0 &&
        g_hover_selection.batch_index < static_cast<int>(g_creation.batches.size())) {
        const auto& batch = g_creation.batches[g_hover_selection.batch_index];
        ImGui::TextColored(ImVec4(batch.color.r, batch.color.g, batch.color.b, 1.0f), "%s", batch.label.c_str());
        if (!batch.description.empty()) ImGui::TextWrapped("%s", batch.description.c_str());
        ImGui::Separator();
        ImGui::TextWrapped("%s", batch.properties.c_str());
        ImGui::TextDisabled("Click to edit");
    } else if (g_hover_selection.kind == HoverKind::SDF) {
        const auto* obj = g_sdf.object_by_id(g_hover_selection.sdf_object_id);
        if (obj) {
            ImGui::TextColored(ImVec4(0.95f, 0.84f, 0.58f, 1.0f), "%s", obj->label.c_str());
            ImGui::TextWrapped("%s", obj->summary.c_str());
            ImGui::Separator();
            ImGui::TextWrapped("%s", obj->material.name);
            ImGui::TextWrapped("Conduct x%.2f | Sink x%.2f | Contact x%.2f | Leak x%.2f",
                               obj->material.conductivity_scale,
                               obj->material.heat_capacity_scale,
                               obj->material.contact_transfer_scale,
                               obj->material.heat_loss_scale);
            if (obj->material.magnetic_mode != ng::SDFField::MagneticMode::NONE) {
                float angle = std::atan2(obj->material.magnetic_dir.y, obj->material.magnetic_dir.x) * 180.0f / 3.14159265f;
                ImGui::TextWrapped("Magnetic: %s | strength %.2f | angle %.0f deg",
                                   obj->material.magnetic_mode == ng::SDFField::MagneticMode::PERMANENT ? "permanent" : "soft",
                                   obj->material.magnetic_mode == ng::SDFField::MagneticMode::PERMANENT ? obj->material.magnetic_strength : obj->material.magnetic_susceptibility,
                                   angle);
            }
            ImGui::TextDisabled("Click to edit");
        }
    }
    ImGui::EndTooltip();
}

static void draw_selection_editor() {
    if (!g_pinned_selection_open || g_pinned_selection.kind == HoverKind::NONE) return;

    if (g_pinned_selection.kind == HoverKind::BATCH &&
        g_pinned_selection.batch_index >= 0 &&
        g_pinned_selection.batch_index < static_cast<int>(g_creation.batches.size())) {
        ng::BatchRecord& batch = g_creation.batches[g_pinned_selection.batch_index];
        ImGui::Begin("Selected Material", &g_pinned_selection_open, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::TextColored(ImVec4(batch.color.r, batch.color.g, batch.color.b, 1.0f), "%s", batch.label.c_str());
        if (!batch.description.empty()) ImGui::TextWrapped("%s", batch.description.c_str());
        ImGui::Separator();
        ImGui::TextWrapped("Techniques: %s", batch.techniques.c_str());

        if (batch.solver == ng::SpawnSolver::MPM && batch.mpm_count > 0) {
            bool changed = false;
            changed |= ImGui::SliderFloat("Stiffness (E)", &batch.youngs_modulus, 500.0f, 50000.0f, "%.0f");
            changed |= ImGui::SliderFloat("Poisson", &batch.poisson_ratio, 0.05f, 0.45f, "%.2f");
            changed |= ImGui::SliderFloat("Temperature", &batch.temperature, 0.0f, 1400.0f, "%.0f K");

            bool fiber_controls = batch.fiber_strength > 0.0f ||
                                  batch.mpm_type == ng::MPMMaterial::ANISO ||
                                  batch.mpm_type == ng::MPMMaterial::FRACTURE ||
                                  batch.mpm_type == ng::MPMMaterial::COMPOSITE ||
                                  batch.mpm_type == ng::MPMMaterial::BRITTLE ||
                                  batch.mpm_type == ng::MPMMaterial::TOUGH ||
                                  batch.mpm_type == ng::MPMMaterial::SPLINTER ||
                                  batch.mpm_type == ng::MPMMaterial::FILAMENT_GLASS ||
                                  batch.mpm_type == ng::MPMMaterial::CHEESE_PULL;
            if (fiber_controls) {
                changed |= ImGui::SliderFloat("Fiber Strength", &batch.fiber_strength, 0.0f, 8.0f, "%.1f");
                float fiber_angle = std::atan2(batch.fiber_dir.y, batch.fiber_dir.x) * 180.0f / 3.14159265f;
                if (ImGui::SliderFloat("Fiber Angle", &fiber_angle, -180.0f, 180.0f, "%.0f deg")) {
                    batch.fiber_dir = ng::vec2(std::cos(fiber_angle * 3.14159265f / 180.0f),
                                               std::sin(fiber_angle * 3.14159265f / 180.0f));
                    changed = true;
                }
            }

            ImGui::Separator();
            ImGui::TextDisabled("Thermal Coupling [new]");
            changed |= ImGui::SliderFloat("Outgassing", &batch.outgassing_scale, 0.0f, 2.5f, "%.2f");
            changed |= ImGui::SliderFloat("Heat Release", &batch.heat_release_scale, 0.0f, 2.5f, "%.2f");
            changed |= ImGui::SliderFloat("Cooling Scale", &batch.cooling_scale, 0.05f, 2.0f, "%.2f");
            changed |= ImGui::SliderFloat("Loft / Air Carry", &batch.loft_scale, 0.0f, 2.0f, "%.2f");

            if (changed) {
                refresh_batch_summary(batch);
                g_mpm.update_batch_material(g_particles, batch.mpm_offset, batch.mpm_count,
                                            batch.youngs_modulus, batch.poisson_ratio,
                                            batch.fiber_strength, batch.temperature, batch.fiber_dir,
                                            batch.outgassing_scale, batch.heat_release_scale,
                                            batch.cooling_scale, batch.loft_scale);
            }
        } else if (batch.solver == ng::SpawnSolver::SPH) {
            ImGui::TextWrapped("This is an SPH batch. Per-batch SPH physics is still global, so tweak it in the SPH Fluid panel for now.");
        }

        ImGui::Separator();
        ImGui::TextWrapped("Properties: %s", batch.properties.c_str());
        ImGui::TextWrapped("Try size %.2f", batch.recommended_size);
        ImGui::TextWrapped("%s", batch.recommended_note.c_str());
        ImGui::End();
    } else if (g_pinned_selection.kind == HoverKind::SDF) {
        const ng::SDFField::ObjectRecord* obj = g_sdf.object_by_id(g_pinned_selection.sdf_object_id);
        if (!obj) {
            g_pinned_selection_open = false;
            return;
        }

        ImGui::Begin("Selected Scene Object", &g_pinned_selection_open, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::TextColored(ImVec4(0.95f, 0.84f, 0.58f, 1.0f), "%s", obj->label.c_str());
        ImGui::TextWrapped("%s", obj->summary.c_str());
        ImGui::Separator();
        ImGui::TextWrapped("Techniques: %s", obj->techniques.c_str());

        ng::SDFField::MaterialPreset mat = obj->material;
        static const char* object_palette_names[] = { "Global", "Silver", "Rose Gold", "Bronze", "Brass" };
        int palette_idx = sdf_palette_combo_from_code(mat.palette_code);
        bool changed = false;
        if (ImGui::Combo("Palette", &palette_idx, object_palette_names, 5)) {
            mat.palette_code = sdf_palette_code_from_combo(palette_idx);
            changed = true;
        }
        changed |= ImGui::SliderFloat("Conductivity", &mat.conductivity_scale, 0.05f, 8.0f, "%.2f");
        changed |= ImGui::SliderFloat("Heat Sink", &mat.heat_capacity_scale, 0.1f, 8.0f, "%.2f");
        changed |= ImGui::SliderFloat("Contact Transfer", &mat.contact_transfer_scale, 0.05f, 4.0f, "%.2f");
        changed |= ImGui::SliderFloat("Heat Leak", &mat.heat_loss_scale, 0.05f, 2.0f, "%.2f");
        static const char* magnetic_mode_names[] = { "None", "Permanent", "Soft" };
        int magnetic_mode_idx = static_cast<int>(mat.magnetic_mode);
        if (ImGui::Combo("Magnetic Mode", &magnetic_mode_idx, magnetic_mode_names, 3)) {
            mat.magnetic_mode = static_cast<ng::SDFField::MagneticMode>(magnetic_mode_idx);
            changed = true;
        }
        if (mat.magnetic_mode != ng::SDFField::MagneticMode::NONE) {
            float magnetic_angle = std::atan2(mat.magnetic_dir.y, mat.magnetic_dir.x) * 180.0f / 3.14159265f;
            if (ImGui::SliderFloat("Magnetic Angle", &magnetic_angle, -180.0f, 180.0f, "%.0f deg")) {
                mat.magnetic_dir = ng::vec2(std::cos(magnetic_angle * 3.14159265f / 180.0f),
                                            std::sin(magnetic_angle * 3.14159265f / 180.0f));
                changed = true;
            }
            if (mat.magnetic_mode == ng::SDFField::MagneticMode::PERMANENT) {
                changed |= ImGui::SliderFloat("Mag Strength", &mat.magnetic_strength, 0.0f, 4.0f, "%.2f");
            } else {
                changed |= ImGui::SliderFloat("Mag Susceptibility", &mat.magnetic_susceptibility, 0.0f, 4.0f, "%.2f");
            }
        }
        if (changed) {
            g_sdf.set_object_material(g_pinned_selection.sdf_object_id, mat);
        }
        ImGui::End();
    }
}

int main(int, char**) {
    ng::Engine engine;
    g_wc.title = "Noita-Gish Engine"; g_wc.width = 1280; g_wc.height = 720; g_wc.vsync = true;
    if (!engine.init(g_wc)) { LOG_FATAL("Engine init failed"); return 1; }
    engine.window().init_imgui();

    init_systems();
    ng::load_scene(g_scene, g_particles, g_sph, g_mpm, g_mpm_grid, g_sdf, &g_creation);
    apply_scene_runtime_defaults(g_scene);
    g_camera.set_viewport(g_wc.width, g_wc.height);
    g_creation.select_preset(0); // Initialize custom preset

    LOG_INFO("SPACE = creation menu | Right-click to place | See Controls panel for all keys");

    ng::u64 prev_time = SDL_GetPerformanceCounter();
    ng::u64 freq = SDL_GetPerformanceFrequency();
    ng::f32 frame_ms = 0.0f, total_time = 0.0f;
    ng::u32 frame = 0;

    while (!engine.window().should_close()) {
        ng::u64 now = SDL_GetPerformanceCounter();
        ng::f32 dt = static_cast<ng::f32>(now - prev_time) / static_cast<ng::f32>(freq);
        prev_time = now;
        frame_ms = dt * 1000.0f;
        total_time += dt;

        engine.window().poll_events();
        engine.input().update();

        bool imgui_mouse = ImGui::GetIO().WantCaptureMouse;

        // Game keys always processed (no text input fields in our UI)
        {
            if (engine.input().key_pressed(SDL_SCANCODE_SPACE)) g_creation.active = !g_creation.active;
            if (engine.input().key_pressed(SDL_SCANCODE_P)) g_paused = !g_paused;
            if (engine.input().key_pressed(SDL_SCANCODE_R)) reload_scene();
            if (engine.input().key_pressed(SDL_SCANCODE_U)) g_show_ui = !g_show_ui;
            if (engine.input().key_pressed(SDL_SCANCODE_X)) g_selection_mode = !g_selection_mode;
            if (engine.input().key_pressed(SDL_SCANCODE_DELETE) || engine.input().key_pressed(SDL_SCANCODE_BACKSPACE))
                clear_non_sdf_objects();

            if (!g_creation.active) {
                if (engine.input().key_pressed(SDL_SCANCODE_1)) g_mode = InteractMode::PUSH;
                if (engine.input().key_pressed(SDL_SCANCODE_2)) g_mode = InteractMode::SPRING_DRAG;
                if (engine.input().key_pressed(SDL_SCANCODE_3)) g_mode = InteractMode::DRAG;
                if (engine.input().key_pressed(SDL_SCANCODE_4)) g_mode = InteractMode::SWEEP_DRAG;
                if (engine.input().key_pressed(SDL_SCANCODE_5)) g_mode = InteractMode::DROP_BALL;
                if (engine.input().key_pressed(SDL_SCANCODE_6)) g_mode = InteractMode::LAUNCHER2;
                if (engine.input().key_pressed(SDL_SCANCODE_7)) g_mode = InteractMode::DRAW_WALL;
                if (engine.input().key_pressed(SDL_SCANCODE_8)) g_mode = InteractMode::ERASE_WALL;
                if (engine.input().key_pressed(SDL_SCANCODE_9)) g_mode = InteractMode::FOOT_CONTROL;
            }

            if (engine.input().key_pressed(SDL_SCANCODE_V)) {
                auto& p = const_cast<ng::SPHParams&>(g_sph.params());
                p.vis_mode = (p.vis_mode + 1) % 8;
            }
            if (engine.input().key_pressed(SDL_SCANCODE_N))
                g_mpm.params().vis_mode = (g_mpm.params().vis_mode + 1) % kMpmVisModeCount;
            if (engine.input().key_pressed(SDL_SCANCODE_I)) g_metaball.enabled = !g_metaball.enabled;
            if (engine.input().key_pressed(SDL_SCANCODE_TAB)) g_color_mode = 1 - g_color_mode;

            if (engine.input().key_pressed(SDL_SCANCODE_F1)) { g_scene = ng::SceneID::DEFAULT; reload_scene(); }
            if (engine.input().key_pressed(SDL_SCANCODE_F2)) { g_scene = ng::SceneID::THERMAL_FURNACE; reload_scene(); }
            if (engine.input().key_pressed(SDL_SCANCODE_F3)) { g_scene = ng::SceneID::FRACTURE_TEST; reload_scene(); }
            if (engine.input().key_pressed(SDL_SCANCODE_F4)) { g_scene = ng::SceneID::MELTING; reload_scene(); }
            if (engine.input().key_pressed(SDL_SCANCODE_F5)) { g_scene = ng::SceneID::DAM_BREAK; reload_scene(); }
            if (engine.input().key_pressed(SDL_SCANCODE_F6)) { g_scene = ng::SceneID::STIFF_OBJECTS; reload_scene(); }
            if (engine.input().key_pressed(SDL_SCANCODE_F7)) { g_scene = ng::SceneID::HEAT_RAMP; reload_scene(); }
            if (engine.input().key_pressed(SDL_SCANCODE_F8)) { g_scene = ng::SceneID::FIRE_FORGE; reload_scene(); }
            if (engine.input().key_pressed(SDL_SCANCODE_F9)) { g_scene = ng::SceneID::CODIM_THREADS; reload_scene(); }
            if (engine.input().key_pressed(SDL_SCANCODE_F10)) { g_scene = ng::SceneID::EMPTY_BOX; reload_scene(); }
            if (engine.input().key_pressed(SDL_SCANCODE_F11)) { g_scene = ng::SceneID::OVEN_OPEN; reload_scene(); }
            if (engine.input().key_pressed(SDL_SCANCODE_F12)) { g_scene = ng::SceneID::POT_HEATER; reload_scene(); }
        }

        ng::f32 foot_wheel_delta = 0.0f;

        // Camera
        if (!imgui_mouse) {
            ng::f32 scroll = engine.window().scroll_delta();
            const bool allow_temp_hotkeys = (!g_creation.active && !g_selection_mode && !imgui_mouse);
            bool interaction_resize_mode = (g_mode == InteractMode::PUSH ||
                                            g_mode == InteractMode::DRAG ||
                                            g_mode == InteractMode::SPRING_DRAG);
            bool launcher_resize_mode = (g_mode == InteractMode::DROP_BALL || g_mode == InteractMode::LAUNCHER2);
            bool foot_control_mode = (g_mode == InteractMode::FOOT_CONTROL && ng::foot_demo_active());
            bool magnet_resize_mode = allow_temp_hotkeys && engine.input().key_down(SDL_SCANCODE_M);
            bool ctrl_down = engine.input().key_down(SDL_SCANCODE_LCTRL) || engine.input().key_down(SDL_SCANCODE_RCTRL);
            bool alt_down = engine.input().key_down(SDL_SCANCODE_LALT) || engine.input().key_down(SDL_SCANCODE_RALT);
            if (g_creation.active) {
                // Scroll resizes preview
                if (scroll != 0.0f) g_creation.size = glm::clamp(g_creation.size + scroll * 0.03f, 0.05f, 1.5f);
            } else if (foot_control_mode) {
                foot_wheel_delta = scroll;
            } else if (magnet_resize_mode && alt_down) {
                if (scroll != 0.0f) resize_magnet_falloff_radius(scroll * 0.03f);
            } else if (magnet_resize_mode && ctrl_down) {
                if (scroll != 0.0f) resize_magnet_main_radius(scroll * 0.03f);
            } else if (launcher_resize_mode && alt_down) {
                if (scroll != 0.0f) g_ball_cone_deg = glm::clamp(g_ball_cone_deg + scroll * 2.5f, 0.0f, 65.0f);
            } else if (launcher_resize_mode && ctrl_down) {
                if (scroll != 0.0f) g_ball_radius = glm::clamp(g_ball_radius + scroll * 0.02f, 0.05f, 0.45f);
            } else if (interaction_resize_mode && alt_down) {
                if (scroll != 0.0f) resize_drag_falloff_radius(scroll * 0.03f);
            } else if (interaction_resize_mode && ctrl_down) {
                if (scroll != 0.0f) resize_drag_main_radius(scroll * 0.03f);
            } else {
                if (scroll != 0.0f) g_camera.zoom(1.0f + scroll * 0.1f);
            }
            if (engine.input().mouse_down(SDL_BUTTON_MIDDLE)) {
                ng::vec2 delta = engine.input().mouse_delta();
                g_camera.pan(ng::vec2(-delta.x, delta.y));
            }
        }
        g_camera.set_viewport(engine.window().width(), engine.window().height());
        ng::vec2 mouse_world = g_camera.screen_to_world(engine.input().mouse_pos());
        const bool allow_temp_hotkeys = (!g_creation.active && !g_selection_mode && !imgui_mouse);
        const bool heat_hotkey = allow_temp_hotkeys && engine.input().key_down(SDL_SCANCODE_G);
        const bool cool_hotkey = allow_temp_hotkeys && engine.input().key_down(SDL_SCANCODE_H);
        const bool magnet_hotkey = allow_temp_hotkeys && engine.input().key_down(SDL_SCANCODE_M);
        const InteractMode active_mode = g_mode;
        const bool foot_mode_active = (!g_creation.active && !g_selection_mode && !imgui_mouse &&
                                       active_mode == InteractMode::FOOT_CONTROL && ng::foot_demo_active());
        const bool launcher_mode_active = (!g_creation.active && !g_selection_mode && !imgui_mouse &&
                                           !magnet_hotkey &&
                                           (active_mode == InteractMode::DROP_BALL ||
                                            active_mode == InteractMode::LAUNCHER2));
        const bool ctrl_down = engine.input().key_down(SDL_SCANCODE_LCTRL) || engine.input().key_down(SDL_SCANCODE_RCTRL);
        const bool alt_down = engine.input().key_down(SDL_SCANCODE_LALT) || engine.input().key_down(SDL_SCANCODE_RALT);
        ng::vec2 mouse_delta_screen = engine.input().mouse_delta();
        ng::vec2 mouse_prev_screen = engine.input().mouse_pos() - mouse_delta_screen;
        ng::vec2 mouse_delta_world = mouse_world - g_camera.screen_to_world(mouse_prev_screen);
        g_creation.preview_pos = mouse_world;
        g_show_size_hint = g_creation.active && engine.input().key_down(SDL_SCANCODE_Q);
        if (imgui_mouse) g_hover_selection = {};
        else update_hover_selection(mouse_world);
        if (g_selection_mode && !g_creation.active && !imgui_mouse &&
            engine.input().mouse_pressed(SDL_BUTTON_LEFT)) {
            if (g_hover_selection.kind != HoverKind::NONE) {
                g_pinned_selection = g_hover_selection;
                g_pinned_selection_open = true;
            } else {
                g_pinned_selection = {};
                g_pinned_selection_open = false;
            }
        }

        // --- Creation placement (right-click) ---
        if (g_creation.active && !imgui_mouse && engine.input().mouse_pressed(SDL_BUTTON_RIGHT)) {
            ng::place_object(g_creation, g_particles, g_sph, g_mpm, g_mpm_grid);
        }

        ng::FootControlInput foot_input;
        foot_input.mouse_world = mouse_world;
        foot_input.mouse_delta_world = mouse_delta_world;
        foot_input.dt = dt;
        foot_input.wheel_delta = foot_mode_active ? foot_wheel_delta : 0.0f;
        foot_input.lmb_down = foot_mode_active && engine.input().mouse_down(SDL_BUTTON_LEFT);
        foot_input.lmb_pressed = foot_mode_active && engine.input().mouse_pressed(SDL_BUTTON_LEFT);
        foot_input.rmb_down = foot_mode_active && engine.input().mouse_down(SDL_BUTTON_RIGHT);
        foot_input.shift_down = engine.input().key_down(SDL_SCANCODE_LSHIFT) || engine.input().key_down(SDL_SCANCODE_RSHIFT);
        foot_input.ctrl_down = ctrl_down;
        foot_input.cycle_prev_pressed = foot_mode_active && engine.input().key_pressed(SDL_SCANCODE_LEFTBRACKET);
        foot_input.cycle_next_pressed = foot_mode_active && engine.input().key_pressed(SDL_SCANCODE_RIGHTBRACKET);
        foot_input.curl_down = foot_mode_active && engine.input().key_down(SDL_SCANCODE_Z);
        foot_input.straighten_down = foot_mode_active && engine.input().key_down(SDL_SCANCODE_S);
        foot_input.contract_down = foot_mode_active && engine.input().key_down(SDL_SCANCODE_C);
        foot_input.extend_down = foot_mode_active && engine.input().key_down(SDL_SCANCODE_B);
        ng::update_foot_demo(foot_input, g_particles, g_mpm, &g_sdf);
        if (ng::foot_demo_active() &&
            (active_mode == InteractMode::DRAG || active_mode == InteractMode::SPRING_DRAG)) {
            g_mpm.clear_kinematic_targets();
        }

        // --- Color mode: batch colors vs debug vis ---
        {
            bool batch_mode = (g_color_mode == 1);
            g_mpm.params().keep_colors = batch_mode;
            const_cast<ng::SPHParams&>(g_sph.params()).keep_colors = batch_mode;
        }

        // --- Tool interaction (left-click, only when creation menu is off) ---
        ng::MouseForce mouse_force;
        bool spring_mode_active = (!g_creation.active && !g_selection_mode && !imgui_mouse &&
                                   !magnet_hotkey &&
                                   (active_mode == InteractMode::DRAG || active_mode == InteractMode::SPRING_DRAG));
        if (!spring_mode_active || !engine.input().mouse_down(SDL_BUTTON_LEFT)) {
            clear_spring_drag();
        } else if (engine.input().mouse_pressed(SDL_BUTTON_LEFT)) {
            g_drag_anchor_world = mouse_world;
            g_drag_capture_active = true;
            sync_drag_falloff_radius();
            g_sph.begin_spring_drag(g_particles, mouse_world, g_tool_radius, g_drag_falloff_radius);
            g_mpm.begin_spring_drag(g_particles, mouse_world, g_tool_radius, g_drag_falloff_radius);
        }
        if (!g_creation.active && !g_selection_mode && !imgui_mouse && !magnet_hotkey) {
            if (launcher_mode_active && engine.input().mouse_pressed(SDL_BUTTON_RIGHT)) {
                g_ball_drag_anchor = mouse_world;
                if (alt_down && !ctrl_down) {
                    g_ball_drag_mode = ProjectileDragMode::CONE;
                    g_ball_cone_drag_start_deg = g_ball_cone_deg;
                } else {
                    g_ball_drag_mode = ProjectileDragMode::AIM;
                }
            }
            if (launcher_mode_active &&
                g_ball_drag_mode == ProjectileDragMode::AIM &&
                engine.input().mouse_down(SDL_BUTTON_RIGHT)) {
                ng::vec2 aim = mouse_world - g_ball_drag_anchor;
                if (glm::length(aim) > 0.05f) {
                    g_ball_launch_vector = aim;
                    g_ball_has_aim = true;
                }
            }
            if (launcher_mode_active &&
                g_ball_drag_mode == ProjectileDragMode::CONE &&
                engine.input().mouse_down(SDL_BUTTON_RIGHT)) {
                ng::vec2 ref = glm::normalize(current_projectile_vector());
                ng::vec2 delta = mouse_world - g_ball_drag_anchor;
                if (glm::length(delta) > 0.03f) {
                    ng::f32 cos_theta = glm::clamp(glm::dot(glm::normalize(delta), ref), -1.0f, 1.0f);
                    g_ball_cone_deg = glm::clamp(glm::degrees(std::acos(cos_theta)), 0.0f, 65.0f);
                } else {
                    g_ball_cone_deg = g_ball_cone_drag_start_deg;
                }
            }
            if (launcher_mode_active &&
                g_ball_drag_mode != ProjectileDragMode::NONE &&
                engine.input().mouse_released(SDL_BUTTON_RIGHT)) {
                g_ball_drag_mode = ProjectileDragMode::NONE;
            }

            if ((active_mode == InteractMode::DROP_BALL || active_mode == InteractMode::LAUNCHER2) &&
                engine.input().mouse_pressed(SDL_BUTTON_LEFT)) {
                fire_projectile(mouse_world);
            } else if (engine.input().mouse_down(SDL_BUTTON_LEFT) || engine.input().mouse_down(SDL_BUTTON_RIGHT)) {
                switch (active_mode) {
                case InteractMode::PUSH:
                    sync_drag_falloff_radius();
                    if (engine.input().mouse_down(SDL_BUTTON_LEFT) && !engine.input().mouse_down(SDL_BUTTON_RIGHT)) {
                        mouse_force = { mouse_world, g_drag_falloff_radius, g_tool_radius, g_tool_force, ng::vec2(0.0f), 0.0f, 0 };
                    } else if (engine.input().mouse_down(SDL_BUTTON_RIGHT) && !engine.input().mouse_down(SDL_BUTTON_LEFT)) {
                        mouse_force = { mouse_world, g_drag_falloff_radius, g_tool_radius, -g_tool_force, ng::vec2(0.0f), 0.0f, 0 };
                    }
                    break;
                case InteractMode::DRAG: {
                    mouse_force = { mouse_world, g_tool_radius, 0.0f, g_drag_force,
                                    ng::vec2(0.0f), g_spring_damping * 0.45f, 1 };
                    break;
                }
                case InteractMode::SWEEP_DRAG: {
                    ng::vec2 delta = engine.input().mouse_delta();
                    if (glm::length(delta) > 0.5f) {
                        g_drag_dir = glm::normalize(ng::vec2(delta.x, -delta.y));
                    }
                    mouse_force = { mouse_world, g_tool_radius, g_tool_radius * g_drag_inner_ratio,
                                    g_drag_force, g_drag_dir, 0.0f, 2 };
                    break;
                }
                case InteractMode::SPRING_DRAG:
                    mouse_force = { mouse_world, g_tool_radius, g_tool_radius * g_drag_inner_ratio,
                                    g_spring_force, ng::vec2(0.0f), g_spring_damping, 3 };
                    break;
                case InteractMode::DRAW_WALL:
                    g_sdf.stamp_circle(mouse_world, 0.1f, true);
                    break;
                case InteractMode::ERASE_WALL:
                    g_sdf.stamp_circle(mouse_world, 0.15f, false);
                    break;
                case InteractMode::DROP_BALL:
                    break;
                case InteractMode::FOOT_CONTROL:
                    break;
                case InteractMode::PULL:
                case InteractMode::MAGNET:
                    break;
                }
            }
        }

        // Heat gun
        ng::MPMParams& mp = g_mpm.params();
        ng::MagneticField::Params& magp = g_magnetic.params();
        mp.heat_gun_pos = mouse_world;
        mp.heat_gun_power = 0.0f;
        mp.magnet_pos = mouse_world;
        mp.magnet_force = 0.0f;
        magp.cursor_pos = mouse_world;
        magp.cursor_dir = ng::vec2(0.0f, 1.0f);
        magp.cursor_radius = mp.magnet_radius;
        magp.cursor_falloff_radius = mp.magnet_falloff_radius;
        magp.cursor_strength = 0.0f;
        const bool magnet_tool_active = (!g_creation.active && !g_selection_mode && !imgui_mouse &&
                                         magnet_hotkey);
        if (!imgui_mouse) {
            if (heat_hotkey) mp.heat_gun_power = 800.0f;
            if (cool_hotkey) mp.heat_gun_power = -800.0f;
            if (magnet_tool_active) {
                // Cursor field comes from the selected preset below.
                // The "Real Magnetics" toggle (magp.enabled) only controls
                // whether scene magnets participate; the cursor field is independent.
                // rasterize shader multiplies by source_scale (~18), so
                // peak M = cursor_strength * source_scale. Scene magnets use
                // strength ~6.5, so 8-10 here makes the cursor slightly stronger
                // than the average scene magnet — enough to visibly distort the
                // field even in strong-field scenes.
                magp.cursor_strength = g_magnet_strength * 0.09f;
            }
        }

        // --- Physics ---
        g_magnetic.step(g_sdf, &g_particles);
        if (!g_paused) {
            // Auto-enable thermal when any MPM particles exist (for heat propagation)
            if (g_mpm.particle_count() > 0 && !mp.enable_thermal) {
                mp.enable_thermal = true;
            }
            ng::SPHParams& sp_live = const_cast<ng::SPHParams&>(g_sph.params());
            const bool sph_has_thermal = has_sph_thermal_batches();
            sp_live.enable_thermal = mp.enable_thermal || sph_has_thermal || std::abs(mp.heat_gun_power) > 1e-4f;
            sp_live.ambient_temp = g_air.config().ambient_temp;
            sp_live.heat_source_pos = mp.heat_source_pos;
            sp_live.heat_source_radius = mp.heat_source_radius;
            sp_live.heat_source_temp = mp.heat_source_temp;
            sp_live.heat_gun_pos = mp.heat_gun_pos;
            sp_live.heat_gun_radius = mp.heat_gun_radius;
            sp_live.heat_gun_power = mp.heat_gun_power;
            sp_live.particle_cooling_rate = mp.particle_cooling_rate;
            sp_live.time = total_time;
            // Sync ambient temp from Eulerian air to MPM
            mp.ambient_temp = g_air.config().ambient_temp;
            mp.physically_based_heat = g_air.config().physically_based_heat;
            ng::f32 scene_heat_radius = mp.heat_source_radius;
            ng::f32 scene_heat_temp = mp.heat_source_temp;
            if (!g_show_heat_gizmos) {
                mp.heat_source_radius = 0.0f;
                mp.heat_source_temp = 0.0f;
                sp_live.heat_source_radius = 0.0f;
                sp_live.heat_source_temp = 0.0f;
            }

            ng::f32 physics_dt = 1.0f / 120.0f;
            g_mpm_grid.contact_temp_buf().clear();
            if (sp_live.enable_thermal && g_particles.range(ng::SolverType::SPH).count > 0) {
                g_sph.scatter_contact_heat(g_particles, g_mpm_grid);
            }
            g_mpm.step(g_particles, g_mpm_grid, physics_dt, &g_sdf, &g_magnetic, &g_air,
                       mouse_force.world_pos, mouse_force.radius, mouse_force.force,
                       mouse_force.drag_dir, mouse_force.mode, mouse_force.inner_radius,
                       mouse_force.damping);
            g_sph.step(g_particles, g_hash, physics_dt, &g_sdf, mouse_force, &g_mpm_grid, &g_air);
            g_air.clear_particle_injection_sources();
            update_pressure_vessels(physics_dt);

            // Eulerian air: inject heat from particles, solve, apply drag
            auto& mpm_r = g_particles.range(ng::SolverType::MPM);
            auto& sph_r = g_particles.range(ng::SolverType::SPH);
            g_mpm.jp_buf().bind_base(47);
            g_mpm.damage_buf().bind_base(49);
            g_mpm.phase_buf().bind_base(50);
            g_mpm.mat_params_buf().bind_base(51);
            g_mpm.thermal_coupling_buf().bind_base(54);
            g_air.update_airtight_from_particles(g_particles, mpm_r.offset, mpm_r.count, physics_dt, &g_sdf);
            g_air.inject_from_particles(g_particles, mpm_r.offset, mpm_r.count, physics_dt);
            if (sp_live.enable_thermal && sph_r.count > 0) {
                g_sph.thermal_coupling_buf().bind_base(54);
                g_air.inject_from_particles(g_particles, sph_r.offset, sph_r.count, physics_dt);
            }

            // Blow tool: hold B + left-click to inject wind at cursor
            if (!imgui_mouse && engine.input().key_down(SDL_SCANCODE_B) &&
                engine.input().mouse_down(SDL_BUTTON_LEFT)) {
                // Blow in direction of mouse movement, or upward if no movement
                ng::vec2 md = engine.input().mouse_delta();
                ng::vec2 blow_dir;
                if (glm::length(md) > 1.0f) {
                    blow_dir = glm::normalize(ng::vec2(md.x, -md.y));
                } else {
                    blow_dir = ng::vec2(0.0f, 1.0f); // Default: blow upward
                }
                g_air.blow_at(mouse_world, blow_dir, g_blow_radius, g_blow_strength, physics_dt);
            }
            if (g_scene == ng::SceneID::WIND_TUNNEL ||
                g_scene == ng::SceneID::OVEN_OPEN_WIND ||
                g_scene == ng::SceneID::POT_HEATER_WIND) {
                g_air.blow_at(ng::vec2(-2.7f, -0.1f), ng::vec2(1.0f, 0.0f), 1.25f, 7.5f, physics_dt);
                g_air.blow_at(ng::vec2(-2.7f, 1.2f), ng::vec2(1.0f, 0.0f), 1.15f, 6.0f, physics_dt);
            }
            // Heat gun also heats the air directly
            g_air.inject_heat_at(mp.heat_gun_pos, mp.heat_gun_radius,
                                 mp.heat_gun_power, physics_dt);
            ng::i32 air_vis_mode = (g_air_vis == 0) ? 1 : (g_air_vis == 1) ? 0 : g_air_vis;
            g_air.set_visualization_mode(air_vis_mode);
            g_air.set_particle_visualization_mode(g_mpm.params().vis_mode);
            g_air.step(physics_dt, &g_sdf);
            // Apply air drag to MPM particles (light particles feel air more)
            g_mpm.thermal_coupling_buf().bind_base(54);
            g_air.apply_drag_to_particles(g_particles, mpm_r.offset, mpm_r.count, physics_dt, 0.5f);
            if (sp_live.enable_thermal && sph_r.count > 0) {
                g_sph.thermal_coupling_buf().bind_base(54);
                g_air.apply_drag_to_particles(g_particles, sph_r.offset, sph_r.count, physics_dt, 0.22f);
            }
            mp.heat_source_radius = scene_heat_radius;
            mp.heat_source_temp = scene_heat_temp;
        }

        // --- Set highlight range in solver params (works in BOTH color modes) ---
        {
            ng::u32 hs = 0, he = 0;
            int active_batch = g_creation.highlighted_batch;
            ng::u32 selected_sdf_object_id = 0;
            if (g_selection_mode) {
                const HoverSelection& selection_ref =
                    g_pinned_selection_open ? g_pinned_selection : g_hover_selection;
                if (selection_ref.kind == HoverKind::BATCH) {
                    active_batch = selection_ref.batch_index;
                } else if (selection_ref.kind == HoverKind::SDF) {
                    active_batch = -1;
                    selected_sdf_object_id = selection_ref.sdf_object_id;
                }
            }
            if (active_batch >= 0 &&
                active_batch < static_cast<ng::i32>(g_creation.batches.size())) {
                auto& b = g_creation.batches[active_batch];
                if (b.sph_count > 0) { hs = b.sph_offset; he = b.sph_offset + b.sph_count; }
                if (b.mpm_count > 0) { hs = b.mpm_offset; he = b.mpm_offset + b.mpm_count; }
            }
            ng::MPMParams& mp2 = g_mpm.params();
            mp2.highlight_start = hs; mp2.highlight_end = he; mp2.time = total_time;
            ng::SPHParams& sp2 = const_cast<ng::SPHParams&>(g_sph.params());
            sp2.highlight_start = hs; sp2.highlight_end = he; sp2.time = total_time;

            // --- Render ---
            glClearColor(0.02f, 0.02f, 0.04f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            // air_vis: 0=smoke+fire(mode1), 1=off(mode0), 2=temp(mode2), 3=density(mode3)
            ng::i32 air_mode = (g_air_vis == 0) ? 1 : (g_air_vis == 1) ? 0 : g_air_vis;
            g_sdf_renderer.render(g_sdf, g_camera, g_air.smoke_texture(), g_air.temp_texture(),
                                  g_air.velocity_texture(), g_air.bio_field_texture(), g_air.automata_texture(),
                                  g_air.bio_field_view_gain(), g_air.automata_view_gain(),
                                  air_mode, g_sdf_palette,
                                  g_fire_vis_start_temp, g_fire_vis_temp_range, g_fire_vis_softness,
                                  selected_sdf_object_id);
        }

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        auto& sph_range = g_particles.range(ng::SolverType::SPH);
        auto& mpm_range = g_particles.range(ng::SolverType::MPM);

        ng::f32 sph_pt = glm::clamp(g_camera.zoom_level() * g_sph.params().smoothing_radius * 0.6f, 1.0f, 20.0f);
        if (g_metaball.enabled) {
            g_metaball.splat(g_particles, g_camera, sph_range.offset, sph_range.count,
                g_sph.params().smoothing_radius * g_metaball.kernel_scale);
            g_metaball.render_surface(g_metaball.threshold,
                static_cast<ng::SurfaceStyle>(g_sph_surface_style));
        }
        if (!g_metaball.enabled || g_metaball.keep_particles) {
            g_particle_renderer.render(g_particles, g_camera, sph_range.offset, sph_range.count, sph_pt);
        }

        ng::f32 mpm_pt = glm::clamp(g_camera.zoom_level() * g_mpm_grid.dx() * 0.4f, 1.0f, 20.0f);
        if (g_mpm_skin_enabled) {
            g_metaball.splat(g_particles, g_camera, mpm_range.offset, mpm_range.count,
                g_mpm_grid.dx() * g_mpm_skin_kernel);
            g_metaball.render_surface(g_mpm_skin_threshold,
                static_cast<ng::SurfaceStyle>(g_mpm_surface_style));
        }
        if (!g_mpm_skin_enabled || g_metaball.keep_particles) {
            g_particle_renderer.render(g_particles, g_camera, mpm_range.offset, mpm_range.count, mpm_pt);
        }

        // Bloom: render highlighted / hovered batch to FBO, blur, composite softly
        int bloom_batch = g_creation.highlighted_batch;
        if (g_selection_mode) {
            const HoverSelection& selection_ref =
                g_pinned_selection_open ? g_pinned_selection : g_hover_selection;
            bloom_batch = (selection_ref.kind == HoverKind::BATCH) ? selection_ref.batch_index : -1;
        }
        if (bloom_batch >= 0 &&
            bloom_batch < static_cast<ng::i32>(g_creation.batches.size())) {
            auto& b = g_creation.batches[bloom_batch];
            if (b.sph_count > 0)
                g_bloom.capture(g_particles, g_camera, b.sph_offset, b.sph_count, sph_pt * 2.5f);
            if (b.mpm_count > 0)
                g_bloom.capture(g_particles, g_camera, b.mpm_offset, b.mpm_count, mpm_pt * 2.5f);
            g_bloom.apply(0.6f); // Visible golden glow
        }

        // Preview outline
        render_preview(total_time);
        const bool pull_lmb = (!g_creation.active && !g_selection_mode && !imgui_mouse &&
                               active_mode == InteractMode::PUSH &&
                               engine.input().mouse_down(SDL_BUTTON_LEFT) &&
                               !engine.input().mouse_down(SDL_BUTTON_RIGHT) &&
                               !magnet_hotkey);
        const bool push_rmb = (!g_creation.active && !g_selection_mode && !imgui_mouse &&
                               active_mode == InteractMode::PUSH &&
                               engine.input().mouse_down(SDL_BUTTON_RIGHT) &&
                               !engine.input().mouse_down(SDL_BUTTON_LEFT) &&
                               !magnet_hotkey);
        render_tool_indicator(mouse_world, total_time, active_mode, pull_lmb, push_rmb);
        const bool magnet_preview_active = (!g_creation.active && !g_selection_mode && !imgui_mouse &&
                                            magnet_hotkey);
        if (magnet_preview_active) {
            g_preview_shader.bind();
            g_preview_shader.set_mat4("u_view_proj", g_camera.view_proj());
            g_preview_shader.set_vec2("u_center", mouse_world);
            g_preview_shader.set_int("u_shape", 0);
            g_preview_shader.set_float("u_time", total_time);
            g_preview_shader.set_float("u_rotation", 0.0f);
            g_preview_shader.set_vec2("u_size", ng::vec2(g_mpm.params().magnet_radius));
            g_preview_shader.set_vec4("u_color", ng::vec4(0.78f, 0.92f, 1.0f, 0.68f));
            glBindVertexArray(g_preview_vao);
            glLineWidth(2.0f);
            glDrawArrays(GL_LINE_LOOP, 0, 64);
            if (g_mpm.params().magnet_falloff_radius > g_mpm.params().magnet_radius + 1e-4f) {
                g_preview_shader.set_vec2("u_size", ng::vec2(g_mpm.params().magnet_falloff_radius));
                g_preview_shader.set_vec4("u_color", ng::vec4(0.62f, 0.84f, 1.0f, 0.42f));
                glDrawArrays(GL_LINE_LOOP, 0, 64);
            }
            glBindVertexArray(0);
            g_preview_shader.unbind();
        }

        // Heat field visualization: filled gradient + dashed outline
        if (g_show_heat_gizmos && (mp.enable_thermal || mp.heat_gun_power != 0.0f)) {
            // Filled gradient glow (additive)
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE);
            glEnable(GL_PROGRAM_POINT_SIZE);

            g_heat_glow_shader.bind();
            g_heat_glow_shader.set_mat4("u_view_proj", g_camera.view_proj());

            if (mp.enable_thermal) {
                // Upload heat source position temporarily to position buffer slot
                ng::vec2 hpos = mp.heat_source_pos;
                // Use a temporary single-vertex approach: draw a huge point at heat source
                // We need to set position in the shader. Use u_offset=-1 trick... or just
                // upload to an unused spot in the buffer.
                // Simplest: render using the preview shader in filled mode instead.
            }
            g_heat_glow_shader.unbind();
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            // Dashed outlines for precise boundary
            g_preview_shader.bind();
            g_preview_shader.set_mat4("u_view_proj", g_camera.view_proj());
            g_preview_shader.set_int("u_shape", 0);
            g_preview_shader.set_float("u_time", total_time);
            g_preview_shader.set_float("u_rotation", 0.0f);

            if (mp.enable_thermal) {
                g_preview_shader.set_vec2("u_center", mp.heat_source_pos);
                g_preview_shader.set_vec2("u_size", ng::vec2(mp.heat_source_radius));
                ng::f32 t_norm = glm::clamp(mp.heat_source_temp / 1500.0f, 0.0f, 1.0f);
                g_preview_shader.set_vec4("u_color", ng::vec4(1.0f, 0.3f + 0.4f * t_norm, 0.1f, 0.5f));
                glBindVertexArray(g_preview_vao);
                glDrawArrays(GL_LINE_LOOP, 0, 64);
            }

            if (mp.heat_gun_power != 0.0f) {
                g_preview_shader.set_vec2("u_center", mouse_world);
                g_preview_shader.set_vec2("u_size", ng::vec2(mp.heat_gun_radius));
                if (mp.heat_gun_power > 0)
                    g_preview_shader.set_vec4("u_color", ng::vec4(1.0f, 0.4f, 0.1f, 0.8f));
                else
                    g_preview_shader.set_vec4("u_color", ng::vec4(0.2f, 0.5f, 1.0f, 0.8f));
                glBindVertexArray(g_preview_vao);
                glDrawArrays(GL_LINE_LOOP, 0, 64);
            }
            glBindVertexArray(0);
            g_preview_shader.unbind();
        }

        render_magnetic_field_shader_overlay(total_time);

        // --- ImGui ---
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        // (magnetic field overlay moved to AFTER the Engine panel so it
        //  picks up the just-clicked "Force Solver Always" checkbox state
        //  with zero latency — see end of ImGui frame.)
        draw_active_tooltip(active_mode, heat_hotkey, cool_hotkey, magnet_hotkey,
                            pull_lmb || (active_mode != InteractMode::PUSH && engine.input().mouse_down(SDL_BUTTON_LEFT)),
                            push_rmb);
        draw_creation_menu();
        draw_selection_tooltip();
        draw_selection_editor();

        if (g_show_ui) {
            draw_ui_top_bar(frame_ms);
            ImGuiViewport* viewport = ImGui::GetMainViewport();
            ng::f32 start_x = viewport->WorkPos.x + 10.0f;
            ng::f32 cursor_x = start_x;
            ng::f32 cursor_y = viewport->WorkPos.y + 56.0f;
            ng::f32 row_height = 0.0f;
            ng::f32 max_x = viewport->WorkPos.x + viewport->WorkSize.x - 10.0f;

            const ImVec2 interaction_size(360.0f, 520.0f);
            const ImVec2 environment_size(350.0f, 430.0f);
            const ImVec2 backends_size(380.0f, 700.0f);
            const ImVec2 appearance_size(350.0f, 650.0f);
            const ImVec2 advanced_size(360.0f, 700.0f);
            const ImVec2 presets_size(340.0f, 230.0f);

            if (g_show_interaction_window) {
                ImVec2 pos = next_panel_position(interaction_size, cursor_x, cursor_y, row_height, start_x, max_x);
                draw_interaction_window(pos, interaction_size, kInteractionAccent);
            }
            if (g_show_environment_window) {
                ImVec2 pos = next_panel_position(environment_size, cursor_x, cursor_y, row_height, start_x, max_x);
                draw_environment_window(pos, environment_size, kEnvironmentAccent);
            }
            if (g_show_backends_window) {
                ImVec2 pos = next_panel_position(backends_size, cursor_x, cursor_y, row_height, start_x, max_x);
                draw_backends_window(pos, backends_size, kBackendsAccent);
            }
            if (g_show_appearance_window) {
                ImVec2 pos = next_panel_position(appearance_size, cursor_x, cursor_y, row_height, start_x, max_x);
                draw_appearance_window(pos, appearance_size, kAppearanceAccent);
            }
            if (g_show_advanced_window) {
                ImVec2 pos = next_panel_position(advanced_size, cursor_x, cursor_y, row_height, start_x, max_x);
                draw_advanced_window(pos, advanced_size, kAdvancedAccent);
            }
            if (g_show_presets_window) {
                ImVec2 pos = next_panel_position(presets_size, cursor_x, cursor_y, row_height, start_x, max_x);
                draw_presets_window(pos, presets_size, kPresetsAccent);
            }
        }

        if (g_show_pipeline_prev && !g_show_pipeline) {
            ng::reset_pipeline_viewer_session();
        }
        g_show_pipeline_prev = g_show_pipeline;
        if (g_show_pipeline)
            ng::draw_pipeline_viewer(&g_show_pipeline);

        // Magnetic field debug overlay — drawn LAST so it sees the
        // just-processed ImGui state (including the "Force Solver Always"
        // checkbox toggled this frame). If debug-force is on, re-run
        // the solve once using the current frame's params before drawing,
        // so the viz has zero-latency response to the checkbox. Normal
        // frames pay nothing: this path only runs when debug mode is on.
        if (g_magnetic.params().debug_force_active) {
            g_magnetic.step(g_sdf, &g_particles);
        }
        draw_magnetic_field_overlay(active_mode);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        engine.window().swap();
        frame++;
        if (frame % 120 == 0)
            LOG_INFO("F%u | %.1fms | SPH:%u MPM:%u", frame, frame_ms,
                g_sph.particle_count(), g_mpm.particle_count());
    }

    LOG_INFO("Clean shutdown.");
    return 0;
}
