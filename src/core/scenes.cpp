#include "core/scenes.h"
#include "core/log.h"
#include "core/creation_menu.h"
#include "core/foot_demo.h"
#include "physics/common/particle_buffer.h"
#include "physics/sph/sph_solver.h"
#include "physics/mpm/mpm_solver.h"
#include "physics/common/grid.h"
#include "physics/sdf/sdf_field.h"

#include <cmath>
#include <utility>

namespace ng {

// ---- helpers ----
static void reset_creation(CreationState* creation) {
    if (!creation) return;
    creation->batches.clear();
    creation->highlighted_batch = -1;
    creation->batch_counter = 0;
}

static void push_scene_batch(CreationState* creation, BatchRecord&& batch) {
    if (!creation) return;
    creation->batches.push_back(std::move(batch));
    creation->batch_counter = static_cast<u32>(creation->batches.size());
}

static void add_floor_and_walls(
    SDFField& sdf,
    SDFField::MaterialPresetID preset = SDFField::MaterialPresetID::GLOBAL_DEFAULT) {
    sdf.add_segment(vec2(-2.5f, -1.5f), vec2(2.5f, -1.5f), 0.15f, preset,
                    "Floor", "Rigid floor collider with solid heat diffusion.");
    sdf.add_segment(vec2(-2.5f, -1.5f), vec2(-2.5f, 2.5f), 0.15f, preset,
                    "Left Wall", "Rigid SDF wall.");
    sdf.add_segment(vec2(2.5f, -1.5f),  vec2(2.5f, 2.5f),  0.15f, preset,
                    "Right Wall", "Rigid SDF wall.");
}

static void add_floor_and_walls_extents(
    SDFField& sdf,
    vec2 world_min,
    vec2 world_max,
    SDFField::MaterialPresetID preset = SDFField::MaterialPresetID::GLOBAL_DEFAULT,
    f32 thickness = 0.15f) {
    sdf.add_segment(vec2(world_min.x, world_min.y), vec2(world_max.x, world_min.y), thickness, preset,
                    "Floor", "Rigid floor collider with solid heat diffusion.");
    sdf.add_segment(vec2(world_min.x, world_min.y), vec2(world_min.x, world_max.y), thickness, preset,
                    "Left Wall", "Rigid SDF wall.");
    sdf.add_segment(vec2(world_max.x, world_min.y), vec2(world_max.x, world_max.y), thickness, preset,
                    "Right Wall", "Rigid SDF wall.");
}

static void add_codim_threads_geometry(
    SDFField& sdf,
    SDFField::MaterialPresetID preset = SDFField::MaterialPresetID::GLOBAL_DEFAULT) {
    sdf.add_segment(vec2(-2.5f, -1.5f), vec2(2.5f, -1.5f), 0.15f, preset,
                    "Thread Floor", "Floor for codimensional drip tests.");
    sdf.add_segment(vec2(-2.5f, -1.5f), vec2(-2.5f, 2.5f), 0.15f, preset,
                    "Left Wall", "Side wall.");
    sdf.add_segment(vec2(2.5f, -1.5f), vec2(2.5f, 2.5f), 0.15f, preset,
                    "Right Wall", "Side wall.");
    sdf.add_box(vec2(-1.5f, 1.0f), vec2(0.6f, 0.06f), preset,
                "Upper Shelf", "Platform that sheds codimensional drips.");
    sdf.add_box(vec2(1.5f, 0.3f), vec2(0.6f, 0.06f), preset,
                "Lower Shelf", "Lower platform for drip and splash tests.");
    sdf.add_segment(vec2(-0.3f, -1.5f), vec2(-0.3f, 0.0f), 0.08f, preset,
                    "Gap Left Rail", "Narrow slot boundary for thread formation.");
    sdf.add_segment(vec2(0.3f, -1.5f), vec2(0.3f, 0.0f), 0.08f, preset,
                    "Gap Right Rail", "Narrow slot boundary for thread formation.");
    sdf.add_segment(vec2(-0.8f, 0.5f), vec2(-0.3f, 0.0f), 0.06f, preset,
                    "Funnel Left", "Guides fluid into the narrow slit.");
    sdf.add_segment(vec2(0.8f, 0.5f), vec2(0.3f, 0.0f), 0.06f, preset,
                    "Funnel Right", "Guides fluid into the narrow slit.");
}

static void add_spiral(SDFField& sdf, vec2 center, f32 r0, f32 r1, f32 turns, f32 thickness,
                       SDFField::MaterialPresetID preset,
                       const char* label, const char* summary) {
    const i32 steps = 40;
    vec2 prev = center + vec2(r0, 0.0f);
    for (i32 i = 1; i <= steps; ++i) {
        f32 t = static_cast<f32>(i) / static_cast<f32>(steps);
        f32 angle = turns * 6.2831853f * t;
        f32 radius = glm::mix(r0, r1, t);
        vec2 cur = center + vec2(std::cos(angle), std::sin(angle)) * radius;
        sdf.add_segment(prev, cur, thickness, preset, label, summary);
        prev = cur;
    }
}

static void register_scene_sph_batch(CreationState* creation, const char* label, const char* description,
                                     ParticleBuffer& particles, u32 before_count, vec4 color,
                                     f32 recommended_size = 0.4f,
                                     const char* recommended_note = "Medium fluid blocks are easiest to read.",
                                     MPMMaterial material = MPMMaterial::SPH_WATER,
                                     f32 initial_temp = 300.0f) {
    if (!creation) return;
    u32 after_count = particles.range(SolverType::SPH).count;
    if (after_count <= before_count) return;

    const MaterialPreset* preset = preset_for_material(SpawnSolver::SPH, material);
    BatchRecord batch = make_batch_record(
        label,
        description ? description : (preset ? preset->description : "Scene-authored SPH liquid."),
        SpawnSolver::SPH, material, color,
        0.0f, 0.0f, initial_temp, vec2(1.0f, 0.0f),
        recommended_size, recommended_note, true);
    batch.temperature = initial_temp;
    if (preset) {
        batch.gas_constant = preset->gas_constant;
        batch.viscosity = preset->viscosity;
        vec4 thermal_defaults = default_thermal_coupling(material);
        batch.outgassing_scale = thermal_defaults.x * preset->outgassing_scale;
        batch.heat_release_scale = thermal_defaults.y * preset->heat_release_scale;
        batch.cooling_scale = thermal_defaults.z * preset->cooling_scale;
        batch.loft_scale = thermal_defaults.w * preset->loft_scale;
        batch.recommended_size = preset->recommended_size;
        batch.recommended_note = preset->recommended_note ? preset->recommended_note : recommended_note;
        batch.description = description ? description : preset->description;
    }
    batch.properties = property_summary(batch.solver, batch.mpm_type,
                                        batch.youngs_modulus, batch.poisson_ratio,
                                        batch.temperature, batch.fiber_dir);
    {
        char fluid_buf[192];
        snprintf(fluid_buf, sizeof(fluid_buf),
                 " | k %.2f | visc %.3f | outgas %.2f | air heat %.2f | cool %.2f | loft %.2f",
                 batch.gas_constant, batch.viscosity, batch.outgassing_scale,
                 batch.heat_release_scale, batch.cooling_scale, batch.loft_scale);
        batch.properties += fluid_buf;
    }
    batch.sph_offset = particles.range(SolverType::SPH).offset + before_count;
    batch.sph_count = after_count - before_count;
    push_scene_batch(creation, std::move(batch));
}

static void register_scene_mpm_batch(CreationState* creation, const char* label, const char* description,
                                     ParticleBuffer& particles, u32 before_count,
                                     MPMMaterial material, f32 youngs_modulus,
                                     f32 poisson_ratio, f32 initial_temp, vec2 fiber_dir,
                                     f32 density_scale = 1.0f) {
    if (!creation) return;
    u32 after_count = particles.range(SolverType::MPM).count;
    if (after_count <= before_count) return;

    const MaterialPreset* preset = preset_for_material(SpawnSolver::MPM, material);
    BatchRecord batch = make_batch_record(
        label,
        description ? description : (preset ? preset->description : "Scene-authored MPM object."),
        SpawnSolver::MPM,
        material,
        preset ? preset->color : default_material_color(SpawnSolver::MPM, material),
        youngs_modulus,
        poisson_ratio,
        initial_temp,
        fiber_dir,
        preset ? preset->recommended_size : 0.28f,
        preset ? preset->recommended_note : "Medium chunks are a good first test size.",
        true);
    batch.fiber_strength = preset ? preset->fiber_strength : 0.0f;
    batch.density_scale = density_scale;
    if (preset) {
        vec4 thermal_defaults = default_thermal_coupling(material);
        batch.outgassing_scale = thermal_defaults.x * preset->outgassing_scale;
        batch.heat_release_scale = thermal_defaults.y * preset->heat_release_scale;
        batch.cooling_scale = thermal_defaults.z * preset->cooling_scale;
        batch.loft_scale = thermal_defaults.w * preset->loft_scale;
    }
    batch.properties = property_summary(batch.solver, batch.mpm_type,
                                        batch.youngs_modulus, batch.poisson_ratio,
                                        batch.temperature, batch.fiber_dir,
                                        batch.density_scale);
    {
        char thermal_buf[160];
        snprintf(thermal_buf, sizeof(thermal_buf),
                 " | outgas %.2f | air heat %.2f | cool %.2f | loft %.2f",
                 batch.outgassing_scale, batch.heat_release_scale, batch.cooling_scale,
                 batch.loft_scale);
        batch.properties += thermal_buf;
    }
    batch.mpm_offset = particles.range(SolverType::MPM).offset + before_count;
    batch.mpm_count = after_count - before_count;
    push_scene_batch(creation, std::move(batch));
}

static void set_mpm_batch_velocity(ParticleBuffer& particles, u32 global_offset, u32 count, vec2 velocity) {
    if (count == 0) return;
    std::vector<vec2> velocities(count, velocity);
    particles.upload_velocities(global_offset, velocities.data(), count);
}

// ---- scenes ----
static void load_default(ParticleBuffer& particles, SPHSolver& sph,
                         MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                         CreationState* creation) {
    // Terrain
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(0.0f, -0.5f), vec2(0.5f, 0.08f));               // platform
    sdf.add_circle(vec2(1.0f, 0.0f), 0.25f);                         // circle obstacle
    sdf.add_segment(vec2(-1.5f, -0.2f), vec2(-0.5f, 0.3f), 0.08f);   // ramp
    sdf.rebuild();

    // SPH fluid block
    f32 sph_spacing = sph.params().smoothing_radius * 0.5f;
    u32 sph_before = particles.range(SolverType::SPH).count;
    sph.spawn_block(particles, vec2(-2.2f, -1.2f), vec2(-0.5f, 0.8f), sph_spacing);
    apply_sph_batch_properties(particles, sph,
                               particles.range(SolverType::SPH).offset + sph_before,
                               particles.range(SolverType::SPH).count - sph_before,
                               MPMMaterial::SPH_WATER, 300.0f, 8.0f, 0.10f,
                               vec4(0.15f, 0.42f, 0.88f, 1.0f));
    register_scene_sph_batch(creation, "Water Basin", "SPH fluid block for splash tests.",
                             particles, sph_before, vec4(0.15f, 0.42f, 0.88f, 1.0f));

    // MPM materials (with stiff E to hold shape)
    f32 mpm_spacing = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 20000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(1.5f, 1.5f), 0.3f, mpm_spacing,
                     MPMMaterial::ELASTIC);
    register_scene_mpm_batch(creation, "Elastic Ball", "Rigid elastic ball.",
                             particles, mpm_before, MPMMaterial::ELASTIC,
                             20000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 15000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.3f, 1.5f), vec2(0.3f, 2.0f), mpm_spacing,
                    MPMMaterial::SNOW);
    register_scene_mpm_batch(creation, "Snow Block", "Packed snow block.",
                             particles, mpm_before, MPMMaterial::SNOW,
                             15000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 8000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.4f, -0.3f), vec2(0.4f, 0.3f), mpm_spacing,
                    MPMMaterial::THERMAL, 300.0f);
    register_scene_mpm_batch(creation, "Thermal Wax", "Thermally softening block.",
                             particles, mpm_before, MPMMaterial::THERMAL,
                             8000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 20000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.0f, 0.5f), vec2(1.6f, 0.9f), mpm_spacing,
                    MPMMaterial::ANISO, 300.0f, vec2(1, 0));
    register_scene_mpm_batch(creation, "Anisotropic Beam", "Fiber-reinforced structural solid.",
                             particles, mpm_before, MPMMaterial::ANISO,
                             20000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.3f);
    mpm.params().heat_source_radius = 0.4f;
    mpm.params().heat_source_temp = 600.0f;
}

static void load_thermal_furnace(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                 MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                 CreationState* creation) {
    sdf.clear();
    sdf.add_segment(vec2(-1.5f, -1.5f), vec2(1.5f, -1.5f), 0.15f);
    sdf.add_segment(vec2(-1.5f, -1.5f), vec2(-1.5f, 1.5f), 0.15f);
    sdf.add_segment(vec2(1.5f, -1.5f),  vec2(1.5f, 1.5f),  0.15f);
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.3f);
    mpm.params().heat_source_radius = 0.6f;
    mpm.params().heat_source_temp = 1200.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 15000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.6f, -0.8f), vec2(0.6f, 0.0f), sp,
                    MPMMaterial::THERMAL, 300.0f);
    register_scene_mpm_batch(creation, "Thermal Ingot", "Heated thermal solid.",
                             particles, mpm_before, MPMMaterial::THERMAL,
                             15000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 25000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(0.0f, 0.8f), 0.3f, sp, MPMMaterial::ELASTIC);
    register_scene_mpm_batch(creation, "Elastic Test Ball", "Dense impactor for the furnace.",
                             particles, mpm_before, MPMMaterial::ELASTIC,
                             25000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));
    mpm.params().youngs_modulus = old_E;
}

static void load_fracture_test(ParticleBuffer& particles, SPHSolver& /*sph*/,
                               MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                               CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.rebuild();

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    // Stiff fracture pillar
    mpm.params().youngs_modulus = 25000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.15f, -1.3f), vec2(0.15f, 0.5f), sp,
                    MPMMaterial::FRACTURE, 300.0f, vec2(0, 1));
    register_scene_mpm_batch(creation, "Fracture Pillar", "Stiff fracture column.",
                             particles, mpm_before, MPMMaterial::FRACTURE,
                             25000.0f, mpm.params().poisson_ratio, 300.0f, vec2(0, 1));

    // Heavy stiff ball
    mpm.params().youngs_modulus = 30000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(0.0f, 2.0f), 0.35f, sp, MPMMaterial::ELASTIC);
    register_scene_mpm_batch(creation, "Impact Ball", "Heavy elastic ball for shatter tests.",
                             particles, mpm_before, MPMMaterial::ELASTIC,
                             30000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));
    mpm.params().youngs_modulus = old_E;
}

static void load_melting(ParticleBuffer& particles, SPHSolver& /*sph*/,
                         MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                         CreationState* creation) {
    // Floor + walls
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.rebuild();

    // Hot plate at bottom center
    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.3f);
    mpm.params().heat_source_radius = 1.0f;
    mpm.params().heat_source_temp = 800.0f;

    f32 mpm_spacing = grid.dx() * 0.5f;

    // Large phase block (cold / frozen)
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.5f, -1.2f), vec2(0.5f, -0.2f), mpm_spacing,
                    MPMMaterial::PHASE, 100.0f);
    register_scene_mpm_batch(creation, "Phase Block", "Cold phase-change material.",
                             particles, mpm_before, MPMMaterial::PHASE,
                             mpm.params().youngs_modulus, mpm.params().poisson_ratio, 100.0f, vec2(1, 0));
}

static void load_dam_break(ParticleBuffer& particles, SPHSolver& sph,
                           MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                           CreationState* creation) {
    // Floor + walls + dam wall segment
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_segment(vec2(-0.3f, -1.5f), vec2(-0.3f, 0.5f), 0.08f);  // dam wall
    sdf.rebuild();

    // Big SPH block on left side
    f32 sph_spacing = sph.params().smoothing_radius * 0.5f;
    u32 sph_before = particles.range(SolverType::SPH).count;
    sph.spawn_block(particles, vec2(-2.3f, -1.3f), vec2(-0.5f, 1.5f), sph_spacing);
    apply_sph_batch_properties(particles, sph,
                               particles.range(SolverType::SPH).offset + sph_before,
                               particles.range(SolverType::SPH).count - sph_before,
                               MPMMaterial::SPH_WATER, 300.0f, 8.0f, 0.10f,
                               vec4(0.12f, 0.36f, 0.86f, 1.0f));
    register_scene_sph_batch(creation, "Reservoir", "Large SPH water reservoir.",
                             particles, sph_before, vec4(0.12f, 0.36f, 0.86f, 1.0f),
                             0.7f, "Use a large reservoir if you want the dam break wave to read strongly.");

    f32 mpm_spacing = grid.dx() * 0.5f;

    // Elastic ball on right side
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(1.0f, -0.5f), 0.25f, mpm_spacing,
                     MPMMaterial::ELASTIC);
    register_scene_mpm_batch(creation, "Elastic Ball", "Ball on the dry side.",
                             particles, mpm_before, MPMMaterial::ELASTIC,
                             mpm.params().youngs_modulus, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    // Snow block on right side
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.5f, -1.3f), vec2(2.2f, -0.5f), mpm_spacing,
                    MPMMaterial::SNOW);
    register_scene_mpm_batch(creation, "Snow Stack", "Packed snow target.",
                             particles, mpm_before, MPMMaterial::SNOW,
                             mpm.params().youngs_modulus, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));
}

static void load_stiff_objects(ParticleBuffer& particles, SPHSolver& sph,
                              MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                              CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    // Shelves for objects to sit on
    sdf.add_segment(vec2(-2.0f, 0.0f), vec2(-0.5f, 0.0f), 0.08f);
    sdf.add_segment(vec2(0.5f, 0.0f), vec2(2.0f, 0.0f), 0.08f);
    sdf.add_segment(vec2(-1.5f, 1.0f), vec2(1.5f, 1.0f), 0.08f);
    sdf.rebuild();

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    // Very stiff elastic cube on left shelf
    mpm.params().youngs_modulus = 40000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.8f, 0.1f), vec2(-1.2f, 0.7f), sp, MPMMaterial::ELASTIC);
    register_scene_mpm_batch(creation, "Rigid Cube", "Very stiff elastic cube.",
                             particles, mpm_before, MPMMaterial::ELASTIC,
                             40000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    // Stiff rubber ball on right shelf
    mpm.params().youngs_modulus = 30000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(1.2f, 0.4f), 0.25f, sp, MPMMaterial::ELASTIC);
    register_scene_mpm_batch(creation, "Rubber Ball", "Dense elastic ball.",
                             particles, mpm_before, MPMMaterial::ELASTIC,
                             30000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    // Hard ice chunks on top shelf
    mpm.params().youngs_modulus = 35000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.5f, 1.1f), vec2(0.5f, 1.6f), sp, MPMMaterial::SNOW);
    register_scene_mpm_batch(creation, "Ice Slab", "Hard snow / ice slab.",
                             particles, mpm_before, MPMMaterial::SNOW,
                             35000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    // Stiff fracture column on floor
    mpm.params().youngs_modulus = 30000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.2f, -1.3f), vec2(0.2f, -0.2f), sp,
                    MPMMaterial::FRACTURE, 300.0f, vec2(0, 1));
    register_scene_mpm_batch(creation, "Fracture Column", "Stiff fracture column.",
                             particles, mpm_before, MPMMaterial::FRACTURE,
                             30000.0f, mpm.params().poisson_ratio, 300.0f, vec2(0, 1));

    // Some SPH fluid to interact with
    f32 sph_sp = sph.params().smoothing_radius * 0.5f;
    u32 sph_before = particles.range(SolverType::SPH).count;
    sph.spawn_block(particles, vec2(0.8f, 1.1f), vec2(1.4f, 1.8f), sph_sp);
    apply_sph_batch_properties(particles, sph,
                               particles.range(SolverType::SPH).offset + sph_before,
                               particles.range(SolverType::SPH).count - sph_before,
                               MPMMaterial::SPH_WATER, 300.0f, 8.0f, 0.10f,
                               vec4(0.15f, 0.44f, 0.9f, 1.0f));
    register_scene_sph_batch(creation, "Top Fluid", "SPH fluid block for splash interaction.",
                             particles, sph_before, vec4(0.15f, 0.44f, 0.9f, 1.0f));

    mpm.params().youngs_modulus = old_E;
}

static void load_heat_ramp(ParticleBuffer& particles, SPHSolver& /*sph*/,
                           MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                           CreationState* creation) {
    sdf.clear();
    // Long ramp from upper-left to lower-right
    sdf.add_segment(vec2(-2.2f, 1.0f), vec2(2.2f, -1.0f), 0.1f);
    // Walls
    sdf.add_segment(vec2(-2.5f, -1.5f), vec2(2.5f, -1.5f), 0.15f); // floor
    sdf.add_segment(vec2(-2.5f, -1.5f), vec2(-2.5f, 2.5f), 0.15f); // left
    sdf.add_segment(vec2(2.5f, -1.5f), vec2(2.5f, 2.5f), 0.15f);   // right
    // Catch basin at bottom-right
    sdf.add_segment(vec2(1.5f, -1.5f), vec2(1.5f, -0.8f), 0.1f);
    sdf.rebuild();

    // Heat source along the bottom of the ramp
    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.5f, -0.5f);
    mpm.params().heat_source_radius = 1.5f;
    mpm.params().heat_source_temp = 900.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    // Phase-change block at top of ramp (starts cold, will melt as it slides into heat zone)
    mpm.params().youngs_modulus = 20000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-2.0f, 1.3f), vec2(-1.2f, 1.9f), sp,
                    MPMMaterial::PHASE, 100.0f);
    register_scene_mpm_batch(creation, "Cold Phase Block", "Cold block that melts on the ramp.",
                             particles, mpm_before, MPMMaterial::PHASE,
                             20000.0f, mpm.params().poisson_ratio, 100.0f, vec2(1, 0));

    // Thermal block beside it
    mpm.params().youngs_modulus = 15000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.0f, 1.5f), vec2(-0.4f, 2.0f), sp,
                    MPMMaterial::THERMAL, 200.0f);
    register_scene_mpm_batch(creation, "Thermal Block", "Heat-softening solid.",
                             particles, mpm_before, MPMMaterial::THERMAL,
                             15000.0f, mpm.params().poisson_ratio, 200.0f, vec2(1, 0));

    // Stiff elastic ball to push things down the ramp
    mpm.params().youngs_modulus = 30000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(-1.5f, 2.5f), 0.3f, sp, MPMMaterial::ELASTIC);
    register_scene_mpm_batch(creation, "Driver Ball", "Elastic ball that pushes materials down the ramp.",
                             particles, mpm_before, MPMMaterial::ELASTIC,
                             30000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_fire_forge(ParticleBuffer& particles, SPHSolver& sph,
                           MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                           CreationState* creation) {
    // Forge scene: fire pit, clay hardening, sparks, water quenching
    sdf.clear();
    sdf.add_segment(vec2(-2.5f, -1.5f), vec2(2.5f, -1.5f), 0.15f); // floor
    sdf.add_segment(vec2(-2.5f, -1.5f), vec2(-2.5f, 2.5f), 0.15f);
    sdf.add_segment(vec2(2.5f, -1.5f), vec2(2.5f, 2.5f), 0.15f);
    // Fire pit (depression)
    sdf.add_segment(vec2(-1.0f, -1.5f), vec2(-1.0f, -1.0f), 0.1f);
    sdf.add_segment(vec2(1.0f, -1.5f), vec2(1.0f, -1.0f), 0.1f);
    // Anvil (platform on right)
    sdf.add_box(vec2(1.8f, -0.8f), vec2(0.4f, 0.15f));
    sdf.rebuild();

    // Heat source in the fire pit
    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.3f);
    mpm.params().heat_source_radius = 0.8f;
    mpm.params().heat_source_temp = 1000.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    // Ember fuel in the fire pit (will spark and float)
    mpm.params().youngs_modulus = 3000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.6f, -1.3f), vec2(0.6f, -1.0f), sp,
                    MPMMaterial::EMBER, 450.0f); // Near ignition
    register_scene_mpm_batch(creation, "Ember Bed", "Pre-heated embers for the forge.",
                             particles, mpm_before, MPMMaterial::EMBER,
                             3000.0f, mpm.params().poisson_ratio, 450.0f, vec2(1, 0));

    // Wet clay on the anvil (will harden when heated)
    mpm.params().youngs_modulus = 2000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.5f, -0.5f), vec2(2.1f, -0.1f), sp,
                    MPMMaterial::HARDEN, 300.0f);
    register_scene_mpm_batch(creation, "Wet Clay Brick", "Heat-curing clay block.",
                             particles, mpm_before, MPMMaterial::HARDEN,
                             2000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    // Some wood to burn
    mpm.params().youngs_modulus = 8000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.3f, -0.8f), vec2(0.3f, -0.3f), sp,
                    MPMMaterial::BURNING, 300.0f, vec2(0, 1));
    register_scene_mpm_batch(creation, "Fuel Wood", "Burning structural fuel.",
                             particles, mpm_before, MPMMaterial::BURNING,
                             8000.0f, mpm.params().poisson_ratio, 300.0f, vec2(0, 1));

    // Water for quenching (SPH)
    f32 sph_sp = sph.params().smoothing_radius * 0.5f;
    u32 sph_before = particles.range(SolverType::SPH).count;
    sph.spawn_block(particles, vec2(-2.2f, -1.3f), vec2(-1.2f, 0.0f), sph_sp);
    apply_sph_batch_properties(particles, sph,
                               particles.range(SolverType::SPH).offset + sph_before,
                               particles.range(SolverType::SPH).count - sph_before,
                               MPMMaterial::SPH_WATER, 300.0f, 8.0f, 0.10f,
                               vec4(0.16f, 0.44f, 0.9f, 1.0f));
    register_scene_sph_batch(creation, "Quench Water", "Water bath for quenching.",
                             particles, sph_before, vec4(0.16f, 0.44f, 0.9f, 1.0f));

    mpm.params().youngs_modulus = old_E;
}

static void load_codim_threads(ParticleBuffer& particles, SPHSolver& sph,
                              MPMSolver& /*mpm*/, UniformGrid& /*grid*/, SDFField& sdf,
                              CreationState* creation) {
    // Showcase codimensional SPH: thin streams, drips, threads
    sdf.clear();
    add_codim_threads_geometry(sdf, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT);
    sdf.rebuild();

    // Enable codim and increase surface tension for visible thread formation
    SPHParams sph_p = sph.params();
    sph_p.codim_enabled = true;
    sph_p.codim_threshold = 0.3f;
    sph_p.surface_tension = 1.5f;
    sph_p.viscosity = 0.3f;
    sph_p.gas_constant = 6.0f;
    sph.set_params(sph_p);

    // Fluid blob on the high platform (will drip off)
    f32 sp = sph.params().smoothing_radius * 0.5f;
    u32 sph_before = particles.range(SolverType::SPH).count;
    sph.spawn_block(particles, vec2(-2.0f, 1.1f), vec2(-1.0f, 1.8f), sp);
    apply_sph_batch_properties(particles, sph,
                               particles.range(SolverType::SPH).offset + sph_before,
                               particles.range(SolverType::SPH).count - sph_before,
                               MPMMaterial::SPH_WATER, 300.0f, sph_p.gas_constant, sph_p.viscosity,
                               vec4(0.14f, 0.42f, 0.90f, 1.0f));
    register_scene_sph_batch(creation, "Upper Drip Pool", "High pool that forms thin drips.",
                             particles, sph_before, vec4(0.14f, 0.42f, 0.90f, 1.0f),
                             0.5f, "Bigger pools on shelves give the clearest thread formation.");

    // Fluid in the funnel (will thread through the narrow gap)
    sph_before = particles.range(SolverType::SPH).count;
    sph.spawn_block(particles, vec2(-0.6f, 0.5f), vec2(0.6f, 1.5f), sp);
    apply_sph_batch_properties(particles, sph,
                               particles.range(SolverType::SPH).offset + sph_before,
                               particles.range(SolverType::SPH).count - sph_before,
                               MPMMaterial::SPH_WATER, 300.0f, sph_p.gas_constant, sph_p.viscosity,
                               vec4(0.12f, 0.46f, 0.92f, 1.0f));
    register_scene_sph_batch(creation, "Funnel Pool", "Fluid source feeding the slit funnel.",
                             particles, sph_before, vec4(0.12f, 0.46f, 0.92f, 1.0f),
                             0.65f, "Use a broad fluid block here if you want long codimensional streams.");

    // Small blob on the right platform
    sph_before = particles.range(SolverType::SPH).count;
    sph.spawn_block(particles, vec2(1.1f, 0.4f), vec2(1.9f, 0.9f), sp);
    apply_sph_batch_properties(particles, sph,
                               particles.range(SolverType::SPH).offset + sph_before,
                               particles.range(SolverType::SPH).count - sph_before,
                               MPMMaterial::SPH_WATER, 300.0f, sph_p.gas_constant, sph_p.viscosity,
                               vec4(0.16f, 0.50f, 0.92f, 1.0f));
    register_scene_sph_batch(creation, "Right Pool", "Secondary drip pool.",
                             particles, sph_before, vec4(0.16f, 0.50f, 0.92f, 1.0f),
                             0.4f, "Small shelf pools are good for short thread and droplet tests.");
}

static void load_sph_thermal_bench(ParticleBuffer& particles, SPHSolver& sph,
                                   MPMSolver& mpm, UniformGrid& /*grid*/, SDFField& sdf,
                                   CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-2.2f, -1.30f), vec2(2.2f, -1.30f), 0.10f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Thermal Plate", "Wide hot plate for comparing boiling, burning, thickening, and flash-off SPH liquids.");

    const f32 centers[5] = {-1.92f, -0.96f, 0.0f, 0.96f, 1.92f};
    const char* cup_names[5] = {
        "Water Cup", "Boil Cup", "Oil Cup", "Syrup Cup", "Flash Cup"
    };
    for (i32 lane = 0; lane < 5; ++lane) {
        const f32 cx = centers[lane];
        sdf.add_segment(vec2(cx - 0.36f, -1.30f), vec2(cx - 0.36f, -0.34f), 0.05f,
                        SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                        cup_names[lane], "Cup wall for the SPH thermal benchmark.");
        sdf.add_segment(vec2(cx + 0.36f, -1.30f), vec2(cx + 0.36f, -0.34f), 0.05f,
                        SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                        cup_names[lane], "Cup wall for the SPH thermal benchmark.");
    }
    sdf.rebuild();

    SPHParams sph_p = sph.params();
    sph_p.codim_enabled = true;
    sph_p.surface_tension = 1.05f;
    sph_p.enable_thermal = true;
    sph_p.immiscible_interfaces = true;
    sph_p.interface_repulsion = 16.0f;
    sph_p.interface_tension = 1.10f;
    sph_p.cross_mix = 0.22f;
    sph_p.cross_thermal_mix = 0.34f;
    sph.set_params(sph_p);

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.18f);
    mpm.params().heat_source_radius = 2.30f;
    mpm.params().heat_source_temp = 760.0f;

    const f32 sp = sph.params().smoothing_radius * 0.5f;
    struct BenchFluid {
        const char* label;
        const char* summary;
        MPMMaterial material;
        f32 temp;
        f32 gas_constant;
        f32 viscosity;
        vec4 color;
    };
    const BenchFluid fluids[5] = {
        {"Water Ref", "Expected: stays the calmest. It should splash and convect, but not boil or ignite aggressively.", MPMMaterial::SPH_WATER, 300.0f, 8.0f, 0.10f, vec4(0.15f, 0.40f, 0.85f, 1.0f)},
        {"Boiling Water", "Expected: becomes livelier and vents steam once hot, with bubbling lift stronger than plain water.", MPMMaterial::SPH_BOILING_WATER, 300.0f, 8.0f, 0.09f, vec4(0.44f, 0.72f, 0.98f, 1.0f)},
        {"Burning Oil", "Expected: ignites, self-heats, smokes, and keeps feeding hot vapor longer than the water cups.", MPMMaterial::SPH_BURNING_OIL, 330.0f, 8.0f, 0.14f, vec4(0.78f, 0.54f, 0.14f, 1.0f)},
        {"Thermal Syrup", "Expected: starts thick, loosens when heated, then clings and globs more again as it cools.", MPMMaterial::SPH_THERMAL_SYRUP, 315.0f, 6.0f, 0.55f, vec4(0.86f, 0.54f, 0.22f, 1.0f)},
        {"Flash Fluid", "Expected: low-boiling and unstable. It should vaporize and loft apart much more aggressively than the other cups.", MPMMaterial::SPH_FLASH_FLUID, 295.0f, 11.0f, 0.03f, vec4(0.82f, 0.72f, 0.50f, 1.0f)}
    };

    for (i32 lane = 0; lane < 5; ++lane) {
        const f32 cx = centers[lane];
        u32 sph_before = particles.range(SolverType::SPH).count;
        sph.spawn_block(particles, vec2(cx - 0.26f, -1.16f), vec2(cx + 0.26f, -0.72f), sp);
        u32 global_offset = particles.range(SolverType::SPH).offset + sph_before;
        u32 count = particles.range(SolverType::SPH).count - sph_before;
        apply_sph_batch_properties(particles, sph, global_offset, count,
                                   fluids[lane].material, fluids[lane].temp,
                                   fluids[lane].gas_constant, fluids[lane].viscosity,
                                   fluids[lane].color);
        register_scene_sph_batch(creation, fluids[lane].label, fluids[lane].summary,
                                 particles, sph_before, fluids[lane].color,
                                 0.32f, "Use medium cup-fills so the thermal behavior reads before the liquid simply splashes out.",
                                 fluids[lane].material, fluids[lane].temp);
    }
}

static void load_oil_over_water(ParticleBuffer& particles, SPHSolver& sph,
                                MPMSolver& mpm, UniformGrid& /*grid*/, SDFField& sdf,
                                CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-2.1f, -1.30f), vec2(2.1f, -1.30f), 0.10f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Thermal Plate", "Wide hot plate for layered liquid tests.");
    sdf.add_segment(vec2(-0.58f, -1.30f), vec2(-0.58f, 0.06f), 0.06f,
                    SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Left Pot Wall", "Pot wall holding the layered oil-over-water column.");
    sdf.add_segment(vec2(0.58f, -1.30f), vec2(0.58f, 0.06f), 0.06f,
                    SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Right Pot Wall", "Pot wall holding the layered oil-over-water column.");
    sdf.rebuild();

    SPHParams sph_p = sph.params();
    sph_p.codim_enabled = true;
    sph_p.surface_tension = 1.18f;
    sph_p.enable_thermal = true;
    sph_p.immiscible_interfaces = true;
    sph_p.interface_repulsion = 22.0f;
    sph_p.interface_tension = 1.35f;
    sph_p.cross_mix = 0.10f;
    sph_p.cross_thermal_mix = 0.18f;
    sph_p.viscosity = 0.10f;
    sph_p.gas_constant = 8.0f;
    sph.set_params(sph_p);

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.18f);
    mpm.params().heat_source_radius = 0.92f;
    mpm.params().heat_source_temp = 760.0f;

    const f32 sp = sph.params().smoothing_radius * 0.5f;

    u32 sph_before = particles.range(SolverType::SPH).count;
    sph.spawn_block(particles, vec2(-0.42f, -1.16f), vec2(0.42f, -0.74f), sp);
    apply_sph_batch_properties(particles, sph,
                               particles.range(SolverType::SPH).offset + sph_before,
                               particles.range(SolverType::SPH).count - sph_before,
                               MPMMaterial::SPH_BOILING_WATER, 300.0f, 8.0f, 0.09f,
                               vec4(0.44f, 0.72f, 0.98f, 1.0f));
    register_scene_sph_batch(creation,
                             "Water Layer",
                             "Expected: the heavier water stays mostly below while heating from the plate, then becomes livelier and boils under the oil.",
                             particles, sph_before, vec4(0.44f, 0.72f, 0.98f, 1.0f),
                             0.44f, "Use medium fills so the lower water layer can visibly churn under the oil cap.",
                             MPMMaterial::SPH_BOILING_WATER, 300.0f);

    sph_before = particles.range(SolverType::SPH).count;
    sph.spawn_block(particles, vec2(-0.42f, -0.74f), vec2(0.42f, -0.38f), sp);
    apply_sph_batch_properties(particles, sph,
                               particles.range(SolverType::SPH).offset + sph_before,
                               particles.range(SolverType::SPH).count - sph_before,
                               MPMMaterial::SPH_BURNING_OIL, 330.0f, 8.0f, 0.14f,
                               vec4(0.78f, 0.54f, 0.14f, 1.0f));
    register_scene_sph_batch(creation,
                             "Oil Layer",
                             "Expected: the lighter oil floats above the water, heats fast, and can ignite/smoke while the water below boils and pushes at the interface.",
                             particles, sph_before, vec4(0.78f, 0.54f, 0.14f, 1.0f),
                             0.36f, "This scene is for interface behavior: oil should stay above water unless the boil becomes very violent.",
                             MPMMaterial::SPH_BURNING_OIL, 330.0f);
}

static void load_thermal_verify_sdf_junction(ParticleBuffer& /*particles*/, SPHSolver& /*sph*/,
                                             MPMSolver& mpm, UniformGrid& /*grid*/, SDFField& sdf,
                                             CreationState* /*creation*/) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(0.0f, -1.30f), vec2(0.0f, 1.18f), 0.085f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Junction Stem",
                    "Expected: this bench is only about thermal ringdown. After you hide the authored hotspot, the junction should cool from the outside in instead of reheating itself from leftover hot air.");
    sdf.add_segment(vec2(-1.52f, -0.12f), vec2(0.0f, -0.12f), 0.095f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Left Junction Arm",
                    "Expected: you may still see a short-lived hot halo near the joint, but it should not sharpen into a self-feeding burner after the source is gone.");
    sdf.add_segment(vec2(0.0f, -0.12f), vec2(1.52f, -0.12f), 0.095f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Right Junction Arm",
                    "Expected: this mirrored arm should cool in the same way as the left one. The key failure this scene catches is a joint that keeps glowing because hot air is feeding it back indefinitely.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -0.12f);
    mpm.params().heat_source_radius = 0.24f;
    mpm.params().heat_source_temp = 980.0f;
}

static void load_thermal_verify_hot_blocks(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                           MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                           CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_box(vec2(-1.55f, -0.72f), vec2(0.34f, 0.08f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Left Plinth", "Pedestal for the inert hot-metal witness.");
    sdf.add_box(vec2(0.0f, -0.72f), vec2(0.34f, 0.08f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Center Plinth", "Pedestal for the ember witness.");
    sdf.add_box(vec2(1.55f, -0.72f), vec2(0.34f, 0.08f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Right Plinth", "Pedestal for the burning-fuel witness.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_radius = 0.0f;
    mpm.params().heat_source_temp = 0.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 72000.0f;
    u32 before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.82f, -0.62f), vec2(-1.28f, -0.30f), sp,
                    MPMMaterial::THERMO_METAL, 760.0f);
    register_scene_mpm_batch(creation, "Hot Metal Witness",
                             "Expected: this inert hot block should warm the air around it and maybe redden briefly, but the plume should ring down and cool away on its own instead of becoming a standing burner.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             72000.0f, mpm.params().poisson_ratio, 760.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 7000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.24f, -0.62f), vec2(0.24f, -0.32f), sp,
                    MPMMaterial::EMBER, 620.0f);
    register_scene_mpm_batch(creation, "Ember Witness",
                             "Expected: this lane should stay hot longer than the metal lane and keep a compact hot/smoky pocket, but it should still decay without turning the whole room into a permanent source.",
                             particles, before, MPMMaterial::EMBER,
                             7000.0f, mpm.params().poisson_ratio, 620.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 12000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.22f, -0.62f), vec2(1.88f, -0.28f), sp,
                    MPMMaterial::BURNING, 440.0f, vec2(0.0f, 1.0f));
    register_scene_mpm_batch(creation, "Burning Fuel Witness",
                             "Expected: this lane should sustain visible combustion longer than the hot metal lane and should out-smoke the ember lane, but once the fuel is spent it still needs to cool out instead of feeding heat forever.",
                             particles, before, MPMMaterial::BURNING,
                             12000.0f, mpm.params().poisson_ratio, 440.0f, vec2(0.0f, 1.0f));

    mpm.params().youngs_modulus = old_E;
}

static void load_thermal_verify_cross_ignition(ParticleBuffer& particles, SPHSolver& sph,
                                               MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                               CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    const f32 centers[3] = { -1.50f, 0.0f, 1.50f };
    for (i32 lane = 0; lane < 3; ++lane) {
        const f32 cx = centers[lane];
        sdf.add_segment(vec2(cx - 0.42f, -1.30f), vec2(cx - 0.42f, -0.26f), 0.05f,
                        SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                        lane == 0 ? "Wood Lane Cup" :
                        lane == 1 ? "Metal Lane Cup" :
                                    "Stoneware Lane Cup",
                        "Cup wall for the cross-ignition verification lane.");
        sdf.add_segment(vec2(cx + 0.42f, -1.30f), vec2(cx + 0.42f, -0.26f), 0.05f,
                        SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                        lane == 0 ? "Wood Lane Cup" :
                        lane == 1 ? "Metal Lane Cup" :
                                    "Stoneware Lane Cup",
                        "Cup wall for the cross-ignition verification lane.");
        sdf.add_box(vec2(cx - 0.22f, 0.13f), vec2(0.12f, 0.05f),
                    SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    lane == 0 ? "Wood Left Ledge" :
                    lane == 1 ? "Metal Left Ledge" :
                                "Stoneware Left Ledge",
                    "Split ledge leaves the center open so the flame plume and the conductor post stay visible.");
        sdf.add_box(vec2(cx + 0.22f, 0.13f), vec2(0.12f, 0.05f),
                    SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    lane == 0 ? "Wood Right Ledge" :
                    lane == 1 ? "Metal Right Ledge" :
                                "Stoneware Right Ledge",
                    "Split ledge leaves the center open so the flame plume and the conductor post stay visible.");
        sdf.add_segment(vec2(cx, -0.18f), vec2(cx, 0.20f), 0.055f,
                        SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                        lane == 0 ? "Wood Lane Conductor" :
                        lane == 1 ? "Metal Lane Conductor" :
                                    "Stoneware Lane Conductor",
                        "Expected: this silver post is the solid-coupling witness path. Hot oil below should heat the post, and the witness sitting on it tells you whether SPH-to-solid-to-MPM transfer is working.");
    }
    sdf.rebuild();

    SPHParams sph_p = sph.params();
    sph_p.codim_enabled = true;
    sph_p.enable_thermal = true;
    sph_p.surface_tension = 1.02f;
    sph_p.immiscible_interfaces = true;
    sph_p.interface_repulsion = 16.0f;
    sph_p.interface_tension = 1.05f;
    sph_p.cross_mix = 0.18f;
    sph_p.cross_thermal_mix = 0.34f;
    sph_p.viscosity = 0.12f;
    sph.set_params(sph_p);

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_radius = 0.0f;
    mpm.params().heat_source_temp = 0.0f;

    const f32 sph_sp = sph.params().smoothing_radius * 0.5f;
    const f32 mpm_sp = grid.dx() * 0.5f;

    for (i32 lane = 0; lane < 3; ++lane) {
        const f32 cx = centers[lane];
        u32 before_sph = particles.range(SolverType::SPH).count;
        sph.spawn_block(particles, vec2(cx - 0.28f, -1.16f), vec2(cx + 0.28f, -0.72f), sph_sp);
        apply_sph_batch_properties(particles, sph,
                                   particles.range(SolverType::SPH).offset + before_sph,
                                   particles.range(SolverType::SPH).count - before_sph,
                                   MPMMaterial::SPH_BURNING_OIL, 520.0f, 8.0f, 0.14f,
                                   vec4(0.78f, 0.54f, 0.14f, 1.0f));
        register_scene_sph_batch(creation,
                                 lane == 0 ? "Oil Cup / Wood Lane" :
                                 lane == 1 ? "Oil Cup / Metal Lane" :
                                             "Oil Cup / Stoneware Lane",
                                 lane == 0
                                     ? "Expected: this pre-lit oil cup should heat the center silver post and eventually light the wood witness above it."
                                     : lane == 1
                                         ? "Expected: this cup should heat the metal witness strongly through the silver post, but the witness should stay non-flammable."
                                         : "Expected: this cup should warm the stoneware witness through the post much more slowly, giving a clean inertia-vs-ignition comparison.",
                                 particles, before_sph, vec4(0.78f, 0.54f, 0.14f, 1.0f),
                                 0.34f, "Medium cup fills make the cross-ignition timing easier to compare.",
                                 MPMMaterial::SPH_BURNING_OIL, 520.0f);
    }

    const f32 old_E = mpm.params().youngs_modulus;
    mpm.params().youngs_modulus = 11000.0f;
    u32 before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.76f, 0.18f), vec2(-1.24f, 0.44f), mpm_sp,
                    MPMMaterial::BURNING, 300.0f, vec2(0.0f, 1.0f));
    register_scene_mpm_batch(creation, "Wood Witness",
                             "Expected: this witness should be the only one that reliably catches and keeps burning. If it never lights, the SPH-to-solid-to-MPM path is still too weak.",
                             particles, before, MPMMaterial::BURNING,
                             11000.0f, mpm.params().poisson_ratio, 300.0f, vec2(0.0f, 1.0f));

    mpm.params().youngs_modulus = 72000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.30f, 0.18f), vec2(0.30f, 0.42f), mpm_sp,
                    MPMMaterial::THERMO_METAL, 300.0f);
    register_scene_mpm_batch(creation, "Metal Witness",
                             "Expected: this witness should get visibly hotter through the post, but it should not behave like fuel. It is here to catch runaway reheating bugs.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             72000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 26000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.22f, 0.18f), vec2(1.78f, 0.44f), mpm_sp,
                    MPMMaterial::STONEWARE, 300.0f);
    register_scene_mpm_batch(creation, "Stoneware Witness",
                             "Expected: this witness should warm most slowly and stay non-flammable, giving you a clean inertia-vs-ignition comparison against the wood and metal lanes.",
                             particles, before, MPMMaterial::STONEWARE,
                             26000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_thermal_verify_bridge_witness(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                               MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                               CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    struct BeamLane {
        f32 y;
        SDFField::MaterialPresetID preset;
        const char* name;
        const char* summary;
    };
    const BeamLane lanes[3] = {
        {  0.96f, SDFField::MaterialPresetID::SILVER_CONDUCTIVE, "Silver Lane", "Expected: this conductive lane should get heat to the cold witness fastest." },
        {  0.14f, SDFField::MaterialPresetID::BRONZE_BALANCED,   "Bronze Lane", "Expected: this balanced lane should sit between silver and brass in both speed and hold." },
        { -0.68f, SDFField::MaterialPresetID::BRASS_HEAT_SINK,   "Brass Lane",  "Expected: this sink lane should respond slowest, but stay warm locally the longest after the ember has started cooling." }
    };
    for (const BeamLane& lane : lanes) {
        sdf.add_segment(vec2(-1.74f, lane.y), vec2(1.02f, lane.y), 0.09f,
                        lane.preset, lane.name, lane.summary);
    }
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_radius = 0.0f;
    mpm.params().heat_source_temp = 0.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;

    for (const BeamLane& lane : lanes) {
        mpm.params().youngs_modulus = 12000.0f;
        u32 before = particles.range(SolverType::MPM).count;
        mpm.spawn_block(particles, vec2(-1.62f, lane.y + 0.11f), vec2(-1.18f, lane.y + 0.31f), sp,
                        MPMMaterial::EMBER, 640.0f);
        register_scene_mpm_batch(creation,
                                 lane.preset == SDFField::MaterialPresetID::SILVER_CONDUCTIVE ? "Silver Source Block" :
                                 lane.preset == SDFField::MaterialPresetID::BRONZE_BALANCED   ? "Bronze Source Block" :
                                                                                                "Brass Source Block",
                                 "Expected: all three ember source blocks start equally hot and rest just above the beam. Any difference in witness timing should come mainly from the SDF lane, not from the source exploding on contact.",
                                 particles, before, MPMMaterial::EMBER,
                                 12000.0f, mpm.params().poisson_ratio, 640.0f, vec2(1, 0));

        mpm.params().youngs_modulus = 68000.0f;
        before = particles.range(SolverType::MPM).count;
        mpm.spawn_block(particles, vec2(0.56f, lane.y + 0.11f), vec2(0.92f, lane.y + 0.27f), sp,
                        MPMMaterial::THERMO_METAL, 300.0f);
        register_scene_mpm_batch(creation,
                                 lane.preset == SDFField::MaterialPresetID::SILVER_CONDUCTIVE ? "Silver Witness" :
                                 lane.preset == SDFField::MaterialPresetID::BRONZE_BALANCED   ? "Bronze Witness" :
                                                                                                "Brass Witness",
                                 lane.summary,
                                 particles, before, MPMMaterial::THERMO_METAL,
                                 68000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));
    }

    mpm.params().youngs_modulus = old_E;
}

static void load_thermal_verify_impact_ringdown(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                                MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                                CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_box(vec2(0.0f, -1.02f), vec2(0.16f, 0.52f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Impact Pedestal", "Small pedestal so the strike happens above the floor instead of being hidden in a ground contact.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_radius = 0.0f;
    mpm.params().heat_source_temp = 0.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 76000.0f;
    u32 before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.36f, -0.42f), vec2(0.36f, -0.08f), sp,
                    MPMMaterial::THERMO_METAL, 300.0f);
    register_scene_mpm_batch(creation, "Target Slab",
                             "Expected: this witness should flash hot around the impact zone, but the air hotspot should ring down instead of becoming a permanent burner after one fast hit.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             76000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 68000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(-1.86f, -0.22f), 0.16f, sp,
                     MPMMaterial::THERMO_METAL, 300.0f);
    register_scene_mpm_batch(creation, "Impact Striker",
                             "Expected: this striker is only here to create a sharp local hit. After contact, any hot plume in the air should fade instead of feeding itself forever.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             68000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));
    BatchRecord* striker = creation && !creation->batches.empty() ? &creation->batches.back() : nullptr;
    if (striker && striker->mpm_count > 0) {
        set_mpm_batch_velocity(particles, striker->mpm_offset, striker->mpm_count, vec2(28.0f, 0.0f));
    }

    mpm.params().youngs_modulus = old_E;
}

static void load_ferro_spike_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                   MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                   CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-2.15f, -1.30f), vec2(2.15f, -1.30f), 0.10f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Bench Plate", "Flat plate for ferrofluid spike tests.");
    sdf.add_segment(vec2(-0.80f, -1.30f), vec2(-0.80f, -0.40f), 0.05f,
                    SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Left Tray", "Shallow tray for the wide ferro puddle.");
    sdf.add_segment(vec2(0.80f, -1.30f), vec2(0.80f, -0.40f), 0.05f,
                    SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Right Tray", "Shallow tray for the deep ferro puddle.");
    sdf.rebuild();

    mpm.params().enable_thermal = false;
    mpm.params().magnet_radius = 0.62f;
    mpm.params().magnet_spike_strength = 1.45f;
    mpm.params().magnet_chain_rate = 8.0f;
    mpm.params().magnet_spike_freq = 17.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    mpm.params().youngs_modulus = 5200.0f;

    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.55f, -1.18f), vec2(-0.35f, -0.92f), sp,
                    MPMMaterial::FERRO_FLUID, 300.0f);
    register_scene_mpm_batch(creation,
                             "Wide Ferro Puddle",
                             "Expected: when you hold M above it, this wide shallow puddle should form several low comb-like ridges instead of only sliding as one blob.",
                             particles, mpm_before, MPMMaterial::FERRO_FLUID,
                             5200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.35f, -1.18f), vec2(1.05f, -0.78f), sp,
                    MPMMaterial::FERRO_FLUID, 300.0f);
    register_scene_mpm_batch(creation,
                             "Deep Ferro Puddle",
                             "Expected: when you hold M above it, this deeper puddle should pull into taller finger-like peaks with a stronger central tower.",
                             particles, mpm_before, MPMMaterial::FERRO_FLUID,
                             5200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_magnetic_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-2.10f, -1.30f), vec2(2.10f, -1.30f), 0.10f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Bench Plate", "Neutral thermal plate for the real magnetic field benchmark.");
    sdf.add_box(vec2(0.0f, -1.02f), vec2(0.72f, 0.18f),
                SDFField::MaterialPresetID::MAGNET_X,
                "Bar Magnet", "Permanent horizontal magnet source for the solved magnetic field.");
    sdf.add_box(vec2(-1.35f, -0.42f), vec2(0.18f, 0.60f),
                SDFField::MaterialPresetID::MAGNET_Y,
                "Pole Magnet", "Vertical permanent magnet so the field bench has two clearly different source directions.");
    sdf.rebuild();

    mpm.params().enable_thermal = false;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 18000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.12f, -0.82f), vec2(-0.82f, -0.54f), sp,
                    MPMMaterial::MAG_SOFT_IRON, 300.0f, vec2(1, 0), 0.24f);
    register_scene_mpm_batch(creation,
                             "Soft Iron Block",
                             "Expected: once Real Magnetics is on, this benchmark block should clearly climb toward the nearby pole magnet and prefer the stronger field region instead of behaving like a normal elastic block.",
                             particles, mpm_before, MPMMaterial::MAG_SOFT_IRON,
                             18000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 0.24f);

    mpm.params().youngs_modulus = 8600.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.10f, -0.52f), vec2(0.92f, -0.40f), sp,
                    MPMMaterial::MAGNETIC_RUBBER, 300.0f, vec2(1, 0), 0.18f);
    register_scene_mpm_batch(creation,
                             "Magnetic Rubber Strip",
                             "Expected: once Real Magnetics is on, this softer strip should bend and drift toward the bar magnet more readily than the soft iron block.",
                             particles, mpm_before, MPMMaterial::MAGNETIC_RUBBER,
                             6200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 0.18f);

    mpm.params().youngs_modulus = 5200.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.36f, -1.02f), vec2(0.94f, -0.86f), sp,
                    MPMMaterial::FERRO_FLUID, 300.0f, vec2(1, 0), 0.14f);
    register_scene_mpm_batch(creation,
                             "Ferrofluid Ref",
                             "Expected: this is still only a first bridge to real ferro. It should now drift much more clearly under the solved field, even though the full free-surface spike model is a later pass.",
                             particles, mpm_before, MPMMaterial::FERRO_FLUID,
                             5200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 0.14f);

    mpm.params().youngs_modulus = old_E;
}

static void load_magnetic_climb_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                      MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                      CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-2.10f, -1.30f), vec2(2.10f, -1.30f), 0.10f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Bench Plate", "Flat plate for the induced soft-iron ferrofluid climb test.");
    sdf.add_box(vec2(0.05f, -1.02f), vec2(1.08f, 0.18f),
                SDFField::MaterialPresetID::MAGNET_X,
                "Drive Magnet", "Permanent horizontal magnet underneath the climb rig.");
    sdf.add_box(vec2(0.88f, -0.50f), vec2(0.10f, 0.44f),
                SDFField::MaterialPresetID::SOFT_IRON,
                "Soft Iron Pole", "Induced soft-iron post that should concentrate the field above the right side of the magnet.");
    sdf.add_box(vec2(0.64f, -0.02f), vec2(0.34f, 0.08f),
                SDFField::MaterialPresetID::SOFT_IRON,
                "Soft Iron Cap", "Induced soft-iron cap that should intensify the field near the pole tip and help ferrofluid wick upward.");
    sdf.rebuild();

    mpm.params().enable_thermal = false;
    mpm.params().magnet_spike_strength = 1.55f;
    mpm.params().magnet_chain_rate = 9.5f;
    mpm.params().magnet_spike_freq = 16.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    mpm.params().youngs_modulus = 5200.0f;

    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.92f, -0.84f), vec2(-0.36f, -0.70f), sp,
                    MPMMaterial::FERRO_FLUID, 300.0f, vec2(1, 0), 0.14f);
    register_scene_mpm_batch(creation,
                             "Reference Ferro Puddle",
                             "Expected: this left puddle should mostly stay squat and only creep modestly, because it is away from the soft-iron pole piece.",
                             particles, mpm_before, MPMMaterial::FERRO_FLUID,
                             5200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 0.14f);

    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.28f, -0.84f), vec2(0.88f, -0.70f), sp,
                    MPMMaterial::FERRO_FLUID, 300.0f, vec2(1, 0), 0.14f);
    register_scene_mpm_batch(creation,
                             "Climb Ferro Puddle",
                             "Expected: with Real Magnetics on, this puddle should pull harder toward the soft-iron pole piece and start climbing or necking upward along the induced iron path instead of staying a flat puddle.",
                             particles, mpm_before, MPMMaterial::FERRO_FLUID,
                             5200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 0.14f);

    mpm.params().youngs_modulus = old_E;
}

static void load_magnetic_floor_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                      MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                      CreationState* creation) {
    sdf.clear();
    sdf.add_segment(vec2(-2.45f, -1.50f), vec2(-2.45f, 2.20f), 0.15f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Left Wall", "Rigid side wall for the magnetic floor benchmark.");
    sdf.add_segment(vec2(2.45f, -1.50f), vec2(2.45f, 2.20f), 0.15f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Right Wall", "Rigid side wall for the magnetic floor benchmark.");
    sdf.add_box(vec2(0.0f, -1.20f), vec2(2.05f, 0.18f),
                SDFField::MaterialPresetID::SOFT_IRON,
                "Soft Iron Floor", "Magnetizable floor: it should become induced by the drive magnet and distort the magnetic lines even before you hold M.");
    sdf.add_box(vec2(0.0f, -1.56f), vec2(1.48f, 0.10f),
                SDFField::MaterialPresetID::MAGNET_X,
                "Drive Magnet", "Permanent magnet embedded under the floor to magnetize the soft-iron plate above.");
    sdf.add_box(vec2(-0.15f, -0.78f), vec2(0.26f, 0.48f),
                SDFField::MaterialPresetID::SOFT_IRON,
                "Floor Pole", "Soft-iron post touching the magnetizable floor so the induced path is easier to see.");
    sdf.rebuild();

    mpm.params().enable_thermal = false;
    mpm.params().magnet_spike_strength = 1.35f;
    mpm.params().magnet_chain_rate = 8.5f;
    mpm.params().magnet_spike_freq = 15.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 48000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.55f, -0.94f), vec2(-0.98f, -0.44f), sp,
                    MPMMaterial::MAG_SOFT_IRON, 300.0f, vec2(1, 0), 0.32f);
    register_scene_mpm_batch(creation,
                             "Rigid Iron Slug",
                             "Expected: this very stiff magnetic solid should stay body-like on the floor and drift toward induced hot-spots, not pinch down into a tiny dot.",
                             particles, mpm_before, MPMMaterial::MAG_SOFT_IRON,
                             48000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 0.32f);

    mpm.params().youngs_modulus = 4200.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.72f, -1.02f), vec2(1.58f, -0.78f), sp,
                    MPMMaterial::FERRO_FLUID, 300.0f, vec2(1, 0), 0.16f);
    register_scene_mpm_batch(creation,
                             "Ferro Floor Puddle",
                             "Expected: this softer magnetic fluid should spread on the induced floor, neck toward stronger regions, and show much more shape change than the rigid iron slug.",
                             particles, mpm_before, MPMMaterial::FERRO_FLUID,
                             4200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 0.16f);

    mpm.params().youngs_modulus = old_E;
}

static void load_rigid_magnetic_floor(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                      MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                      CreationState* creation) {
    sdf.clear();
    sdf.add_segment(vec2(-2.45f, -1.50f), vec2(-2.45f, 2.20f), 0.15f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Left Wall", "Rigid side wall for the rigid magnetic floor test.");
    sdf.add_segment(vec2(2.45f, -1.50f), vec2(2.45f, 2.20f), 0.15f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Right Wall", "Rigid side wall for the rigid magnetic floor test.");
    sdf.add_box(vec2(0.0f, -1.18f), vec2(2.05f, 0.20f),
                SDFField::MaterialPresetID::SOFT_IRON,
                "Soft Iron Floor", "Rigid magnetizable floor. Its M field should light up before you touch M, and the field lines should bend around the floor fittings.");
    sdf.add_box(vec2(0.0f, -1.56f), vec2(1.56f, 0.10f),
                SDFField::MaterialPresetID::MAGNET_X,
                "Drive Magnet", "Permanent magnet embedded under the floor to induce the entire soft-iron plate.");
    sdf.add_box(vec2(-1.18f, -0.82f), vec2(0.32f, 0.34f),
                SDFField::MaterialPresetID::SOFT_IRON,
                "Rigid Iron Slug", "Rigid soft-iron test slug. Use this to validate field concentration without MPM collapse artifacts.");
    sdf.add_box(vec2(-0.10f, -0.78f), vec2(0.14f, 0.44f),
                SDFField::MaterialPresetID::SOFT_IRON,
                "Floor Pole", "Soft-iron pole piece rooted in the floor to visibly concentrate the induced field.");
    sdf.add_box(vec2(0.72f, -0.96f), vec2(0.15f, 0.20f),
                SDFField::MaterialPresetID::SOFT_IRON,
                "Floor Tooth A", "First rigid tooth on the soft-iron floor so field concentration is easier to spot.");
    sdf.add_box(vec2(1.10f, -0.96f), vec2(0.15f, 0.20f),
                SDFField::MaterialPresetID::SOFT_IRON,
                "Floor Tooth B", "Second rigid tooth on the soft-iron floor for extra line distortion.");
    sdf.rebuild();

    mpm.params().enable_thermal = false;
    mpm.params().magnet_spike_strength = 1.45f;
    mpm.params().magnet_chain_rate = 8.8f;
    mpm.params().magnet_spike_freq = 15.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 4200.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.56f, -1.04f), vec2(1.62f, -0.78f), sp,
                    MPMMaterial::FERRO_FLUID, 300.0f, vec2(1, 0), 0.16f);
    register_scene_mpm_batch(creation,
                             "Ferro Floor Puddle",
                             "Expected: this is the deformable comparison sample. It should neck and creep on the induced floor, while the rigid SDF iron pieces hold their shape and just concentrate the field.",
                             particles, mpm_before, MPMMaterial::FERRO_FLUID,
                             4200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 0.16f);

    mpm.params().youngs_modulus = old_E;
}

static void load_mag_cursor_unit(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                 MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                 CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-2.10f, -1.28f), vec2(2.10f, -1.28f), 0.10f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Bench Plate", "Non-magnetic plate for testing the cursor magnet only.");
    sdf.rebuild();

    mpm.params().enable_thermal = false;
    mpm.params().magnet_radius = 0.56f;
    mpm.params().magnet_spike_strength = 1.05f;
    mpm.params().magnet_chain_rate = 6.4f;
    mpm.params().magnet_spike_freq = 14.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    mpm.params().youngs_modulus = 4800.0f;

    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.36f, -1.06f), vec2(0.36f, -0.84f), sp,
                    MPMMaterial::FERRO_FLUID, 300.0f, vec2(1, 0), 0.14f);
    register_scene_mpm_batch(creation,
                             "Cursor Ferro Puddle",
                             "Expected: with no scene magnets at all, holding M above the puddle should pull it toward the brush axis and create a single centered hump or short comb, not a long-lived ring on the brush rim.",
                             particles, mpm_before, MPMMaterial::FERRO_FLUID,
                             4800.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 0.14f);

    mpm.params().youngs_modulus = old_E;
}

static void load_mag_permanent_pole(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                    MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                    CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-2.10f, -1.30f), vec2(2.10f, -1.30f), 0.10f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Bench Plate", "Simple plate for a single permanent-pole magnetic test.");
    sdf.add_box(vec2(0.0f, -0.94f), vec2(0.20f, 0.62f),
                SDFField::MaterialPresetID::MAGNET_Y,
                "Pole Magnet", "Single permanent vertical magnet. The field should be strongest near the exposed top pole.");
    sdf.rebuild();

    mpm.params().enable_thermal = false;
    mpm.params().magnet_spike_strength = 1.00f;
    mpm.params().magnet_chain_rate = 6.0f;
    mpm.params().magnet_spike_freq = 13.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    mpm.params().youngs_modulus = 4700.0f;

    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.34f, -0.08f), vec2(0.34f, 0.10f), sp,
                    MPMMaterial::FERRO_FLUID, 300.0f, vec2(1, 0), 0.14f);
    register_scene_mpm_batch(creation,
                             "Pole Ferro Puddle",
                             "Expected: with only one permanent magnet and no soft iron, the puddle should collect above the pole tip and form one dominant hump or short neck instead of pinning to a circular ring.",
                             particles, mpm_before, MPMMaterial::FERRO_FLUID,
                             4700.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 0.14f);

    mpm.params().youngs_modulus = old_E;
}

static void load_mag_soft_iron_field(ParticleBuffer& /*particles*/, SPHSolver& /*sph*/,
                                     MPMSolver& mpm, UniformGrid& /*grid*/, SDFField& sdf,
                                     CreationState* /*creation*/) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-2.12f, -1.30f), vec2(2.12f, -1.30f), 0.10f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Bench Plate", "Field-only magnetic scene: no magnetic MPM, only rigid SDF magnets.");
    sdf.add_box(vec2(-0.68f, -0.98f), vec2(0.82f, 0.18f),
                SDFField::MaterialPresetID::MAGNET_X,
                "Drive Magnet", "Permanent drive magnet for the field-only distortion test.");
    sdf.add_box(vec2(0.56f, -0.52f), vec2(0.12f, 0.52f),
                SDFField::MaterialPresetID::SOFT_IRON,
                "Soft Iron Pole", "Rigid soft-iron pole. Magnetic lines should bend into it and intensify near the tip.");
    sdf.add_box(vec2(0.28f, 0.02f), vec2(0.34f, 0.08f),
                SDFField::MaterialPresetID::SOFT_IRON,
                "Soft Iron Cap", "Soft-iron cap that should spread the pole field into a stronger local hot region.");
    sdf.rebuild();

    mpm.params().enable_thermal = false;
}

static void load_mag_soft_iron_body(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                    MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                    CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-2.10f, -1.30f), vec2(2.10f, -1.30f), 0.10f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Bench Plate", "Non-magnetic plate for isolating MPM soft-iron body response.");
    sdf.add_box(vec2(0.0f, -1.00f), vec2(0.22f, 0.64f),
                SDFField::MaterialPresetID::MAGNET_Y,
                "Drive Pole", "Single permanent pole used to test whether the soft-iron MPM body translates without collapsing.");
    sdf.rebuild();

    mpm.params().enable_thermal = false;
    mpm.params().magnet_spike_strength = 0.85f;
    mpm.params().magnet_chain_rate = 5.0f;
    mpm.params().magnet_spike_freq = 12.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    mpm.params().youngs_modulus = 56000.0f;

    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.62f, -0.38f), vec2(1.08f, -0.02f), sp,
                    MPMMaterial::MAG_SOFT_IRON, 300.0f, vec2(1, 0), 0.26f);
    register_scene_mpm_batch(creation,
                             "Soft Iron Body",
                             "Expected: this block should drift toward the pole and may cant slightly, but it should stay body-like instead of collapsing into a dot or short hair.",
                             particles, mpm_before, MPMMaterial::MAG_SOFT_IRON,
                             56000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 0.26f);

    mpm.params().youngs_modulus = old_E;
}

static void load_open_oven(ParticleBuffer& particles, SPHSolver& /*sph*/,
                           MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                           CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    // Open-top oven cavity
    sdf.add_segment(vec2(-1.4f, -1.4f), vec2(1.4f, -1.4f), 0.18f);
    sdf.add_segment(vec2(-1.4f, -1.4f), vec2(-1.4f, 1.4f), 0.18f);
    sdf.add_segment(vec2(1.4f, -1.4f), vec2(1.4f, 1.4f), 0.18f);
    // Inner shelf
    sdf.add_segment(vec2(-0.9f, -0.55f), vec2(0.9f, -0.55f), 0.08f);
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.18f);
    mpm.params().heat_source_radius = 0.65f;
    mpm.params().heat_source_temp = 1150.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 9000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.75f, -0.45f), vec2(-0.2f, -0.05f), sp,
                    MPMMaterial::SPLINTER, 300.0f, vec2(0, 1));
    register_scene_mpm_batch(creation, "Kiln Reed Stack", "Heat-cures and splinters in the oven.",
                             particles, mpm_before, MPMMaterial::SPLINTER,
                             9000.0f, mpm.params().poisson_ratio, 300.0f, vec2(0, 1));

    mpm.params().youngs_modulus = 14000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.1f, -0.4f), vec2(0.7f, 0.05f), sp,
                    MPMMaterial::CERAMIC, 300.0f);
    register_scene_mpm_batch(creation, "Ceramic Brick", "Ceramic test piece.",
                             particles, mpm_before, MPMMaterial::CERAMIC,
                             14000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 8000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.25f, -1.2f), vec2(0.25f, -0.92f), sp,
                    MPMMaterial::EMBER, 430.0f);
    register_scene_mpm_batch(creation, "Coal Bed", "Hot embers feeding the oven.",
                             particles, mpm_before, MPMMaterial::EMBER,
                             8000.0f, mpm.params().poisson_ratio, 430.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_pot_heater(ParticleBuffer& particles, SPHSolver& /*sph*/,
                            MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                            CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);

    // Closed lower firebox / heater chamber
    sdf.add_segment(vec2(-1.4f, -1.35f), vec2(1.4f, -1.35f), 0.18f);
    sdf.add_segment(vec2(-1.4f, -1.35f), vec2(-1.4f, -0.15f), 0.18f);
    sdf.add_segment(vec2(1.4f, -1.35f), vec2(1.4f, -0.15f), 0.18f);
    sdf.add_segment(vec2(-1.4f, -0.15f), vec2(1.4f, -0.15f), 0.18f);

    // Open pot sitting on top of the heater lid
    sdf.add_segment(vec2(-0.9f, 0.15f), vec2(0.9f, 0.15f), 0.12f);
    sdf.add_segment(vec2(-0.9f, 0.15f), vec2(-0.9f, 1.2f), 0.12f);
    sdf.add_segment(vec2(0.9f, 0.15f), vec2(0.9f, 1.2f), 0.12f);
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -0.8f);
    mpm.params().heat_source_radius = 0.55f;
    mpm.params().heat_source_temp = 1250.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 2500.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.55f, 0.28f), vec2(0.55f, 0.58f), sp,
                    MPMMaterial::FLAMMABLE_FLUID, 300.0f);
    register_scene_mpm_batch(creation, "Pot Liquid", "Flammable liquid inside the pot.",
                             particles, mpm_before, MPMMaterial::FLAMMABLE_FLUID,
                             2500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 16000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.25f, 0.65f), vec2(0.25f, 1.0f), sp,
                    MPMMaterial::GLASS, 300.0f);
    register_scene_mpm_batch(creation, "Glass Charge", "Glass test piece above the liquid.",
                             particles, mpm_before, MPMMaterial::GLASS,
                             16000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 7000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.3f, -1.18f), vec2(0.3f, -0.92f), sp,
                    MPMMaterial::BURNING, 380.0f, vec2(0, 1));
    register_scene_mpm_batch(creation, "Fuel Pack", "Burning fuel under the pot.",
                             particles, mpm_before, MPMMaterial::BURNING,
                             7000.0f, mpm.params().poisson_ratio, 380.0f, vec2(0, 1));

    mpm.params().youngs_modulus = old_E;
}

static void load_codim_threads_cold(ParticleBuffer& particles, SPHSolver& sph,
                                    MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                    CreationState* creation) {
    sdf.clear();
    add_codim_threads_geometry(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.rebuild();

    SPHParams sph_p = sph.params();
    sph_p.codim_enabled = true;
    sph_p.codim_threshold = 0.3f;
    sph_p.surface_tension = 1.5f;
    sph_p.viscosity = 0.3f;
    sph_p.gas_constant = 6.0f;
    sph.set_params(sph_p);

    mpm.params().enable_thermal = false;
    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;
    mpm.params().youngs_modulus = 26000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.32f, 0.06f), vec2(0.32f, 0.26f), sp,
                    MPMMaterial::TOUGH, 260.0f, vec2(1, 0));
    register_scene_mpm_batch(creation, "Blast Lid", "A movable lid for gunpowder and pressure tests.",
                             particles, mpm_before, MPMMaterial::TOUGH,
                             26000.0f, mpm.params().poisson_ratio, 260.0f, vec2(1, 0));
    mpm.params().youngs_modulus = old_E;
}

static void load_zero_g_soft_lab(ParticleBuffer& /*particles*/, SPHSolver& /*sph*/,
                                 MPMSolver& mpm, UniformGrid& /*grid*/, SDFField& sdf,
                                 CreationState* /*creation*/) {
    sdf.clear();
    sdf.add_segment(vec2(-2.5f, -1.5f), vec2(2.5f, -1.5f), 0.10f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Lower Rail", "Soft-lab lower guide rail.");
    sdf.add_segment(vec2(-2.5f, 2.2f), vec2(2.5f, 2.2f), 0.10f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Upper Rail", "Soft-lab upper guide rail.");
    sdf.add_segment(vec2(-2.2f, -1.2f), vec2(-2.2f, 2.0f), 0.10f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Left Frame", "Zero-g lab frame.");
    sdf.add_segment(vec2(2.2f, -1.2f), vec2(2.2f, 2.0f), 0.10f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Right Frame", "Zero-g lab frame.");
    sdf.add_segment(vec2(-0.9f, 0.2f), vec2(0.9f, 0.2f), 0.06f, SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                    "Center Guide", "Conductive guide rail for soft material tests.");
    sdf.rebuild();
    mpm.params().enable_thermal = false;
}

static void load_thermal_bridge(ParticleBuffer& /*particles*/, SPHSolver& /*sph*/,
                                MPMSolver& mpm, UniformGrid& /*grid*/, SDFField& sdf,
                                CreationState* /*creation*/) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_box(vec2(-1.7f, -0.2f), vec2(0.55f, 1.05f), SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Left Heat Sink", "Large brass sink: slow to heat, stores a lot of energy.");
    sdf.add_box(vec2(1.7f, -0.2f), vec2(0.55f, 1.05f), SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Right Heat Sink", "Large brass sink: strong thermal inertia.");
    sdf.add_box(vec2(0.0f, 0.45f), vec2(1.2f, 0.12f), SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                "Silver Bridge", "Highly conductive bridge that passes heat between the sinks.");
    sdf.add_segment(vec2(-1.15f, -0.15f), vec2(-0.55f, 0.33f), 0.08f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Left Support", "Decorative support.");
    sdf.add_segment(vec2(1.15f, -0.15f), vec2(0.55f, 0.33f), 0.08f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Right Support", "Decorative support.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(-1.75f, -1.1f);
    mpm.params().heat_source_radius = 0.45f;
    mpm.params().heat_source_temp = 1200.0f;
}

static void load_thermal_bridge_strong(ParticleBuffer& /*particles*/, SPHSolver& /*sph*/,
                                       MPMSolver& mpm, UniformGrid& /*grid*/, SDFField& sdf,
                                       CreationState* /*creation*/) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_box(vec2(-1.7f, -0.15f), vec2(0.62f, 1.12f), SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Left Heat Sink XL", "Large sink tuned for strong internal conduction and high thermal storage.");
    sdf.add_box(vec2(1.7f, -0.15f), vec2(0.62f, 1.12f), SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Right Heat Sink XL", "Large sink tuned for strong internal conduction and high thermal storage.");
    sdf.add_box(vec2(0.0f, 0.55f), vec2(1.38f, 0.16f), SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                "Silver Bridge XL", "Aggressively conductive bridge linking the two sinks.");
    sdf.add_segment(vec2(-1.22f, -0.22f), vec2(-0.62f, 0.39f), 0.10f, SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                    "Left Brace", "Conductive brace feeding heat into the bridge.");
    sdf.add_segment(vec2(1.22f, -0.22f), vec2(0.62f, 0.39f), 0.10f, SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                    "Right Brace", "Conductive brace feeding heat into the bridge.");
    sdf.rebuild();

    SDFField::MaterialPreset sink = SDFField::material_preset(SDFField::MaterialPresetID::BRASS_HEAT_SINK);
    sink.name = "Brass Heat Sink+";
    sink.summary = "High-capacity sink with much stronger internal conduction than the default brass preset.";
    sink.conductivity_scale = 3.6f;
    sink.heat_capacity_scale = 4.6f;
    sink.contact_transfer_scale = 1.05f;
    sink.heat_loss_scale = 0.55f;
    sdf.set_object_material(4, sink);
    sdf.set_object_material(5, sink);

    SDFField::MaterialPreset bridge = SDFField::material_preset(SDFField::MaterialPresetID::SILVER_CONDUCTIVE);
    bridge.name = "Silver Bridge+";
    bridge.summary = "Very strong bridge conduction for whole-piece heat spread.";
    bridge.conductivity_scale = 6.0f;
    bridge.heat_capacity_scale = 1.2f;
    bridge.contact_transfer_scale = 1.8f;
    bridge.heat_loss_scale = 0.75f;
    sdf.set_object_material(6, bridge);
    sdf.set_object_material(7, bridge);
    sdf.set_object_material(8, bridge);

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(-1.82f, -1.08f);
    mpm.params().heat_source_radius = 0.58f;
    mpm.params().heat_source_temp = 1350.0f;
}

static void load_spiral_metals(ParticleBuffer& /*particles*/, SPHSolver& /*sph*/,
                               MPMSolver& mpm, UniformGrid& /*grid*/, SDFField& sdf,
                               CreationState* /*creation*/) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    add_spiral(sdf, vec2(-1.05f, 0.45f), 0.12f, 0.92f, 2.8f, 0.06f,
               SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
               "Silver Spiral", "Highly conductive spiral guide.");
    add_spiral(sdf, vec2(1.05f, 0.45f), 0.18f, 0.95f, -2.6f, 0.08f,
               SDFField::MaterialPresetID::BRASS_HEAT_SINK,
               "Brass Spiral", "High-capacity spiral that absorbs heat slowly.");
    sdf.add_box(vec2(0.0f, -0.55f), vec2(0.4f, 0.1f), SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Center Plinth", "Warm alloy plinth between the spirals.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.2f);
    mpm.params().heat_source_radius = 0.55f;
    mpm.params().heat_source_temp = 1100.0f;
}

static void load_thin_pipe(ParticleBuffer& /*particles*/, SPHSolver& /*sph*/,
                           MPMSolver& mpm, UniformGrid& /*grid*/, SDFField& sdf,
                           CreationState* /*creation*/) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-0.16f, -1.35f), vec2(-0.16f, 2.45f), 0.08f,
                    SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                    "Pipe Left Wall", "Thin upright pipe wall for constrained stack / ignition tests.");
    sdf.add_segment(vec2(0.16f, -1.35f), vec2(0.16f, 2.45f), 0.08f,
                    SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                    "Pipe Right Wall", "Thin upright pipe wall for constrained stack / ignition tests.");
    sdf.add_segment(vec2(-0.16f, -1.35f), vec2(0.16f, -1.35f), 0.08f,
                    SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                    "Pipe Base", "Closed bottom cap; top stays open.");
    sdf.rebuild();
    mpm.params().enable_thermal = false;
}

static void load_empty_zero_g(ParticleBuffer& /*particles*/, SPHSolver& /*sph*/,
                              MPMSolver& mpm, UniformGrid& /*grid*/, SDFField& sdf,
                              CreationState* /*creation*/) {
    sdf.clear();
    sdf.rebuild();
    mpm.params().enable_thermal = false;
}

static void load_floor_only(ParticleBuffer& /*particles*/, SPHSolver& /*sph*/,
                            MPMSolver& mpm, UniformGrid& /*grid*/, SDFField& sdf,
                            CreationState* /*creation*/) {
    sdf.clear();
    sdf.add_segment(vec2(-2.5f, -1.5f), vec2(2.5f, -1.5f), 0.15f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Floor", "Single floor plane with no side walls.");
    sdf.rebuild();
    mpm.params().enable_thermal = false;
}

static void load_glaze_kiln(ParticleBuffer& particles, SPHSolver& /*sph*/,
                            MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                            CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.55f, -1.35f), vec2(1.55f, -1.35f), 0.18f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Kiln Base", "Heavy kiln floor.");
    sdf.add_segment(vec2(-1.55f, -1.35f), vec2(-1.55f, 1.25f), 0.18f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Kiln Left Wall", "Insulating kiln wall.");
    sdf.add_segment(vec2(1.55f, -1.35f), vec2(1.55f, 1.25f), 0.18f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Kiln Right Wall", "Insulating kiln wall.");
    sdf.add_segment(vec2(-0.95f, -0.42f), vec2(0.95f, -0.42f), 0.08f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Kiln Shelf", "Hot shelf for glaze and shell-core pottery tests.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.18f);
    mpm.params().heat_source_radius = 0.68f;
    mpm.params().heat_source_temp = 1220.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 11000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.88f, -0.34f), vec2(-0.18f, 0.10f), sp,
                    MPMMaterial::GLAZE_CLAY, 300.0f);
    register_scene_mpm_batch(creation, "Glaze Tile Blank", "Shell-first pottery blank for glaze-skin and crack tests.",
                             particles, mpm_before, MPMMaterial::GLAZE_CLAY,
                             11000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 12000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.12f, -0.28f), vec2(0.78f, 0.16f), sp,
                    MPMMaterial::GLAZE_CLAY, 300.0f);
    register_scene_mpm_batch(creation, "Glaze Brick Blank", "Thicker pottery blank that shows shell/core mismatch more slowly.",
                             particles, mpm_before, MPMMaterial::GLAZE_CLAY,
                             12000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 8500.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.26f, -1.18f), vec2(0.26f, -0.92f), sp,
                    MPMMaterial::EMBER, 460.0f);
    register_scene_mpm_batch(creation, "Kiln Coals", "Hot embers feeding the kiln chamber.",
                             particles, mpm_before, MPMMaterial::EMBER,
                             8500.0f, mpm.params().poisson_ratio, 460.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_bake_oven(ParticleBuffer& particles, SPHSolver& /*sph*/,
                           MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                           CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.45f, -1.30f), vec2(1.45f, -1.30f), 0.16f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Oven Deck", "Baking deck.");
    sdf.add_segment(vec2(-1.45f, -1.30f), vec2(-1.45f, 1.20f), 0.16f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Oven Left Wall", "Open-top oven wall.");
    sdf.add_segment(vec2(1.45f, -1.30f), vec2(1.45f, 1.20f), 0.16f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Oven Right Wall", "Open-top oven wall.");
    sdf.add_segment(vec2(-0.95f, -0.36f), vec2(0.95f, -0.36f), 0.08f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Baking Stone", "Hot stone for loaf tests.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.12f);
    mpm.params().heat_source_radius = 0.74f;
    mpm.params().heat_source_temp = 980.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 8200.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.86f, -0.30f), vec2(0.06f, 0.18f), sp,
                    MPMMaterial::CRUST_DOUGH, 300.0f);
    register_scene_mpm_batch(creation, "Crust Loaf", "Shell-drying loaf that crusts outside while the core keeps expanding.",
                             particles, mpm_before, MPMMaterial::CRUST_DOUGH,
                             8200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 7200.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.14f, -0.30f), vec2(0.92f, 0.14f), sp,
                    MPMMaterial::BREAD, 300.0f);
    register_scene_mpm_batch(creation, "Bread Dough Ref", "Earlier gas-only bread prototype for side-by-side comparison.",
                             particles, mpm_before, MPMMaterial::BREAD,
                             7200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 8000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.24f, -1.16f), vec2(0.24f, -0.92f), sp,
                    MPMMaterial::EMBER, 430.0f);
    register_scene_mpm_batch(creation, "Baking Coals", "Steady ember bed for the oven.",
                             particles, mpm_before, MPMMaterial::EMBER,
                             8000.0f, mpm.params().poisson_ratio, 430.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_stress_forge(ParticleBuffer& particles, SPHSolver& /*sph*/,
                              MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                              CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_box(vec2(-1.12f, -0.68f), vec2(0.22f, 0.72f), SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Left Support", "Hot-side support with more thermal soak.");
    sdf.add_box(vec2(1.12f, -0.68f), vec2(0.22f, 0.72f), SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                "Right Support", "Cooler conductive support.");
    sdf.add_segment(vec2(-1.36f, -1.28f), vec2(-0.58f, -1.28f), 0.08f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Forge Burner Lip", "Helps localize the heat under the left side.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(-1.00f, -1.04f);
    mpm.params().heat_source_radius = 0.54f;
    mpm.params().heat_source_temp = 1280.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 28000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.08f, 0.06f), vec2(1.08f, 0.28f), sp,
                    MPMMaterial::THERMO_METAL, 300.0f);
    register_scene_mpm_batch(creation, "Steel Bridge", "Thermoelastic bar that heats soft, expands, then cools back harder and stress-prone.",
                             particles, mpm_before, MPMMaterial::THERMO_METAL,
                             28000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 18000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(0.0f, 0.72f), 0.18f, sp, MPMMaterial::THERMO_METAL, 300.0f);
    register_scene_mpm_batch(creation, "Steel Test Bead", "Compact metal sample for direct heat/cool poking.",
                             particles, mpm_before, MPMMaterial::THERMO_METAL,
                             18000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_reactive_hearth(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                 MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                 CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-0.72f, -1.28f), vec2(-0.72f, -0.18f), 0.12f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Hearth Left Wall", "Confining wall for reactive burn tests.");
    sdf.add_segment(vec2(0.72f, -1.28f), vec2(0.72f, -0.18f), 0.12f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Hearth Right Wall", "Confining wall for reactive burn tests.");
    sdf.add_segment(vec2(-0.72f, -0.18f), vec2(0.72f, -0.18f), 0.10f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Upper Rail", "Low cap rail so the smoke and rupture read against a boundary.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.08f);
    mpm.params().heat_source_radius = 0.58f;
    mpm.params().heat_source_temp = 1180.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 9000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.26f, -1.16f), vec2(0.26f, -0.92f), sp,
                    MPMMaterial::EMBER, 440.0f);
    register_scene_mpm_batch(creation, "Hearth Coals", "Coal bed feeding the reactive resin.",
                             particles, mpm_before, MPMMaterial::EMBER,
                             9000.0f, mpm.params().poisson_ratio, 440.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 15000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(0.0f, -0.54f), 0.22f, sp, MPMMaterial::REACTIVE_BURN, 300.0f);
    register_scene_mpm_batch(creation, "Resin Charge", "Staged-burn resin that chars on the outside, gases inside, then blisters and pops.",
                             particles, mpm_before, MPMMaterial::REACTIVE_BURN,
                             15000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 24000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.42f, 0.02f), vec2(0.42f, 0.16f), sp,
                    MPMMaterial::TOUGH, 300.0f, vec2(1, 0));
    register_scene_mpm_batch(creation, "Test Lid", "Movable lid to catch the smoky pressure burst.",
                             particles, mpm_before, MPMMaterial::TOUGH,
                             24000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_glaze_rack(ParticleBuffer& particles, SPHSolver& /*sph*/,
                            MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                            CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.65f, -1.34f), vec2(1.65f, -1.34f), 0.18f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Rack Base", "Hot base below the glaze rack.");
    sdf.add_segment(vec2(-1.05f, 0.32f), vec2(1.05f, 0.32f), 0.08f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Rack Bar", "Upper bar for glaze ornaments.");
    sdf.add_segment(vec2(-0.72f, 0.32f), vec2(-0.72f, -0.12f), 0.05f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Left Hook", "Left hanger.");
    sdf.add_segment(vec2(0.72f, 0.32f), vec2(0.72f, -0.12f), 0.05f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Right Hook", "Right hanger.");
    sdf.add_segment(vec2(-1.05f, -0.52f), vec2(1.05f, -0.52f), 0.08f, SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Lower Shelf", "Warm shelf for glaze pooling tests.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.10f);
    mpm.params().heat_source_radius = 0.72f;
    mpm.params().heat_source_temp = 1210.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 9000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.96f, -0.42f), vec2(-0.26f, -0.04f), sp,
                    MPMMaterial::GLAZE_DRIP, 300.0f);
    register_scene_mpm_batch(creation, "Drip Tile", "Runny glaze shell demo that should pool and droop while the core keeps ceramic body.",
                             particles, mpm_before, MPMMaterial::GLAZE_DRIP,
                             9000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 11200.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(0.66f, -0.08f), 0.23f, sp, MPMMaterial::GLAZE_DRIP, 300.0f);
    register_scene_mpm_batch(creation, "Drip Ornament", "Rounded glaze piece for shell flow and glossy pooling.",
                             particles, mpm_before, MPMMaterial::GLAZE_DRIP,
                             11200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 11000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.16f, -0.42f), vec2(0.16f, -0.10f), sp,
                    MPMMaterial::GLAZE_CLAY, 300.0f);
    register_scene_mpm_batch(creation, "Glaze Clay Ref", "Reference shell-first pottery beside the drippier glaze variant.",
                             particles, mpm_before, MPMMaterial::GLAZE_CLAY,
                             11000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_steam_oven(ParticleBuffer& particles, SPHSolver& /*sph*/,
                            MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                            CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.38f, -1.28f), vec2(1.38f, -1.28f), 0.16f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Steam Deck", "Hot oven floor.");
    sdf.add_segment(vec2(-1.38f, -1.28f), vec2(-1.38f, 1.08f), 0.16f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Steam Left Wall", "Steam oven wall.");
    sdf.add_segment(vec2(1.38f, -1.28f), vec2(1.38f, 1.08f), 0.16f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Steam Right Wall", "Steam oven wall.");
    sdf.add_segment(vec2(-0.94f, -0.34f), vec2(0.94f, -0.34f), 0.08f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Steam Stone", "Raised stone for buns and loaves.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.08f);
    mpm.params().heat_source_radius = 0.78f;
    mpm.params().heat_source_temp = 940.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 7600.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(-0.62f, -0.06f), 0.24f, sp, MPMMaterial::STEAM_BUN, 300.0f);
    register_scene_mpm_batch(creation, "Steam Bun", "Springier bun tuned for rounder lift and softer skin venting.",
                             particles, mpm_before, MPMMaterial::STEAM_BUN,
                             7600.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 8200.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.10f, -0.28f), vec2(0.54f, 0.12f), sp,
                    MPMMaterial::CRUST_DOUGH, 300.0f);
    register_scene_mpm_batch(creation, "Crust Loaf Ref", "Bread-like loaf with earlier shell set for comparison against the bun.",
                             particles, mpm_before, MPMMaterial::CRUST_DOUGH,
                             8200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 7200.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.68f, -0.24f), vec2(1.10f, 0.08f), sp,
                    MPMMaterial::BREAD, 300.0f);
    register_scene_mpm_batch(creation, "Bread Ref", "Gas-only bread baseline beside the shell/core doughs.",
                             particles, mpm_before, MPMMaterial::BREAD,
                             7200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_seed_roaster(ParticleBuffer& particles, SPHSolver& /*sph*/,
                              MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                              CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-0.98f, -1.26f), vec2(0.98f, -1.26f), 0.14f, SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Roaster Plate", "Hot plate under the seed bed.");
    sdf.add_segment(vec2(-0.98f, -1.26f), vec2(-0.98f, -0.38f), 0.12f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Pan Left", "Shallow roasting pan wall.");
    sdf.add_segment(vec2(0.98f, -1.26f), vec2(0.98f, -0.38f), 0.12f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Pan Right", "Shallow roasting pan wall.");
    sdf.add_segment(vec2(-0.42f, -0.04f), vec2(0.42f, -0.04f), 0.08f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Light Lid", "A light cap to make the little pops visible.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.08f);
    mpm.params().heat_source_radius = 0.60f;
    mpm.params().heat_source_temp = 1120.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 9000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.38f, -1.12f), vec2(0.38f, -0.84f), sp,
                    MPMMaterial::REACTIVE_BURN, 300.0f);
    register_scene_mpm_batch(creation, "Seed Bed", "Many small reactive kernels meant to toast, smoke, and pop under the light lid.",
                             particles, mpm_before, MPMMaterial::REACTIVE_BURN,
                             9000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 18000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.34f, -0.01f), vec2(0.34f, 0.10f), sp,
                    MPMMaterial::TOUGH, 300.0f, vec2(1, 0));
    register_scene_mpm_batch(creation, "Roaster Lid", "Light tough lid for reading lots of tiny pressure kicks.",
                             particles, mpm_before, MPMMaterial::TOUGH,
                             18000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_crazing_shelf(ParticleBuffer& particles, SPHSolver& /*sph*/,
                               MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                               CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.45f, -1.28f), vec2(1.45f, -1.28f), 0.16f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Shelf Base", "Warm kiln base under the comparison shelf.");
    sdf.add_segment(vec2(-1.12f, -0.30f), vec2(1.12f, -0.30f), 0.09f, SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Kiln Shelf", "Heat-soaking shelf for shell and glaze comparisons.");
    sdf.add_segment(vec2(-0.96f, 0.46f), vec2(0.96f, 0.46f), 0.06f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Top Rail", "Warm upper rail to keep the chamber readable.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.08f);
    mpm.params().heat_source_radius = 0.74f;
    mpm.params().heat_source_temp = 1210.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 11000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.00f, -0.24f), vec2(-0.44f, 0.12f), sp,
                    MPMMaterial::GLAZE_CLAY, 300.0f);
    register_scene_mpm_batch(creation, "Shell Tile", "Layered glaze-clay reference piece for shell-first firing.",
                             particles, mpm_before, MPMMaterial::GLAZE_CLAY,
                             11000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 13500.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.18f, -0.22f), vec2(0.36f, 0.10f), sp,
                    MPMMaterial::GLAZE_CLAY, 300.0f);
    register_scene_mpm_batch(creation, "Crazing Tile", "Stiffer glaze-shell tile intended to crack and craze more than it drips.",
                             particles, mpm_before, MPMMaterial::GLAZE_CLAY,
                             13500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 9200.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(0.90f, -0.02f), 0.23f, sp, MPMMaterial::GLAZE_DRIP, 300.0f);
    register_scene_mpm_batch(creation, "Glaze Bead", "Round glaze-drip sample meant to pool and droop while keeping a ceramic core.",
                             particles, mpm_before, MPMMaterial::GLAZE_DRIP,
                             9200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_pocket_oven(ParticleBuffer& particles, SPHSolver& /*sph*/,
                             MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                             CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.42f, -1.28f), vec2(1.42f, -1.28f), 0.16f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Pocket Deck", "Baking deck for side-by-side bread tests.");
    sdf.add_segment(vec2(-1.42f, -1.28f), vec2(-1.42f, 1.08f), 0.16f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Pocket Left Wall", "Open-top oven wall.");
    sdf.add_segment(vec2(1.42f, -1.28f), vec2(1.42f, 1.08f), 0.16f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Pocket Right Wall", "Open-top oven wall.");
    sdf.add_segment(vec2(-0.98f, -0.34f), vec2(0.98f, -0.34f), 0.08f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Pocket Stone", "Raised stone for buns, pockets, and loaves.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.10f);
    mpm.params().heat_source_radius = 0.80f;
    mpm.params().heat_source_temp = 960.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 6800.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.02f, -0.24f), vec2(-0.36f, -0.02f), sp,
                    MPMMaterial::STEAM_BUN, 300.0f);
    register_scene_mpm_batch(creation, "Pita Pocket", "Flat shell/core dough tuned to balloon outward and wrinkle after venting.",
                             particles, mpm_before, MPMMaterial::STEAM_BUN,
                             6800.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 7600.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(0.02f, -0.04f), 0.22f, sp, MPMMaterial::STEAM_BUN, 300.0f);
    register_scene_mpm_batch(creation, "Steam Bun", "Rounder steam-bun reference for a puffier shell/core rise.",
                             particles, mpm_before, MPMMaterial::STEAM_BUN,
                             7600.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 8200.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.46f, -0.28f), vec2(1.06f, 0.12f), sp,
                    MPMMaterial::CRUST_DOUGH, 300.0f);
    register_scene_mpm_batch(creation, "Crust Loaf", "Chunkier shell-setting loaf for comparison against the pocket and bun.",
                             particles, mpm_before, MPMMaterial::CRUST_DOUGH,
                             8200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_tempering_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                 MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                 CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_box(vec2(-1.12f, -0.74f), vec2(0.22f, 0.78f), SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Hot Jaw", "Hot-side support with extra soak.");
    sdf.add_box(vec2(1.12f, -0.74f), vec2(0.22f, 0.78f), SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                "Cool Jaw", "Cooler conductive support to promote gradients.");
    sdf.add_segment(vec2(-1.40f, -1.28f), vec2(-0.56f, -1.28f), 0.08f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Bench Burner", "Localized burner for uneven heating.");
    sdf.add_segment(vec2(0.62f, -1.18f), vec2(1.38f, -1.18f), 0.10f, SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Quench Block", "Dense sink block to pull heat down on the right.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(-1.02f, -1.04f);
    mpm.params().heat_source_radius = 0.54f;
    mpm.params().heat_source_temp = 1280.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 28000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.08f, 0.06f), vec2(1.08f, 0.26f), sp,
                    MPMMaterial::THERMO_METAL, 300.0f);
    register_scene_mpm_batch(creation, "Quench Bar", "Long thermoelastic bar for left-hot right-cool stress buildup.",
                             particles, mpm_before, MPMMaterial::THERMO_METAL,
                             28000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 22000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.90f, 0.58f), vec2(0.42f, 0.70f), sp,
                    MPMMaterial::THERMO_METAL, 300.0f);
    register_scene_mpm_batch(creation, "Spring Strip", "Slim strip for faster bend-and-recover reads under uneven heat.",
                             particles, mpm_before, MPMMaterial::THERMO_METAL,
                             22000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 34000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.10f, 0.82f), vec2(1.02f, 0.96f), sp,
                    MPMMaterial::THERMO_METAL, 300.0f);
    register_scene_mpm_batch(creation, "Tempering Blade", "Heavier blade blank for bowing and cool-set hardening tests.",
                             particles, mpm_before, MPMMaterial::THERMO_METAL,
                             34000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_pressure_pantry(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                 MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                 CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.36f, -1.26f), vec2(1.36f, -1.26f), 0.14f, SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Pantry Plate", "Shared hot plate under the burst compartments.");
    sdf.add_segment(vec2(-1.36f, -1.26f), vec2(-1.36f, -0.28f), 0.12f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Pantry Left", "Left outer wall.");
    sdf.add_segment(vec2(1.36f, -1.26f), vec2(1.36f, -0.28f), 0.12f, SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Pantry Right", "Right outer wall.");
    sdf.add_segment(vec2(-0.46f, -1.26f), vec2(-0.46f, -0.20f), 0.08f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Left Divider", "Separates gunpowder from the center pellets.");
    sdf.add_segment(vec2(0.46f, -1.26f), vec2(0.46f, -0.20f), 0.08f, SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Right Divider", "Separates pellets from the seed tray.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.08f);
    mpm.params().heat_source_radius = 0.82f;
    mpm.params().heat_source_temp = 1160.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 9000.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.18f, -1.12f), vec2(-0.66f, -0.90f), sp,
                    MPMMaterial::FIRECRACKER, 300.0f);
    register_scene_mpm_batch(creation, "Gunpowder Bed", "Granular fast-reacting charge meant to kick its lid sharply.",
                             particles, mpm_before, MPMMaterial::FIRECRACKER,
                             9000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 17000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(0.0f, -0.98f), 0.17f, sp, MPMMaterial::FIRECRACKER, 300.0f);
    register_scene_mpm_batch(creation, "Pellet Cluster", "Compact firecracker-like charge for a smokier delayed burst.",
                             particles, mpm_before, MPMMaterial::FIRECRACKER,
                             17000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 7600.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.66f, -1.12f), vec2(1.18f, -0.92f), sp,
                    MPMMaterial::REACTIVE_BURN, 300.0f);
    register_scene_mpm_batch(creation, "Popcorn Tray", "Reactive seed-like charge for repeated little jumps and pops.",
                             particles, mpm_before, MPMMaterial::REACTIVE_BURN,
                             7600.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 18500.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.12f, -0.18f), vec2(-0.72f, -0.06f), sp,
                    MPMMaterial::TOUGH, 300.0f, vec2(1, 0));
    register_scene_mpm_batch(creation, "Gunpowder Lid", "Light lid above the left charge to show the hard upward kick.",
                             particles, mpm_before, MPMMaterial::TOUGH,
                             18500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.18f, -0.12f), vec2(0.18f, 0.00f), sp,
                    MPMMaterial::TOUGH, 300.0f, vec2(1, 0));
    register_scene_mpm_batch(creation, "Pellet Lid", "Center lid for delayed smoky pellet bursts.",
                             particles, mpm_before, MPMMaterial::TOUGH,
                             18500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.72f, -0.18f), vec2(1.12f, -0.06f), sp,
                    MPMMaterial::TOUGH, 300.0f, vec2(1, 0));
    register_scene_mpm_batch(creation, "Pop Lid", "Right lid for repeated little pressure kicks from the popping tray.",
                             particles, mpm_before, MPMMaterial::TOUGH,
                             18500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_aniso_tear_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                  MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                  CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);

    const f32 lane_centers[3] = {-1.28f, 0.0f, 1.28f};
    const char* lane_labels[3] = {
        "Along-Grain Tear Strip",
        "Cross-Grain Tear Strip",
        "Bias-Grain Tear Strip"
    };
    const char* lane_summaries[3] = {
        "Fibers run with the span. After the strip cures, it should hold longer and tear into longer grain-aligned bands.",
        "Fibers run across the span. This one should give up earlier and crack more directly across the loaded bridge.",
        "Fibers run diagonally. This one should develop a skewed tear path that biases to one side."
    };
    const vec2 lane_fibers[3] = {
        vec2(1.0f, 0.0f),
        vec2(0.0f, 1.0f),
        glm::normalize(vec2(1.0f, 1.0f))
    };
    const char lane_suffix[3] = {'A', 'B', 'C'};

    for (i32 lane = 0; lane < 3; ++lane) {
        const f32 cx = lane_centers[lane];
        char label[64];
        sdf.add_box(vec2(cx - 0.46f, -0.18f), vec2(0.18f, 0.34f),
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    (snprintf(label, sizeof(label), "Left Saddle %c", lane_suffix[lane]), label),
                    "Rigid saddle support for a tear strip.");
        sdf.add_box(vec2(cx + 0.46f, -0.18f), vec2(0.18f, 0.34f),
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    (snprintf(label, sizeof(label), "Right Saddle %c", lane_suffix[lane]), label),
                    "Rigid saddle support for a tear strip.");
        sdf.add_segment(vec2(cx - 0.68f, -0.40f), vec2(cx - 0.68f, 0.56f), 0.04f,
                        SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                        (snprintf(label, sizeof(label), "Left Guide %c", lane_suffix[lane]), label),
                        "Guide rail that keeps the loaded strap from sliding fully off its saddle.");
        sdf.add_segment(vec2(cx + 0.68f, -0.40f), vec2(cx + 0.68f, 0.56f), 0.04f,
                        SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                        (snprintf(label, sizeof(label), "Right Guide %c", lane_suffix[lane]), label),
                        "Guide rail that keeps the loaded strap from sliding fully off its saddle.");
        sdf.add_segment(vec2(cx - 0.70f, -1.22f), vec2(cx + 0.70f, -1.22f), 0.09f,
                        SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                        (snprintf(label, sizeof(label), "Hot Plate %c", lane_suffix[lane]), label),
                        "Shared hot plate that cures each tear strip from below.");
    }
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.02f);
    mpm.params().heat_source_radius = 1.88f;
    mpm.params().heat_source_temp = 1080.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 old_fiber = mpm.params().fiber_strength;

    for (i32 lane = 0; lane < 3; ++lane) {
        const f32 cx = lane_centers[lane];
        const vec2 fiber_dir = lane_fibers[lane];

        mpm.params().youngs_modulus = 6800.0f;
        mpm.params().fiber_strength = 6.0f;
        u32 mpm_before = particles.range(SolverType::MPM).count;
        mpm.spawn_block(particles, vec2(cx - 0.62f, 0.16f), vec2(cx + 0.62f, 0.26f), sp,
                        MPMMaterial::ORTHO_TEAR, 300.0f, fiber_dir);
        register_scene_mpm_batch(creation, lane_labels[lane], lane_summaries[lane],
                                 particles, mpm_before, MPMMaterial::ORTHO_TEAR,
                                 6800.0f, mpm.params().poisson_ratio, 300.0f, fiber_dir);
        if (creation && !creation->batches.empty()) {
            BatchRecord& batch = creation->batches.back();
            batch.fiber_strength = 6.0f;
            batch.properties += " | orthotropic tear bench";
            batch.recommended_size = 0.62f;
            batch.recommended_note = "Let the strips cure first. Along-grain should fray into longer bands, cross-grain should snap more directly, and bias should skew its tear.";
        }

        mpm.params().youngs_modulus = 30000.0f;
        mpm.params().fiber_strength = 0.0f;
        mpm_before = particles.range(SolverType::MPM).count;
        mpm.spawn_circle(particles, vec2(cx, 0.44f), 0.20f, sp,
                         MPMMaterial::TOUGH, 300.0f, vec2(1.0f, 0.0f), 4.2f);
        register_scene_mpm_batch(creation,
                                 lane == 0 ? "Load Bead A" : (lane == 1 ? "Load Bead B" : "Load Bead C"),
                                 "Dense load bead that pulls the strip into tension under identical weight in each lane.",
                                 particles, mpm_before, MPMMaterial::TOUGH,
                                 30000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));
        if (creation && !creation->batches.empty()) {
            creation->batches.back().recommended_note = "Use the heated strip under this bead as the real readout; the bead is just the matching center load.";
        }
    }

    mpm.params().youngs_modulus = old_E;
    mpm.params().fiber_strength = old_fiber;
}

static void load_aniso_bend_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                  MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                  CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);

    const f32 beam_y[3] = {0.86f, 0.26f, -0.34f};
    const char* labels[3] = {
        "Along-Grain Cantilever",
        "Cross-Grain Cantilever",
        "Bias-Grain Cantilever"
    };
    const char* summaries[3] = {
        "Fibers run along the beam. This should be the stiffest and droop the least under the tip load.",
        "Fibers run across the beam. This should sag the most because the span is pulling across the reinforcement.",
        "Fibers run diagonally. This should land between the other two and show a skewed bend profile."
    };
    const vec2 fibers[3] = {
        vec2(1.0f, 0.0f),
        vec2(0.0f, 1.0f),
        glm::normalize(vec2(1.0f, 1.0f))
    };
    const char lane_suffix[3] = {'A', 'B', 'C'};
    for (i32 lane = 0; lane < 3; ++lane) {
        char label[64];
        const f32 y = beam_y[lane];
        sdf.add_box(vec2(-1.64f, y + 0.02f), vec2(0.16f, 0.22f),
                    SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                    (snprintf(label, sizeof(label), "Clamp %c", lane_suffix[lane]), label),
                    "Rigid clamp that anchors one end of the cantilever.");
        sdf.add_segment(vec2(-1.84f, y - 0.20f), vec2(-1.84f, y + 0.30f), 0.04f,
                        SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                        (snprintf(label, sizeof(label), "Backstop %c", lane_suffix[lane]), label),
                        "Backstop that keeps the beam root seated in the clamp.");
    }
    sdf.rebuild();
    mpm.params().enable_thermal = false;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 old_fiber = mpm.params().fiber_strength;

    for (i32 lane = 0; lane < 3; ++lane) {
        const f32 y = beam_y[lane];
        const vec2 fiber_dir = fibers[lane];

        mpm.params().youngs_modulus = 5600.0f;
        mpm.params().fiber_strength = 5.0f;
        u32 mpm_before = particles.range(SolverType::MPM).count;
        mpm.spawn_block(particles, vec2(-1.72f, y - 0.04f), vec2(-0.24f, y + 0.05f), sp,
                        MPMMaterial::ORTHO_BEND, 300.0f, fiber_dir);
        register_scene_mpm_batch(creation, labels[lane], summaries[lane],
                                 particles, mpm_before, MPMMaterial::ORTHO_BEND,
                                 5600.0f, mpm.params().poisson_ratio, 300.0f, fiber_dir);
        if (creation && !creation->batches.empty()) {
            BatchRecord& batch = creation->batches.back();
            batch.fiber_strength = 5.0f;
            batch.properties += " | orthotropic bend bench";
            batch.recommended_size = 0.62f;
            batch.recommended_note = "This one is for pure anisotropic stiffness. Along-grain should droop the least, cross-grain the most, and bias should land in between.";
        }

        mpm.params().youngs_modulus = 28000.0f;
        mpm.params().fiber_strength = 0.0f;
        mpm_before = particles.range(SolverType::MPM).count;
        mpm.spawn_circle(particles, vec2(-0.02f, y + 0.02f), 0.16f, sp,
                         MPMMaterial::TOUGH, 300.0f, vec2(1.0f, 0.0f), 4.0f);
        register_scene_mpm_batch(creation,
                                 lane == 0 ? "Tip Load A" : (lane == 1 ? "Tip Load B" : "Tip Load C"),
                                 "Matched tip load for cantilever sag comparison.",
                                 particles, mpm_before, MPMMaterial::TOUGH,
                                 28000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));
    }

    mpm.params().youngs_modulus = old_E;
    mpm.params().fiber_strength = old_fiber;
}

static void load_porous_bake_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                   MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                   CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.78f, -1.24f), vec2(1.78f, -1.24f), 0.12f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Bake Hearth", "Shared baking hearth for side-by-side crumb tests.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.06f);
    mpm.params().heat_source_radius = 1.18f;
    mpm.params().heat_source_temp = 980.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;

    const f32 centers[4] = {-1.35f, -0.45f, 0.45f, 1.35f};

    mpm.params().youngs_modulus = 7200.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(centers[0], -0.78f), 0.23f, sp, MPMMaterial::BREAD, 300.0f);
    register_scene_mpm_batch(creation, "Bread Ref", "Gas-only bread reference for comparing simple puffing against the newer shell and porous variants.",
                             particles, mpm_before, MPMMaterial::BREAD,
                             7200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 8200.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(centers[1], -0.78f), 0.23f, sp, MPMMaterial::CRUST_DOUGH, 300.0f);
    register_scene_mpm_batch(creation, "Crust Dough", "Shell-setting dough reference: outside sets first while the core keeps trying to lift.",
                             particles, mpm_before, MPMMaterial::CRUST_DOUGH,
                             8200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 8400.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(centers[2], -0.78f), 0.23f, sp, MPMMaterial::CRUMB_LOAF, 300.0f);
    register_scene_mpm_batch(creation, "Crumb Loaf", "Porous loaf benchmark: persistent crumb holes and localized vent seams should show up here.",
                             particles, mpm_before, MPMMaterial::CRUMB_LOAF,
                             8400.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 7600.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(centers[3], -0.78f), 0.23f, sp, MPMMaterial::STEAM_BUN, 300.0f);
    register_scene_mpm_batch(creation, "Steam Bun", "Steamier shell/core dough reference: rounder lift, striping vents, and softer skin than the loafs.",
                             particles, mpm_before, MPMMaterial::STEAM_BUN,
                             7600.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_pottery_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                               MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                               CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.76f, -1.24f), vec2(1.76f, -1.24f), 0.12f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Kiln Shelf", "Shared kiln shelf for pottery side-by-side tests.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.04f);
    mpm.params().heat_source_radius = 1.16f;
    mpm.params().heat_source_temp = 1180.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 11800.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.54f, -0.92f), vec2(-0.86f, -0.58f), sp,
                    MPMMaterial::BISQUE, 300.0f);
    register_scene_mpm_batch(creation, "Bisque Tile", "Porous pottery body benchmark: burnout, shrink, and porous crack patterns should read here.",
                             particles, mpm_before, MPMMaterial::BISQUE,
                             11800.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 11000.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.62f, -0.92f), vec2(0.06f, -0.58f), sp,
                    MPMMaterial::GLAZE_CLAY, 300.0f);
    register_scene_mpm_batch(creation, "Glaze Tile", "Shell-first glaze-clay reference for vitrifying shell and core-lag mismatch.",
                             particles, mpm_before, MPMMaterial::GLAZE_CLAY,
                             11000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 13500.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.30f, -0.92f), vec2(0.98f, -0.58f), sp,
                    MPMMaterial::GLAZE_CLAY, 300.0f);
    register_scene_mpm_batch(creation, "Crazing Tile", "Stiffer glaze tile tuned to crack and craze more than it drips.",
                             particles, mpm_before, MPMMaterial::GLAZE_CLAY,
                             13500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 9200.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(1.44f, -0.76f), 0.20f, sp, MPMMaterial::GLAZE_DRIP, 300.0f);
    register_scene_mpm_batch(creation, "Glaze Bead", "Runny glaze benchmark that should droop and pool while keeping a ceramic core.",
                             particles, mpm_before, MPMMaterial::GLAZE_DRIP,
                             9200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_pastry_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                              MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                              CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.78f, -1.24f), vec2(1.78f, -1.24f), 0.12f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Pastry Hearth", "Shared hearth for browning, lamination, and tear-sheet tests.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.04f);
    mpm.params().heat_source_radius = 1.18f;
    mpm.params().heat_source_temp = 990.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 old_fiber = mpm.params().fiber_strength;

    mpm.params().youngs_modulus = 7800.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.64f, -0.92f), vec2(-0.92f, -0.74f), sp,
                    MPMMaterial::MAILLARD, 300.0f);
    register_scene_mpm_batch(creation, "Maillard Slice", "Expected: surface browns and dries, with only modest blistering instead of a big puff.",
                             particles, mpm_before, MPMMaterial::MAILLARD,
                             7800.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 7200.0f;
    mpm.params().fiber_strength = 2.8f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.64f, -0.92f), vec2(0.08f, -0.72f), sp,
                    MPMMaterial::LAMINATED_PASTRY, 300.0f, vec2(1.0f, 0.0f));
    register_scene_mpm_batch(creation, "Laminated Pastry", "Expected: layers puff apart, brown on the outside, and peel into flaky ribbons rather than one smooth loaf.",
                             particles, mpm_before, MPMMaterial::LAMINATED_PASTRY,
                             7200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 7600.0f;
    mpm.params().fiber_strength = 2.2f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.36f, -0.94f), vec2(1.04f, -0.76f), sp,
                    MPMMaterial::TEAR_SKIN, 300.0f, vec2(1.0f, 0.0f));
    register_scene_mpm_batch(creation, "Tear Skin", "Expected: dries into a sheet, then opens long vent tears instead of puffing like pastry layers.",
                             particles, mpm_before, MPMMaterial::TEAR_SKIN,
                             7600.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 7600.0f;
    mpm.params().fiber_strength = 0.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(1.48f, -0.78f), 0.22f, sp, MPMMaterial::STEAM_BUN, 300.0f);
    register_scene_mpm_batch(creation, "Steam Bun Ref", "Expected: rounder dome rise with striped vents, softer than the layered pastry sheet.",
                             particles, mpm_before, MPMMaterial::STEAM_BUN,
                             7600.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
    mpm.params().fiber_strength = old_fiber;
}

static void load_stoneware_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                 MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                 CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.78f, -1.24f), vec2(1.78f, -1.24f), 0.12f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Stoneware Shelf", "Hot kiln shelf for dense-firing comparisons.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.04f);
    mpm.params().heat_source_radius = 1.20f;
    mpm.params().heat_source_temp = 1220.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 11800.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.66f, -0.94f), vec2(-0.96f, -0.56f), sp,
                    MPMMaterial::BISQUE, 300.0f);
    register_scene_mpm_batch(creation, "Bisque Tile", "Expected: more open porous body, lighter color, and earlier porous cracking than dense stoneware.",
                             particles, mpm_before, MPMMaterial::BISQUE,
                             11800.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 14500.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.72f, -0.94f), vec2(-0.02f, -0.56f), sp,
                    MPMMaterial::STONEWARE, 300.0f);
    register_scene_mpm_batch(creation, "Stoneware Tile", "Expected: shrinks denser, warps a bit, and develops tighter craze patterns instead of open bisque pores.",
                             particles, mpm_before, MPMMaterial::STONEWARE,
                             14500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 13500.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.22f, -0.94f), vec2(0.92f, -0.56f), sp,
                    MPMMaterial::GLAZE_CLAY, 300.0f);
    register_scene_mpm_batch(creation, "Crazing Tile", "Expected: shell-first firing and stronger glaze mismatch cracks than the dense stoneware tile.",
                             particles, mpm_before, MPMMaterial::GLAZE_CLAY,
                             13500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 9200.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(1.44f, -0.76f), 0.20f, sp, MPMMaterial::GLAZE_DRIP, 300.0f);
    register_scene_mpm_batch(creation, "Glaze Drip", "Expected: the shell glaze softens and runs the most, while the core still tries to stay ceramic.",
                             particles, mpm_before, MPMMaterial::GLAZE_DRIP,
                             9200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_adv_bake_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.82f, -1.24f), vec2(1.82f, -1.24f), 0.13f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Advanced Bake Stone", "Hot stone for the advanced bake comparison: shell set, crumb holes, vent channels, and lamination.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.06f);
    mpm.params().heat_source_radius = 1.22f;
    mpm.params().heat_source_temp = 1010.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 old_fiber = mpm.params().fiber_strength;
    const f32 centers[4] = {-1.38f, -0.46f, 0.46f, 1.38f};

    mpm.params().youngs_modulus = 8200.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(centers[0], -0.78f), 0.23f, sp, MPMMaterial::CRUST_DOUGH, 300.0f);
    register_scene_mpm_batch(creation, "Crust Dough Ref", "Expected: shell sets first, then the loaf cracks and vents, but it should stay more shell-like than porous inside.",
                             particles, mpm_before, MPMMaterial::CRUST_DOUGH,
                             8200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 8400.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(centers[1], -0.78f), 0.23f, sp, MPMMaterial::CRUMB_LOAF, 300.0f);
    register_scene_mpm_batch(creation, "Crumb Loaf", "Expected: more persistent round-ish crumb holes and localized vent seams than the crust reference.",
                             particles, mpm_before, MPMMaterial::CRUMB_LOAF,
                             8400.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 9800.0f;
    mpm.params().fiber_strength = 2.4f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(centers[2], -0.78f), 0.23f, sp, MPMMaterial::VENT_CRUMB, 300.0f, vec2(0.0f, 1.0f));
    register_scene_mpm_batch(creation, "Vent Crumb", "Expected: a drier shell with wetter core should open bigger vent channels and tunnel-like crumb instead of only round pores.",
                             particles, mpm_before, MPMMaterial::VENT_CRUMB,
                             9800.0f, mpm.params().poisson_ratio, 300.0f, vec2(0, 1));

    mpm.params().youngs_modulus = 7600.0f;
    mpm.params().fiber_strength = 2.8f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(centers[3] - 0.36f, -0.92f), vec2(centers[3] + 0.36f, -0.72f), sp,
                    MPMMaterial::LAMINATED_PASTRY, 300.0f, vec2(1.0f, 0.0f));
    register_scene_mpm_batch(creation, "Laminated Pastry", "Expected: flatter sheet that puffs apart into layers and flakes, rather than forming crumb tunnels.",
                             particles, mpm_before, MPMMaterial::LAMINATED_PASTRY,
                             7600.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
    mpm.params().fiber_strength = old_fiber;
}

static void load_adv_kiln_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.82f, -1.24f), vec2(1.82f, -1.24f), 0.13f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Advanced Kiln Shelf", "Dense kiln shelf for comparing porous bisque, denser stoneware, vitrified clay, and blister glaze.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.04f);
    mpm.params().heat_source_radius = 1.20f;
    mpm.params().heat_source_temp = 1260.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 centers[4] = {-1.42f, -0.46f, 0.46f, 1.42f};

    mpm.params().youngs_modulus = 11800.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(centers[0] - 0.34f, -0.92f), vec2(centers[0] + 0.34f, -0.56f), sp,
                    MPMMaterial::BISQUE, 300.0f);
    register_scene_mpm_batch(creation, "Bisque Tile", "Expected: more open porous body with lighter firing and earlier porous cracking than the denser kiln bodies.",
                             particles, mpm_before, MPMMaterial::BISQUE,
                             11800.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 14500.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(centers[1] - 0.34f, -0.92f), vec2(centers[1] + 0.34f, -0.56f), sp,
                    MPMMaterial::STONEWARE, 300.0f);
    register_scene_mpm_batch(creation, "Stoneware Tile", "Expected: denser shrink and tighter craze than bisque, but still less extreme than the vitrified clay sample.",
                             particles, mpm_before, MPMMaterial::STONEWARE,
                             14500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 16800.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(centers[2] - 0.34f, -0.92f), vec2(centers[2] + 0.34f, -0.56f), sp,
                    MPMMaterial::VITREOUS_CLAY, 300.0f);
    register_scene_mpm_batch(creation, "Vitreous Tile", "Expected: the strongest shrink and densification, with a harder-fired body and tighter shell-driven cracking than stoneware.",
                             particles, mpm_before, MPMMaterial::VITREOUS_CLAY,
                             16800.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 11200.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(centers[3], -0.76f), 0.20f, sp, MPMMaterial::BLISTER_GLAZE, 300.0f);
    register_scene_mpm_batch(creation, "Blister Glaze", "Expected: glossy shell with trapped blister pockets that vent pits or flakes while the ceramic core still resists a full drip.",
                             particles, mpm_before, MPMMaterial::BLISTER_GLAZE,
                             11200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_open_crumb_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                  MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                  CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.82f, -1.24f), vec2(1.82f, -1.24f), 0.13f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Open Crumb Stone", "Hot bake stone for comparing classic crumb, vent crumb, open crumb, and bun-like retention.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.06f);
    mpm.params().heat_source_radius = 1.24f;
    mpm.params().heat_source_temp = 1035.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 old_fiber = mpm.params().fiber_strength;
    const f32 centers[4] = {-1.38f, -0.46f, 0.46f, 1.38f};

    mpm.params().youngs_modulus = 8400.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(centers[0], -0.78f), 0.23f, sp, MPMMaterial::CRUMB_LOAF, 300.0f);
    register_scene_mpm_batch(creation, "Crumb Loaf Ref", "Expected: smaller persistent crumb holes and localized vent seams, but still denser than the more advanced crumb models.",
                             particles, mpm_before, MPMMaterial::CRUMB_LOAF,
                             8400.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 9800.0f;
    mpm.params().fiber_strength = 2.4f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(centers[1], -0.78f), 0.23f, sp, MPMMaterial::VENT_CRUMB, 300.0f, vec2(0.0f, 1.0f));
    register_scene_mpm_batch(creation, "Vent Crumb Ref", "Expected: stronger chimney-like vents and tunnel crumb, but still less open than the new coalescing crumb sample.",
                             particles, mpm_before, MPMMaterial::VENT_CRUMB,
                             9800.0f, mpm.params().poisson_ratio, 300.0f, vec2(0, 1));

    mpm.params().youngs_modulus = 11200.0f;
    mpm.params().fiber_strength = 2.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(centers[2], -0.78f), 0.24f, sp, MPMMaterial::OPEN_CRUMB, 300.0f, vec2(0.0f, 1.0f));
    register_scene_mpm_batch(creation, "Open Crumb", "Expected: larger retained pore chambers and stronger post-rise scaffold than Crumb Loaf or Vent Crumb.",
                             particles, mpm_before, MPMMaterial::OPEN_CRUMB,
                             11200.0f, mpm.params().poisson_ratio, 300.0f, vec2(0, 1));

    mpm.params().youngs_modulus = 7600.0f;
    mpm.params().fiber_strength = 0.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(centers[3], -0.78f), 0.23f, sp, MPMMaterial::STEAM_BUN, 300.0f);
    register_scene_mpm_batch(creation, "Steam Bun Ref", "Expected: rounder rise and softer striped venting, but less large open crumb than the new coalescing loaf.",
                             particles, mpm_before, MPMMaterial::STEAM_BUN,
                             7600.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
    mpm.params().fiber_strength = old_fiber;
}

static void load_sinter_lock_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                   MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                   CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.82f, -1.24f), vec2(1.82f, -1.24f), 0.13f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Sinter Lock Shelf", "Kiln shelf for comparing open bisque, stoneware, vitrified clay, and stronger shrink-lock firing.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.04f);
    mpm.params().heat_source_radius = 1.20f;
    mpm.params().heat_source_temp = 1290.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 centers[4] = {-1.42f, -0.46f, 0.46f, 1.42f};

    mpm.params().youngs_modulus = 11800.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(centers[0] - 0.34f, -0.92f), vec2(centers[0] + 0.34f, -0.56f), sp,
                    MPMMaterial::BISQUE, 300.0f);
    register_scene_mpm_batch(creation, "Bisque Ref", "Expected: more porous body and earlier hot collapse than the denser fired samples.",
                             particles, mpm_before, MPMMaterial::BISQUE,
                             11800.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 14500.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(centers[1] - 0.34f, -0.92f), vec2(centers[1] + 0.34f, -0.56f), sp,
                    MPMMaterial::STONEWARE, 300.0f);
    register_scene_mpm_batch(creation, "Stoneware Ref", "Expected: denser shrink and tighter craze than bisque, but still less locked than the new sinter-lock body.",
                             particles, mpm_before, MPMMaterial::STONEWARE,
                             14500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 16800.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(centers[2] - 0.34f, -0.92f), vec2(centers[2] + 0.34f, -0.56f), sp,
                    MPMMaterial::VITREOUS_CLAY, 300.0f);
    register_scene_mpm_batch(creation, "Vitreous Ref", "Expected: strong densification and tighter shell-driven cracking, but still less locked than the new fired-lock sample.",
                             particles, mpm_before, MPMMaterial::VITREOUS_CLAY,
                             16800.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 18600.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(centers[3] - 0.34f, -0.92f), vec2(centers[3] + 0.34f, -0.56f), sp,
                    MPMMaterial::SINTER_LOCK, 300.0f);
    register_scene_mpm_batch(creation, "Sinter-Lock Clay", "Expected: strongest shrink-set shape retention, denser fired body, and less pure hot collapse than the other kiln samples.",
                             particles, mpm_before, MPMMaterial::SINTER_LOCK,
                             18600.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_moisture_binder_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                       MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                       CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.82f, -1.24f), vec2(1.82f, -1.24f), 0.13f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Binder Stone", "Bake stone for comparing baseline crumb, binder-set crumb walls, channeled vent crumb, and open crumb retention.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.05f);
    mpm.params().heat_source_radius = 1.24f;
    mpm.params().heat_source_temp = 1040.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 old_fiber = mpm.params().fiber_strength;
    const f32 centers[4] = {-1.38f, -0.46f, 0.46f, 1.38f};

    mpm.params().youngs_modulus = 8400.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(centers[0], -0.78f), 0.23f, sp, MPMMaterial::CRUMB_LOAF, 300.0f);
    register_scene_mpm_batch(creation, "Crumb Loaf Ref", "Expected: modest pore retention and vent seams, but less strong wall support than the binder-set crumbs.",
                             particles, mpm_before, MPMMaterial::CRUMB_LOAF,
                             8400.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 10400.0f;
    mpm.params().fiber_strength = 2.2f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(centers[1], -0.78f), 0.24f, sp, MPMMaterial::BINDER_CRUMB, 300.0f, vec2(0.0f, 1.0f));
    register_scene_mpm_batch(creation, "Binder Crumb", "Expected: stronger crumb scaffold and thicker baked walls around the pores than the baseline loaf.",
                             particles, mpm_before, MPMMaterial::BINDER_CRUMB,
                             10400.0f, mpm.params().poisson_ratio, 300.0f, vec2(0, 1));

    mpm.params().youngs_modulus = 11200.0f;
    mpm.params().fiber_strength = 2.8f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(centers[2], -0.78f), 0.24f, sp, MPMMaterial::CHANNEL_CRUMB, 300.0f, vec2(0.0f, 1.0f));
    register_scene_mpm_batch(creation, "Channel Crumb", "Expected: stronger vent tunnels and channels that still leave a connected loaf body instead of a simple split.",
                             particles, mpm_before, MPMMaterial::CHANNEL_CRUMB,
                             11200.0f, mpm.params().poisson_ratio, 300.0f, vec2(0, 1));

    mpm.params().youngs_modulus = 11200.0f;
    mpm.params().fiber_strength = 2.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(centers[3], -0.78f), 0.24f, sp, MPMMaterial::OPEN_CRUMB, 300.0f, vec2(0.0f, 1.0f));
    register_scene_mpm_batch(creation, "Open Crumb Ref", "Expected: larger pore chambers than the baseline loaf, but less wall-like binder retention than the new binder and channel variants.",
                             particles, mpm_before, MPMMaterial::OPEN_CRUMB,
                             11200.0f, mpm.params().poisson_ratio, 300.0f, vec2(0, 1));

    mpm.params().youngs_modulus = old_E;
    mpm.params().fiber_strength = old_fiber;
}

static void load_burnout_pottery_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                       MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                       CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.82f, -1.24f), vec2(1.82f, -1.24f), 0.13f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Burnout Shelf", "Kiln shelf for comparing porous bisque, burnout-supported pottery, vitrification, and fired-lock retention.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.04f);
    mpm.params().heat_source_radius = 1.22f;
    mpm.params().heat_source_temp = 1280.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 centers[4] = {-1.42f, -0.46f, 0.46f, 1.42f};

    mpm.params().youngs_modulus = 11800.0f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(centers[0] - 0.34f, -0.92f), vec2(centers[0] + 0.34f, -0.56f), sp,
                    MPMMaterial::BISQUE, 300.0f);
    register_scene_mpm_batch(creation, "Bisque Ref", "Expected: porous body and earlier hot slump than the more supported burnout and vitrified bodies.",
                             particles, mpm_before, MPMMaterial::BISQUE,
                             11800.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 15200.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(centers[1] - 0.34f, -0.92f), vec2(centers[1] + 0.34f, -0.56f), sp,
                    MPMMaterial::BURNOUT_CLAY, 300.0f);
    register_scene_mpm_batch(creation, "Burnout Clay", "Expected: moisture loss and pore-former burnout, but with a firmer firing binder that keeps the body connected longer than bisque.",
                             particles, mpm_before, MPMMaterial::BURNOUT_CLAY,
                             15200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 16800.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(centers[2] - 0.34f, -0.92f), vec2(centers[2] + 0.34f, -0.56f), sp,
                    MPMMaterial::VITREOUS_CLAY, 300.0f);
    register_scene_mpm_batch(creation, "Vitreous Ref", "Expected: denser shrink and harder-fired body than burnout or bisque, but less shape lock than sinter-lock clay.",
                             particles, mpm_before, MPMMaterial::VITREOUS_CLAY,
                             16800.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 18600.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(centers[3] - 0.34f, -0.92f), vec2(centers[3] + 0.34f, -0.56f), sp,
                    MPMMaterial::SINTER_LOCK, 300.0f);
    register_scene_mpm_batch(creation, "Sinter-Lock Ref", "Expected: strongest fired shape lock and densest body, with less pure hot collapse than the other kiln bodies.",
                             particles, mpm_before, MPMMaterial::SINTER_LOCK,
                             18600.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

static void load_vented_skin_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                   MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                   CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-1.82f, -1.24f), vec2(1.82f, -1.24f), 0.13f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Vent Plate", "Hot strip bench for comparing full tearing, vent-first sheets, delamination, and orthotropic splitting.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.05f);
    mpm.params().heat_source_radius = 1.24f;
    mpm.params().heat_source_temp = 990.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 old_fiber = mpm.params().fiber_strength;
    const f32 centers[4] = {-1.38f, -0.46f, 0.46f, 1.38f};

    mpm.params().youngs_modulus = 7600.0f;
    mpm.params().fiber_strength = 2.2f;
    u32 mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(centers[0] - 0.34f, -0.84f), vec2(centers[0] + 0.34f, -0.70f), sp,
                    MPMMaterial::TEAR_SKIN, 300.0f, vec2(1.0f, 0.0f));
    register_scene_mpm_batch(creation, "Tear Skin Ref", "Expected: weak bands form and the strip tears apart more directly once venting localizes.",
                             particles, mpm_before, MPMMaterial::TEAR_SKIN,
                             7600.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 9200.0f;
    mpm.params().fiber_strength = 2.8f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(centers[1] - 0.34f, -0.84f), vec2(centers[1] + 0.34f, -0.70f), sp,
                    MPMMaterial::VENTED_SKIN, 300.0f, vec2(1.0f, 0.0f));
    register_scene_mpm_batch(creation, "Vented Skin", "Expected: vents and slots open first, but the sheet should keep more membrane body than Tear Skin.",
                             particles, mpm_before, MPMMaterial::VENTED_SKIN,
                             9200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 7200.0f;
    mpm.params().fiber_strength = 2.8f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(centers[2] - 0.34f, -0.84f), vec2(centers[2] + 0.34f, -0.70f), sp,
                    MPMMaterial::LAMINATED_PASTRY, 300.0f, vec2(1.0f, 0.0f));
    register_scene_mpm_batch(creation, "Laminated Pastry Ref", "Expected: layer lift and delamination rather than simple vent slits or direct tearing.",
                             particles, mpm_before, MPMMaterial::LAMINATED_PASTRY,
                             7200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 6800.0f;
    mpm.params().fiber_strength = 6.0f;
    mpm_before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(centers[3] - 0.34f, -0.84f), vec2(centers[3] + 0.34f, -0.70f), sp,
                    MPMMaterial::ORTHO_TEAR, 300.0f, vec2(1.0f, 0.0f));
    register_scene_mpm_batch(creation, "Ortho Tear Ref", "Expected: more directional split behavior than the other sheets, especially once the strip warms through.",
                             particles, mpm_before, MPMMaterial::ORTHO_TEAR,
                             6800.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
    mpm.params().fiber_strength = old_fiber;
}

static void load_aniso_strong_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                    MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                    CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);

    const f32 lane_centers[3] = {-1.26f, 0.28f, 1.82f};
    const vec2 fibers[3] = {
        vec2(1.0f, 0.0f),
        vec2(0.0f, 1.0f),
        glm::normalize(vec2(1.0f, 1.0f))
    };
    const char* bend_labels[3] = {
        "Strong Along-Grain Beam",
        "Strong Cross-Grain Beam",
        "Strong Bias-Grain Beam"
    };
    const char* bend_summaries[3] = {
        "High-stiffness orthotropic beam. This should keep the flattest cantilever profile and the smallest tip sag.",
        "High-stiffness orthotropic beam with fibers across the span. This should sag the most and curl more near the tip.",
        "High-stiffness orthotropic beam with diagonal fibers. This should land between the other two and show a skewed bend shape."
    };
    const char* tear_labels[3] = {
        "Strong Along-Grain Strap",
        "Strong Cross-Grain Strap",
        "Strong Bias-Grain Strap"
    };
    const char* tear_summaries[3] = {
        "Cured orthotropic strap with a narrow neck. This should hold longest and fray into longer grain-aligned bands.",
        "Cured orthotropic strap with fibers across the pull. This should split sooner and more directly through the neck.",
        "Cured orthotropic strap with diagonal fibers. This should tear on a skewed path and fail asymmetrically."
    };
    const char lane_suffix[3] = {'A', 'B', 'C'};

    for (i32 lane = 0; lane < 3; ++lane) {
        const f32 cx = lane_centers[lane];
        char label[64];

        sdf.add_box(vec2(cx - 0.90f, 1.06f), vec2(0.16f, 0.22f),
                    SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                    (snprintf(label, sizeof(label), "Strong Clamp %c", lane_suffix[lane]), label),
                    "Rigid clamp for the stronger orthotropic bend benchmark.");
        sdf.add_segment(vec2(cx - 1.04f, 0.84f), vec2(cx - 1.04f, 1.38f), 0.035f,
                        SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                        (snprintf(label, sizeof(label), "Strong Backstop %c", lane_suffix[lane]), label),
                        "Backstop that keeps the beam root from slipping out of the clamp.");

        sdf.add_box(vec2(cx - 0.42f, -0.78f), vec2(0.18f, 0.24f),
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    (snprintf(label, sizeof(label), "Strong Left Saddle %c", lane_suffix[lane]), label),
                    "Support saddle for a stronger orthotropic tear strap.");
        sdf.add_box(vec2(cx + 0.42f, -0.78f), vec2(0.18f, 0.24f),
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    (snprintf(label, sizeof(label), "Strong Right Saddle %c", lane_suffix[lane]), label),
                    "Support saddle for a stronger orthotropic tear strap.");
        sdf.add_segment(vec2(cx - 0.56f, -1.24f), vec2(cx + 0.56f, -1.24f), 0.08f,
                        SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                        (snprintf(label, sizeof(label), "Strong Hot Plate %c", lane_suffix[lane]), label),
                        "Mild cure plate for the stronger tear comparison. It should harden the neck without liquefying the strip.");
    }
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.08f);
    mpm.params().heat_source_radius = 2.02f;
    mpm.params().heat_source_temp = 760.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 old_fiber = mpm.params().fiber_strength;

    for (i32 lane = 0; lane < 3; ++lane) {
        const f32 cx = lane_centers[lane];
        const vec2 fiber_dir = fibers[lane];

        mpm.params().youngs_modulus = 17200.0f;
        mpm.params().fiber_strength = 10.4f;
        u32 mpm_before = particles.range(SolverType::MPM).count;
        mpm.spawn_block(particles, vec2(cx - 0.70f, 1.18f), vec2(cx - 0.08f, 1.42f), sp,
                        MPMMaterial::ORTHO_BEND, 300.0f, fiber_dir);
        register_scene_mpm_batch(creation, bend_labels[lane], bend_summaries[lane],
                                 particles, mpm_before, MPMMaterial::ORTHO_BEND,
                                 17200.0f, mpm.params().poisson_ratio, 300.0f, fiber_dir);
        if (creation && !creation->batches.empty()) {
            BatchRecord& batch = creation->batches.back();
            batch.fiber_strength = 10.4f;
            batch.properties += " | strong anisotropic bend";
            batch.recommended_size = 0.66f;
            batch.recommended_note = "This stronger beam bench is for gross shape reads, not failure. Along-grain should stay flattest, cross-grain should sag most, and bias should sit between them.";
        }

        mpm.params().youngs_modulus = 30000.0f;
        mpm.params().fiber_strength = 0.0f;
        mpm_before = particles.range(SolverType::MPM).count;
        mpm.spawn_circle(particles, vec2(cx + 0.00f, 1.33f), 0.085f, sp,
                         MPMMaterial::TOUGH, 300.0f, vec2(1.0f, 0.0f), 0.72f);
        register_scene_mpm_batch(creation,
                                 lane == 0 ? "Strong Tip Load A" : (lane == 1 ? "Strong Tip Load B" : "Strong Tip Load C"),
                                 "Light matched tip load used to reveal directional stiffness without immediately breaking the beam.",
                                 particles, mpm_before, MPMMaterial::TOUGH,
                                 30000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

        mpm.params().youngs_modulus = 11200.0f;
        mpm.params().fiber_strength = 8.0f;
        mpm_before = particles.range(SolverType::MPM).count;
        mpm.spawn_block(particles, vec2(cx - 0.52f, -0.46f), vec2(cx - 0.12f, -0.30f), sp,
                        MPMMaterial::ORTHO_TEAR, 300.0f, fiber_dir);
        mpm.spawn_block(particles, vec2(cx - 0.12f, -0.42f), vec2(cx + 0.12f, -0.34f), sp,
                        MPMMaterial::ORTHO_TEAR, 300.0f, fiber_dir);
        mpm.spawn_block(particles, vec2(cx + 0.12f, -0.46f), vec2(cx + 0.52f, -0.30f), sp,
                        MPMMaterial::ORTHO_TEAR, 300.0f, fiber_dir);
        register_scene_mpm_batch(creation, tear_labels[lane], tear_summaries[lane],
                                 particles, mpm_before, MPMMaterial::ORTHO_TEAR,
                                 11200.0f, mpm.params().poisson_ratio, 300.0f, fiber_dir);
        if (creation && !creation->batches.empty()) {
            BatchRecord& batch = creation->batches.back();
            batch.fiber_strength = 8.0f;
            batch.properties += " | strong anisotropic tear";
            batch.recommended_size = 0.58f;
            batch.recommended_note = "Let the lower straps warm on the plate first. The narrow neck should make the tear path easy to read: along-grain should fray longer, cross-grain should split sooner, and bias should break off-axis.";
        }

        mpm.params().youngs_modulus = 30000.0f;
        mpm.params().fiber_strength = 0.0f;
        mpm_before = particles.range(SolverType::MPM).count;
        mpm.spawn_circle(particles, vec2(cx, -0.02f), 0.125f, sp,
                         MPMMaterial::TOUGH, 300.0f, vec2(1.0f, 0.0f), 2.0f);
        register_scene_mpm_batch(creation,
                                 lane == 0 ? "Strong Center Load A" : (lane == 1 ? "Strong Center Load B" : "Strong Center Load C"),
                                 "Compact matched center load for the stronger tear straps.",
                                 particles, mpm_before, MPMMaterial::TOUGH,
                                 30000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));
    }

    mpm.params().youngs_modulus = old_E;
    mpm.params().fiber_strength = old_fiber;
}

static void load_oobleck_impactor_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                        MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                        CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(-1.55f, -0.55f), vec2(0.22f, 0.05f),
                SDFField::MaterialPresetID::GLOBAL_DEFAULT,
                "Calm Shelf", "Raised shelf for the slow-slump oobleck puck.");
    sdf.add_box(vec2(0.0f, -0.45f), vec2(0.28f, 0.05f),
                SDFField::MaterialPresetID::GLOBAL_DEFAULT,
                "Impact Pad", "Central pad for the drop-hammer oobleck test.");
    sdf.rebuild();

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 6200.0f;
    u32 before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.92f, -0.40f), vec2(-1.18f, -0.06f), sp,
                    MPMMaterial::OOBLECK, 300.0f);
    register_scene_mpm_batch(creation, "Oobleck Calm Puck",
                             "Baseline oobleck puck. This one should slump and spread a bit when left mostly alone.",
                             particles, before, MPMMaterial::OOBLECK,
                             8600.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.42f, -0.28f), vec2(0.42f, 0.10f), sp,
                    MPMMaterial::OOBLECK, 300.0f);
    register_scene_mpm_batch(creation, "Oobleck Strike Pad",
                             "Main oobleck target. The fast drop-hammer should make this one jam harder than the calm puck.",
                             particles, before, MPMMaterial::OOBLECK,
                             8600.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.12f, -1.22f), vec2(1.94f, -0.84f), sp,
                    MPMMaterial::OOBLECK, 300.0f);
    register_scene_mpm_batch(creation, "Oobleck Side-Hit Puck",
                             "Side-hit target. The horizontal slug should make a sharper, more visibly jammed shear zone.",
                             particles, before, MPMMaterial::OOBLECK,
                             8600.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 36000.0f;
    before = particles.range(SolverType::MPM).count;
    u32 drop_offset = particles.range(SolverType::MPM).offset + before;
    mpm.spawn_circle(particles, vec2(0.0f, 1.18f), 0.17f, sp,
                     MPMMaterial::THERMO_METAL, 300.0f, vec2(1, 0), 7.5f);
    u32 drop_count = particles.range(SolverType::MPM).count - before;
    set_mpm_batch_velocity(particles, drop_offset, drop_count, vec2(0.0f, -10.8f));
    register_scene_mpm_batch(creation, "Drop Hammer",
                             "Dense steel slug for the vertical jamming test.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             36000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 7.5f);

    before = particles.range(SolverType::MPM).count;
    u32 side_offset = particles.range(SolverType::MPM).offset + before;
    mpm.spawn_circle(particles, vec2(2.28f, -0.34f), 0.15f, sp,
                     MPMMaterial::THERMO_METAL, 300.0f, vec2(1, 0), 7.0f);
    u32 side_count = particles.range(SolverType::MPM).count - before;
    set_mpm_batch_velocity(particles, side_offset, side_count, vec2(-12.6f, 0.0f));
    register_scene_mpm_batch(creation, "Side Hammer",
                             "Fast lateral steel slug for the shear-thickening read.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             36000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 7.0f);

    mpm.params().youngs_modulus = old_E;
}

static void load_impact_memory_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                     MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                     CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(-1.45f, -0.58f), vec2(0.22f, 0.05f),
                SDFField::MaterialPresetID::GLOBAL_DEFAULT,
                "Fresh Shelf", "Shelf for the untouched gel baseline.");
    sdf.add_box(vec2(1.30f, -0.58f), vec2(0.26f, 0.05f),
                SDFField::MaterialPresetID::GLOBAL_DEFAULT,
                "Repeat Shelf", "Shelf for repeated-hit memory gel.");
    sdf.rebuild();

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 9200.0f;
    u32 before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.82f, -0.42f), vec2(-1.08f, -0.02f), sp,
                    MPMMaterial::IMPACT_GEL, 300.0f);
    register_scene_mpm_batch(creation, "Fresh Gel Pad",
                             "Untouched impact-memory gel baseline. Compare this with the struck pads as they harden and start holding dents.",
                             particles, before, MPMMaterial::IMPACT_GEL,
                             9200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.34f, -1.22f), vec2(0.34f, -0.30f), sp,
                    MPMMaterial::IMPACT_GEL, 300.0f);
    register_scene_mpm_batch(creation, "Single-Hit Column",
                             "Tall gel column for a clean one-hit memory test.",
                             particles, before, MPMMaterial::IMPACT_GEL,
                             9200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.92f, -0.42f), vec2(1.68f, -0.02f), sp,
                    MPMMaterial::IMPACT_GEL, 300.0f);
    register_scene_mpm_batch(creation, "Repeated-Hit Pad",
                             "Two hammers land here one after another so the gel has a chance to harden and remember the earlier deformation.",
                             particles, before, MPMMaterial::IMPACT_GEL,
                             9200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = 34000.0f;
    before = particles.range(SolverType::MPM).count;
    u32 hammer_a_offset = particles.range(SolverType::MPM).offset + before;
    mpm.spawn_circle(particles, vec2(0.0f, 1.30f), 0.16f, sp,
                     MPMMaterial::THERMO_METAL, 300.0f, vec2(1, 0), 6.8f);
    u32 hammer_a_count = particles.range(SolverType::MPM).count - before;
    set_mpm_batch_velocity(particles, hammer_a_offset, hammer_a_count, vec2(0.0f, -10.8f));
    register_scene_mpm_batch(creation, "Single-Hit Hammer",
                             "First dense striker for the center column.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             34000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 6.8f);

    before = particles.range(SolverType::MPM).count;
    u32 hammer_b_offset = particles.range(SolverType::MPM).offset + before;
    mpm.spawn_circle(particles, vec2(1.30f, 1.50f), 0.15f, sp,
                     MPMMaterial::THERMO_METAL, 300.0f, vec2(1, 0), 6.2f);
    u32 hammer_b_count = particles.range(SolverType::MPM).count - before;
    set_mpm_batch_velocity(particles, hammer_b_offset, hammer_b_count, vec2(0.0f, -9.6f));
    register_scene_mpm_batch(creation, "Repeat Hammer A",
                             "First of the repeated strikers for the right pad.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             34000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 6.2f);

    before = particles.range(SolverType::MPM).count;
    u32 hammer_c_offset = particles.range(SolverType::MPM).offset + before;
    mpm.spawn_circle(particles, vec2(1.30f, 0.78f), 0.13f, sp,
                     MPMMaterial::THERMO_METAL, 300.0f, vec2(1, 0), 5.8f);
    u32 hammer_c_count = particles.range(SolverType::MPM).count - before;
    set_mpm_batch_velocity(particles, hammer_c_offset, hammer_c_count, vec2(0.0f, -9.0f));
    register_scene_mpm_batch(creation, "Repeat Hammer B",
                             "Second delayed striker for the right pad.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             34000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 5.8f);

    mpm.params().youngs_modulus = old_E;
}

static void load_blast_armor_lane(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                  MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                  CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(-1.70f, -1.18f), vec2(0.30f, 0.06f),
                SDFField::MaterialPresetID::BRONZE_BALANCED,
                "Metal Plinth", "Stable bronze plinth for the thermo-metal plate.");
    sdf.add_box(vec2(-0.14f, -1.18f), vec2(0.30f, 0.06f),
                SDFField::MaterialPresetID::BRONZE_BALANCED,
                "Stoneware Plinth", "Stable bronze plinth for the stoneware tile.");
    sdf.add_box(vec2(1.26f, -1.18f), vec2(0.38f, 0.06f),
                SDFField::MaterialPresetID::BRONZE_BALANCED,
                "Layered Armor Plinth", "Stable bronze plinth for the layered target.");
    sdf.add_box(vec2(2.08f, -0.10f), vec2(0.12f, 1.18f),
                SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Catch Wall", "Heat-sink backstop for fragments and hot debris.");
    sdf.rebuild();

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;
    u32 before = 0;

    mpm.params().youngs_modulus = 52000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.82f, -1.12f), vec2(-1.58f, 0.18f), sp,
                    MPMMaterial::THERMO_METAL, 300.0f, vec2(1, 0), 7.6f);
    register_scene_mpm_batch(creation, "Thermo-Metal Witness Plate",
                             "Thin upright metal plate for shaped-charge and blast bending tests. It should dent, heat, and throw fragments much later than the ceramic targets.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             52000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 7.6f);

    mpm.params().youngs_modulus = 42000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.28f, -1.12f), vec2(0.00f, 0.14f), sp,
                    MPMMaterial::STONEWARE, 300.0f, vec2(1, 0), 3.2f);
    register_scene_mpm_batch(creation, "Stoneware Armor Tile",
                             "Dense fired tile. This is a good demolition / HESH target because it should stay rigid for a while, then crack and shed chunky fragments.",
                             particles, before, MPMMaterial::STONEWARE,
                             42000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 3.2f);

    mpm.params().youngs_modulus = 36000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.94f, -1.12f), vec2(1.10f, 0.10f), sp,
                    MPMMaterial::CERAMIC, 300.0f, vec2(1, 0), 2.8f);
    register_scene_mpm_batch(creation, "Ceramic Strike Face",
                             "Thin brittle front face. Compare this with the tougher backing behind it to read front-face breakup and back-face survival.",
                             particles, before, MPMMaterial::CERAMIC,
                             36000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 2.8f);

    mpm.params().youngs_modulus = 46000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.10f, -1.12f), vec2(1.42f, 0.12f), sp,
                    MPMMaterial::TOUGH, 300.0f, vec2(1, 0), 4.8f);
    register_scene_mpm_batch(creation, "Tough Backer",
                             "Higher-retention backing slab. This one should stay together longer and transmit more impulse than the ceramic strike face in front of it.",
                             particles, before, MPMMaterial::TOUGH,
                             46000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 4.8f);

    mpm.params().youngs_modulus = 30000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.58f, -1.08f), vec2(1.82f, -0.10f), sp,
                    MPMMaterial::BRITTLE, 300.0f, vec2(1, 0), 2.4f);
    register_scene_mpm_batch(creation, "Brittle Witness Sheet",
                             "Thin fragile witness sheet behind the layered target. Use this to read whether the armor stack still throws a dangerous fragment cloud.",
                             particles, before, MPMMaterial::BRITTLE,
                             30000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 2.4f);

    mpm.params().youngs_modulus = old_E;
}

static void load_breach_chamber(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_segment(vec2(-0.30f, -1.50f), vec2(-0.30f, -0.64f), 0.10f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Door Jamb Lower", "Lower chamber wall segment bracing the blast door.");
    sdf.add_segment(vec2(-0.30f, 0.72f), vec2(-0.30f, 2.15f), 0.10f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Door Jamb Upper", "Upper chamber wall segment bracing the blast door.");
    sdf.add_segment(vec2(-0.30f, 2.15f), vec2(2.05f, 2.15f), 0.12f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Chamber Roof", "Closed roof to make blast reflections easy to see.");
    sdf.add_segment(vec2(2.05f, -1.50f), vec2(2.05f, 2.15f), 0.12f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Rear Chamber Wall", "Heavy heat-sink wall at the back of the chamber.");
    sdf.add_box(vec2(1.16f, -0.18f), vec2(0.58f, 0.06f),
                SDFField::MaterialPresetID::BRONZE_BALANCED,
                "Interior Shelf", "Shelf for the chamber witness blocks.");
    sdf.rebuild();

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;
    u32 before = 0;

    mpm.params().youngs_modulus = 62000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.52f, -0.60f), vec2(-0.20f, 0.68f), sp,
                    MPMMaterial::THERMO_METAL, 300.0f, vec2(1, 0), 7.4f);
    register_scene_mpm_batch(creation, "Blast Door",
                             "Freestanding metal blast door in the chamber opening. This is the main breaching target for the room test.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             62000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 7.4f);

    mpm.params().youngs_modulus = 32000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(0.54f, -1.02f), 0.18f, sp,
                     MPMMaterial::CERAMIC, 300.0f, vec2(1, 0), 2.7f);
    register_scene_mpm_batch(creation, "Ceramic Witness Pot A",
                             "Front witness pot inside the room. Good for checking whether the initial shock front and fragments make it past the door.",
                             particles, before, MPMMaterial::CERAMIC,
                             32000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 2.7f);

    before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(0.98f, -1.02f), 0.18f, sp,
                     MPMMaterial::CERAMIC, 300.0f, vec2(1, 0), 2.7f);
    register_scene_mpm_batch(creation, "Ceramic Witness Pot B",
                             "Second witness pot deeper in the room for reflected-wave comparison.",
                             particles, before, MPMMaterial::CERAMIC,
                             32000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 2.7f);

    mpm.params().youngs_modulus = 40000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.82f, -0.10f), vec2(1.46f, 0.30f), sp,
                    MPMMaterial::STONEWARE, 300.0f, vec2(1, 0), 3.1f);
    register_scene_mpm_batch(creation, "Stoneware Chamber Crate",
                             "Heavy fired crate on the shelf. This makes reflected blast, sustained heat, and secondary impacts easier to read than a single floor target.",
                             particles, before, MPMMaterial::STONEWARE,
                             40000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 3.1f);

    mpm.params().youngs_modulus = 50000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.62f, -1.12f), vec2(1.90f, 0.20f), sp,
                    MPMMaterial::TOUGH, 300.0f, vec2(1, 0), 5.0f);
    register_scene_mpm_batch(creation, "Rear Bulkhead Witness",
                             "Tough bulkhead witness near the rear wall. Compare how much of the room impulse survives all the way to the back.",
                             particles, before, MPMMaterial::TOUGH,
                             50000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 5.0f);

    mpm.params().youngs_modulus = old_E;
}

static void load_spall_plate_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                   MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                   CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(-1.56f, -1.18f), vec2(0.34f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Lane A Plinth", "Plinth for the metal + brittle liner lane.");
    sdf.add_box(vec2(0.00f, -1.18f), vec2(0.34f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Lane B Plinth", "Plinth for the stoneware + ceramic lane.");
    sdf.add_box(vec2(1.56f, -1.18f), vec2(0.34f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Lane C Plinth", "Plinth for the bulkhead + spall lane.");
    sdf.add_box(vec2(2.10f, -0.10f), vec2(0.12f, 1.18f),
                SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                "Lane Catch Wall", "Conductive catch wall for hot fragments and spall.");
    sdf.rebuild();

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;
    u32 before = 0;

    mpm.params().youngs_modulus = 56000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.72f, -1.12f), vec2(-1.56f, 0.10f), sp,
                    MPMMaterial::THERMO_METAL, 300.0f, vec2(1, 0), 7.4f);
    register_scene_mpm_batch(creation, "Lane A Front Plate",
                             "Thin metal front plate. Good for reading APHE / HEAT penetration and how much spall it throws into the liner behind it.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             56000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 7.4f);

    mpm.params().youngs_modulus = 30000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.50f, -1.08f), vec2(-1.20f, -0.02f), sp,
                    MPMMaterial::BRITTLE, 300.0f, vec2(1, 0), 2.5f);
    register_scene_mpm_batch(creation, "Lane A Spall Liner",
                             "Brittle witness liner behind the metal plate. If a charge gets through, this should break up early and make the fragment cone obvious.",
                             particles, before, MPMMaterial::BRITTLE,
                             30000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 2.5f);

    mpm.params().youngs_modulus = 42000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.12f, -1.12f), vec2(0.08f, 0.10f), sp,
                    MPMMaterial::STONEWARE, 300.0f, vec2(1, 0), 3.2f);
    register_scene_mpm_batch(creation, "Lane B Fired Face",
                             "Dense fired face. This lane is meant for broader impact and face-crack comparison instead of purely metallic response.",
                             particles, before, MPMMaterial::STONEWARE,
                             42000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 3.2f);

    mpm.params().youngs_modulus = 34000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.08f, -1.08f), vec2(0.34f, 0.02f), sp,
                    MPMMaterial::CERAMIC, 300.0f, vec2(1, 0), 2.8f);
    register_scene_mpm_batch(creation, "Lane B Ceramic Backer",
                             "Ceramic backing slab. This should hold shape a bit longer than the brittle liner but still show chunky back-face shedding.",
                             particles, before, MPMMaterial::CERAMIC,
                             34000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 2.8f);

    mpm.params().youngs_modulus = 52000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.34f, -1.12f), vec2(1.58f, 0.12f), sp,
                    MPMMaterial::TOUGH, 300.0f, vec2(1, 0), 5.0f);
    register_scene_mpm_batch(creation, "Lane C Bulkhead",
                             "Thick tough bulkhead. This lane is the control for deep impulse without much early front-face breakup.",
                             particles, before, MPMMaterial::TOUGH,
                             52000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 5.0f);

    mpm.params().youngs_modulus = 28000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.60f, -1.08f), vec2(1.84f, -0.12f), sp,
                    MPMMaterial::BRITTLE, 300.0f, vec2(1, 0), 2.3f);
    register_scene_mpm_batch(creation, "Lane C Rear Witness",
                             "Rear brittle witness for the heavy bulkhead lane. Use this to see whether the target stops most of the fragment cone or only delays it.",
                             particles, before, MPMMaterial::BRITTLE,
                             28000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 2.3f);

    mpm.params().youngs_modulus = old_E;
}

static void load_open_blast_range_xl(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                     MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                     CreationState* creation) {
    sdf.clear();
    sdf.add_segment(vec2(-5.60f, -1.55f), vec2(5.60f, -1.55f), 0.16f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Range Floor", "Wide open blast floor so fireballs and smoke have much more room to expand.");
    sdf.add_box(vec2(-3.90f, -1.18f), vec2(0.38f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Metal Plinth", "Wide-spaced plinth for the thermo-metal witness plate.");
    sdf.add_box(vec2(-0.55f, -1.18f), vec2(0.38f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Stoneware Plinth", "Wide-spaced plinth for the fired ceramic target.");
    sdf.add_box(vec2(2.60f, -1.18f), vec2(0.46f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Layered Armor Plinth", "Wide-spaced plinth for layered armor and spall reading.");
    sdf.add_box(vec2(5.00f, -0.12f), vec2(0.12f, 1.40f),
                SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Far Catch Wall", "Remote heat-sink wall to catch fragments after they have crossed a much bigger air gap.");
    sdf.rebuild();

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;
    u32 before = 0;

    mpm.params().youngs_modulus = 56000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-4.04f, -1.12f), vec2(-3.76f, 0.26f), sp,
                    MPMMaterial::THERMO_METAL, 300.0f, vec2(1, 0), 7.8f);
    register_scene_mpm_batch(creation, "XL Thermo-Metal Witness",
                             "Same idea as the smaller blast lane, but with much more room around it so shock, fire, and hot gas do not immediately fill the whole scene.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             56000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 7.8f);

    mpm.params().youngs_modulus = 43000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.72f, -1.12f), vec2(-0.38f, 0.18f), sp,
                    MPMMaterial::STONEWARE, 300.0f, vec2(1, 0), 3.3f);
    register_scene_mpm_batch(creation, "XL Stoneware Slab",
                             "Dense fired slab with enough air around it to watch delayed heating, cracking, and fragment travel more clearly.",
                             particles, before, MPMMaterial::STONEWARE,
                             43000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 3.3f);

    mpm.params().youngs_modulus = 36500.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(2.10f, -1.12f), vec2(2.30f, 0.12f), sp,
                    MPMMaterial::CERAMIC, 300.0f, vec2(1, 0), 2.8f);
    register_scene_mpm_batch(creation, "XL Ceramic Front Face",
                             "Front strike face for long-range fragment and spall tests.",
                             particles, before, MPMMaterial::CERAMIC,
                             36500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 2.8f);

    mpm.params().youngs_modulus = 48000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(2.30f, -1.10f), vec2(2.72f, 0.14f), sp,
                    MPMMaterial::TOUGH, 300.0f, vec2(1, 0), 4.9f);
    register_scene_mpm_batch(creation, "XL Tough Backer",
                             "Tough backing plate with more open space behind it so the surviving fragment cloud is easier to read.",
                             particles, before, MPMMaterial::TOUGH,
                             48000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 4.9f);

    mpm.params().youngs_modulus = 28500.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(3.12f, -1.08f), vec2(3.38f, -0.04f), sp,
                    MPMMaterial::BRITTLE, 300.0f, vec2(1, 0), 2.4f);
    register_scene_mpm_batch(creation, "XL Rear Witness",
                             "Rear brittle sheet placed farther downstream so delayed spall and fragment cones stay visible.",
                             particles, before, MPMMaterial::BRITTLE,
                             28500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 2.4f);

    mpm.params().youngs_modulus = old_E;
}

static void load_breach_hall_xl(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                CreationState* creation) {
    sdf.clear();
    add_floor_and_walls_extents(sdf, vec2(-5.60f, -1.60f), vec2(5.60f, 3.60f),
                                SDFField::MaterialPresetID::BRONZE_BALANCED, 0.16f);
    sdf.add_segment(vec2(-2.55f, -1.60f), vec2(-2.55f, -0.58f), 0.11f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Door Jamb Lower", "Lower jamb of the wide breach hall opening.");
    sdf.add_segment(vec2(-2.55f, 0.86f), vec2(-2.55f, 2.85f), 0.11f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Door Jamb Upper", "Upper jamb of the wide breach hall opening.");
    sdf.add_segment(vec2(-2.55f, 2.85f), vec2(4.95f, 2.85f), 0.12f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Hall Roof", "Long roof so repeated pressure reflections can develop over a larger distance.");
    sdf.add_segment(vec2(4.95f, -1.60f), vec2(4.95f, 2.85f), 0.14f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Rear Bulkhead", "Large heat-sink bulkhead at the far end of the hall.");
    sdf.add_box(vec2(1.35f, -0.12f), vec2(1.05f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Interior Shelf", "Long witness shelf inside the larger breach hall.");
    sdf.rebuild();

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;
    u32 before = 0;

    mpm.params().youngs_modulus = 66000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-2.88f, -0.56f), vec2(-2.48f, 0.92f), sp,
                    MPMMaterial::THERMO_METAL, 300.0f, vec2(1, 0), 7.6f);
    register_scene_mpm_batch(creation, "XL Blast Door",
                             "Taller metal blast door set in a much bigger hall so door flex, rupture, and delayed reflected loading are easier to separate visually.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             66000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 7.6f);

    mpm.params().youngs_modulus = 32500.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(0.10f, -1.02f), 0.20f, sp,
                     MPMMaterial::CERAMIC, 300.0f, vec2(1, 0), 2.8f);
    register_scene_mpm_batch(creation, "XL Witness Pot A",
                             "First ceramic witness pot with much more standoff from the door.",
                             particles, before, MPMMaterial::CERAMIC,
                             32500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 2.8f);

    before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(1.10f, -1.02f), 0.20f, sp,
                     MPMMaterial::CERAMIC, 300.0f, vec2(1, 0), 2.8f);
    register_scene_mpm_batch(creation, "XL Witness Pot B",
                             "Second witness pot deeper in the hall for reflected-wave comparison.",
                             particles, before, MPMMaterial::CERAMIC,
                             32500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 2.8f);

    mpm.params().youngs_modulus = 40500.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.00f, -0.10f), vec2(2.10f, 0.36f), sp,
                    MPMMaterial::STONEWARE, 300.0f, vec2(1, 0), 3.2f);
    register_scene_mpm_batch(creation, "XL Stoneware Crate",
                             "Long-shelf fired crate that reads reflected blast and delayed heat loading in the expanded hall.",
                             particles, before, MPMMaterial::STONEWARE,
                             40500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 3.2f);

    mpm.params().youngs_modulus = 50500.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(4.10f, -1.14f), vec2(4.48f, 0.30f), sp,
                    MPMMaterial::TOUGH, 300.0f, vec2(1, 0), 5.1f);
    register_scene_mpm_batch(creation, "XL Rear Bulkhead Witness",
                             "Tough witness near the rear wall so late-stage impulse and reflected fragments have a clear long-distance target.",
                             particles, before, MPMMaterial::TOUGH,
                             50500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 5.1f);

    mpm.params().youngs_modulus = old_E;
}

static void load_spall_gallery_xl(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                  MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                  CreationState* creation) {
    sdf.clear();
    sdf.add_segment(vec2(-5.80f, -1.55f), vec2(5.80f, -1.55f), 0.16f,
                    SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Gallery Floor", "Long floor for spall and witness-lane comparisons.");
    sdf.add_box(vec2(-4.10f, -1.18f), vec2(0.36f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Lane A Plinth", "Far-left plinth for the metal plus brittle witness lane.");
    sdf.add_box(vec2(-0.40f, -1.18f), vec2(0.36f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Lane B Plinth", "Center plinth for the fired ceramic lane.");
    sdf.add_box(vec2(3.20f, -1.18f), vec2(0.36f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Lane C Plinth", "Right plinth for the tough bulkhead lane.");
    sdf.add_box(vec2(5.15f, -0.08f), vec2(0.12f, 1.48f),
                SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                "Gallery Catch Wall", "Far conductive backstop so hot fragments have to travel farther before hitting it.");
    sdf.rebuild();

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;
    u32 before = 0;

    mpm.params().youngs_modulus = 57500.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-4.28f, -1.12f), vec2(-4.08f, 0.14f), sp,
                    MPMMaterial::THERMO_METAL, 300.0f, vec2(1, 0), 7.5f);
    register_scene_mpm_batch(creation, "Gallery Lane A Front Plate",
                             "Long-range version of the metal witness lane so fragment cones stay readable instead of immediately hitting a side wall.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             57500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 7.5f);

    mpm.params().youngs_modulus = 30500.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-4.02f, -1.08f), vec2(-3.66f, 0.00f), sp,
                    MPMMaterial::BRITTLE, 300.0f, vec2(1, 0), 2.5f);
    register_scene_mpm_batch(creation, "Gallery Lane A Spall Liner",
                             "Rear brittle liner with extra open distance behind it so the surviving fragment plume remains visible.",
                             particles, before, MPMMaterial::BRITTLE,
                             30500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 2.5f);

    mpm.params().youngs_modulus = 43000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.56f, -1.12f), vec2(-0.30f, 0.12f), sp,
                    MPMMaterial::STONEWARE, 300.0f, vec2(1, 0), 3.2f);
    register_scene_mpm_batch(creation, "Gallery Lane B Fired Face",
                             "Fired ceramic lane with more lateral separation from the other targets.",
                             particles, before, MPMMaterial::STONEWARE,
                             43000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 3.2f);

    mpm.params().youngs_modulus = 34500.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.30f, -1.08f), vec2(0.06f, 0.04f), sp,
                    MPMMaterial::CERAMIC, 300.0f, vec2(1, 0), 2.8f);
    register_scene_mpm_batch(creation, "Gallery Lane B Backer",
                             "Ceramic backing slab for the center lane.",
                             particles, before, MPMMaterial::CERAMIC,
                             34500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 2.8f);

    mpm.params().youngs_modulus = 52500.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(3.00f, -1.12f), vec2(3.28f, 0.14f), sp,
                    MPMMaterial::TOUGH, 300.0f, vec2(1, 0), 5.0f);
    register_scene_mpm_batch(creation, "Gallery Lane C Bulkhead",
                             "Tough bulkhead lane for delayed impulse and fragment filtering.",
                             particles, before, MPMMaterial::TOUGH,
                             52500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 5.0f);

    mpm.params().youngs_modulus = 28500.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(3.32f, -1.08f), vec2(3.64f, -0.10f), sp,
                    MPMMaterial::BRITTLE, 300.0f, vec2(1, 0), 2.3f);
    register_scene_mpm_batch(creation, "Gallery Lane C Rear Witness",
                             "Rear brittle witness sheet that now has a much longer spall flight path before the catch wall.",
                             particles, before, MPMMaterial::BRITTLE,
                             28500.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 2.3f);

    mpm.params().youngs_modulus = old_E;
}

static void load_bio_replicator_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                      MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                      CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(-1.55f, -0.15f), vec2(0.55f, 0.07f),
                SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Warm Shelf", "Conductive shelf for morphogen and baking crossover tests.");
    sdf.add_box(vec2(1.45f, 0.45f), vec2(0.48f, 0.07f),
                SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                "Cool Shelf", "Raised shelf for watching budding fronts split away from the ground.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(-1.50f, -1.25f);
    mpm.params().heat_source_radius = 0.62f;
    mpm.params().heat_source_temp = 560.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;
    f32 old_nu = mpm.params().poisson_ratio;

    u32 before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 12400.0f;
    mpm.params().poisson_ratio = 0.37f;
    mpm.spawn_circle(particles, vec2(-1.45f, -0.85f), 0.38f, sp,
                     MPMMaterial::TOPO_GOO, 318.0f, vec2(1, 0), 1.0f,
                     vec4(0.06f, 0.34f, 0.92f, 0.05f));
    register_scene_mpm_batch(creation, "Replicator Goo",
                             "Expected: the blob should round up, form multiple active lobes, and sometimes split or re-merge as the bio field self-organizes.",
                             particles, before, MPMMaterial::TOPO_GOO,
                             12400.0f, 0.37f, 318.0f, vec2(1, 0), 1.0f);

    before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 8600.0f;
    mpm.params().poisson_ratio = 0.30f;
    mpm.spawn_circle(particles, vec2(0.35f, -0.82f), 0.42f, sp,
                     MPMMaterial::CRUMB_LOAF, 330.0f, vec2(1, 0), 1.0f,
                     vec4(0.92f, 0.78f, 0.76f, 0.14f));
    register_scene_mpm_batch(creation, "Morph Crumb",
                             "Expected: patterned pore islands and split fronts should appear instead of one uniform puff everywhere.",
                             particles, before, MPMMaterial::CRUMB_LOAF,
                             8600.0f, 0.30f, 330.0f, vec2(1, 0), 1.0f);

    before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 9800.0f;
    mpm.params().poisson_ratio = 0.28f;
    mpm.spawn_block(particles, vec2(1.05f, 0.55f), vec2(2.00f, 0.88f), sp,
                    MPMMaterial::MAILLARD, 338.0f, vec2(1, 0), 1.0f,
                    vec4(0.30f, 0.70f, 0.62f, 0.08f));
    register_scene_mpm_batch(creation, "Pattern Skin",
                             "Expected: thin browning fronts and island-like blister bands should travel across the sheet under the bio field.",
                             particles, before, MPMMaterial::MAILLARD,
                             9800.0f, 0.28f, 338.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = old_E;
    mpm.params().poisson_ratio = old_nu;
}

static void load_mycelium_morph_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                      MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                      CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(0.0f, -0.05f), vec2(0.95f, 0.07f),
                SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Culture Shelf", "Warm platform that helps seed mushroom-like lobe growth.");
    sdf.add_segment(vec2(-1.75f, -1.50f), vec2(-1.20f, 0.60f), 0.08f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Left Ramp", "Ramp for watching soft budding material crawl and shear.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.20f);
    mpm.params().heat_source_radius = 0.58f;
    mpm.params().heat_source_temp = 520.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;
    f32 old_nu = mpm.params().poisson_ratio;

    u32 before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 7600.0f;
    mpm.params().poisson_ratio = 0.29f;
    mpm.spawn_circle(particles, vec2(-0.65f, -0.78f), 0.34f, sp,
                     MPMMaterial::MUSHROOM, 328.0f, vec2(1, 0), 1.0f,
                     vec4(0.36f, 0.42f, 0.76f, 0.20f));
    mpm.spawn_circle(particles, vec2(0.20f, -0.76f), 0.30f, sp,
                     MPMMaterial::MUSHROOM, 322.0f, vec2(1, 0), 1.0f,
                     vec4(0.36f, 0.42f, 0.76f, 0.20f));
    register_scene_mpm_batch(creation, "Mycelium Cells",
                             "Expected: warm caps should brighten at the active front, bud into softer lobes, and bias motion toward the reaction-diffusion ridges.",
                             particles, before, MPMMaterial::MUSHROOM,
                             7600.0f, 0.29f, 328.0f, vec2(1, 0), 1.0f);

    before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 11800.0f;
    mpm.params().poisson_ratio = 0.36f;
    mpm.spawn_block(particles, vec2(1.05f, -0.92f), vec2(1.85f, -0.38f), sp,
                    MPMMaterial::TOPO_GOO, 316.0f, vec2(1, 0), 1.0f,
                    vec4(0.06f, 0.34f, 0.92f, 0.05f));
    register_scene_mpm_batch(creation, "Crawler Slab",
                             "Expected: the slab should grow biased ridges and then pull itself into several rounded living-looking cells instead of one uniform melt.",
                             particles, before, MPMMaterial::TOPO_GOO,
                             11800.0f, 0.36f, 316.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = old_E;
    mpm.params().poisson_ratio = old_nu;
}

static void load_morphogenesis_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                     MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                     CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(-1.35f, -0.12f), vec2(0.62f, 0.07f),
                SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Warm Plinth", "Steady conductive base for differential-growth tissue and edge-budding sheet tests.");
    sdf.add_box(vec2(1.35f, 0.46f), vec2(0.56f, 0.07f),
                SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                "Cool Bridge", "Raised platform where frills and folds can separate cleanly from the ground.");
    sdf.add_segment(vec2(-0.25f, -1.40f), vec2(0.35f, 0.25f), 0.07f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Growth Ramp", "Slope for watching biased morphogen drift turn into buckling or edge-colony motion.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(-1.30f, -1.18f);
    mpm.params().heat_source_radius = 0.56f;
    mpm.params().heat_source_temp = 500.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;
    f32 old_nu = mpm.params().poisson_ratio;
    f32 old_fiber = mpm.params().fiber_strength;

    u32 before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 11200.0f;
    mpm.params().poisson_ratio = 0.38f;
    mpm.params().fiber_strength = 2.2f;
    mpm.spawn_circle(particles, vec2(-1.25f, -0.82f), 0.38f, sp,
                     MPMMaterial::MORPH_TISSUE, 314.0f, vec2(1, 0), 1.0f,
                     vec4(0.18f, 0.34f, 0.88f, 0.10f));
    register_scene_mpm_batch(creation, "Morph Tissue Dome",
                             "Expected: the dome should swell unevenly, buckle into several soft folds, and keep a more living-looking body instead of simply slumping.",
                             particles, before, MPMMaterial::MORPH_TISSUE,
                             11200.0f, 0.38f, 314.0f, vec2(1, 0), 1.0f);

    before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 9200.0f;
    mpm.params().poisson_ratio = 0.40f;
    mpm.params().fiber_strength = 2.4f;
    mpm.spawn_block(particles, vec2(0.82f, 0.08f), vec2(1.92f, 0.34f), sp,
                    MPMMaterial::CELL_SHEET, 314.0f, vec2(1, 0), 1.0f,
                    vec4(0.16f, 0.28f, 0.88f, 0.08f));
    register_scene_mpm_batch(creation, "Cell Sheet Strip",
                             "Expected: active edges should wrinkle and bud outward into colony-like lobes while the middle stays more sheet-like.",
                             particles, before, MPMMaterial::CELL_SHEET,
                             9200.0f, 0.40f, 314.0f, vec2(1, 0), 1.0f);

    before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 9800.0f;
    mpm.params().poisson_ratio = 0.36f;
    mpm.params().fiber_strength = 2.0f;
    mpm.spawn_circle(particles, vec2(0.15f, -0.68f), 0.24f, sp,
                     MPMMaterial::MORPH_TISSUE, 308.0f, vec2(0, 1), 1.0f,
                     vec4(0.18f, 0.34f, 0.88f, 0.10f));
    register_scene_mpm_batch(creation, "Morph Tissue Seed",
                             "Expected: the smaller seed should bias itself along the ramp and split into softer lobes faster than the large dome.",
                             particles, before, MPMMaterial::MORPH_TISSUE,
                             9800.0f, 0.36f, 308.0f, vec2(0, 1), 1.0f);

    mpm.params().youngs_modulus = old_E;
    mpm.params().poisson_ratio = old_nu;
    mpm.params().fiber_strength = old_fiber;
}

static void load_root_garden_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                   MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                   CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(0.0f, -0.10f), vec2(0.78f, 0.07f),
                SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Warm Bed", "Conductive base that keeps the center active while roots thicken and arch.");
    sdf.add_segment(vec2(-1.85f, -1.50f), vec2(-1.20f, 0.50f), 0.08f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Left Trellis", "Inclined guide for root-like strands and anisotropic crawling.");
    sdf.add_segment(vec2(1.10f, -1.20f), vec2(1.75f, 0.20f), 0.07f,
                    SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                    "Right Trellis", "Cooler guide that reveals whether the roots keep their bundles once they leave the warm bed.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.20f);
    mpm.params().heat_source_radius = 0.60f;
    mpm.params().heat_source_temp = 470.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;
    f32 old_nu = mpm.params().poisson_ratio;
    f32 old_fiber = mpm.params().fiber_strength;

    u32 before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 14800.0f;
    mpm.params().poisson_ratio = 0.31f;
    mpm.params().fiber_strength = 4.8f;
    mpm.spawn_block(particles, vec2(-1.55f, -1.02f), vec2(-1.10f, -0.18f), sp,
                    MPMMaterial::ROOT_WEAVE, 308.0f, vec2(0, 1), 1.0f,
                    vec4(0.12f, 0.24f, 0.92f, 0.05f));
    register_scene_mpm_batch(creation, "Root Weave Stem",
                             "Expected: the vertical stem should creep along the left trellis, keep directional bundles, and stiffen into a root-like arch instead of turning into a puddle.",
                             particles, before, MPMMaterial::ROOT_WEAVE,
                             14800.0f, 0.31f, 308.0f, vec2(0, 1), 1.0f);

    before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 13600.0f;
    mpm.params().poisson_ratio = 0.30f;
    mpm.params().fiber_strength = 5.2f;
    mpm.spawn_block(particles, vec2(0.42f, -0.92f), vec2(1.32f, -0.62f), sp,
                    MPMMaterial::ROOT_WEAVE, 310.0f, vec2(1, 0), 1.0f,
                    vec4(0.12f, 0.24f, 0.92f, 0.05f));
    register_scene_mpm_batch(creation, "Root Weave Mat",
                             "Expected: the horizontal patch should split into directional bundles and send root-like strands toward the right trellis.",
                             particles, before, MPMMaterial::ROOT_WEAVE,
                             13600.0f, 0.30f, 310.0f, vec2(1, 0), 1.0f);

    before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 7600.0f;
    mpm.params().poisson_ratio = 0.30f;
    mpm.params().fiber_strength = 2.0f;
    mpm.spawn_circle(particles, vec2(-0.10f, -0.70f), 0.26f, sp,
                     MPMMaterial::MORPH_TISSUE, 312.0f, vec2(1, 0), 1.0f,
                     vec4(0.18f, 0.34f, 0.88f, 0.10f));
    register_scene_mpm_batch(creation, "Soft Bud Cluster",
                             "Expected: the soft cluster should show faster lobe growth, while the root weaves keep stronger anisotropic structure nearby.",
                             particles, before, MPMMaterial::MORPH_TISSUE,
                             7600.0f, 0.30f, 312.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = old_E;
    mpm.params().poisson_ratio = old_nu;
    mpm.params().fiber_strength = old_fiber;
}

static void load_cell_colony_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                   MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                   CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(-1.45f, -0.08f), vec2(0.42f, 0.06f),
                SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Left Island", "Warm island for a first colony patch.");
    sdf.add_box(vec2(0.0f, 0.18f), vec2(0.38f, 0.06f),
                SDFField::MaterialPresetID::BRONZE_BALANCED,
                "Center Island", "Mid-height island for smoother colony separation.");
    sdf.add_box(vec2(1.42f, -0.02f), vec2(0.40f, 0.06f),
                SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                "Right Island", "Cooler island for comparing whether lobes stay attached or pinch off.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(-0.10f, -1.15f);
    mpm.params().heat_source_radius = 0.48f;
    mpm.params().heat_source_temp = 440.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;
    f32 old_nu = mpm.params().poisson_ratio;
    f32 old_fiber = mpm.params().fiber_strength;

    u32 before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 9200.0f;
    mpm.params().poisson_ratio = 0.40f;
    mpm.params().fiber_strength = 2.4f;
    mpm.spawn_circle(particles, vec2(-1.42f, -0.74f), 0.30f, sp,
                     MPMMaterial::CELL_SHEET, 314.0f, vec2(1, 0), 1.0f,
                     vec4(0.16f, 0.28f, 0.88f, 0.08f));
    register_scene_mpm_batch(creation, "Colony Disk A",
                             "Expected: this disk should bud at the edge first, then pinch into several soft colony lobes instead of staying circular.",
                             particles, before, MPMMaterial::CELL_SHEET,
                             9200.0f, 0.40f, 314.0f, vec2(1, 0), 1.0f);

    before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 10400.0f;
    mpm.params().poisson_ratio = 0.39f;
    mpm.params().fiber_strength = 2.6f;
    mpm.spawn_block(particles, vec2(-0.28f, -0.42f), vec2(0.28f, -0.18f), sp,
                    MPMMaterial::CELL_SHEET, 312.0f, vec2(1, 0), 1.0f,
                    vec4(0.16f, 0.28f, 0.88f, 0.08f));
    register_scene_mpm_batch(creation, "Colony Strip B",
                             "Expected: the strip should wrinkle into a chain of cells, closer to a smoothlife-style membrane edge than a plain block.",
                             particles, before, MPMMaterial::CELL_SHEET,
                             10400.0f, 0.39f, 312.0f, vec2(1, 0), 1.0f);

    before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 10200.0f;
    mpm.params().poisson_ratio = 0.38f;
    mpm.params().fiber_strength = 2.1f;
    mpm.spawn_circle(particles, vec2(1.42f, -0.72f), 0.26f, sp,
                     MPMMaterial::MORPH_TISSUE, 316.0f, vec2(0, 1), 1.0f,
                     vec4(0.18f, 0.34f, 0.88f, 0.10f));
    register_scene_mpm_batch(creation, "Colony Seed C",
                             "Expected: this softer seed should swell and fold, then compete with the sheet colonies so you can compare tissue buckling versus edge budding.",
                             particles, before, MPMMaterial::MORPH_TISSUE,
                             10200.0f, 0.38f, 316.0f, vec2(0, 1), 1.0f);

    mpm.params().youngs_modulus = old_E;
    mpm.params().poisson_ratio = old_nu;
    mpm.params().fiber_strength = old_fiber;
}

static void load_automata_air_coupling_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                             MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                             CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(-1.50f, -0.10f), vec2(0.58f, 0.07f),
                SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Warm Left Shelf", "Expected: active fronts should seed from here and leak into the nearby air before the blob follows.");
    sdf.add_box(vec2(1.38f, 0.42f), vec2(0.52f, 0.07f),
                SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                "Cool Right Shelf", "Expected: weaker growth here should make the field stretch through air instead of staying glued to the particles.");
    sdf.add_segment(vec2(-0.10f, -1.42f), vec2(0.42f, 0.40f), 0.07f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Air Guide Ramp", "Expected: drive vectors should lean along this ramp and reveal where the automata field wants the tissue to crawl.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(-1.45f, -1.16f);
    mpm.params().heat_source_radius = 0.56f;
    mpm.params().heat_source_temp = 470.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;
    f32 old_nu = mpm.params().poisson_ratio;
    f32 old_fiber = mpm.params().fiber_strength;

    u32 before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 9800.0f;
    mpm.params().poisson_ratio = 0.40f;
    mpm.params().fiber_strength = 2.6f;
    mpm.spawn_block(particles, vec2(-1.72f, -0.92f), vec2(-0.94f, -0.46f), sp,
                    MPMMaterial::CELL_SHEET, 314.0f, vec2(1, 0), 1.0f,
                    vec4(0.20f, 0.34f, 0.92f, 0.10f));
    register_scene_mpm_batch(creation, "Sheet Seeder",
                             "Expected: start with Air View = Automata Drive. The field should reach a little ahead of the active edge and the sheet should gradually follow instead of only flickering in place.",
                             particles, before, MPMMaterial::CELL_SHEET,
                             9800.0f, 0.40f, 314.0f, vec2(1, 0), 1.0f);

    before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 11600.0f;
    mpm.params().poisson_ratio = 0.37f;
    mpm.params().fiber_strength = 2.4f;
    mpm.spawn_circle(particles, vec2(0.10f, -0.68f), 0.28f, sp,
                     MPMMaterial::MORPH_TISSUE, 314.0f, vec2(0, 1), 1.0f,
                     vec4(0.20f, 0.34f, 0.92f, 0.10f));
    register_scene_mpm_batch(creation, "Morph Seed",
                             "Expected: the central seed should show uneven pulsing and then crawl toward the stronger air-side drive rather than only changing color.",
                             particles, before, MPMMaterial::MORPH_TISSUE,
                             11600.0f, 0.37f, 314.0f, vec2(0, 1), 1.0f);

    before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 14200.0f;
    mpm.params().poisson_ratio = 0.31f;
    mpm.params().fiber_strength = 4.6f;
    mpm.spawn_block(particles, vec2(1.10f, -0.94f), vec2(1.58f, -0.14f), sp,
                    MPMMaterial::ROOT_WEAVE, 308.0f, vec2(0, 1), 1.0f,
                    vec4(0.14f, 0.24f, 0.92f, 0.06f));
    register_scene_mpm_batch(creation, "Root Probe",
                             "Expected: the anisotropic probe should bend with the drive field but retain a more directional bundle response than the soft cell sheet.",
                             particles, before, MPMMaterial::ROOT_WEAVE,
                             14200.0f, 0.31f, 308.0f, vec2(0, 1), 1.0f);

    mpm.params().youngs_modulus = old_E;
    mpm.params().poisson_ratio = old_nu;
    mpm.params().fiber_strength = old_fiber;
}

static void load_automata_fire_regrowth_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                              MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                              CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(0.0f, -0.12f), vec2(1.05f, 0.07f),
                SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Regrowth Bed", "Expected: local fire damage should carve out the field here, then nearby fronts should creep back in from the cooler edges.");
    sdf.add_box(vec2(1.55f, 0.52f), vec2(0.42f, 0.07f),
                SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                "Witness Shelf", "Raised shelf for comparing unburnt regrowth against a cooler untouched reference patch.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(-0.55f, -0.98f);
    mpm.params().heat_source_radius = 0.34f;
    mpm.params().heat_source_temp = 860.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;
    f32 old_nu = mpm.params().poisson_ratio;
    f32 old_fiber = mpm.params().fiber_strength;

    u32 before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 9800.0f;
    mpm.params().poisson_ratio = 0.40f;
    mpm.params().fiber_strength = 2.5f;
    mpm.spawn_block(particles, vec2(-1.10f, -0.88f), vec2(0.20f, -0.44f), sp,
                    MPMMaterial::CELL_SHEET, 315.0f, vec2(1, 0), 1.0f,
                    vec4(0.20f, 0.34f, 0.92f, 0.10f));
    register_scene_mpm_batch(creation, "Burnable Sheet",
                             "Expected: the hot left side should lose colony occupancy first, then the automata front should regrow from the cooler right edge and pull the sheet back into the damaged gap.",
                             particles, before, MPMMaterial::CELL_SHEET,
                             9800.0f, 0.40f, 315.0f, vec2(1, 0), 1.0f);

    before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 11400.0f;
    mpm.params().poisson_ratio = 0.37f;
    mpm.params().fiber_strength = 2.4f;
    mpm.spawn_circle(particles, vec2(0.82f, -0.70f), 0.26f, sp,
                     MPMMaterial::MORPH_TISSUE, 314.0f, vec2(0, 1), 1.0f,
                     vec4(0.20f, 0.34f, 0.92f, 0.10f));
    register_scene_mpm_batch(creation, "Regrowth Seed",
                             "Expected: this dome should keep a cleaner untouched colony front so you can compare live regrowth against the burnt patch on the left.",
                             particles, before, MPMMaterial::MORPH_TISSUE,
                             11400.0f, 0.37f, 314.0f, vec2(0, 1), 1.0f);

    before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 7600.0f;
    mpm.params().poisson_ratio = 0.29f;
    mpm.spawn_circle(particles, vec2(1.55f, 0.72f), 0.20f, sp,
                     MPMMaterial::MUSHROOM, 320.0f, vec2(1, 0), 1.0f,
                     vec4(0.32f, 0.42f, 0.78f, 0.20f));
    register_scene_mpm_batch(creation, "Cool Witness Cap",
                             "Expected: this cooler witness patch should mostly keep its field while the hot patch below gets erased and then tries to grow back.",
                             particles, before, MPMMaterial::MUSHROOM,
                             7600.0f, 0.29f, 320.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = old_E;
    mpm.params().poisson_ratio = old_nu;
    mpm.params().fiber_strength = old_fiber;
}

static void load_ash_regrowth_bench(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                    MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                    CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(-0.78f, -0.10f), vec2(1.18f, 0.07f),
                SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Ash Bed", "Expected: the hot-left sample should char into weak ash first, then regrow back from the cooler side while the ash scaffold partially slumps.");
    sdf.add_box(vec2(1.48f, 0.42f), vec2(0.54f, 0.07f),
                SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                "Cool Witness Shelf", "Raised shelf for a cooler control patch that should stay much more intact than the hot samples below.");
    sdf.add_segment(vec2(0.58f, -1.10f), vec2(1.00f, -0.08f), 0.06f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Spill Ramp", "Expected: once the ash gets fragile here it should spill and regrow back in a visibly lopsided way.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(-1.18f, -0.98f);
    mpm.params().heat_source_radius = 0.30f;
    mpm.params().heat_source_temp = 940.0f;

    f32 sp = grid.dx() * 0.5f;
    f32 old_E = mpm.params().youngs_modulus;
    f32 old_nu = mpm.params().poisson_ratio;
    f32 old_fiber = mpm.params().fiber_strength;

    u32 before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 11200.0f;
    mpm.params().poisson_ratio = 0.34f;
    mpm.params().fiber_strength = 2.3f;
    mpm.spawn_block(particles, vec2(-1.58f, -0.92f), vec2(-0.18f, -0.46f), sp,
                    MPMMaterial::ASH_REGROWTH, 316.0f, vec2(1, 0), 1.0f,
                    vec4(0.68f, 0.78f, 0.70f, 1.0f));
    register_scene_mpm_batch(creation, "Ash Slab",
                             "Expected: the hot-left face should turn into fragile ash, slump and shed first, then regrow back from the cooler right edge into a deformed slab instead of an instant perfect reset.",
                             particles, before, MPMMaterial::ASH_REGROWTH,
                             11200.0f, 0.34f, 316.0f, vec2(1, 0), 1.0f);

    before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 9800.0f;
    mpm.params().poisson_ratio = 0.33f;
    mpm.params().fiber_strength = 2.0f;
    mpm.spawn_block(particles, vec2(0.54f, -0.92f), vec2(0.90f, -0.12f), sp,
                    MPMMaterial::ASH_REGROWTH, 314.0f, vec2(0, 1), 1.0f,
                    vec4(0.68f, 0.78f, 0.70f, 1.0f));
    register_scene_mpm_batch(creation, "Ash Pillar",
                             "Expected: this tall pillar should ash over and spill much more asymmetrically on the ramp, so its regrowth comes back as a crooked, partially collapsed column.",
                             particles, before, MPMMaterial::ASH_REGROWTH,
                             9800.0f, 0.33f, 314.0f, vec2(0, 1), 1.0f);

    before = particles.range(SolverType::MPM).count;
    mpm.params().youngs_modulus = 10400.0f;
    mpm.params().poisson_ratio = 0.35f;
    mpm.params().fiber_strength = 2.1f;
    mpm.spawn_circle(particles, vec2(1.48f, 0.66f), 0.22f, sp,
                     MPMMaterial::ASH_REGROWTH, 308.0f, vec2(1, 0), 1.0f,
                     vec4(0.68f, 0.78f, 0.70f, 1.0f));
    register_scene_mpm_batch(creation, "Cool Witness Patch",
                             "Expected: the witness patch should stay mostly alive and body-like, giving you a clean comparison against the ash-and-regrow samples on the heated bed.",
                             particles, before, MPMMaterial::ASH_REGROWTH,
                             10400.0f, 0.35f, 308.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = old_E;
    mpm.params().poisson_ratio = old_nu;
    mpm.params().fiber_strength = old_fiber;
}

static void load_hybrid_regrowth_wall(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                      MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                      CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(-0.84f, -0.08f), vec2(1.22f, 0.07f),
                SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Warm Regrowth Bed", "Expected: the left side should char first, then the wall should try to recover from the cooler right edge instead of snapping back instantly.");
    sdf.add_box(vec2(1.52f, 0.46f), vec2(0.52f, 0.07f),
                SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                "Cool Witness Shelf", "Reference perch for a cleaner untouched colony front.");
    sdf.add_segment(vec2(0.42f, -1.08f), vec2(0.98f, -0.10f), 0.06f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Spill Brace", "Once the ash gets fragile here it should spill and regrow back asymmetrically.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(-1.20f, -0.98f);
    mpm.params().heat_source_radius = 0.30f;
    mpm.params().heat_source_temp = 930.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 old_nu = mpm.params().poisson_ratio;
    const f32 old_fiber = mpm.params().fiber_strength;
    u32 before = 0;

    mpm.params().youngs_modulus = 11800.0f;
    mpm.params().poisson_ratio = 0.34f;
    mpm.params().fiber_strength = 2.6f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.60f, -0.92f), vec2(-0.24f, -0.34f), sp,
                    MPMMaterial::ASH_REGROWTH, 316.0f, vec2(1, 0), 1.0f,
                    vec4(0.10f, 0.20f, 0.92f, 0.05f));
    register_scene_mpm_batch(creation, "Hybrid Regrowth Wall",
                             "Expected: the heated left side should blacken into weak ash, slump around the spill brace, then regrow from the cooler right edge into a warped wall instead of a perfect reset.",
                             particles, before, MPMMaterial::ASH_REGROWTH,
                             11800.0f, 0.34f, 316.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = 9400.0f;
    mpm.params().poisson_ratio = 0.40f;
    mpm.params().fiber_strength = 2.8f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.52f, -0.30f), vec2(-0.38f, -0.08f), sp,
                    MPMMaterial::CELL_SHEET, 320.0f, vec2(1, 0), 1.0f,
                    vec4(0.18f, 0.34f, 0.80f, 0.14f));
    register_scene_mpm_batch(creation, "Hybrid Bio Cap",
                             "Expected: the cap should lose occupancy first near the heat source, then re-colonize across the cooled ash face and help the wall pull itself back together.",
                             particles, before, MPMMaterial::CELL_SHEET,
                             9400.0f, 0.40f, 320.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = 14200.0f;
    mpm.params().poisson_ratio = 0.31f;
    mpm.params().fiber_strength = 5.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.64f, -0.92f), vec2(0.92f, -0.12f), sp,
                    MPMMaterial::ROOT_WEAVE, 308.0f, vec2(0, 1), 1.0f,
                    vec4(0.12f, 0.24f, 0.92f, 0.05f));
    register_scene_mpm_batch(creation, "Root Brace",
                             "Expected: this directional brace should keep stronger structure along the spill side, so the returning wall reads less symmetric than a plain block.",
                             particles, before, MPMMaterial::ROOT_WEAVE,
                             14200.0f, 0.31f, 308.0f, vec2(0, 1), 1.0f);

    mpm.params().youngs_modulus = 7600.0f;
    mpm.params().poisson_ratio = 0.29f;
    mpm.params().fiber_strength = 0.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(1.52f, 0.68f), 0.20f, sp,
                     MPMMaterial::MUSHROOM, 318.0f, vec2(1, 0), 1.0f,
                     vec4(0.30f, 0.38f, 0.76f, 0.20f));
    register_scene_mpm_batch(creation, "Cool Witness Patch",
                             "Expected: the witness patch should stay much more intact than the heated wall, so it gives a clean live-versus-charred comparison.",
                             particles, before, MPMMaterial::MUSHROOM,
                             7600.0f, 0.29f, 318.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = old_E;
    mpm.params().poisson_ratio = old_nu;
    mpm.params().fiber_strength = old_fiber;
}

static void load_hybrid_kiln_process(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                     MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                     CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(-1.42f, -0.10f), vec2(0.56f, 0.07f),
                SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Kiln Hearth", "Expected: the left-side kiln pieces should shell-set first, then drip, craze, or harden differently as the heat climbs.");
    sdf.add_box(vec2(0.18f, 0.34f), vec2(0.96f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Upper Witness Shelf", "Raised shelf for cooler baking and glaze comparison.");
    sdf.add_segment(vec2(1.86f, -1.08f), vec2(2.22f, -0.10f), 0.05f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Draft Ramp", "As fired pieces soften or drip, this ramp should bias the flow and make the shell/core split easier to read.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(-1.18f, -1.02f);
    mpm.params().heat_source_radius = 0.36f;
    mpm.params().heat_source_temp = 1040.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 old_nu = mpm.params().poisson_ratio;
    const f32 old_fiber = mpm.params().fiber_strength;
    u32 before = 0;

    mpm.params().youngs_modulus = 12200.0f;
    mpm.params().poisson_ratio = 0.22f;
    mpm.params().fiber_strength = 0.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.84f, -0.90f), vec2(-1.16f, -0.34f), sp,
                    MPMMaterial::GLAZE_CLAY, 300.0f, vec2(1, 0), 1.0f,
                    vec4(0.38f, 0.92f, 0.78f, 0.04f));
    register_scene_mpm_batch(creation, "Kiln Shell Blank",
                             "Expected: this thicker shell-first blank should glaze and stiffen at the surface before the core catches up.",
                             particles, before, MPMMaterial::GLAZE_CLAY,
                             12200.0f, 0.22f, 300.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = 11200.0f;
    mpm.params().poisson_ratio = 0.24f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(-0.46f, -0.62f), 0.24f, sp,
                     MPMMaterial::BLISTER_GLAZE, 300.0f, vec2(1, 0), 1.0f,
                     vec4(0.62f, 0.84f, 0.80f, 0.05f));
    register_scene_mpm_batch(creation, "Blister Glaze Witness",
                             "Expected: the hotter shell should trap volatiles, pit, and blister instead of only cracking or dripping smoothly.",
                             particles, before, MPMMaterial::BLISTER_GLAZE,
                             11200.0f, 0.24f, 300.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = 8400.0f;
    mpm.params().poisson_ratio = 0.30f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.02f, -0.02f), vec2(0.88f, 0.24f), sp,
                    MPMMaterial::CRUST_DOUGH, 302.0f, vec2(1, 0), 1.0f,
                    vec4(0.78f, 0.72f, 0.80f, 0.10f));
    register_scene_mpm_batch(creation, "Crust Dough Loaf",
                             "Expected: the loaf should set a shell and keep a bread-like body longer than a plain dough slab while the hotter kiln pieces on the left continue firing.",
                             particles, before, MPMMaterial::CRUST_DOUGH,
                             8400.0f, 0.30f, 302.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = 14800.0f;
    mpm.params().poisson_ratio = 0.20f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.48f, -0.92f), vec2(1.92f, -0.22f), sp,
                    MPMMaterial::STONEWARE, 300.0f, vec2(1, 0), 1.0f,
                    vec4(0.34f, 0.88f, 0.76f, 0.04f));
    register_scene_mpm_batch(creation, "Stoneware Ramp Witness",
                             "Expected: the fired body on the draft ramp should stay denser and chunkier while the glazier pieces show more shell-local effects.",
                             particles, before, MPMMaterial::STONEWARE,
                             14800.0f, 0.20f, 300.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = old_E;
    mpm.params().poisson_ratio = old_nu;
    mpm.params().fiber_strength = old_fiber;
}

static void load_hybrid_soft_heat_range(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                        MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                        CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_box(vec2(-1.68f, -1.18f), vec2(0.36f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Lane A Plinth", "Hybrid HEAT lane: tuned for softer shaped-charge and layered-target comparisons.");
    sdf.add_box(vec2(0.02f, -1.18f), vec2(0.38f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Lane B Plinth", "Ceramic-versus-metal lane for wider heat spread and delayed breakage.");
    sdf.add_box(vec2(1.78f, -1.18f), vec2(0.36f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Lane C Plinth", "Soft target lane so weaker charges still have something readable to push apart.");
    sdf.add_box(vec2(2.26f, -0.08f), vec2(0.12f, 1.18f),
                SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                "Catch Wall", "Downrange wall for hot fragments and softened rear witnesses.");
    sdf.rebuild();

    mpm.params().enable_thermal = false;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 old_nu = mpm.params().poisson_ratio;
    const f32 old_fiber = mpm.params().fiber_strength;
    u32 before = 0;

    mpm.params().youngs_modulus = 56000.0f;
    mpm.params().poisson_ratio = 0.26f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.84f, -1.12f), vec2(-1.68f, 0.14f), sp,
                    MPMMaterial::THERMO_METAL, 300.0f, vec2(1, 0), 7.4f);
    register_scene_mpm_batch(creation, "Lane A Metal Face",
                             "Expected: use Soft HEAT, Medium HEAT, or Above Med HEAT charges here. The front plate should stay intact just long enough to focus the jet into the softer liner behind it.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             56000.0f, 0.26f, 300.0f, vec2(1, 0), 7.4f);

    mpm.params().youngs_modulus = 30000.0f;
    mpm.params().poisson_ratio = 0.22f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.62f, -1.08f), vec2(-1.28f, -0.06f), sp,
                    MPMMaterial::BRITTLE, 300.0f, vec2(1, 0), 2.5f);
    register_scene_mpm_batch(creation, "Lane A Spall Liner",
                             "Expected: if the jet or hot fragment cone gets through, this liner should break first and make the penetration path obvious.",
                             particles, before, MPMMaterial::BRITTLE,
                             30000.0f, 0.22f, 300.0f, vec2(1, 0), 2.5f);

    mpm.params().youngs_modulus = 42500.0f;
    mpm.params().poisson_ratio = 0.20f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.12f, -1.12f), vec2(0.08f, 0.10f), sp,
                    MPMMaterial::STONEWARE, 300.0f, vec2(1, 0), 3.3f);
    register_scene_mpm_batch(creation, "Lane B Fired Face",
                             "Expected: the denser fired face should crack chunkier and spread heat laterally more than the thin metal lane.",
                             particles, before, MPMMaterial::STONEWARE,
                             42500.0f, 0.20f, 300.0f, vec2(1, 0), 3.3f);

    mpm.params().youngs_modulus = 34000.0f;
    mpm.params().poisson_ratio = 0.24f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.08f, -1.08f), vec2(0.34f, 0.02f), sp,
                    MPMMaterial::CERAMIC, 300.0f, vec2(1, 0), 2.8f);
    register_scene_mpm_batch(creation, "Lane B Ceramic Backer",
                             "Expected: this backer should shed chunky rear fragments if the lane gets fully breached, but it should outlast the brittle liner in lane A.",
                             particles, before, MPMMaterial::CERAMIC,
                             34000.0f, 0.24f, 300.0f, vec2(1, 0), 2.8f);

    mpm.params().youngs_modulus = 9200.0f;
    mpm.params().poisson_ratio = 0.35f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.46f, -1.10f), vec2(1.98f, -0.48f), sp,
                    MPMMaterial::CRUST_DOUGH, 300.0f, vec2(1, 0), 1.0f,
                    vec4(0.62f, 0.62f, 0.84f, 0.10f));
    register_scene_mpm_batch(creation, "Lane C Soft Witness",
                             "Expected: weaker shaped charges should still gouge and bake this softer loaf-like witness, so low-power presets stay readable too.",
                             particles, before, MPMMaterial::CRUST_DOUGH,
                             9200.0f, 0.35f, 300.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = old_E;
    mpm.params().poisson_ratio = old_nu;
    mpm.params().fiber_strength = old_fiber;
}

static void load_hybrid_pressure_pottery(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                         MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                         CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(-1.42f, -0.14f), vec2(0.48f, 0.07f),
                SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Cook Plate", "Expected: pressure seeds here should heat slowly, then kick upward into the ceramic lids instead of venting immediately.");
    sdf.add_box(vec2(0.16f, -0.14f), vec2(0.48f, 0.07f),
                SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Cook Plate B", "Second pressure lane with a denser fired cap.");
    sdf.add_box(vec2(1.74f, -0.14f), vec2(0.48f, 0.07f),
                SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Cook Plate C", "Third lane for the shell-first glaze piece.");
    sdf.add_box(vec2(2.18f, -0.08f), vec2(0.10f, 1.10f),
                SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                "Rear Catch", "Catch wall for secondary pottery fragments.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(-0.10f, -1.00f);
    mpm.params().heat_source_radius = 0.46f;
    mpm.params().heat_source_temp = 860.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 old_nu = mpm.params().poisson_ratio;
    const f32 old_fiber = mpm.params().fiber_strength;
    u32 before = 0;

    mpm.params().youngs_modulus = 26000.0f;
    mpm.params().poisson_ratio = 0.20f;
    mpm.params().fiber_strength = 0.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(-1.42f, -0.78f), 0.12f, sp,
                     MPMMaterial::SEALED_CHARGE, 300.0f, vec2(1, 0), 1.0f,
                     vec4(1.30f, 1.18f, 0.74f, 0.02f));
    register_scene_mpm_batch(creation, "Pressure Seed A",
                             "Expected: the sealed bead should cook under the lid and throw a hotter, tighter burst upward than the older porous firecracker materials.",
                             particles, before, MPMMaterial::SEALED_CHARGE,
                             26000.0f, 0.20f, 300.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = 11800.0f;
    mpm.params().poisson_ratio = 0.22f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.68f, -0.52f), vec2(-1.16f, -0.26f), sp,
                    MPMMaterial::GLAZE_CLAY, 300.0f, vec2(1, 0), 1.0f,
                    vec4(0.40f, 0.92f, 0.80f, 0.04f));
    register_scene_mpm_batch(creation, "Glaze Lid",
                             "Expected: this glaze-clay lid should shell-set, crack, and then get kicked apart by the pressure seed under it.",
                             particles, before, MPMMaterial::GLAZE_CLAY,
                             11800.0f, 0.22f, 300.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = 27000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(0.16f, -0.78f), 0.12f, sp,
                     MPMMaterial::SEALED_CHARGE, 300.0f, vec2(1, 0), 1.0f,
                     vec4(1.34f, 1.22f, 0.72f, 0.02f));
    register_scene_mpm_batch(creation, "Pressure Seed B",
                             "Expected: the second seed gives a denser fired cap a hotter local poke so you can compare shell-setting versus chunkier rupture.",
                             particles, before, MPMMaterial::SEALED_CHARGE,
                             27000.0f, 0.20f, 300.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = 14800.0f;
    mpm.params().poisson_ratio = 0.20f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.08f, -0.54f), vec2(0.40f, -0.22f), sp,
                    MPMMaterial::STONEWARE, 300.0f, vec2(1, 0), 1.0f,
                    vec4(0.32f, 0.88f, 0.78f, 0.04f));
    register_scene_mpm_batch(creation, "Stoneware Cap",
                             "Expected: the stoneware cap should stay chunkier and harder than the glaze lid before it finally breaks loose.",
                             particles, before, MPMMaterial::STONEWARE,
                             14800.0f, 0.20f, 300.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = 28000.0f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(1.74f, -0.78f), 0.12f, sp,
                     MPMMaterial::SEALED_CHARGE, 300.0f, vec2(1, 0), 1.0f,
                     vec4(1.38f, 1.26f, 0.70f, 0.02f));
    register_scene_mpm_batch(creation, "Pressure Seed C",
                             "Expected: the right seed should vent into the hotter reactive glaze cap and throw more shell-local debris than the denser stoneware lane.",
                             particles, before, MPMMaterial::SEALED_CHARGE,
                             28000.0f, 0.20f, 300.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = 11200.0f;
    mpm.params().poisson_ratio = 0.24f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(1.74f, -0.38f), 0.18f, sp,
                     MPMMaterial::BLISTER_GLAZE, 300.0f, vec2(1, 0), 1.0f,
                     vec4(0.62f, 0.84f, 0.82f, 0.05f));
    register_scene_mpm_batch(creation, "Blister Cap",
                             "Expected: blister glaze should pit, vent, and flake differently from the cleaner glaze lid and denser stoneware cap once the pressure seed finally opens beneath it.",
                             particles, before, MPMMaterial::BLISTER_GLAZE,
                             11200.0f, 0.24f, 300.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = old_E;
    mpm.params().poisson_ratio = old_nu;
    mpm.params().fiber_strength = old_fiber;
}

static void load_hybrid_ferro_splash(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                     MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                     CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-2.10f, -1.30f), vec2(2.10f, -1.30f), 0.10f,
                    SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                    "Bench Plate", "Expected: ferro splash should climb and distort around the magnets here instead of behaving like a plain puddle.");
    sdf.add_box(vec2(-0.22f, -0.98f), vec2(0.22f, 0.62f),
                SDFField::MaterialPresetID::MAGNET_Y,
                "Pole Magnet", "Permanent pole for the main splash and spike test.");
    sdf.add_box(vec2(0.86f, -0.52f), vec2(0.12f, 0.52f),
                SDFField::MaterialPresetID::SOFT_IRON,
                "Soft Iron Pole", "Rigid soft-iron guide that should bend the field and collect the splash differently from the bare pole.");
    sdf.add_box(vec2(0.58f, 0.02f), vec2(0.34f, 0.08f),
                SDFField::MaterialPresetID::SOFT_IRON,
                "Soft Iron Cap", "Cap that should broaden the hot magnetic region above the right-side guide.");
    sdf.rebuild();

    mpm.params().enable_thermal = false;
    mpm.params().magnet_spike_strength = 1.15f;
    mpm.params().magnet_chain_rate = 6.4f;
    mpm.params().magnet_spike_freq = 13.5f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 old_nu = mpm.params().poisson_ratio;
    u32 before = 0;

    mpm.params().youngs_modulus = 4700.0f;
    mpm.params().poisson_ratio = 0.33f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.58f, -0.10f), vec2(0.38f, 0.10f), sp,
                    MPMMaterial::FERRO_FLUID, 300.0f, vec2(1, 0), 0.16f);
    register_scene_mpm_batch(creation, "Ferro Splash Slurry",
                             "Expected: the puddle should collect over the pole, then distort and split differently once the right-side soft-iron guide starts concentrating the field.",
                             particles, before, MPMMaterial::FERRO_FLUID,
                             4700.0f, 0.33f, 300.0f, vec2(1, 0), 0.16f);

    mpm.params().youngs_modulus = 9800.0f;
    mpm.params().poisson_ratio = 0.36f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.24f, -0.34f), vec2(1.72f, -0.04f), sp,
                    MPMMaterial::MAGNETIC_RUBBER, 300.0f, vec2(1, 0), 0.20f);
    register_scene_mpm_batch(creation, "Magnetic Rubber Guide",
                             "Expected: the compliant strip should bend and drift under the same field that is shaping the ferro puddle, giving you an easier body-versus-fluid comparison.",
                             particles, before, MPMMaterial::MAGNETIC_RUBBER,
                             9800.0f, 0.36f, 300.0f, vec2(1, 0), 0.20f);

    mpm.params().youngs_modulus = 54000.0f;
    mpm.params().poisson_ratio = 0.22f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.72f, -0.42f), vec2(-1.28f, -0.08f), sp,
                    MPMMaterial::MAG_SOFT_IRON, 300.0f, vec2(1, 0), 0.24f);
    register_scene_mpm_batch(creation, "Soft Iron Witness",
                             "Expected: the soft-iron witness should drift toward the pole but stay body-like, so you can compare its motion with the more fluid ferro splash in the middle.",
                             particles, before, MPMMaterial::MAG_SOFT_IRON,
                             54000.0f, 0.22f, 300.0f, vec2(1, 0), 0.24f);

    mpm.params().youngs_modulus = old_E;
    mpm.params().poisson_ratio = old_nu;
}

static void load_hybrid_oobleck_armor(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                      MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                      CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(-1.58f, -1.18f), vec2(0.34f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Lane A Plinth", "Impact lane for the oobleck-first shield.");
    sdf.add_box(vec2(0.04f, -1.18f), vec2(0.34f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Lane B Plinth", "Impact lane for the gel-first shield.");
    sdf.add_box(vec2(1.66f, -1.18f), vec2(0.34f, 0.06f),
                SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                "Lane C Plinth", "Control lane for a hard backer with no adaptive pad.");
    sdf.rebuild();

    mpm.params().enable_thermal = false;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 old_nu = mpm.params().poisson_ratio;
    u32 before = 0;

    mpm.params().youngs_modulus = 8600.0f;
    mpm.params().poisson_ratio = 0.33f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.92f, -1.12f), vec2(-1.54f, -0.56f), sp,
                    MPMMaterial::OOBLECK, 300.0f, vec2(1, 0), 1.0f,
                    vec4(0.02f, 0.18f, 0.88f, 0.02f));
    register_scene_mpm_batch(creation, "Oobleck Front Pad",
                             "Expected: the front pad should jam on impact and spare more of the backer than the bare control lane.",
                             particles, before, MPMMaterial::OOBLECK,
                             8600.0f, 0.33f, 300.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = 52000.0f;
    mpm.params().poisson_ratio = 0.24f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.50f, -1.08f), vec2(-1.24f, 0.08f), sp,
                    MPMMaterial::TOUGH, 300.0f, vec2(1, 0), 4.8f);
    register_scene_mpm_batch(creation, "Oobleck Backer",
                             "Expected: compare how much of this tougher backer survives versus the gel lane and the bare control lane after the hammers land.",
                             particles, before, MPMMaterial::TOUGH,
                             52000.0f, 0.24f, 300.0f, vec2(1, 0), 4.8f);

    mpm.params().youngs_modulus = 10400.0f;
    mpm.params().poisson_ratio = 0.35f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.30f, -1.12f), vec2(0.08f, -0.54f), sp,
                    MPMMaterial::IMPACT_GEL, 300.0f, vec2(1, 0), 1.0f,
                    vec4(0.02f, 0.16f, 0.92f, 0.02f));
    register_scene_mpm_batch(creation, "Gel Front Pad",
                             "Expected: the memory gel lane should keep more of the dent and impact history instead of only jamming for the instant of collision.",
                             particles, before, MPMMaterial::IMPACT_GEL,
                             10400.0f, 0.35f, 300.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = 52000.0f;
    mpm.params().poisson_ratio = 0.24f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.12f, -1.08f), vec2(0.38f, 0.08f), sp,
                    MPMMaterial::TOUGH, 300.0f, vec2(1, 0), 4.8f);
    register_scene_mpm_batch(creation, "Gel Backer",
                             "Expected: this backer should show a different failure pattern than the oobleck lane because the front pad remembers and redistributes later hits.",
                             particles, before, MPMMaterial::TOUGH,
                             52000.0f, 0.24f, 300.0f, vec2(1, 0), 4.8f);

    mpm.params().youngs_modulus = 52000.0f;
    mpm.params().poisson_ratio = 0.24f;
    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.56f, -1.08f), vec2(1.88f, 0.08f), sp,
                    MPMMaterial::TOUGH, 300.0f, vec2(1, 0), 4.8f);
    register_scene_mpm_batch(creation, "Bare Control Backer",
                             "Expected: this control lane should take the harshest direct hit because it has no adaptive pad in front of it.",
                             particles, before, MPMMaterial::TOUGH,
                             52000.0f, 0.24f, 300.0f, vec2(1, 0), 4.8f);

    mpm.params().youngs_modulus = 36000.0f;
    mpm.params().poisson_ratio = 0.24f;
    before = particles.range(SolverType::MPM).count;
    u32 drop_a_offset = particles.range(SolverType::MPM).offset + before;
    mpm.spawn_circle(particles, vec2(-1.40f, 1.18f), 0.16f, sp,
                     MPMMaterial::THERMO_METAL, 300.0f, vec2(1, 0), 7.2f);
    u32 drop_a_count = particles.range(SolverType::MPM).count - before;
    set_mpm_batch_velocity(particles, drop_a_offset, drop_a_count, vec2(0.0f, -11.0f));
    register_scene_mpm_batch(creation, "Oobleck Hammer",
                             "Dense striker for the oobleck lane.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             36000.0f, 0.24f, 300.0f, vec2(1, 0), 7.2f);

    before = particles.range(SolverType::MPM).count;
    u32 drop_b_offset = particles.range(SolverType::MPM).offset + before;
    mpm.spawn_circle(particles, vec2(0.04f, 1.26f), 0.16f, sp,
                     MPMMaterial::THERMO_METAL, 300.0f, vec2(1, 0), 7.2f);
    u32 drop_b_count = particles.range(SolverType::MPM).count - before;
    set_mpm_batch_velocity(particles, drop_b_offset, drop_b_count, vec2(0.0f, -11.0f));
    register_scene_mpm_batch(creation, "Gel Hammer",
                             "Dense striker for the gel lane.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             36000.0f, 0.24f, 300.0f, vec2(1, 0), 7.2f);

    before = particles.range(SolverType::MPM).count;
    u32 drop_c_offset = particles.range(SolverType::MPM).offset + before;
    mpm.spawn_circle(particles, vec2(1.72f, 1.22f), 0.16f, sp,
                     MPMMaterial::THERMO_METAL, 300.0f, vec2(1, 0), 7.2f);
    u32 drop_c_count = particles.range(SolverType::MPM).count - before;
    set_mpm_batch_velocity(particles, drop_c_offset, drop_c_count, vec2(0.0f, -11.0f));
    register_scene_mpm_batch(creation, "Control Hammer",
                             "Same striker for the bare control lane.",
                             particles, before, MPMMaterial::THERMO_METAL,
                             36000.0f, 0.24f, 300.0f, vec2(1, 0), 7.2f);

    mpm.params().youngs_modulus = old_E;
    mpm.params().poisson_ratio = old_nu;
}

// ================================================================
// Fix-verification test scenes. Each one isolates one of the Tier S/A/B/C
// physics fixes so you can eyeball before/after behavior without a full
// regression run.
// ================================================================

// Tests S1 (regrow advancement), S2 (ash softens instead of stiffens), B3 (ash
// fracture sensitivity). A hot column should char to ash, crumble visibly under
// its own weight and a drop ball, then regrow from its cooler top after the
// heat source is released. The witness column on the cool shelf provides a
// side-by-side reference.
static void load_fix_test_bio_heal(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                   MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                   CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(-1.10f, -1.12f), vec2(0.90f, 0.06f),
                SDFField::MaterialPresetID::BRASS_HEAT_SINK,
                "Hot Brass Bed", "Primary heat-soak plate beneath the test column. Expected: the column should ash from the bottom up.");
    sdf.add_box(vec2(1.55f, 0.30f), vec2(0.48f, 0.06f),
                SDFField::MaterialPresetID::SILVER_CONDUCTIVE,
                "Cool Witness Shelf", "Raised shelf that should stay much cooler so the witness column keeps its green/alive color.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(-0.20f, -1.05f);
    mpm.params().heat_source_radius = 0.60f;
    mpm.params().heat_source_temp = 960.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;
    const f32 old_nu = mpm.params().poisson_ratio;
    const f32 old_fiber = mpm.params().fiber_strength;

    mpm.params().youngs_modulus = 10800.0f;
    mpm.params().poisson_ratio = 0.34f;
    mpm.params().fiber_strength = 2.4f;
    u32 before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-0.36f, -0.92f), vec2(-0.04f, 0.30f), sp,
                    MPMMaterial::ASH_REGROWTH, 316.0f, vec2(0, 1), 1.0f);
    register_scene_mpm_batch(creation, "Ash Column (Hot)",
                             "S1+S2+B3: bottom should char into weak crumbly ash, middle should soften, top should regrow once the heat gun is released. Ash should now feel genuinely brittle instead of oddly stiff.",
                             particles, before, MPMMaterial::ASH_REGROWTH,
                             10800.0f, 0.34f, 316.0f, vec2(0, 1), 1.0f);

    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(1.38f, 0.40f), vec2(1.72f, 1.12f), sp,
                    MPMMaterial::ASH_REGROWTH, 308.0f, vec2(0, 1), 1.0f);
    register_scene_mpm_batch(creation, "Ash Witness (Cool)",
                             "Reference column that should stay mostly alive and green. Use it to judge how much color restoration you see on the hot column after recovery.",
                             particles, before, MPMMaterial::ASH_REGROWTH,
                             10800.0f, 0.34f, 308.0f, vec2(0, 1), 1.0f);

    mpm.params().youngs_modulus = old_E;
    mpm.params().poisson_ratio = old_nu;
    mpm.params().fiber_strength = old_fiber;
}

// Tests A1 (saturating blast temp write) + A2 (combustion_hold 0.82->0.55). A
// small sealed charge in open air should detonate when heated; the resulting
// plume should cool back within ~1-2s rather than lingering for 3-5s. A ring
// of passive ceramic witnesses lets you eyeball whether heat sticks around
// them unnaturally.
static void load_fix_test_blast_cooldown(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                         MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                         CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_box(vec2(0.0f, -1.20f), vec2(0.30f, 0.06f),
                SDFField::MaterialPresetID::BRONZE_BALANCED,
                "Charge Plinth", "Low plinth so the blast front spreads cleanly to both sides for cooldown observation.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(0.0f, -1.12f);
    mpm.params().heat_source_radius = 0.20f;
    mpm.params().heat_source_temp = 820.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 14000.0f;
    u32 before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(0.0f, -1.04f), 0.16f, sp,
                     MPMMaterial::SEALED_CHARGE, 300.0f, vec2(1, 0), 1.0f);
    register_scene_mpm_batch(creation, "Sealed Charge",
                             "A1+A2: should rupture and push a pressure wave, then the residual plume should cool back within ~1-2s. If it glows for 4+ seconds the cap/cooling fix didn't take.",
                             particles, before, MPMMaterial::SEALED_CHARGE,
                             14000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 1.0f);

    mpm.params().youngs_modulus = 38000.0f;
    const vec2 ring_positions[6] = {
        vec2(-1.60f, -0.60f), vec2(-1.10f, 0.10f), vec2(-0.40f, 0.80f),
        vec2(0.60f, 0.80f),   vec2(1.30f, 0.10f),  vec2(1.80f, -0.60f)
    };
    for (int i = 0; i < 6; ++i) {
        before = particles.range(SolverType::MPM).count;
        mpm.spawn_circle(particles, ring_positions[i], 0.08f, sp,
                         MPMMaterial::CERAMIC, 300.0f, vec2(1, 0), 2.4f);
        register_scene_mpm_batch(creation, "Ceramic Witness",
                                 "Passive ceramic pebble at fixed radius. Use the set together to read whether the plume leaves a persistent hot-shell or cools away cleanly.",
                                 particles, before, MPMMaterial::CERAMIC,
                                 38000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 2.4f);
    }

    mpm.params().youngs_modulus = old_E;
}

// Tests A3 (per-material friction scale) + A4 (lower thresholds + KE-informed
// emit temp). A row of material strips with a heavy elastic ball sliding over
// them — all at modest speeds that the old thresholds would have silently
// zeroed. Metal and ceramic should now visibly heat, while wax and dough stay
// cool. Use G to drop heat if a strip doesn't flex enough to register.
static void load_fix_test_collision_heat(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                         MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                         CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf);
    sdf.add_segment(vec2(-2.35f, -0.60f), vec2(2.35f, -1.00f), 0.08f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Slide Ramp", "Tilted ramp so a dropped ball keeps tangential speed across the sample strips below.");
    sdf.rebuild();

    mpm.params().enable_thermal = true;
    mpm.params().heat_source_pos = vec2(-3.0f, -2.0f);
    mpm.params().heat_source_radius = 0.10f;
    mpm.params().heat_source_temp = 300.0f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;

    struct Strip {
        const char* label;
        const char* summary;
        MPMMaterial material;
        f32 E;
        f32 density;
        f32 x_center;
        vec4 color;
    };
    const Strip strips[5] = {
        {"Metal Strip",    "A3/A4: impact_heat_scale=2.2, should glow under even slow slides.", MPMMaterial::THERMO_METAL, 44000.0f, 6.8f, -1.70f, vec4(0.66f, 0.70f, 0.78f, 1.0f)},
        {"Ceramic Strip",  "A3/A4: impact_heat_scale=1.5, should warm visibly in the first couple of passes.", MPMMaterial::CERAMIC,     32000.0f, 2.6f, -0.85f, vec4(0.86f, 0.84f, 0.80f, 1.0f)},
        {"Stoneware Strip","A3/A4: stoneware family scale=1.5, middle reference between metal and ceramic.",   MPMMaterial::STONEWARE,   36000.0f, 3.0f,  0.00f, vec4(0.78f, 0.66f, 0.52f, 1.0f)},
        {"Memory Wax",     "A3/A4: memory wax has no per-material boost and should stay cool unless deeply compressed.", MPMMaterial::MEMORY_WAX, 9000.0f, 1.4f,  0.85f, vec4(0.94f, 0.80f, 0.60f, 1.0f)},
        {"Bread Strip",    "A3/A4: bread family explicitly dialed down (0.35x). Basically no heating expected from sliding.", MPMMaterial::BREAD, 12000.0f, 1.3f,  1.70f, vec4(0.90f, 0.78f, 0.52f, 1.0f)}
    };

    for (const Strip& s : strips) {
        mpm.params().youngs_modulus = s.E;
        u32 before = particles.range(SolverType::MPM).count;
        mpm.spawn_block(particles, vec2(s.x_center - 0.30f, -1.20f),
                        vec2(s.x_center + 0.30f, -1.02f), sp,
                        s.material, 300.0f, vec2(1, 0), s.density);
        register_scene_mpm_batch(creation, s.label, s.summary,
                                 particles, before, s.material,
                                 s.E, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), s.density);
    }

    // Heavy elastic drop ball, staged above the ramp's left edge so it slides
    // across all five strips.
    mpm.params().youngs_modulus = 80000.0f;
    u32 before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, vec2(-2.10f, 0.60f), 0.18f, sp,
                     MPMMaterial::ELASTIC, 300.0f, vec2(1, 0), 9.0f);
    register_scene_mpm_batch(creation, "Slide Impactor",
                             "Heavy elastic ball. Falls onto the ramp and slides across all five strips so you can compare per-material heating in one run.",
                             particles, before, MPMMaterial::ELASTIC,
                             80000.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0), 9.0f);

    mpm.params().youngs_modulus = old_E;
}

// Tests C1 (demagnetizing-field feedback). A wide shallow ferrofluid puddle
// over a horizontal bar magnet. The spike pattern should self-regulate in
// spacing. A deep central pool provides a taller-spike comparison.
static void load_fix_test_ferro_demag(ParticleBuffer& particles, SPHSolver& /*sph*/,
                                      MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                                      CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-2.15f, -1.30f), vec2(2.15f, -1.30f), 0.10f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Bench Plate", "Flat bronze plate under the ferro puddles.");
    sdf.add_box(vec2(0.0f, -1.50f), vec2(1.80f, 0.10f),
                SDFField::MaterialPresetID::MAGNET_X,
                "Bar Magnet", "Permanent horizontal magnet buried just below the bench. Expected: both ferro puddles above should form clearly separated spike combs.");
    sdf.rebuild();

    mpm.params().enable_thermal = false;
    mpm.params().magnet_radius = 0.70f;
    mpm.params().magnet_spike_strength = 1.35f;

    const f32 sp = grid.dx() * 0.5f;
    const f32 old_E = mpm.params().youngs_modulus;

    mpm.params().youngs_modulus = 5200.0f;
    u32 before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(-1.80f, -1.20f), vec2(-0.10f, -0.92f), sp,
                    MPMMaterial::FERRO_FLUID, 300.0f);
    register_scene_mpm_batch(creation, "Wide Shallow Puddle",
                             "C1: demag feedback should keep spikes from packing onto a single point. Expect a regular comb of ~4-6 low peaks across this strip.",
                             particles, before, MPMMaterial::FERRO_FLUID,
                             5200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, vec2(0.30f, -1.20f), vec2(1.00f, -0.60f), sp,
                    MPMMaterial::FERRO_FLUID, 300.0f);
    register_scene_mpm_batch(creation, "Deep Central Pool",
                             "Tall pool for comparing peak height. Spikes should be taller but still finitely spaced. If spikes collapse into a single tower, the demag feedback isn't firing.",
                             particles, before, MPMMaterial::FERRO_FLUID,
                             5200.0f, mpm.params().poisson_ratio, 300.0f, vec2(1, 0));

    mpm.params().youngs_modulus = old_E;
}

// Tests C6 (symmetric SPH thermal weighting). A single cup with a hot layer
// above a cold layer of the same fluid — they should now actually equilibrate
// toward a single warm temperature. Before the fix, the heavier low-rho
// particles heated faster and the pool never reached a single temperature.
static void load_fix_test_sph_equil(ParticleBuffer& particles, SPHSolver& sph,
                                    MPMSolver& mpm, UniformGrid& /*grid*/, SDFField& sdf,
                                    CreationState* creation) {
    sdf.clear();
    add_floor_and_walls(sdf, SDFField::MaterialPresetID::BRONZE_BALANCED);
    sdf.add_segment(vec2(-0.80f, -1.30f), vec2(-0.80f, 0.10f), 0.06f,
                    SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Cup Left", "Left wall of the mixing cup.");
    sdf.add_segment(vec2(0.80f, -1.30f), vec2(0.80f, 0.10f), 0.06f,
                    SDFField::MaterialPresetID::ROSE_GOLD_LIGHT,
                    "Cup Right", "Right wall of the mixing cup.");
    sdf.rebuild();

    SPHParams sph_p = sph.params();
    sph_p.enable_thermal = true;
    sph_p.surface_tension = 0.80f;
    sph_p.codim_enabled = false;
    sph.set_params(sph_p);

    mpm.params().enable_thermal = false;

    const f32 sp = sph.params().smoothing_radius * 0.5f;

    // Cold base layer (below).
    u32 before = particles.range(SolverType::SPH).count;
    sph.spawn_block(particles, vec2(-0.70f, -1.22f), vec2(0.70f, -0.60f), sp);
    u32 off = particles.range(SolverType::SPH).offset + before;
    u32 cnt = particles.range(SolverType::SPH).count - before;
    apply_sph_batch_properties(particles, sph, off, cnt,
                               MPMMaterial::SPH_WATER, 282.0f, 8.0f, 0.10f,
                               vec4(0.20f, 0.46f, 0.96f, 1.0f));
    register_scene_sph_batch(creation, "Cold Water Base",
                             "C6: starts at 282K. Should warm toward the middle temperature (~320K) as the hot layer above it mixes in, not stay stuck cold.",
                             particles, before, vec4(0.20f, 0.46f, 0.96f, 1.0f),
                             0.42f, "Fill the lower half of the cup.",
                             MPMMaterial::SPH_WATER, 282.0f);

    // Hot upper layer.
    before = particles.range(SolverType::SPH).count;
    sph.spawn_block(particles, vec2(-0.70f, -0.58f), vec2(0.70f, 0.02f), sp);
    off = particles.range(SolverType::SPH).offset + before;
    cnt = particles.range(SolverType::SPH).count - before;
    apply_sph_batch_properties(particles, sph, off, cnt,
                               MPMMaterial::SPH_WATER, 360.0f, 8.0f, 0.10f,
                               vec4(0.98f, 0.40f, 0.22f, 1.0f));
    register_scene_sph_batch(creation, "Hot Water Cap",
                             "C6: starts at 360K. Should cool toward the middle temperature as it mixes with the cold base. If the cap stays visibly hot while the base stays visibly cold, the symmetric weighting fix didn't take.",
                             particles, before, vec4(0.98f, 0.40f, 0.22f, 1.0f),
                             0.42f, "Fill the upper half of the cup.",
                             MPMMaterial::SPH_WATER, 360.0f);
}

// ---- dispatch ----
void load_scene(SceneID id, ParticleBuffer& particles, SPHSolver& sph,
                MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                CreationState* creation) {
    LOG_INFO("Loading scene: %s", scene_names[static_cast<u32>(id)]);
    reset_creation(creation);
    switch (id) {
    case SceneID::DEFAULT:          load_default(particles, sph, mpm, grid, sdf, creation); break;
    case SceneID::THERMAL_FURNACE:  load_thermal_furnace(particles, sph, mpm, grid, sdf, creation); break;
    case SceneID::FRACTURE_TEST:    load_fracture_test(particles, sph, mpm, grid, sdf, creation); break;
    case SceneID::MELTING:          load_melting(particles, sph, mpm, grid, sdf, creation); break;
    case SceneID::DAM_BREAK:        load_dam_break(particles, sph, mpm, grid, sdf, creation); break;
    case SceneID::STIFF_OBJECTS:    load_stiff_objects(particles, sph, mpm, grid, sdf, creation); break;
    case SceneID::HEAT_RAMP:        load_heat_ramp(particles, sph, mpm, grid, sdf, creation); break;
    case SceneID::FIRE_FORGE:       load_fire_forge(particles, sph, mpm, grid, sdf, creation); break;
    case SceneID::CODIM_THREADS:    load_codim_threads(particles, sph, mpm, grid, sdf, creation); break;
    case SceneID::EMPTY_BOX: {
        sdf.clear();
        add_floor_and_walls(sdf);
        sdf.rebuild();
        break;
    }
    case SceneID::BOX_HEAT_IN_AIR: {
        // Box with heat source floating IN AIR (not touching walls)
        // Tests if heating the box walls causes artifacts
        sdf.clear();
        add_floor_and_walls(sdf);
        sdf.rebuild();
        mpm.params().enable_thermal = true;
        mpm.params().heat_source_pos = vec2(0.0f, 0.5f); // In the air, away from walls
        mpm.params().heat_source_radius = 0.4f;
        mpm.params().heat_source_temp = 800.0f;
        break;
    }
    case SceneID::HEAT_NO_WALLS: {
        // No geometry at all, just heat source floating in space
        sdf.clear();
        sdf.rebuild();
        mpm.params().enable_thermal = true;
        mpm.params().heat_source_pos = vec2(0.0f, 0.0f);
        mpm.params().heat_source_radius = 0.5f;
        mpm.params().heat_source_temp = 800.0f;
        break;
    }
    case SceneID::WIND_TUNNEL: {
        // No heat sources. Obstacles for testing airflow.
        sdf.clear();
        add_floor_and_walls(sdf);
        // Obstacles for interesting flow patterns
        sdf.add_circle(vec2(0.0f, 0.0f), 0.3f);                       // Central cylinder
        sdf.add_box(vec2(-1.0f, -0.5f), vec2(0.15f, 0.4f));           // Vertical plate
        sdf.add_segment(vec2(0.8f, -0.5f), vec2(1.5f, 0.2f), 0.06f);  // Angled ramp
        sdf.rebuild();
        mpm.params().enable_thermal = false;
        break;
    }
    case SceneID::OVEN_OPEN:
    case SceneID::OVEN_OPEN_WIND:
        load_open_oven(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::POT_HEATER:
    case SceneID::POT_HEATER_WIND:
        load_pot_heater(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::CODIM_THREADS_COLD:
        load_codim_threads_cold(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::ZERO_G_SOFT_LAB:
        load_zero_g_soft_lab(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::THERMAL_BRIDGE:
        load_thermal_bridge(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::SPIRAL_METALS:
        load_spiral_metals(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::THERMAL_BRIDGE_STRONG:
        load_thermal_bridge_strong(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::THIN_PIPE:
        load_thin_pipe(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::EMPTY_ZERO_G:
        load_empty_zero_g(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::FLOOR_ONLY:
        load_floor_only(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::GLAZE_KILN:
        load_glaze_kiln(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::BAKE_OVEN:
        load_bake_oven(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::STRESS_FORGE:
        load_stress_forge(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::REACTIVE_HEARTH:
        load_reactive_hearth(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::GLAZE_RACK:
        load_glaze_rack(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::STEAM_OVEN:
        load_steam_oven(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::SEED_ROASTER:
        load_seed_roaster(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::CRAZING_SHELF:
        load_crazing_shelf(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::POCKET_OVEN:
        load_pocket_oven(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::TEMPERING_BENCH:
        load_tempering_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::PRESSURE_PANTRY:
        load_pressure_pantry(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::ANISO_TEAR_BENCH:
        load_aniso_tear_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::ANISO_BEND_BENCH:
        load_aniso_bend_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::POROUS_BAKE_BENCH:
        load_porous_bake_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::POTTERY_BENCH:
        load_pottery_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::PASTRY_BENCH:
        load_pastry_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::STONEWARE_BENCH:
        load_stoneware_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::ANISO_STRONG_BENCH:
        load_aniso_strong_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::ADV_BAKE_BENCH:
        load_adv_bake_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::ADV_KILN_BENCH:
        load_adv_kiln_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::OPEN_CRUMB_BENCH:
        load_open_crumb_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::SINTER_LOCK_BENCH:
        load_sinter_lock_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::MOISTURE_BINDER_BENCH:
        load_moisture_binder_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::BURNOUT_POTTERY_BENCH:
        load_burnout_pottery_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::VENTED_SKIN_BENCH:
        load_vented_skin_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::SPH_THERMAL_BENCH:
        load_sph_thermal_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::OIL_OVER_WATER:
        load_oil_over_water(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::FERRO_SPIKE_BENCH:
        load_ferro_spike_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::MAGNETIC_BENCH:
        load_magnetic_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::MAGNETIC_CLIMB_BENCH:
        load_magnetic_climb_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::MAGNETIC_FLOOR_BENCH:
        load_magnetic_floor_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::RIGID_MAGNETIC_FLOOR:
        load_rigid_magnetic_floor(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::MAG_CURSOR_UNIT:
        load_mag_cursor_unit(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::MAG_PERMANENT_POLE:
        load_mag_permanent_pole(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::MAG_SOFT_IRON_FIELD:
        load_mag_soft_iron_field(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::MAG_SOFT_IRON_BODY:
        load_mag_soft_iron_body(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::OOBLECK_IMPACTOR_BENCH:
        load_oobleck_impactor_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::IMPACT_MEMORY_BENCH:
        load_impact_memory_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::BLAST_ARMOR_LANE:
        load_blast_armor_lane(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::BREACH_CHAMBER:
        load_breach_chamber(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::SPALL_PLATE_BENCH:
        load_spall_plate_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::OPEN_BLAST_RANGE_XL:
        load_open_blast_range_xl(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::BREACH_HALL_XL:
        load_breach_hall_xl(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::SPALL_GALLERY_XL:
        load_spall_gallery_xl(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::BIO_REPLICATOR_BENCH:
        load_bio_replicator_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::MYCELIUM_MORPH_BENCH:
        load_mycelium_morph_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::MORPHOGENESIS_BENCH:
        load_morphogenesis_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::ROOT_GARDEN_BENCH:
        load_root_garden_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::CELL_COLONY_BENCH:
        load_cell_colony_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::AUTOMATA_AIR_COUPLING_BENCH:
        load_automata_air_coupling_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::AUTOMATA_FIRE_REGROWTH_BENCH:
        load_automata_fire_regrowth_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::AUTOMATA_MAX_COUPLING_BENCH:
        load_automata_air_coupling_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::ASH_REGROWTH_BENCH:
        load_ash_regrowth_bench(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::FOOT_DEMO_BENCH:
        load_foot_demo_scene(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::HYBRID_REGROWTH_WALL:
        load_hybrid_regrowth_wall(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::HYBRID_KILN_PROCESS:
        load_hybrid_kiln_process(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::HYBRID_SOFT_HEAT_RANGE:
        load_hybrid_soft_heat_range(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::HYBRID_PRESSURE_POTTERY:
        load_hybrid_pressure_pottery(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::HYBRID_FERRO_SPLASH:
        load_hybrid_ferro_splash(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::HYBRID_OOBLECK_ARMOR:
        load_hybrid_oobleck_armor(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::THERMAL_VERIFY_SDF_JUNCTION:
        load_thermal_verify_sdf_junction(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::THERMAL_VERIFY_HOT_BLOCKS:
        load_thermal_verify_hot_blocks(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::THERMAL_VERIFY_CROSS_IGNITION:
        load_thermal_verify_cross_ignition(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::THERMAL_VERIFY_BRIDGE_WITNESS:
        load_thermal_verify_bridge_witness(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::THERMAL_VERIFY_IMPACT_RINGDOWN:
        load_thermal_verify_impact_ringdown(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::FIX_TEST_BIO_HEAL:
        load_fix_test_bio_heal(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::FIX_TEST_BLAST_COOLDOWN:
        load_fix_test_blast_cooldown(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::FIX_TEST_COLLISION_HEAT:
        load_fix_test_collision_heat(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::FIX_TEST_FERRO_DEMAG:
        load_fix_test_ferro_demag(particles, sph, mpm, grid, sdf, creation);
        break;
    case SceneID::FIX_TEST_SPH_EQUIL:
        load_fix_test_sph_equil(particles, sph, mpm, grid, sdf, creation);
        break;
    }
}

} // namespace ng
