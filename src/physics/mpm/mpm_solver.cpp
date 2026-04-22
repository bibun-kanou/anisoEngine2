#include "physics/mpm/mpm_solver.h"
#include "physics/magnetic/magnetic_field.h"
#include "physics/sdf/sdf_field.h"
#include "physics/eulerian/euler_fluid.h"
#include "core/log.h"

#include <glad/gl.h>
#include <vector>
#include <cmath>

namespace ng {

vec4 default_thermal_coupling(MPMMaterial material) {
    switch (material) {
        case MPMMaterial::BURNING: return vec4(0.95f, 1.18f, 0.46f, 0.28f);
        case MPMMaterial::EMBER: return vec4(0.90f, 1.35f, 0.16f, 0.96f);
        case MPMMaterial::FLAMMABLE_FLUID: return vec4(1.15f, 1.1f, 0.48f, 0.36f);
        case MPMMaterial::FIRECRACKER: return vec4(1.35f, 1.45f, 0.70f, 0.08f);
        case MPMMaterial::REACTIVE_BURN: return vec4(1.25f, 1.25f, 0.60f, 0.18f);
        case MPMMaterial::ANISO: return vec4(1.0f, 1.0f, 0.70f, 0.16f);
        case MPMMaterial::COMPOSITE: return vec4(1.0f, 1.0f, 0.62f, 0.14f);
        case MPMMaterial::BREAD:
        case MPMMaterial::CRUST_DOUGH:
        case MPMMaterial::STEAM_BUN:
            return vec4(1.0f, 1.0f, 0.80f, 0.28f);
        case MPMMaterial::CHEESE_PULL:
            return vec4(1.0f, 1.0f, 0.72f, 0.18f);
        case MPMMaterial::GLASS:
        case MPMMaterial::FILAMENT_GLASS:
            return vec4(1.0f, 1.0f, 0.82f, 0.08f);
        case MPMMaterial::THERMO_METAL:
            return vec4(1.0f, 1.0f, 0.88f, 0.05f);
        case MPMMaterial::MEMORY_WAX:
            return vec4(0.10f, 0.85f, 0.72f, 0.04f);
        case MPMMaterial::FERRO_FLUID:
            return vec4(0.00f, 0.75f, 0.72f, 0.06f);
        case MPMMaterial::MAILLARD:
            return vec4(0.30f, 0.65f, 0.60f, 0.08f);
        case MPMMaterial::MUSHROOM:
            return vec4(0.60f, 0.35f, 0.76f, 0.22f);
        case MPMMaterial::CRUMB_LOAF:
            return vec4(0.95f, 0.72f, 0.78f, 0.12f);
        case MPMMaterial::BISQUE:
            return vec4(0.55f, 0.82f, 0.78f, 0.06f);
        case MPMMaterial::TEAR_SKIN:
            return vec4(0.75f, 0.70f, 0.82f, 0.10f);
        case MPMMaterial::LAMINATED_PASTRY:
            return vec4(0.82f, 0.68f, 0.80f, 0.10f);
        case MPMMaterial::STONEWARE:
            return vec4(0.34f, 0.88f, 0.76f, 0.04f);
        case MPMMaterial::ORTHO_BEND:
            return vec4(1.0f, 1.0f, 0.82f, 0.08f);
        case MPMMaterial::ORTHO_TEAR:
            return vec4(0.96f, 0.90f, 0.86f, 0.08f);
        case MPMMaterial::VENT_CRUMB:
            return vec4(0.82f, 0.68f, 0.84f, 0.08f);
        case MPMMaterial::VITREOUS_CLAY:
            return vec4(0.28f, 0.94f, 0.78f, 0.03f);
        case MPMMaterial::BLISTER_GLAZE:
            return vec4(0.62f, 0.84f, 0.80f, 0.05f);
        case MPMMaterial::OPEN_CRUMB:
            return vec4(0.88f, 0.60f, 0.88f, 0.08f);
        case MPMMaterial::SINTER_LOCK:
            return vec4(0.24f, 0.98f, 0.82f, 0.02f);
        case MPMMaterial::BINDER_CRUMB:
            return vec4(0.78f, 0.74f, 0.84f, 0.08f);
        case MPMMaterial::CHANNEL_CRUMB:
            return vec4(0.86f, 0.66f, 0.88f, 0.08f);
        case MPMMaterial::BURNOUT_CLAY:
            return vec4(0.42f, 0.90f, 0.80f, 0.04f);
        case MPMMaterial::VENTED_SKIN:
            return vec4(0.70f, 0.72f, 0.82f, 0.06f);
        case MPMMaterial::SPH_WATER:
            return vec4(0.30f, 0.22f, 0.98f, 0.08f);
        case MPMMaterial::SPH_VISCOUS_GOO:
            return vec4(0.10f, 0.20f, 0.95f, 0.05f);
        case MPMMaterial::SPH_LIGHT_OIL:
            return vec4(0.22f, 0.34f, 0.88f, 0.12f);
        case MPMMaterial::SPH_BURNING_OIL:
            return vec4(0.58f, 1.28f, 0.42f, 0.30f);
        case MPMMaterial::SPH_BOILING_WATER:
            return vec4(1.05f, 0.38f, 0.82f, 0.20f);
        case MPMMaterial::SPH_THERMAL_SYRUP:
            return vec4(0.18f, 0.18f, 0.84f, 0.06f);
        case MPMMaterial::SPH_FLASH_FLUID:
            return vec4(1.30f, 0.56f, 0.40f, 0.42f);
        case MPMMaterial::MAG_SOFT_IRON:
            return vec4(0.02f, 0.95f, 0.86f, 0.01f);
        case MPMMaterial::MAGNETIC_RUBBER:
            return vec4(0.02f, 0.92f, 0.84f, 0.02f);
        case MPMMaterial::TOPO_GOO:
            return vec4(0.04f, 0.28f, 0.92f, 0.04f);
        case MPMMaterial::OOBLECK:
            return vec4(0.02f, 0.20f, 0.88f, 0.02f);
        case MPMMaterial::IMPACT_GEL:
            return vec4(0.02f, 0.18f, 0.90f, 0.03f);
        case MPMMaterial::SEALED_CHARGE:
            return vec4(1.55f, 1.70f, 0.60f, 0.03f);
        case MPMMaterial::MORPH_TISSUE:
            return vec4(0.18f, 0.34f, 0.88f, 0.10f);
        case MPMMaterial::ROOT_WEAVE:
            return vec4(0.12f, 0.24f, 0.92f, 0.05f);
        case MPMMaterial::CELL_SHEET:
            return vec4(0.16f, 0.28f, 0.88f, 0.08f);
        case MPMMaterial::ASH_REGROWTH:
            return vec4(0.08f, 0.18f, 0.92f, 0.04f);
        default:
            return vec4(1.0f, 1.0f, 1.0f, 0.18f);
    }
}

static vec4 material_spawn_color(MPMMaterial material) {
    switch (material) {
        case MPMMaterial::FLUID:   return vec4(0.1f, 0.3f, 0.8f, 1.0f);
        case MPMMaterial::ELASTIC: return vec4(0.2f, 0.7f, 0.3f, 1.0f);
        case MPMMaterial::SNOW:    return vec4(0.85f, 0.88f, 0.95f, 1.0f);
        case MPMMaterial::ANISO:   return vec4(0.8f, 0.5f, 0.2f, 1.0f);
        case MPMMaterial::THERMAL: return vec4(0.8f, 0.6f, 0.3f, 1.0f);
        case MPMMaterial::FRACTURE:return vec4(0.6f, 0.4f, 0.15f, 1.0f);
        case MPMMaterial::PHASE:   return vec4(0.3f, 0.6f, 0.9f, 1.0f);
        case MPMMaterial::BURNING: return vec4(0.5f, 0.3f, 0.15f, 1.0f);
        case MPMMaterial::EMBER:   return vec4(0.8f, 0.5f, 0.2f, 1.0f);
        case MPMMaterial::HARDEN:  return vec4(0.5f, 0.5f, 0.6f, 1.0f);
        case MPMMaterial::CERAMIC: return vec4(0.85f, 0.8f, 0.75f, 1.0f);
        case MPMMaterial::COMPOSITE:return vec4(0.6f, 0.45f, 0.3f, 1.0f);
        case MPMMaterial::BRITTLE: return vec4(0.62f, 0.52f, 0.42f, 1.0f);
        case MPMMaterial::TOUGH:   return vec4(0.42f, 0.3f, 0.2f, 1.0f);
        case MPMMaterial::GLASS:   return vec4(0.72f, 0.82f, 0.9f, 1.0f);
        case MPMMaterial::BLOOM:   return vec4(0.88f, 0.62f, 0.44f, 1.0f);
        case MPMMaterial::FLAMMABLE_FLUID: return vec4(0.78f, 0.62f, 0.18f, 1.0f);
        case MPMMaterial::FOAM:    return vec4(0.78f, 0.76f, 0.64f, 1.0f);
        case MPMMaterial::SPLINTER:return vec4(0.84f, 0.68f, 0.42f, 1.0f);
        case MPMMaterial::BREAD:   return vec4(0.86f, 0.74f, 0.50f, 1.0f);
        case MPMMaterial::PUFF_CLAY:return vec4(0.82f, 0.66f, 0.48f, 1.0f);
        case MPMMaterial::FIRECRACKER:return vec4(0.82f, 0.16f, 0.12f, 1.0f);
        case MPMMaterial::GLAZE_CLAY:return vec4(0.92f, 0.78f, 0.64f, 1.0f);
        case MPMMaterial::CRUST_DOUGH:return vec4(0.90f, 0.76f, 0.52f, 1.0f);
        case MPMMaterial::THERMO_METAL:return vec4(0.62f, 0.64f, 0.70f, 1.0f);
        case MPMMaterial::REACTIVE_BURN:return vec4(0.56f, 0.20f, 0.12f, 1.0f);
        case MPMMaterial::GLAZE_DRIP:return vec4(0.96f, 0.82f, 0.62f, 1.0f);
        case MPMMaterial::STEAM_BUN:return vec4(0.96f, 0.84f, 0.58f, 1.0f);
        case MPMMaterial::FILAMENT_GLASS:return vec4(0.88f, 0.92f, 1.0f, 1.0f);
        case MPMMaterial::CHEESE_PULL:return vec4(0.98f, 0.84f, 0.44f, 1.0f);
        case MPMMaterial::MEMORY_WAX:return vec4(0.92f, 0.76f, 0.42f, 1.0f);
        case MPMMaterial::FERRO_FLUID:return vec4(0.95f, 0.55f, 0.15f, 1.0f);
        case MPMMaterial::MAILLARD:return vec4(0.96f, 0.84f, 0.54f, 1.0f);
        case MPMMaterial::MUSHROOM:return vec4(0.72f, 0.68f, 0.52f, 1.0f);
        case MPMMaterial::CRUMB_LOAF:return vec4(0.95f, 0.80f, 0.54f, 1.0f);
        case MPMMaterial::BISQUE:return vec4(0.90f, 0.78f, 0.68f, 1.0f);
        case MPMMaterial::TEAR_SKIN:return vec4(0.96f, 0.88f, 0.68f, 1.0f);
        case MPMMaterial::LAMINATED_PASTRY:return vec4(0.98f, 0.86f, 0.58f, 1.0f);
        case MPMMaterial::STONEWARE:return vec4(0.72f, 0.62f, 0.56f, 1.0f);
        case MPMMaterial::ORTHO_BEND:return vec4(0.78f, 0.56f, 0.26f, 1.0f);
        case MPMMaterial::ORTHO_TEAR:return vec4(0.86f, 0.64f, 0.34f, 1.0f);
        case MPMMaterial::VENT_CRUMB:return vec4(0.97f, 0.82f, 0.58f, 1.0f);
        case MPMMaterial::VITREOUS_CLAY:return vec4(0.74f, 0.66f, 0.60f, 1.0f);
        case MPMMaterial::BLISTER_GLAZE:return vec4(0.98f, 0.84f, 0.64f, 1.0f);
        case MPMMaterial::OPEN_CRUMB:return vec4(0.99f, 0.84f, 0.62f, 1.0f);
        case MPMMaterial::SINTER_LOCK:return vec4(0.68f, 0.60f, 0.56f, 1.0f);
        case MPMMaterial::BINDER_CRUMB:return vec4(0.97f, 0.84f, 0.60f, 1.0f);
        case MPMMaterial::CHANNEL_CRUMB:return vec4(0.98f, 0.86f, 0.64f, 1.0f);
        case MPMMaterial::BURNOUT_CLAY:return vec4(0.88f, 0.76f, 0.66f, 1.0f);
        case MPMMaterial::VENTED_SKIN:return vec4(0.96f, 0.90f, 0.72f, 1.0f);
        case MPMMaterial::SPH_WATER:return vec4(0.15f, 0.40f, 0.85f, 1.0f);
        case MPMMaterial::SPH_VISCOUS_GOO:return vec4(0.30f, 0.80f, 0.20f, 1.0f);
        case MPMMaterial::SPH_LIGHT_OIL:return vec4(0.70f, 0.60f, 0.20f, 1.0f);
        case MPMMaterial::SPH_BURNING_OIL:return vec4(0.70f, 0.46f, 0.10f, 1.0f);
        case MPMMaterial::SPH_BOILING_WATER:return vec4(0.44f, 0.72f, 0.98f, 1.0f);
        case MPMMaterial::SPH_THERMAL_SYRUP:return vec4(0.86f, 0.54f, 0.22f, 1.0f);
        case MPMMaterial::SPH_FLASH_FLUID:return vec4(0.82f, 0.72f, 0.50f, 1.0f);
        case MPMMaterial::MAG_SOFT_IRON:return vec4(0.56f, 0.60f, 0.66f, 1.0f);
        case MPMMaterial::MAGNETIC_RUBBER:return vec4(0.44f, 0.48f, 0.56f, 1.0f);
        case MPMMaterial::TOPO_GOO:return vec4(0.18f, 0.78f, 0.58f, 1.0f);
        case MPMMaterial::OOBLECK:return vec4(0.76f, 0.68f, 0.46f, 1.0f);
        case MPMMaterial::IMPACT_GEL:return vec4(0.66f, 0.46f, 0.88f, 1.0f);
        case MPMMaterial::SEALED_CHARGE:return vec4(0.70f, 0.16f, 0.10f, 1.0f);
        case MPMMaterial::MORPH_TISSUE:return vec4(0.30f, 0.76f, 0.56f, 1.0f);
        case MPMMaterial::ROOT_WEAVE:return vec4(0.40f, 0.72f, 0.30f, 1.0f);
        case MPMMaterial::CELL_SHEET:return vec4(0.92f, 0.60f, 0.70f, 1.0f);
        case MPMMaterial::ASH_REGROWTH:return vec4(0.96f, 0.97f, 0.94f, 1.0f);
    }
    return vec4(0.5f);
}

static bool material_uses_zero_jp(MPMMaterial material) {
    return material == MPMMaterial::HARDEN ||
           material == MPMMaterial::CERAMIC ||
           material == MPMMaterial::COMPOSITE ||
           material == MPMMaterial::BLOOM ||
           material == MPMMaterial::SPLINTER ||
           material == MPMMaterial::FOAM ||
           material == MPMMaterial::BREAD ||
           material == MPMMaterial::PUFF_CLAY ||
           material == MPMMaterial::FIRECRACKER ||
           material == MPMMaterial::GLAZE_CLAY ||
           material == MPMMaterial::CRUST_DOUGH ||
           material == MPMMaterial::THERMO_METAL ||
           material == MPMMaterial::REACTIVE_BURN ||
           material == MPMMaterial::GLAZE_DRIP ||
           material == MPMMaterial::STEAM_BUN ||
           material == MPMMaterial::FILAMENT_GLASS ||
           material == MPMMaterial::CHEESE_PULL ||
           material == MPMMaterial::CRUMB_LOAF ||
           material == MPMMaterial::BISQUE ||
           material == MPMMaterial::TEAR_SKIN ||
           material == MPMMaterial::LAMINATED_PASTRY ||
           material == MPMMaterial::STONEWARE ||
           material == MPMMaterial::ORTHO_TEAR ||
           material == MPMMaterial::VENT_CRUMB ||
           material == MPMMaterial::VITREOUS_CLAY ||
           material == MPMMaterial::BLISTER_GLAZE ||
           material == MPMMaterial::OPEN_CRUMB ||
           material == MPMMaterial::SINTER_LOCK ||
           material == MPMMaterial::BINDER_CRUMB ||
           material == MPMMaterial::CHANNEL_CRUMB ||
           material == MPMMaterial::BURNOUT_CLAY ||
           material == MPMMaterial::VENTED_SKIN ||
           material == MPMMaterial::TOPO_GOO ||
           material == MPMMaterial::OOBLECK ||
           material == MPMMaterial::IMPACT_GEL ||
           material == MPMMaterial::SEALED_CHARGE ||
           material == MPMMaterial::MORPH_TISSUE ||
           material == MPMMaterial::ROOT_WEAVE ||
           material == MPMMaterial::CELL_SHEET ||
           material == MPMMaterial::ASH_REGROWTH;
}

static float material_initial_phase(MPMMaterial material) {
    switch (material) {
        case MPMMaterial::TOPO_GOO:
            return 0.78f; // starts cohesive and self-healing
        case MPMMaterial::VENT_CRUMB:
        case MPMMaterial::OPEN_CRUMB:
        case MPMMaterial::BINDER_CRUMB:
        case MPMMaterial::CHANNEL_CRUMB:
        case MPMMaterial::BURNOUT_CLAY:
        case MPMMaterial::VENTED_SKIN:
            return 1.0f; // moisture-rich dough body
        default:
            return 0.0f;
    }
}

static float material_initial_damage(MPMMaterial material) {
    switch (material) {
        case MPMMaterial::ASH_REGROWTH:
            return 1.0f; // Starts fully alive, then heat can kill it back toward zero.
        default:
            return 0.0f;
    }
}

static float block_shell_seed(vec2 pos, vec2 min, vec2 max, f32 spacing) {
    float dist_x = glm::min(pos.x - min.x, max.x - pos.x);
    float dist_y = glm::min(pos.y - min.y, max.y - pos.y);
    float dist_to_edge = glm::min(dist_x, dist_y);
    float shell_band = glm::max(spacing * 2.5f, 0.04f);
    return 1.0f - glm::smoothstep(shell_band * 0.45f, shell_band * 1.35f, dist_to_edge);
}

static float circle_shell_seed(vec2 pos, vec2 center, f32 radius, f32 spacing) {
    float dist_to_surface = radius - glm::length(pos - center);
    float shell_band = glm::max(spacing * 2.8f, 0.04f);
    return 1.0f - glm::smoothstep(shell_band * 0.4f, shell_band * 1.25f, dist_to_surface);
}

void MPMSolver::init(UniformGrid& grid) {
    u32 mpm_cap = 200000;
    jp_buf_.create(mpm_cap * sizeof(f32));
    fiber_buf_.create(mpm_cap * sizeof(vec2));
    damage_buf_.create(mpm_cap * sizeof(f32));
    phase_buf_.create(mpm_cap * sizeof(f32));
    mat_params_buf_.create(mpm_cap * sizeof(vec4));
    thermal_coupling_buf_.create(mpm_cap * sizeof(vec4));
    spring_anchor_buf_.create(mpm_cap * sizeof(vec2));
    spring_weight_buf_.create(mpm_cap * sizeof(f32));

    // Initialize Jp to 1.0, damage and phase to 0.0
    std::vector<f32> ones(mpm_cap, 1.0f);
    jp_buf_.upload(ones.data(), mpm_cap * sizeof(f32));
    damage_buf_.clear();
    phase_buf_.clear();
    mat_params_buf_.clear();
    thermal_coupling_buf_.clear();
    spring_anchor_buf_.clear();
    spring_weight_buf_.clear();
    spring_drag_active_ = false;
    kinematic_targets_active_ = false;
    kinematic_target_force_ = 0.0f;
    kinematic_target_damping_ = 0.0f;
    spring_origin_ = vec2(0.0f);

    clear_shader_.load("shaders/physics/mpm_clear.comp");
    if (!p2g_shader_.load("shaders/physics/mpm_p2g.comp"))
        LOG_ERROR("FAILED to load mpm_p2g.comp!");
    grid_op_shader_.load("shaders/physics/mpm_grid_op.comp");
    if (!impact_contact_shader_.load("shaders/physics/mpm_impact_contact.comp"))
        LOG_ERROR("FAILED to load mpm_impact_contact.comp!");
    if (!g2p_shader_.load("shaders/physics/mpm_g2p.comp"))
        LOG_ERROR("FAILED to load mpm_g2p.comp!");
    thermal_shader_.load("shaders/physics/mpm_thermal.comp");

    LOG_INFO("MPM solver initialized (E=%.0f, nu=%.2f, %u material types)",
        params_.youngs_modulus, params_.poisson_ratio,
        static_cast<u32>(MPMMaterial::ASH_REGROWTH) + 1u);
}

void MPMSolver::spawn_from_positions(ParticleBuffer& particles, const std::vector<vec2>& positions,
                                     const std::vector<f32>& shell_seeds, f32 mass,
                                     MPMMaterial material, f32 initial_temp, vec2 fiber_dir,
                                     vec4 thermal_coupling) {
    u32 count = static_cast<u32>(positions.size());
    if (count == 0) return;

    std::vector<vec2> velocities(count, vec2(0.0f));
    std::vector<f32>  masses(count, mass);
    std::vector<vec4> colors(count, material_spawn_color(material));
    std::vector<u32>  mat_ids(count, static_cast<u32>(material));
    std::vector<f32>  temps(count, initial_temp);
    std::vector<vec4> deform_grads(count, vec4(1, 0, 0, 1));
    std::vector<vec4> affine_moms(count, vec4(0, 0, 0, 0));
    std::vector<f32>  jps(count, material_uses_zero_jp(material) ? 0.0f : 1.0f);
    std::vector<vec2> fibers(count, glm::length(fiber_dir) > 1e-6f ? glm::normalize(fiber_dir) : vec2(0.0f));
    std::vector<vec4> mat_params(count);
    std::vector<vec4> thermal_params(count);
    std::vector<f32> damages(count, material_initial_damage(material));
    std::vector<f32> phases(count, material_initial_phase(material));

    for (u32 i = 0; i < count; ++i) {
        float shell_seed = (i < shell_seeds.size()) ? glm::clamp(shell_seeds[i], 0.0f, 1.0f) : 0.0f;
        mat_params[i] = vec4(params_.youngs_modulus, params_.poisson_ratio, params_.fiber_strength, shell_seed);
        vec4 thermal_defaults = default_thermal_coupling(material);
        thermal_params[i] = vec4(
            thermal_defaults.x * thermal_coupling.x,
            thermal_defaults.y * thermal_coupling.y,
            thermal_defaults.z * thermal_coupling.z,
            thermal_defaults.w * thermal_coupling.w);
    }

    u32 offset = particles.allocate(SolverType::MPM, count);
    if (offset == UINT32_MAX) return;

    particles.upload_positions(offset, positions.data(), count);
    particles.upload_velocities(offset, velocities.data(), count);
    particles.upload_masses(offset, masses.data(), count);
    particles.upload_colors(offset, colors.data(), count);
    particles.upload_material_ids(offset, mat_ids.data(), count);
    particles.temperatures().upload(temps.data(), count * sizeof(f32), offset * sizeof(f32));

    u32 local_offset = offset - particles.range(SolverType::MPM).offset;
    particles.deform_grads().upload(deform_grads.data(), count * sizeof(vec4), local_offset * sizeof(vec4));
    particles.affine_moms().upload(affine_moms.data(), count * sizeof(vec4), local_offset * sizeof(vec4));
    jp_buf_.upload(jps.data(), count * sizeof(f32), local_offset * sizeof(f32));
    fiber_buf_.upload(fibers.data(), count * sizeof(vec2), local_offset * sizeof(vec2));
    damage_buf_.upload(damages.data(), count * sizeof(f32), local_offset * sizeof(f32));
    phase_buf_.upload(phases.data(), count * sizeof(f32), local_offset * sizeof(f32));
    mat_params_buf_.upload(mat_params.data(), count * sizeof(vec4), local_offset * sizeof(vec4));
    thermal_coupling_buf_.upload(thermal_params.data(), count * sizeof(vec4), local_offset * sizeof(vec4));

    particle_count_ = particles.range(SolverType::MPM).count;
    LOG_INFO("Spawned %u MPM particles (mat=%u, total=%u)", count, static_cast<u32>(material), particle_count_);
}

void MPMSolver::spawn_block(ParticleBuffer& particles, vec2 min, vec2 max, f32 spacing,
                              MPMMaterial material, f32 initial_temp, vec2 fiber_dir, f32 density_scale,
                              vec4 thermal_coupling) {
    std::vector<vec2> positions;
    std::vector<f32> shell_seeds;

    f32 rho = 1000.0f * density_scale; // Default density
    f32 vol = spacing * spacing;
    f32 mass = rho * vol;

    for (f32 y = min.y; y < max.y; y += spacing) {
        for (f32 x = min.x; x < max.x; x += spacing) {
            vec2 spawn_pos(x, y);
            positions.push_back(spawn_pos);
            shell_seeds.push_back(block_shell_seed(spawn_pos, min, max, spacing));
        }
    }

    spawn_from_positions(particles, positions, shell_seeds, mass, material, initial_temp, fiber_dir, thermal_coupling);
}

void MPMSolver::spawn_circle(ParticleBuffer& particles, vec2 center, f32 radius, f32 spacing,
                               MPMMaterial material, f32 initial_temp, vec2 fiber_dir, f32 density_scale,
                               vec4 thermal_coupling) {
    // Reuse spawn_block with circular filtering
    vec2 min = center - vec2(radius);
    vec2 max = center + vec2(radius);

    std::vector<vec2> positions;
    std::vector<f32> shell_seeds;

    f32 mass = 1000.0f * density_scale * spacing * spacing;

    for (f32 y = min.y; y < max.y; y += spacing) {
        for (f32 x = min.x; x < max.x; x += spacing) {
            if (glm::length(vec2(x, y) - center) > radius) continue;
            vec2 spawn_pos(x, y);
            positions.push_back(spawn_pos);
            shell_seeds.push_back(circle_shell_seed(spawn_pos, center, radius, spacing));
        }
    }

    spawn_from_positions(particles, positions, shell_seeds, mass, material, initial_temp, fiber_dir, thermal_coupling);
}

void MPMSolver::spawn_points(ParticleBuffer& particles, const std::vector<vec2>& positions,
                             const std::vector<f32>& shell_seeds, f32 spacing,
                             MPMMaterial material, f32 initial_temp, vec2 fiber_dir, f32 density_scale,
                             vec4 thermal_coupling) {
    f32 mass = 1000.0f * density_scale * spacing * spacing;
    spawn_from_positions(particles, positions, shell_seeds, mass, material, initial_temp, fiber_dir, thermal_coupling);
}

void MPMSolver::begin_spring_drag(ParticleBuffer& particles, vec2 center, f32 radius, f32 falloff_radius) {
    const auto& range = particles.range(SolverType::MPM);
    if (range.count == 0 || radius <= 0.0f) {
        spring_drag_active_ = false;
        return;
    }

    if (falloff_radius < radius) falloff_radius = radius;

    spring_origin_ = center;

    std::vector<vec2> positions(range.count);
    std::vector<vec2> anchors(range.count, vec2(0.0f));
    std::vector<f32> weights(range.count, 0.0f);
    particles.positions().download(positions.data(), range.count * sizeof(vec2), range.offset * sizeof(vec2));

    bool any = false;
    for (u32 i = 0; i < range.count; ++i) {
        vec2 delta = positions[i] - center;
        f32 dist = glm::length(delta);
        if (dist >= falloff_radius) continue;
        if (dist <= radius || falloff_radius <= radius + 1e-4f) {
            weights[i] = 1.0f;
        } else {
            f32 t = (dist - radius) / (falloff_radius - radius);
            weights[i] = glm::clamp(1.0f - t, 0.0f, 1.0f);
        }
        anchors[i] = delta;
        any = any || weights[i] > 0.0f;
    }

    spring_anchor_buf_.clear();
    spring_weight_buf_.clear();
    if (any) {
        spring_anchor_buf_.upload(anchors.data(), range.count * sizeof(vec2));
        spring_weight_buf_.upload(weights.data(), range.count * sizeof(f32));
    }
    spring_drag_active_ = any;
}

void MPMSolver::end_spring_drag() {
    spring_anchor_buf_.clear();
    spring_weight_buf_.clear();
    spring_origin_ = vec2(0.0f);
    spring_drag_active_ = false;
}

void MPMSolver::set_kinematic_targets(const std::vector<vec2>& targets,
                                      const std::vector<f32>& weights,
                                      f32 force, f32 damping) {
    if (targets.empty() || weights.empty() || targets.size() != weights.size()) {
        clear_kinematic_targets();
        return;
    }

    spring_anchor_buf_.upload(targets.data(), targets.size() * sizeof(vec2));
    spring_weight_buf_.upload(weights.data(), weights.size() * sizeof(f32));
    spring_origin_ = vec2(0.0f);
    kinematic_targets_active_ = true;
    kinematic_target_force_ = glm::max(force, 0.0f);
    kinematic_target_damping_ = glm::max(damping, 0.0f);
}

void MPMSolver::clear_kinematic_targets() {
    kinematic_targets_active_ = false;
    kinematic_target_force_ = 0.0f;
    kinematic_target_damping_ = 0.0f;
    if (!spring_drag_active_) {
        spring_anchor_buf_.clear();
        spring_weight_buf_.clear();
    }
}

void MPMSolver::step(ParticleBuffer& particles, UniformGrid& grid, f32 dt,
                      const SDFField* sdf, const MagneticField* magnetic,
                      const EulerianFluid* air,
                      vec2 mouse_pos, f32 mouse_radius, f32 mouse_force,
                      vec2 mouse_dir, i32 mouse_mode, f32 mouse_inner_radius,
                      f32 mouse_damping) {
    if (particle_count_ == 0) return;

    // 2 substeps for CFL stability at higher E values
    // 5 substeps for CFL at E=40000: dx/sqrt(E/rho)=0.033/6.3=0.0052
    // sub_dt=0.0083/5=0.0017 < 0.0052 OK
    constexpr i32 SUBSTEPS = 5;
    f32 sub_dt = dt / static_cast<f32>(SUBSTEPS);

    for (i32 s = 0; s < SUBSTEPS; s++) {
        sub_step_mpm(particles, grid, sub_dt, sdf, magnetic, air, mouse_pos, mouse_radius, mouse_force,
                     mouse_dir, mouse_mode, mouse_inner_radius, mouse_damping);
    }
}

void MPMSolver::update_batch_material(ParticleBuffer& particles, u32 global_offset, u32 count,
                                      f32 youngs_modulus, f32 poisson_ratio,
                                      f32 fiber_strength, f32 temperature, vec2 fiber_dir,
                                      f32 outgassing_scale, f32 heat_release_scale,
                                      f32 cooling_scale, f32 loft_scale) {
    if (count == 0) return;
    const auto& range = particles.range(SolverType::MPM);
    if (global_offset < range.offset || global_offset + count > range.offset + range.count) return;

    u32 local_offset = global_offset - range.offset;
    vec2 fiber = glm::length(fiber_dir) > 1e-6f ? glm::normalize(fiber_dir) : vec2(0.0f);
    std::vector<vec4> mat_params(count);
    std::vector<vec4> thermal_params(count);
    std::vector<vec2> fibers(count, fiber);
    std::vector<f32> temperatures(count, temperature);

    mat_params_buf_.download(mat_params.data(), count * sizeof(vec4), local_offset * sizeof(vec4));
    thermal_coupling_buf_.download(thermal_params.data(), count * sizeof(vec4), local_offset * sizeof(vec4));
    for (vec4& p : mat_params) {
        p.x = youngs_modulus;
        p.y = poisson_ratio;
        p.z = fiber_strength;
    }
    for (vec4& p : thermal_params) {
        p.x = outgassing_scale;
        p.y = heat_release_scale;
        p.z = cooling_scale;
        p.w = loft_scale;
    }

    mat_params_buf_.upload(mat_params.data(), count * sizeof(vec4), local_offset * sizeof(vec4));
    thermal_coupling_buf_.upload(thermal_params.data(), count * sizeof(vec4), local_offset * sizeof(vec4));
    fiber_buf_.upload(fibers.data(), count * sizeof(vec2), local_offset * sizeof(vec2));
    particles.temperatures().upload(temperatures.data(), count * sizeof(f32), global_offset * sizeof(f32));
}

void MPMSolver::sub_step_mpm(ParticleBuffer& particles, UniformGrid& grid, f32 dt,
                              const SDFField* sdf, const MagneticField* magnetic,
                              const EulerianFluid* air,
                              vec2 mouse_pos, f32 mouse_radius, f32 mouse_force,
                              vec2 mouse_dir, i32 mouse_mode, f32 mouse_inner_radius,
                              f32 mouse_damping) {
    auto& range = particles.range(SolverType::MPM);
    u32 grid_cells = grid.total_cells();

    // Bind particle data + extra MPM buffers
    particles.bind_all();
    jp_buf_.bind_base(47);
    fiber_buf_.bind_base(48);
    damage_buf_.bind_base(49);
    phase_buf_.bind_base(50);
    mat_params_buf_.bind_base(51);
    thermal_coupling_buf_.bind_base(54);
    spring_anchor_buf_.bind_base(52);
    spring_weight_buf_.bind_base(53);
    grid.temp_atomic_buf().bind_base(55);

    // Step 1: Clear grid
    grid.bind_for_grid();
    clear_shader_.bind();
    clear_shader_.set_uint("u_grid_cells", grid_cells);
    clear_shader_.dispatch_1d(grid_cells);
    ComputeShader::barrier_ssbo();

    // Step 2: P2G (particle → grid scatter)
    grid.bind_for_p2g();
    p2g_shader_.bind();
    p2g_shader_.set_uint("u_offset", range.offset);
    p2g_shader_.set_uint("u_count", range.count);
    p2g_shader_.set_float("u_dx", grid.dx());
    p2g_shader_.set_float("u_dt", dt);
    p2g_shader_.set_vec2("u_grid_origin", grid.world_min());
    p2g_shader_.set_ivec2("u_grid_res", grid.resolution());
    f32 p_spacing = grid.dx() * 0.5f;
    p2g_shader_.set_float("u_p_vol", p_spacing * p_spacing);
    p2g_shader_.set_float("u_E", params_.youngs_modulus);
    p2g_shader_.set_float("u_nu", params_.poisson_ratio);
    p2g_shader_.set_float("u_fiber_strength", params_.fiber_strength);
    p2g_shader_.set_float("u_fracture_threshold", params_.fracture_threshold);
    p2g_shader_.set_float("u_fracture_rate", params_.fracture_rate);
    p2g_shader_.set_float("u_melt_temp", params_.melt_temp);
    p2g_shader_.set_float("u_melt_range", params_.melt_range);
    p2g_shader_.set_float("u_latent_heat", params_.latent_heat);
    p2g_shader_.set_int("u_enable_pseudo_25d", params_.pseudo_25d_enabled ? 1 : 0);
    p2g_shader_.set_float("u_pseudo_25d_depth", params_.pseudo_25d_depth);
    p2g_shader_.set_float("u_pseudo_25d_shell_support", params_.pseudo_25d_shell_support);
    p2g_shader_.set_float("u_pseudo_25d_enclosure", params_.pseudo_25d_enclosure);
    p2g_shader_.set_float("u_pseudo_25d_cohesion", params_.pseudo_25d_cohesion);
    p2g_shader_.set_vec2("u_magnet_pos", params_.magnet_pos);
    p2g_shader_.set_float("u_magnet_radius", params_.magnet_radius);
    p2g_shader_.set_float("u_magnet_falloff_radius", params_.magnet_falloff_radius);
    p2g_shader_.set_float("u_magnet_force", params_.magnet_force);
    p2g_shader_.set_float("u_magnet_spike_strength", params_.magnet_spike_strength);
    p2g_shader_.set_float("u_magnet_chain_rate", params_.magnet_chain_rate);
    p2g_shader_.set_float("u_magnet_spike_freq", params_.magnet_spike_freq);
    p2g_shader_.set_float("u_bio_regrowth_rate", air ? air->config().bio_regrowth_rate : 1.0f);
    if (sdf) {
        sdf->bind_for_read(0);
        p2g_shader_.set_int("u_sdf_tex", 0);
        p2g_shader_.set_int("u_use_sdf", 1);
        p2g_shader_.set_vec2("u_sdf_world_min", sdf->world_min());
        p2g_shader_.set_vec2("u_sdf_world_max", sdf->world_max());
    } else {
        p2g_shader_.set_int("u_use_sdf", 0);
    }
    if (magnetic) {
        // Constitutive magnetic response uses the scene-drive field for
        // strength/gradient and the full induced field for direction.
        // That preserves pole distortion from nearby soft iron without
        // letting particle self-induction feed back directly into the
        // bulk force scale and explode the ferro phase.
        magnetic->bind_field_for_read(1);
        magnetic->bind_total_field_for_read(2);
        magnetic->bind_magnetization_for_read(3);
        p2g_shader_.set_int("u_magnetic_drive_tex", 1);
        p2g_shader_.set_int("u_magnetic_total_tex", 2);
        p2g_shader_.set_int("u_magnetic_material_tex", 3);
        p2g_shader_.set_int("u_use_real_magnetics", magnetic->active() ? 1 : 0);
        p2g_shader_.set_vec2("u_magnetic_world_min", magnetic->world_min());
        p2g_shader_.set_vec2("u_magnetic_world_max", magnetic->world_max());
    } else {
        p2g_shader_.set_int("u_use_real_magnetics", 0);
    }
    p2g_shader_.dispatch_1d(range.count);
    ComputeShader::barrier_ssbo();

    // Step 3: Grid operations (momentum → velocity, gravity, BCs)
    grid.bind_for_grid();
    if (sdf) {
        sdf->bind_for_read(0);
        grid_op_shader_.bind();
        grid_op_shader_.set_int("u_sdf_tex", 0);
        grid_op_shader_.set_int("u_use_sdf", 1);
        grid_op_shader_.set_vec2("u_sdf_world_min", sdf->world_min());
        grid_op_shader_.set_vec2("u_sdf_world_max", sdf->world_max());
    } else {
        grid_op_shader_.bind();
        grid_op_shader_.set_int("u_use_sdf", 0);
    }
    grid_op_shader_.set_uint("u_grid_cells", grid_cells);
    grid_op_shader_.set_ivec2("u_grid_res", grid.resolution());
    grid_op_shader_.set_float("u_dx", grid.dx());
    grid_op_shader_.set_float("u_dt", dt);
    grid_op_shader_.set_vec2("u_grid_origin", grid.world_min());
    grid_op_shader_.set_vec2("u_gravity", params_.gravity);
    grid_op_shader_.set_vec2("u_mouse_world", mouse_pos);
    grid_op_shader_.set_float("u_mouse_radius", mouse_radius);
    grid_op_shader_.set_float("u_mouse_force", mouse_force);
    grid_op_shader_.set_vec2("u_mouse_dir", mouse_dir);
    grid_op_shader_.set_int("u_mouse_mode", mouse_mode);
    grid_op_shader_.set_float("u_mouse_inner_radius", mouse_inner_radius);
    grid_op_shader_.dispatch_1d(grid_cells);
    ComputeShader::barrier_ssbo();

    // Step 3.5: Thermal diffusion (if enabled)
    if (params_.enable_thermal) {
        // Transfer particle temperatures to grid (simple scatter, reuse mass-weighted)
        // For now: thermal diffusion happens on grid using existing temperature field
        grid.bind_for_grid();
        thermal_shader_.bind();
        thermal_shader_.set_uint("u_grid_cells", grid_cells);
        thermal_shader_.set_ivec2("u_grid_res", grid.resolution());
        thermal_shader_.set_float("u_dx", grid.dx());
        thermal_shader_.set_float("u_dt", dt);
        thermal_shader_.set_float("u_thermal_k", params_.thermal_k);
        thermal_shader_.set_int("u_physically_based_heat", params_.physically_based_heat ? 1 : 0);
        thermal_shader_.set_float("u_heat_source_x", params_.heat_source_pos.x);
        thermal_shader_.set_float("u_heat_source_y", params_.heat_source_pos.y);
        thermal_shader_.set_float("u_heat_source_radius", params_.heat_source_radius);
        thermal_shader_.set_float("u_heat_source_temp", params_.heat_source_temp);
        thermal_shader_.dispatch_1d(grid_cells);
        ComputeShader::barrier_ssbo();

        // Swap temperature buffers
        glCopyNamedBufferSubData(
            grid.temp2_buf().handle(), grid.temp_buf().handle(),
            0, 0, static_cast<GLsizeiptr>(grid_cells * sizeof(f32)));
        ComputeShader::barrier_ssbo();
    }

    if (params_.enable_thermal) {
        particles.bind_all();
        grid.bind_for_g2p();
        impact_contact_shader_.bind();
        impact_contact_shader_.set_uint("u_offset", range.offset);
        impact_contact_shader_.set_uint("u_count", range.count);
        impact_contact_shader_.set_float("u_dx", grid.dx());
        impact_contact_shader_.set_float("u_dt", dt);
        impact_contact_shader_.set_vec2("u_grid_origin", grid.world_min());
        impact_contact_shader_.set_ivec2("u_grid_res", grid.resolution());
        impact_contact_shader_.set_float("u_ambient_temp", params_.ambient_temp);
        impact_contact_shader_.dispatch_1d(range.count);
        ComputeShader::barrier_ssbo();
    }

    // Step 4: G2P (grid → particle gather + particle-level SDF collision)
    particles.bind_all();
    grid.bind_for_g2p();
    if (sdf) sdf->bind_for_read(0);
    if (magnetic) {
        magnetic->bind_field_for_read(1);
        magnetic->bind_total_field_for_read(2);
        magnetic->bind_magnetization_for_read(3);
    }
    g2p_shader_.bind();
    g2p_shader_.set_uint("u_offset", range.offset);
    g2p_shader_.set_uint("u_count", range.count);
    g2p_shader_.set_float("u_dx", grid.dx());
    g2p_shader_.set_float("u_dt", dt);
    g2p_shader_.set_vec2("u_grid_origin", grid.world_min());
    g2p_shader_.set_ivec2("u_grid_res", grid.resolution());
    g2p_shader_.set_int("u_vis_mode", params_.vis_mode);
    if (sdf) {
        g2p_shader_.set_int("u_sdf_tex", 0);
        g2p_shader_.set_int("u_use_sdf", 1);
        g2p_shader_.set_vec2("u_sdf_world_min", sdf->world_min());
        g2p_shader_.set_vec2("u_sdf_world_max", sdf->world_max());
    } else {
        g2p_shader_.set_int("u_use_sdf", 0);
    }
    if (magnetic) {
        g2p_shader_.set_int("u_magnetic_field_tex", 1);
        g2p_shader_.set_int("u_magnetic_total_tex", 2);
        g2p_shader_.set_int("u_magnetic_material_tex", 3);
        g2p_shader_.set_int("u_use_real_magnetics", magnetic->active() ? 1 : 0);
        g2p_shader_.set_vec2("u_magnetic_world_min", magnetic->world_min());
        g2p_shader_.set_vec2("u_magnetic_world_max", magnetic->world_max());
        g2p_shader_.set_float("u_magnetic_force_scale", magnetic->params().force_scale);
    } else {
        g2p_shader_.set_int("u_use_real_magnetics", 0);
        g2p_shader_.set_float("u_magnetic_force_scale", 0.0f);
    }
    if (air) {
        glBindTextureUnit(4, air->temp_texture());
        g2p_shader_.set_int("u_air_temp_tex", 4);
        g2p_shader_.set_int("u_use_air_heat", 1);
        g2p_shader_.set_vec2("u_air_world_min", air->world_min());
        g2p_shader_.set_vec2("u_air_world_max", air->world_max());
        glBindTextureUnit(5, air->airtight_pressure_texture());
        glBindTextureUnit(6, air->airtight_outside_texture());
        g2p_shader_.set_int("u_airtight_pressure_tex", 5);
        g2p_shader_.set_int("u_airtight_outside_tex", 6);
        g2p_shader_.set_int("u_use_airtight_pressure", 1);
        g2p_shader_.set_ivec2("u_cavity_res", air->airtight_resolution());
        glBindTextureUnit(7, air->bio_a_texture());
        glBindTextureUnit(8, air->bio_b_texture());
        glBindTextureUnit(9, air->automata_texture());
        glBindTextureUnit(10, air->bio_support_texture());
        glBindTextureUnit(11, air->bio_source_texture());
        g2p_shader_.set_int("u_bio_a_tex", 7);
        g2p_shader_.set_int("u_bio_b_tex", 8);
        g2p_shader_.set_int("u_automata_tex", 9);
        g2p_shader_.set_int("u_bio_support_tex", 10);
        g2p_shader_.set_int("u_bio_source_tex", 11);
        g2p_shader_.set_int("u_use_bio_field", air->config().bio_enabled ? 1 : 0);
        g2p_shader_.set_float("u_bio_coupling", air->config().bio_coupling);
        g2p_shader_.set_float("u_bio_regrowth_rate", air->config().bio_regrowth_rate);
        g2p_shader_.set_int("u_use_automata_field", air->config().automata_enabled ? 1 : 0);
        g2p_shader_.set_float("u_automata_coupling", air->config().automata_coupling);
        g2p_shader_.set_float("u_bio_view_gain", air->bio_field_view_gain());
        g2p_shader_.set_float("u_automata_view_gain", air->automata_view_gain());
    } else {
        g2p_shader_.set_int("u_use_air_heat", 0);
        g2p_shader_.set_int("u_use_airtight_pressure", 0);
        g2p_shader_.set_ivec2("u_cavity_res", ivec2(1));
        g2p_shader_.set_int("u_use_bio_field", 0);
        g2p_shader_.set_float("u_bio_coupling", 0.0f);
        g2p_shader_.set_float("u_bio_regrowth_rate", 1.0f);
        g2p_shader_.set_int("u_use_automata_field", 0);
        g2p_shader_.set_float("u_automata_coupling", 0.0f);
        g2p_shader_.set_float("u_bio_view_gain", 1.0f);
        g2p_shader_.set_float("u_automata_view_gain", 1.0f);
    }

    // Heat gun (interactive)
    g2p_shader_.set_vec2("u_heat_pos", params_.heat_gun_pos);
    g2p_shader_.set_float("u_heat_radius", params_.heat_gun_radius);
    g2p_shader_.set_float("u_heat_power", params_.heat_gun_power);

    // Ambient heat source (scene)
    g2p_shader_.set_vec2("u_heat_src_pos", params_.heat_source_pos);
    g2p_shader_.set_float("u_heat_src_radius", params_.enable_thermal ? params_.heat_source_radius : 0.0f);
    g2p_shader_.set_float("u_heat_src_temp", params_.enable_thermal ? params_.heat_source_temp : 0.0f);
    g2p_shader_.set_float("u_ambient_temp", params_.ambient_temp);
    g2p_shader_.set_float("u_particle_cooling", params_.particle_cooling_rate);
    g2p_shader_.set_int("u_enable_pseudo_25d", params_.pseudo_25d_enabled ? 1 : 0);
    g2p_shader_.set_float("u_pseudo_25d_depth", params_.pseudo_25d_depth);
    g2p_shader_.set_vec2("u_mouse_world", mouse_pos);
    g2p_shader_.set_float("u_mouse_force", mouse_force);
    g2p_shader_.set_float("u_mouse_damping", mouse_damping);
    g2p_shader_.set_int("u_mouse_mode", mouse_mode);
    g2p_shader_.set_vec2("u_spring_origin", spring_origin_);
    g2p_shader_.set_int("u_kinematic_targets_active", kinematic_targets_active_ ? 1 : 0);
    g2p_shader_.set_float("u_kinematic_target_force", kinematic_target_force_);
    g2p_shader_.set_float("u_kinematic_target_damping", kinematic_target_damping_);
    g2p_shader_.set_vec2("u_magnet_pos", params_.magnet_pos);
    g2p_shader_.set_float("u_magnet_radius", params_.magnet_radius);
    g2p_shader_.set_float("u_magnet_falloff_radius", params_.magnet_falloff_radius);
    g2p_shader_.set_float("u_magnet_force", params_.magnet_force);
    g2p_shader_.set_float("u_magnet_spike_strength", params_.magnet_spike_strength);
    g2p_shader_.set_float("u_magnet_spike_freq", params_.magnet_spike_freq);

    // Batch color preservation + highlight
    g2p_shader_.set_int("u_keep_colors", params_.keep_colors ? 1 : 0);
    g2p_shader_.set_uint("u_highlight_start", params_.highlight_start);
    g2p_shader_.set_uint("u_highlight_end", params_.highlight_end);
    g2p_shader_.set_float("u_time", params_.time);
    g2p_shader_.set_vec2("u_gravity", params_.gravity);
    g2p_shader_.set_int("u_multi_scale", params_.multi_scale ? 1 : 0);

    g2p_shader_.dispatch_1d(range.count);
    ComputeShader::barrier_ssbo();
}

} // namespace ng
