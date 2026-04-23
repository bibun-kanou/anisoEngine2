#include "core/creation_menu.h"
#include "physics/common/particle_buffer.h"
#include "physics/common/grid.h"
#include "physics/sph/sph_solver.h"
#include "physics/mpm/mpm_solver.h"
#include "core/log.h"

#include <algorithm>
#include <limits>

namespace ng {

namespace {

constexpr f32 kPi = 3.14159265f;

vec2 rotate_vec(vec2 v, f32 radians) {
    f32 c = std::cos(radians);
    f32 s = std::sin(radians);
    return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

f32 sd_box(vec2 p, vec2 half_extents) {
    vec2 d = glm::abs(p) - half_extents;
    vec2 outside = glm::max(d, vec2(0.0f));
    return glm::length(outside) + glm::min(glm::max(d.x, d.y), 0.0f);
}

f32 dist_to_segment(vec2 p, vec2 a, vec2 b) {
    vec2 ab = b - a;
    f32 denom = glm::dot(ab, ab);
    if (denom <= 1e-8f) return glm::length(p - a);
    f32 t = glm::clamp(glm::dot(p - a, ab) / denom, 0.0f, 1.0f);
    return glm::length(p - (a + ab * t));
}

bool point_in_polygon(vec2 p, const std::vector<vec2>& poly) {
    bool inside = false;
    if (poly.size() < 3) return false;
    for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
        const vec2& a = poly[i];
        const vec2& b = poly[j];
        bool intersect = ((a.y > p.y) != (b.y > p.y)) &&
                         (p.x < (b.x - a.x) * (p.y - a.y) / ((b.y - a.y) + 1e-8f) + a.x);
        if (intersect) inside = !inside;
    }
    return inside;
}

f32 polygon_edge_distance(vec2 p, const std::vector<vec2>& poly) {
    if (poly.size() < 2) return 0.0f;
    f32 d = std::numeric_limits<f32>::max();
    for (size_t i = 0; i < poly.size(); ++i) {
        d = glm::min(d, dist_to_segment(p, poly[i], poly[(i + 1) % poly.size()]));
    }
    return d;
}

std::vector<vec2> build_polygon(SpawnShape shape, vec2 half_extents) {
    std::vector<vec2> poly;
    if (shape == SpawnShape::TRIANGLE) {
        poly = {
            vec2(0.0f, half_extents.y),
            vec2(half_extents.x, -0.5f * half_extents.y),
            vec2(-half_extents.x, -0.5f * half_extents.y)
        };
    } else if (shape == SpawnShape::STAR) {
        poly.reserve(10);
        f32 inner_scale = 0.42f;
        for (int i = 0; i < 10; ++i) {
            f32 ang = -kPi * 0.5f + kPi * 0.2f * static_cast<f32>(i);
            f32 r = (i % 2 == 0) ? 1.0f : inner_scale;
            poly.push_back(vec2(std::cos(ang) * half_extents.x * r,
                                std::sin(ang) * half_extents.y * r));
        }
    }
    return poly;
}

struct ShapeHit {
    bool inside = false;
    f32 boundary_dist = 0.0f;
};

ShapeHit eval_shape_local(SpawnShape shape, vec2 p, vec2 half_extents, f32 spacing,
                          const std::vector<vec2>& poly) {
    ShapeHit hit{};
    switch (shape) {
    case SpawnShape::CIRCLE: {
        f32 r = half_extents.x;
        f32 len = glm::length(p);
        if (len <= r) {
            hit.inside = true;
            hit.boundary_dist = r - len;
        }
        break;
    }
    case SpawnShape::SHELL_CIRCLE: {
        f32 outer = half_extents.x;
        f32 thickness = shape_shell_thickness(shape, half_extents, spacing);
        f32 inner = glm::max(outer - thickness, outer * 0.35f);
        inner = glm::min(inner, outer - spacing * 0.6f);
        inner = glm::max(inner, 0.0f);
        f32 len = glm::length(p);
        if (len <= outer && len >= inner) {
            hit.inside = true;
            hit.boundary_dist = glm::min(outer - len, len - inner);
        }
        break;
    }
    case SpawnShape::RECT:
    case SpawnShape::BEAM: {
        f32 sd = sd_box(p, half_extents);
        if (sd <= 0.0f) {
            hit.inside = true;
            hit.boundary_dist = -sd;
        }
        break;
    }
    case SpawnShape::SHELL_RECT: {
        f32 thickness = shape_shell_thickness(shape, half_extents, spacing);
        vec2 inner = glm::max(half_extents - vec2(thickness), half_extents * 0.32f);
        inner = glm::min(inner, half_extents - vec2(spacing * 0.6f));
        inner = glm::max(inner, vec2(0.0f));
        f32 sd_outer = sd_box(p, half_extents);
        f32 sd_inner = sd_box(p, inner);
        if (sd_outer <= 0.0f && sd_inner >= 0.0f) {
            hit.inside = true;
            hit.boundary_dist = glm::min(-sd_outer, sd_inner);
        }
        break;
    }
    case SpawnShape::TRIANGLE:
    case SpawnShape::STAR: {
        if (point_in_polygon(p, poly)) {
            hit.inside = true;
            hit.boundary_dist = polygon_edge_distance(p, poly);
        }
        break;
    }
    default:
        break;
    }
    return hit;
}

void build_shape_points(const CreationState& state, f32 spacing,
                        std::vector<vec2>& positions, std::vector<f32>& shell_seeds) {
    positions.clear();
    shell_seeds.clear();

    vec2 half_extents = shape_half_extents(state.shape, state.size, state.aspect);
    std::vector<vec2> poly = build_polygon(state.shape, half_extents);
    f32 angle = state.shape_angle * kPi / 180.0f;
    f32 bound_radius = glm::length(half_extents) + spacing;
    if (state.shape == SpawnShape::CIRCLE || state.shape == SpawnShape::SHELL_CIRCLE) {
        bound_radius = half_extents.x + spacing;
    }
    f32 shell_band = glm::max(spacing * 2.5f, 0.08f * glm::max(half_extents.x, half_extents.y));

    for (f32 y = state.preview_pos.y - bound_radius; y <= state.preview_pos.y + bound_radius; y += spacing) {
        for (f32 x = state.preview_pos.x - bound_radius; x <= state.preview_pos.x + bound_radius; x += spacing) {
            vec2 world_p(x, y);
            vec2 local = rotate_vec(world_p - state.preview_pos, -angle);
            ShapeHit hit = eval_shape_local(state.shape, local, half_extents, spacing, poly);
            if (!hit.inside) continue;
            positions.push_back(world_p);
            f32 shell_seed = 1.0f - glm::smoothstep(shell_band * 0.35f, shell_band * 1.20f, hit.boundary_dist);
            shell_seeds.push_back(glm::clamp(shell_seed, 0.0f, 1.0f));
        }
    }
}

void refresh_batch_properties(BatchRecord& batch) {
    batch.properties = property_summary(batch.solver, batch.mpm_type,
                                        batch.youngs_modulus, batch.poisson_ratio,
                                        batch.temperature, batch.fiber_dir,
                                        batch.density_scale);
    if (batch.solver == SpawnSolver::SPH) {
        char fluid_buf[192];
        snprintf(fluid_buf, sizeof(fluid_buf),
                 " | k %.2f | visc %.3f | outgas %.2f | air heat %.2f | cool %.2f | loft %.2f",
                 batch.gas_constant, batch.viscosity, batch.outgassing_scale,
                 batch.heat_release_scale, batch.cooling_scale, batch.loft_scale);
        batch.properties += fluid_buf;
    } else {
        char thermal_buf[160];
        snprintf(thermal_buf, sizeof(thermal_buf),
                 " | outgas %.2f | air heat %.2f | cool %.2f | loft %.2f",
                 batch.outgassing_scale, batch.heat_release_scale, batch.cooling_scale, batch.loft_scale);
        batch.properties += thermal_buf;
    }
}

} // namespace

const MaterialPreset* preset_for_material(SpawnSolver solver, MPMMaterial mpm_type) {
    for (const auto& preset : get_presets()) {
        if (preset.solver == solver && preset.mpm_type == mpm_type) return &preset;
    }
    return nullptr;
}

vec4 default_material_color(SpawnSolver solver, MPMMaterial mpm_type) {
    if (const auto* preset = preset_for_material(solver, mpm_type)) return preset->color;
    if (solver == SpawnSolver::SPH) {
        switch (mpm_type) {
            case MPMMaterial::SPH_WATER: return vec4(0.15f, 0.4f, 0.85f, 1.0f);
            case MPMMaterial::SPH_VISCOUS_GOO: return vec4(0.3f, 0.8f, 0.2f, 1.0f);
            case MPMMaterial::SPH_LIGHT_OIL: return vec4(0.7f, 0.6f, 0.2f, 1.0f);
        case MPMMaterial::SPH_BURNING_OIL: return vec4(0.78f, 0.54f, 0.14f, 1.0f);
        case MPMMaterial::SPH_BOILING_WATER: return vec4(0.44f, 0.72f, 0.98f, 1.0f);
        case MPMMaterial::SPH_THERMAL_SYRUP: return vec4(0.86f, 0.54f, 0.22f, 1.0f);
        case MPMMaterial::SPH_FLASH_FLUID: return vec4(0.82f, 0.72f, 0.50f, 1.0f);
        case MPMMaterial::MAG_SOFT_IRON: return vec4(0.56f, 0.60f, 0.66f, 1.0f);
        case MPMMaterial::MAGNETIC_RUBBER: return vec4(0.44f, 0.48f, 0.56f, 1.0f);
        default: return vec4(0.15f, 0.4f, 0.85f, 1.0f);
    }
    }
    switch (mpm_type) {
        case MPMMaterial::FLUID: return vec4(0.1f, 0.3f, 0.8f, 1.0f);
        case MPMMaterial::ELASTIC: return vec4(0.2f, 0.7f, 0.3f, 1.0f);
        case MPMMaterial::SNOW: return vec4(0.85f, 0.88f, 0.95f, 1.0f);
        case MPMMaterial::ANISO: return vec4(0.8f, 0.5f, 0.2f, 1.0f);
        case MPMMaterial::THERMAL: return vec4(0.8f, 0.6f, 0.3f, 1.0f);
        case MPMMaterial::FRACTURE: return vec4(0.6f, 0.4f, 0.15f, 1.0f);
        case MPMMaterial::PHASE: return vec4(0.3f, 0.6f, 0.95f, 1.0f);
        case MPMMaterial::BURNING: return vec4(0.5f, 0.3f, 0.15f, 1.0f);
        case MPMMaterial::EMBER: return vec4(0.95f, 0.6f, 0.18f, 1.0f);
        case MPMMaterial::HARDEN: return vec4(0.55f, 0.45f, 0.35f, 1.0f);
        case MPMMaterial::CERAMIC: return vec4(0.82f, 0.7f, 0.58f, 1.0f);
        case MPMMaterial::COMPOSITE: return vec4(0.6f, 0.45f, 0.3f, 1.0f);
        case MPMMaterial::BRITTLE: return vec4(0.62f, 0.52f, 0.42f, 1.0f);
        case MPMMaterial::TOUGH: return vec4(0.42f, 0.3f, 0.2f, 1.0f);
        case MPMMaterial::GLASS: return vec4(0.72f, 0.82f, 0.9f, 1.0f);
        case MPMMaterial::BLOOM: return vec4(0.88f, 0.62f, 0.44f, 1.0f);
        case MPMMaterial::FLAMMABLE_FLUID: return vec4(0.78f, 0.62f, 0.18f, 1.0f);
        case MPMMaterial::FOAM: return vec4(0.78f, 0.76f, 0.64f, 1.0f);
        case MPMMaterial::SPLINTER: return vec4(0.84f, 0.68f, 0.42f, 1.0f);
        case MPMMaterial::BREAD: return vec4(0.86f, 0.74f, 0.50f, 1.0f);
        case MPMMaterial::PUFF_CLAY: return vec4(0.82f, 0.66f, 0.48f, 1.0f);
        case MPMMaterial::FIRECRACKER: return vec4(0.82f, 0.16f, 0.12f, 1.0f);
        case MPMMaterial::GLAZE_CLAY: return vec4(0.92f, 0.78f, 0.64f, 1.0f);
        case MPMMaterial::CRUST_DOUGH: return vec4(0.90f, 0.76f, 0.52f, 1.0f);
        case MPMMaterial::THERMO_METAL: return vec4(0.62f, 0.64f, 0.70f, 1.0f);
        case MPMMaterial::REACTIVE_BURN: return vec4(0.56f, 0.20f, 0.12f, 1.0f);
        case MPMMaterial::GLAZE_DRIP: return vec4(0.96f, 0.82f, 0.62f, 1.0f);
        case MPMMaterial::STEAM_BUN: return vec4(0.96f, 0.84f, 0.58f, 1.0f);
        case MPMMaterial::FILAMENT_GLASS: return vec4(0.88f, 0.92f, 1.0f, 1.0f);
        case MPMMaterial::CHEESE_PULL: return vec4(0.98f, 0.84f, 0.44f, 1.0f);
        case MPMMaterial::MEMORY_WAX: return vec4(0.92f, 0.76f, 0.42f, 1.0f);
        case MPMMaterial::FERRO_FLUID: return vec4(0.95f, 0.55f, 0.15f, 1.0f);
        case MPMMaterial::HEAVY_FERRO_FLUID: return vec4(0.32f, 0.28f, 0.38f, 1.0f);
        case MPMMaterial::DIAMAGNETIC_FLUID: return vec4(0.58f, 0.82f, 0.95f, 1.0f);
        case MPMMaterial::PARA_MIST: return vec4(0.82f, 0.86f, 0.96f, 1.0f);
        case MPMMaterial::STICKY_FERRO: return vec4(0.18f, 0.16f, 0.22f, 1.0f);
        case MPMMaterial::SUPERCONDUCTOR: return vec4(0.72f, 0.90f, 1.00f, 1.0f);
        case MPMMaterial::CURIE_FERRO: return vec4(0.48f, 0.42f, 0.48f, 1.0f);
        case MPMMaterial::EDDY_COPPER: return vec4(0.78f, 0.46f, 0.22f, 1.0f);
        case MPMMaterial::HARD_MAGNET: return vec4(0.24f, 0.22f, 0.34f, 1.0f);
        case MPMMaterial::SAND_GRANULAR: return vec4(0.88f, 0.78f, 0.50f, 1.0f);
        case MPMMaterial::PHASE_BRITTLE: return vec4(0.82f, 0.86f, 0.92f, 1.0f);
        case MPMMaterial::MAILLARD: return vec4(0.96f, 0.84f, 0.54f, 1.0f);
        case MPMMaterial::MUSHROOM: return vec4(0.72f, 0.68f, 0.52f, 1.0f);
        case MPMMaterial::ORTHO_BEND: return vec4(0.78f, 0.56f, 0.26f, 1.0f);
        case MPMMaterial::ORTHO_TEAR: return vec4(0.86f, 0.64f, 0.34f, 1.0f);
        case MPMMaterial::VENT_CRUMB: return vec4(0.97f, 0.82f, 0.58f, 1.0f);
        case MPMMaterial::VITREOUS_CLAY: return vec4(0.74f, 0.66f, 0.60f, 1.0f);
        case MPMMaterial::BLISTER_GLAZE: return vec4(0.98f, 0.84f, 0.64f, 1.0f);
        case MPMMaterial::OPEN_CRUMB: return vec4(0.99f, 0.84f, 0.62f, 1.0f);
        case MPMMaterial::SINTER_LOCK: return vec4(0.68f, 0.60f, 0.56f, 1.0f);
        case MPMMaterial::MAG_SOFT_IRON: return vec4(0.56f, 0.60f, 0.66f, 1.0f);
        case MPMMaterial::MAGNETIC_RUBBER: return vec4(0.44f, 0.48f, 0.56f, 1.0f);
        case MPMMaterial::TOPO_GOO: return vec4(0.18f, 0.78f, 0.58f, 1.0f);
        case MPMMaterial::OOBLECK: return vec4(0.76f, 0.68f, 0.46f, 1.0f);
        case MPMMaterial::IMPACT_GEL: return vec4(0.66f, 0.46f, 0.88f, 1.0f);
        case MPMMaterial::SEALED_CHARGE: return vec4(0.70f, 0.16f, 0.10f, 1.0f);
        case MPMMaterial::MORPH_TISSUE: return vec4(0.30f, 0.76f, 0.56f, 1.0f);
        case MPMMaterial::ROOT_WEAVE: return vec4(0.40f, 0.72f, 0.30f, 1.0f);
        case MPMMaterial::CELL_SHEET: return vec4(0.92f, 0.60f, 0.70f, 1.0f);
        case MPMMaterial::ASH_REGROWTH: return vec4(0.68f, 0.78f, 0.70f, 1.0f);
        default: return vec4(0.5f);
    }
}

std::string technique_summary(SpawnSolver solver, MPMMaterial mpm_type) {
    if (solver == SpawnSolver::SPH) {
        switch (mpm_type) {
            case MPMMaterial::SPH_WATER:
                return "SPH water: WCSPH pressure, XSPH smoothing, codimensional surface tracking, and SDF collision.";
            case MPMMaterial::SPH_VISCOUS_GOO:
                return "SPH goo: higher effective viscosity, stronger damping, codimensional surface tracking, and SDF collision.";
            case MPMMaterial::SPH_LIGHT_OIL:
                return "SPH oil: lighter low-viscosity liquid with softer cohesion, codimensional tracking, and SDF collision.";
            case MPMMaterial::SPH_BURNING_OIL:
                return "Thermal SPH oil: ignition, self-heating, smoke/vapor emission, hot loft, and liquid pressure flow.";
            case MPMMaterial::SPH_BOILING_WATER:
                return "Thermal SPH water: heat diffusion, boiling state, steam venting, bubbling lift, and codimensional droplets.";
            case MPMMaterial::SPH_THERMAL_SYRUP:
                return "Thermal SPH syrup: heat-thinning, cool-thickening, sticky surface retention, and slower vapor loss.";
            case MPMMaterial::SPH_FLASH_FLUID:
                return "Thermal SPH flash fluid: low-boiling liquid with aggressive vaporization, expansion, and unstable loft.";
            default:
                return "SPH: WCSPH pressure, XSPH smoothing, codimensional surface tracking, SDF collision.";
        }
    }
    switch (mpm_type) {
        case MPMMaterial::FLUID: return "MPM fluid EOS, isotropic deformation reset, SDF collision.";
        case MPMMaterial::ELASTIC: return "MLS-MPM elasticity with corotated stress and APIC transfer.";
        case MPMMaterial::SNOW: return "Snow plasticity, singular value clamping, hardening and fracture-like crumble.";
        case MPMMaterial::ANISO: return "Anisotropic fiber reinforcement with heat-driven weakening.";
        case MPMMaterial::THERMAL: return "Temperature-softening solid with reversible hot deformation.";
        case MPMMaterial::FRACTURE: return "Damage accumulation, stress degradation, chunk-like breakup.";
        case MPMMaterial::PHASE: return "Phase transition, latent heat, thermal expansion, hot flow.";
        case MPMMaterial::BURNING: return "Combustion, self-heating, smoke injection, buoyant hot debris.";
        case MPMMaterial::EMBER: return "Long-glow ember combustion with sustained hot lofting.";
        case MPMMaterial::HARDEN: return "Irreversible heat curing with annealing into a stiffer body.";
        case MPMMaterial::CERAMIC: return "Kiln curing plus brittle fracture with residual chunk stiffness.";
        case MPMMaterial::COMPOSITE: return "Fiber + curing + directional fracture along grain.";
        case MPMMaterial::BRITTLE: return "Compression-resistant brittle solid with tensile/shear crack bias.";
        case MPMMaterial::TOUGH: return "Higher-threshold fracture solid with chunk retention.";
        case MPMMaterial::GLASS: return "Glass transition, viscous hot flow, cool re-hardening, brittle cracking.";
        case MPMMaterial::BLOOM: return "Heat curing, internal expansion, petal-like bursting fracture.";
        case MPMMaterial::FLAMMABLE_FLUID: return "Igniting liquid pool, burning spread, smoke, hot vapor pressure.";
        case MPMMaterial::FOAM: return "Firm solid that softens, expands, and grows porous when heated.";
        case MPMMaterial::SPLINTER: return "Heat-cure stiffening with self-stress and fiber-guided splintering.";
        case MPMMaterial::BREAD: return "Heat-generated gas, soft expansion, partial setting, and tear-open rupture.";
        case MPMMaterial::PUFF_CLAY: return "Hot curing, trapped gas swelling, and crumbly pressure fracture.";
        case MPMMaterial::FIRECRACKER: return "Pressurized shell, rapid gas buildup, venting smoke, burst-like rupture.";
        case MPMMaterial::GLAZE_CLAY: return "Shell/core kiln model, outer vitrification, glaze skin flow, and mismatch cracking.";
        case MPMMaterial::CRUST_DOUGH: return "Core gas growth plus shell drying, crust stiffening, loaf expansion, and tear vents.";
        case MPMMaterial::THERMO_METAL: return "Thermoelastic expansion stress, hot anneal softening, and cool hardening memory.";
        case MPMMaterial::REACTIVE_BURN: return "Staged burning with char shell, pyro gas buildup, smoky blistering, and rupture bursts.";
        case MPMMaterial::GLAZE_DRIP: return "Shell/core pottery with a hotter runnier glaze skin, shell flow, and ceramic core retention.";
        case MPMMaterial::STEAM_BUN: return "Steamier shell/core dough with stronger round lift, springy crumb, and gentler skin venting.";
        case MPMMaterial::FILAMENT_GLASS: return "First-pass codimensional-style melt: hot viscoelastic glass with stretch-aligned filament memory and thread-forming pull.";
        case MPMMaterial::CHEESE_PULL: return "First-pass codimensional-style food melt: warm soft matrix with extensional cohesion for cheese-like pull strands.";
        case MPMMaterial::MEMORY_WAX: return "Thermoplastic memory solid: softens hot, flows, then cool-recovers toward its rest shape.";
        case MPMMaterial::FERRO_FLUID: return "Magnet-responsive MPM liquid with chain-like flow memory and field pulling at the cursor.";
        case MPMMaterial::HEAVY_FERRO_FLUID: return "Much stronger ferrofluid: visibly clumps onto itself, forms taller spikes, and pulls harder under any magnetic field.";
        case MPMMaterial::DIAMAGNETIC_FLUID: return "Diamagnetic (anti-ferro) liquid: negative susceptibility. Pushed AWAY from high-field regions — pools at the edge of magnet zones, hangs back from the cursor, and forms a hollow around any local |H| peak.";
        case MPMMaterial::PARA_MIST: return "Weakly paramagnetic aerosol / mist: low density, gentle drift along field lines. Doesn't clump, forms wispy flowing trails that follow the field direction.";
        case MPMMaterial::STICKY_FERRO: return "High-cohesion ferrofluid: strong magnetization, very damped, stays in dense globules instead of tall spikes. Dark matte look. Good for 'magnetic blobs' feel.";
        case MPMMaterial::SUPERCONDUCTOR: return "Type-I superconductor: below critical temperature (T_c ~180 K) becomes a PERFECT diamagnet (Meissner effect) — expels magnetic field entirely. Above T_c acts like regular metal. Try placing a cold block near a scene magnet: field visibly hollows around it. Heat it with G, the hollow collapses.";
        case MPMMaterial::CURIE_FERRO: return "Iron-like ferromagnet with a sharp Curie transition (T_Curie ~680 K). Below that it's a strong hard magnet; above it, demagnetized paramagnetic. Heat it to 'kill' its magnetism — visible 'spike collapse' when you pass a flame near a magnetized bar.";
        case MPMMaterial::EDDY_COPPER: return "Copper-analog conductor, NOT magnetic. Heats up when it moves through a magnetic field (eddy currents). Drag a block of it through a bar magnet region or hold M nearby and shake — watch it redden. Cools quickly when still.";
        case MPMMaterial::HARD_MAGNET: return "Hard permanent magnet (magnetite-analog) with PERSISTENT magnetization. Expose it to a strong field (hold M near it for a couple seconds) and it locks in a magnetic moment — then it keeps magnetizing scene ferrofluid FOR SECONDS after you let go of M. This is hysteresis: M_prev decays at only ~0.3/s so on human timescales it behaves as a permanent magnet you just made.";
        case MPMMaterial::SAND_GRANULAR: return "True granular sand via Drucker-Prager plasticity (Klar et al. 2016). Return-mapping on the SVD in log-strain space with a 35° friction cone. Forms real angle-of-repose piles, slumps on impact, avalanches when steeper than its critical angle, and — unlike fluids — holds a static heap against gravity.";
        case MPMMaterial::PHASE_BRITTLE: return "Ceramic-like phase-field brittle material. Elastic energy density drives damage growth via dd/dt = k*(psi - psi_c)*(1 - d); stress degrades quadratically (1-d)^2. Clean crack tips, propagating along high-energy paths. Shatters on impact, holds shape under gentle load.";
        case MPMMaterial::MAG_SOFT_IRON: return "Real magnetics benchmark: soft iron body that samples the solved magnetic field and pulls toward stronger |H|^2 regions.";
        case MPMMaterial::MAGNETIC_RUBBER: return "Real magnetics benchmark: compliant magnetizable solid that bends and drifts under the solved magnetic field.";
        case MPMMaterial::MAILLARD: return "Cooking surface model: browning, drying, shell-setting, and steam blistering without a full bread-like rise.";
        case MPMMaterial::MUSHROOM: return "Porous bio solid: wilts with heat, vents spore-like plumes, and tears into soft cap fragments.";
        case MPMMaterial::CRUMB_LOAF: return "Porous loaf microstructure: drying scaffold, persistent crumb pores, and localized vent tearing.";
        case MPMMaterial::BISQUE: return "Porous pottery body: burnout, shell-first sintering, shrink mismatch, and bisque-like cracking.";
        case MPMMaterial::TEAR_SKIN: return "Drying tear sheet: porous weak bands, localized ripping, and hot vent tears.";
        case MPMMaterial::LAMINATED_PASTRY: return "Layered bake model: steam lift between layers, shell browning, and delamination into flaky sheets.";
        case MPMMaterial::STONEWARE: return "Dense firing model: shell-first vitrification, shrinkage, warping, and craze-prone stoneware cracking.";
        case MPMMaterial::ORTHO_BEND: return "Strong orthotropic benchmark: fiber-dominant stiffness with very weak cross-grain response for clean bend comparisons.";
        case MPMMaterial::ORTHO_TEAR: return "Orthotropic tear benchmark: heat-cured fiber sheet that resists along-grain pull but splits rapidly across the grain.";
        case MPMMaterial::VENT_CRUMB: return "Advanced loaf microstructure: shell-first drying, retained core moisture, vent-channel opening, and tunnel-like baked crumb.";
        case MPMMaterial::VITREOUS_CLAY: return "Advanced kiln densification: stronger vitrification, tighter firing shrink, denser body, and harder-fired craze behavior.";
        case MPMMaterial::BLISTER_GLAZE: return "Reactive glaze shell: glossy skin, trapped volatile blisters, vent pits, and ceramic-core glaze mismatch.";
        case MPMMaterial::OPEN_CRUMB: return "Bubble-coalescing loaf: stronger pore retention, larger open crumb chambers, and arch-like baked scaffold instead of only hairline vents.";
        case MPMMaterial::SINTER_LOCK: return "Sinter-lock kiln body: denser firing, stronger shape lock, and real shrink-set body retention instead of plain hot slump.";
        case MPMMaterial::BINDER_CRUMB: return "Moisture+binder scaffold: drying sets a stronger crumb matrix, pores persist, and venting does not immediately collapse the loaf body.";
        case MPMMaterial::CHANNEL_CRUMB: return "Moisture-channel loaf: retained core moisture opens tunnels and channels, but the scaffold tries to keep a connected baked body.";
        case MPMMaterial::BURNOUT_CLAY: return "Burnout pottery body: moisture loss, pore-former burnout, firing binder set, and porous shrink without immediate catastrophic collapse.";
        case MPMMaterial::TOPO_GOO: return "Topology-changing goo: fast shear opens holes and separated lobes, calm contact heals and re-merges, and the body switches between sticking and separating based on stress.";
        case MPMMaterial::OOBLECK: return "Non-Newtonian paste: low-stress slump, shear-thickening jam, yield-style memory, and impact-driven temporary solidification.";
        case MPMMaterial::IMPACT_GEL: return "Interaction-driven gel: repeated hits harden the body, deformation memory lingers longer, and the material gradually forgets when left alone.";
        case MPMMaterial::VENTED_SKIN: return "Vent-before-tear sheet: drying membrane opens vents and slots first, so venting reads differently from full tearing.";
        case MPMMaterial::SEALED_CHARGE: return "Airtight pressure charge: shell-biased confinement traps hot gas, holds pressure longer than porous firecracker pellets, then ruptures into a faster air-driven blast.";
        case MPMMaterial::MORPH_TISSUE: return "Bio-driven morphogenesis proxy: differential growth swells the body unevenly, shell/core mismatch creates buckling, and the tissue keeps soft fold memory.";
        case MPMMaterial::ROOT_WEAVE: return "Chemotactic anisotropic growth: root-like strands creep along morphogen ridges, align into bundles, and stiffen as they lignify.";
        case MPMMaterial::CELL_SHEET: return "Smoothlife-like membrane proxy: active edges bud outward, wrinkle into frills, and pinch into colony-shaped lobes under the bio field.";
        case MPMMaterial::ASH_REGROWTH: return "Ash-guided regrowth proxy: burning turns the body into fragile ash, the ash acts as a living scaffold, and bio/automata fronts pull it back into a deformed regrown shape instead of a perfect reset.";
        default: return "MLS-MPM solid with SDF collision.";
    }
}

std::string property_summary(SpawnSolver solver, MPMMaterial mpm_type,
                             f32 youngs_modulus, f32 poisson_ratio,
                             f32 initial_temp, vec2 fiber_dir,
                             f32 density_scale) {
    char buf[256];
    if (solver == SpawnSolver::SPH) {
        snprintf(buf, sizeof(buf), "SPH liquid batch. Start temp %.0fK | mat %u.",
                 initial_temp, static_cast<u32>(mpm_type));
        return std::string(buf);
    }
    bool show_density = std::abs(density_scale - 1.0f) > 0.05f;
    float rest_density = 1000.0f * glm::max(density_scale, 0.0f);
    float fiber_len = glm::length(fiber_dir);
    if (fiber_len > 1e-5f) {
        float angle = std::atan2(fiber_dir.y, fiber_dir.x) * 180.0f / 3.14159265f;
        if (show_density) {
            snprintf(buf, sizeof(buf), "E %.0f | nu %.2f | start %.0fK | fiber %.0f deg | rho %.0f | mat %u",
                     youngs_modulus, poisson_ratio, initial_temp, angle, rest_density, static_cast<u32>(mpm_type));
        } else {
            snprintf(buf, sizeof(buf), "E %.0f | nu %.2f | start %.0fK | fiber %.0f deg | mat %u",
                     youngs_modulus, poisson_ratio, initial_temp, angle, static_cast<u32>(mpm_type));
        }
    } else {
        if (show_density) {
            snprintf(buf, sizeof(buf), "E %.0f | nu %.2f | start %.0fK | rho %.0f | mat %u",
                     youngs_modulus, poisson_ratio, initial_temp, rest_density, static_cast<u32>(mpm_type));
        } else {
            snprintf(buf, sizeof(buf), "E %.0f | nu %.2f | start %.0fK | mat %u",
                     youngs_modulus, poisson_ratio, initial_temp, static_cast<u32>(mpm_type));
        }
    }
    return std::string(buf);
}

BatchRecord make_batch_record(const char* label, const char* description,
                              SpawnSolver solver, MPMMaterial mpm_type, vec4 color,
                              f32 youngs_modulus, f32 poisson_ratio,
                              f32 initial_temp, vec2 fiber_dir,
                              f32 recommended_size, const char* recommended_note,
                              bool scene_authored) {
    BatchRecord batch{};
    batch.label = label ? label : "Object";
    batch.description = description ? description : "";
    batch.techniques = technique_summary(solver, mpm_type);
    batch.recommended_size = recommended_size;
    batch.recommended_note = recommended_note ? recommended_note : "Medium blobs are a good first test size.";
    batch.color = color;
    batch.youngs_modulus = youngs_modulus;
    batch.poisson_ratio = poisson_ratio;
    batch.temperature = initial_temp;
    batch.fiber_dir = fiber_dir;
    vec4 thermal = default_thermal_coupling(mpm_type);
    batch.outgassing_scale = thermal.x;
    batch.heat_release_scale = thermal.y;
    batch.cooling_scale = thermal.z;
    batch.loft_scale = thermal.w;
    batch.solver = solver;
    batch.mpm_type = mpm_type;
    batch.scene_authored = scene_authored;
    refresh_batch_properties(batch);
    return batch;
}

void apply_sph_batch_properties(ParticleBuffer& particles, SPHSolver& sph,
                                u32 global_offset, u32 count,
                                MPMMaterial material, f32 initial_temp,
                                f32 gas_constant, f32 viscosity,
                                vec4 color,
                                vec4 thermal_coupling) {
    if (count == 0) return;
    const u32 local_offset = global_offset - particles.range(SolverType::SPH).offset;
    std::vector<vec4> colors(count, color);
    std::vector<u32> mat_ids(count, static_cast<u32>(material));
    std::vector<f32> temps(count, initial_temp);
    std::vector<f32> states(count, 0.0f);
    std::vector<vec4> mat_params(count, vec4(gas_constant, viscosity, 1.0f, 0.0f));
    std::vector<vec4> thermal_params(count);
    vec4 thermal_defaults = default_thermal_coupling(material);
    for (u32 i = 0; i < count; ++i) {
        thermal_params[i] = vec4(
            thermal_defaults.x * thermal_coupling.x,
            thermal_defaults.y * thermal_coupling.y,
            thermal_defaults.z * thermal_coupling.z,
            thermal_defaults.w * thermal_coupling.w);
    }
    particles.upload_colors(global_offset, colors.data(), count);
    particles.upload_material_ids(global_offset, mat_ids.data(), count);
    particles.upload_temperatures(global_offset, temps.data(), count);
    particles.pressures().upload(states.data(), count * sizeof(f32), local_offset * sizeof(f32));
    sph.material_param_buf().upload(mat_params.data(), count * sizeof(vec4), local_offset * sizeof(vec4));
    sph.thermal_coupling_buf().upload(thermal_params.data(), count * sizeof(vec4), local_offset * sizeof(vec4));
}

void place_object(CreationState& state, ParticleBuffer& particles,
                  SPHSolver& sph, MPMSolver& mpm, UniformGrid& grid) {
    const auto& preset = state.custom; // Use customized copy

    // Apply fiber angle from customize panel
    f32 rad = state.fiber_angle * kPi / 180.0f;
    vec2 fiber = vec2(std::cos(rad), std::sin(rad));

    vec4 bcol = batch_color(state.batch_counter);
    vec4 color = vec4(glm::mix(vec3(preset.color), vec3(bcol), 0.4f), 1.0f);

    BatchRecord batch = make_batch_record(
        preset.name, preset.description, preset.solver, preset.mpm_type, color,
        preset.youngs_modulus, preset.poisson_ratio, preset.initial_temp, fiber,
        preset.recommended_size, preset.recommended_note, false);
    batch.fiber_strength = preset.fiber_strength;
    batch.gas_constant = preset.gas_constant;
    batch.viscosity = preset.viscosity;
    vec4 thermal_defaults = default_thermal_coupling(preset.mpm_type);
    batch.outgassing_scale = thermal_defaults.x * preset.outgassing_scale;
    batch.heat_release_scale = thermal_defaults.y * preset.heat_release_scale;
    batch.cooling_scale = thermal_defaults.z * preset.cooling_scale;
    batch.loft_scale = thermal_defaults.w * preset.loft_scale;
    refresh_batch_properties(batch);

    if (preset.solver == SpawnSolver::SPH) {
        u32 before = particles.range(SolverType::SPH).count;
        f32 spacing = sph.params().smoothing_radius * 0.5f;

        if (state.shape == SpawnShape::CIRCLE) {
            sph.spawn_circle(particles, state.preview_pos, state.size, spacing);
        } else if (state.shape == SpawnShape::RECT) {
            vec2 half_extents = shape_half_extents(state.shape, state.size, state.aspect);
            sph.spawn_block(particles, state.preview_pos - half_extents,
                            state.preview_pos + half_extents, spacing);
        } else {
            std::vector<vec2> positions;
            std::vector<f32> shell_seeds;
            build_shape_points(state, spacing, positions, shell_seeds);
            sph.spawn_points(particles, positions);
        }

        u32 after = particles.range(SolverType::SPH).count;
        batch.sph_offset = particles.range(SolverType::SPH).offset + before;
        batch.sph_count = after - before;
        apply_sph_batch_properties(particles, sph, batch.sph_offset, batch.sph_count,
                                   preset.mpm_type, preset.initial_temp,
                                   preset.gas_constant, preset.viscosity,
                                   color,
                                   vec4(preset.outgassing_scale, preset.heat_release_scale,
                                        preset.cooling_scale, preset.loft_scale));
        LOG_INFO("Placed SPH '%s' batch#%u (%u)", preset.name, state.batch_counter, batch.sph_count);
    } else {
        u32 before = particles.range(SolverType::MPM).count;

        MPMParams& mp = mpm.params();
        f32 old_E = mp.youngs_modulus, old_nu = mp.poisson_ratio, old_fs = mp.fiber_strength;
        mp.youngs_modulus = preset.youngs_modulus;
        mp.poisson_ratio = preset.poisson_ratio;
        mp.fiber_strength = preset.fiber_strength;

        f32 spacing = grid.dx() * 0.5f;
        if (state.shape == SpawnShape::CIRCLE) {
            mpm.spawn_circle(particles, state.preview_pos, state.size, spacing,
                             preset.mpm_type, preset.initial_temp, fiber, 1.0f,
                             vec4(batch.outgassing_scale, batch.heat_release_scale, batch.cooling_scale, batch.loft_scale));
        } else if (state.shape == SpawnShape::RECT) {
            vec2 half_extents = shape_half_extents(state.shape, state.size, state.aspect);
            mpm.spawn_block(particles, state.preview_pos - half_extents,
                            state.preview_pos + half_extents, spacing,
                            preset.mpm_type, preset.initial_temp, fiber, 1.0f,
                            vec4(batch.outgassing_scale, batch.heat_release_scale, batch.cooling_scale, batch.loft_scale));
        } else {
            std::vector<vec2> positions;
            std::vector<f32> shell_seeds;
            build_shape_points(state, spacing, positions, shell_seeds);
            mpm.spawn_points(particles, positions, shell_seeds, spacing,
                             preset.mpm_type, preset.initial_temp, fiber, 1.0f,
                             vec4(batch.outgassing_scale, batch.heat_release_scale, batch.cooling_scale, batch.loft_scale));
        }

        mp.youngs_modulus = old_E; mp.poisson_ratio = old_nu; mp.fiber_strength = old_fs;

        u32 after = particles.range(SolverType::MPM).count;
        batch.mpm_offset = particles.range(SolverType::MPM).offset + before;
        batch.mpm_count = after - before;

        std::vector<vec4> colors(batch.mpm_count, color);
        particles.upload_colors(batch.mpm_offset, colors.data(), batch.mpm_count);
        LOG_INFO("Placed MPM '%s' batch#%u (%u)", preset.name, state.batch_counter, batch.mpm_count);
    }

    state.batches.push_back(batch);
    state.batch_counter++;
}

} // namespace ng
