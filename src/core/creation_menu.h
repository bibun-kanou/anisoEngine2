#pragma once

#include "core/types.h"
#include "physics/mpm/mpm_solver.h"
#include <vector>
#include <cmath>
#include <string>

namespace ng {

class ParticleBuffer;
class SPHSolver;
class SpatialHash;
class SDFField;
class UniformGrid;
class Camera;

enum class SpawnShape : i32 {
    CIRCLE = 0,
    RECT = 1,
    BEAM = 2,
    TRIANGLE = 3,
    STAR = 4,
    SHELL_CIRCLE = 5,
    SHELL_RECT = 6
};
constexpr int SHAPE_COUNT = 7;
inline const char* spawn_shape_names[] = {
    "Circle",
    "Rectangle",
    "Beam",
    "Triangle",
    "Star",
    "Shell Circle",
    "Shell Rectangle"
};

inline bool shape_uses_aspect(SpawnShape shape) {
    return shape != SpawnShape::CIRCLE && shape != SpawnShape::SHELL_CIRCLE;
}

inline bool shape_uses_rotation(SpawnShape shape) {
    return shape != SpawnShape::CIRCLE && shape != SpawnShape::SHELL_CIRCLE;
}

inline bool shape_is_shell(SpawnShape shape) {
    return shape == SpawnShape::SHELL_CIRCLE || shape == SpawnShape::SHELL_RECT;
}

inline SpawnShape base_preview_shape(SpawnShape shape) {
    switch (shape) {
    case SpawnShape::SHELL_CIRCLE: return SpawnShape::CIRCLE;
    case SpawnShape::SHELL_RECT: return SpawnShape::RECT;
    default: return shape;
    }
}

inline vec2 shape_half_extents(SpawnShape shape, f32 size, f32 aspect) {
    switch (shape) {
    case SpawnShape::CIRCLE:
    case SpawnShape::SHELL_CIRCLE:
        return vec2(size);
    case SpawnShape::RECT:
    case SpawnShape::SHELL_RECT:
        return vec2(size, size * aspect);
    case SpawnShape::BEAM:
        return vec2(size * glm::clamp(aspect, 1.0f, 8.0f), size * 0.18f);
    case SpawnShape::TRIANGLE:
    case SpawnShape::STAR:
        return vec2(size, size * aspect);
    default:
        return vec2(size);
    }
}

inline f32 shape_shell_thickness(SpawnShape shape, vec2 half_extents, f32 spacing_hint = 0.0f) {
    f32 base = glm::max(glm::min(half_extents.x, half_extents.y) * 0.22f, 0.03f);
    if (shape == SpawnShape::SHELL_RECT) base = glm::max(glm::min(half_extents.x, half_extents.y) * 0.18f, 0.03f);
    if (spacing_hint > 0.0f) base = glm::max(base, spacing_hint * 2.25f);
    return base;
}
enum class SpawnSolver : i32 { SPH = 0, MPM = 1 };
enum class PresetCategory : i32 { FLUIDS=0, ELASTIC=1, THERMAL=2, FIRE=3, STRUCTURAL=4 };
constexpr int CATEGORY_COUNT = 5;
inline const char* category_names[] = { "Fluids", "Elastic", "Thermal", "Fire", "Structural" };
enum class PresetQuickTab : i32 {
    ALL = 0,
    KILN_BAKE = 1,
    METAL_BURST = 2,
    SHELL_CORE = 3,
    PRESSURE_POP = 4,
    BIO_COOK = 5,
    FIELD_MEMORY = 6,
    POROUS_TEAR = 7,
    ADV_BAKE_KILN = 8,
    THICKNESS_ENHANCED = 9,
    MOISTURE_BINDER = 10,
    SPH_THERMAL = 11,
    ADAPTIVE_MATTER = 12,
    BIO_AUTOMATA = 13,
    HYBRID_LABS = 14
};
constexpr int QUICK_TAB_COUNT = 15;
inline const char* quick_tab_names[] = {
    "All",
    "Kiln & Bake",
    "Metal & Stress",
    "Shell & Core",
    "Pressure & Pop",
    "Bio & Cooking",
    "Magnetics & Memory",
    "Porous & Tear",
    "Advanced Bake+Kiln",
    "Enhanced by Thickness",
    "Moisture / Pore / Binder",
    "SPH Thermal",
    "Adaptive Matter",
    "Bio / Automata",
    "Hybrid Labs"
};

struct MaterialPreset {
    const char* name;
    const char* description;
    PresetCategory category;
    SpawnSolver solver;
    MPMMaterial mpm_type;

    f32 youngs_modulus;
    f32 poisson_ratio;
    f32 fiber_strength;
    f32 initial_temp;
    vec2 fiber_dir;

    f32 gas_constant;
    f32 viscosity;

    vec4 color;
    f32  recommended_size = 0.28f;
    const char* recommended_note = "Medium blobs are a good first test size.";
    PresetQuickTab quick_tab = PresetQuickTab::ALL;
    f32 outgassing_scale = 1.0f;
    f32 heat_release_scale = 1.0f;
    f32 cooling_scale = 1.0f;
    f32 loft_scale = 1.0f;
    f32 physical_scale = 0.1f; // Real-world size in meters (0.1 = 10cm object)
                                // Gravity scaled by physical_scale / visual_size
                                // So a 10cm wax cube displayed at 0.5 units feels like 10cm
};

inline const std::vector<MaterialPreset>& get_presets() {
    static const std::vector<MaterialPreset> presets = {
        // --- FLUIDS ---
        {"Water",          "Standard SPH water, splashes on impact",
         PresetCategory::FLUIDS, SpawnSolver::SPH, MPMMaterial::SPH_WATER, 0,0,0,300,{1,0}, 8,0.1f, {0.15f,0.4f,0.85f,1}},
        {"Viscous Goo",    "Thick SPH goo with strong damping and slower flow",
         PresetCategory::FLUIDS, SpawnSolver::SPH, MPMMaterial::SPH_VISCOUS_GOO, 0,0,0,300,{1,0}, 5,0.8f, {0.3f,0.8f,0.2f,1}},
        {"Light Oil",      "Thin SPH oil, low viscosity, lighter loft than water",
         PresetCategory::FLUIDS, SpawnSolver::SPH, MPMMaterial::SPH_LIGHT_OIL, 0,0,0,300,{1,0}, 12,0.02f, {0.7f,0.6f,0.2f,1}},
        {"Burning Oil [new][experimental]", "Thermal SPH oil that ignites, self-heats, smokes, and throws hot vapor once lit",
         PresetCategory::FLUIDS, SpawnSolver::SPH, MPMMaterial::SPH_BURNING_OIL, 0,0,0,330,{1,0}, 8,0.14f, {0.78f,0.54f,0.14f,1},
         0.34f, "Use shallow pools or shell circles over a heater. Once lit, this should burn longer than water and keep feeding smoke and hot vapor.", PresetQuickTab::SPH_THERMAL, 0.68f, 1.05f, 0.58f, 0.24f},
        {"Boiling Water [new][experimental]", "Thermal SPH water that churns, vents steam, and becomes visibly livelier once it reaches a boil",
         PresetCategory::FLUIDS, SpawnSolver::SPH, MPMMaterial::SPH_BOILING_WATER, 0,0,0,300,{1,0}, 8,0.09f, {0.44f,0.72f,0.98f,1},
         0.34f, "Use medium pools in cups or bowls. This one is for visible boiling and steam lift rather than combustion.", PresetQuickTab::SPH_THERMAL, 1.05f, 0.38f, 0.82f, 0.20f},
        {"Thermal Syrup [new][experimental]", "SPH syrup that thins when hot, drizzles more easily, then thickens back up as it cools",
         PresetCategory::FLUIDS, SpawnSolver::SPH, MPMMaterial::SPH_THERMAL_SYRUP, 0,0,0,315,{1,0}, 6,0.55f, {0.86f,0.54f,0.22f,1},
         0.28f, "Use short pours, puddles, or dripping beads. Heat should make it runnier, while cooling should make it cling and glob back up.", PresetQuickTab::SPH_THERMAL, 0.18f, 0.18f, 0.84f, 0.06f},
        {"Flash Fluid [new][experimental]", "Low-boiling SPH liquid that flashes into vapor aggressively and can loft itself apart when heated hard",
         PresetCategory::FLUIDS, SpawnSolver::SPH, MPMMaterial::SPH_FLASH_FLUID, 0,0,0,295,{1,0}, 11,0.03f, {0.82f,0.72f,0.50f,1},
         0.26f, "Use small beads or very shallow puddles. This one is intentionally unstable and should feel much more explosive than boiling water.", PresetQuickTab::SPH_THERMAL, 1.30f, 0.56f, 0.40f, 0.42f},
        {"MPM Fluid",      "Grid-based fluid, good for bulk flow",
         PresetCategory::FLUIDS, SpawnSolver::MPM, MPMMaterial::FLUID, 5000,0.3f,0,300,{1,0}, 0,0, {0.1f,0.3f,0.8f,1}},
        {"Lava",           "Hot dense fluid (800K), glows orange",
         PresetCategory::FLUIDS, SpawnSolver::MPM, MPMMaterial::FLUID, 3000,0.3f,0,800,{1,0}, 0,0, {0.9f,0.3f,0.1f,1}},
        {"Lamp Oil [new]", "Flammable MPM liquid that ignites into a burning pool",
         PresetCategory::FLUIDS, SpawnSolver::MPM, MPMMaterial::FLAMMABLE_FLUID, 2200,0.35f,0,300,{1,0}, 0,0, {0.78f,0.62f,0.18f,1}},
        {"Ferrofluid [new][experimental]", "Magnet-responsive liquid. Hold M near it to pull spikes and puddles around. Bright orange for visibility while debugging.",
         PresetCategory::FLUIDS, SpawnSolver::MPM, MPMMaterial::FERRO_FLUID, 4200,0.33f,0,300,{1,0}, 0,0, {0.95f,0.55f,0.15f,1},
         0.26f, "Use a shallow puddle or bead cluster, then hold M near it. Smaller pools show the magnetic pull more clearly.", PresetQuickTab::FIELD_MEMORY, 0.00f, 0.80f, 0.72f, 0.06f},
        {"Soft Iron [new][experimental]", "Magnetizable ferromagnetic solid for the real magnetic field solve. It should pull toward SDF magnets and concentrate in stronger field regions.",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::MAG_SOFT_IRON, 24000,0.22f,0,300,{1,0}, 0,0, {0.56f,0.60f,0.66f,1},
         0.26f, "Use beads, short bars, or shell circles near a permanent magnet scene. This is the cleanest first benchmark for the new solved magnetic field.", PresetQuickTab::FIELD_MEMORY, 0.02f, 0.95f, 0.86f, 0.01f},
        {"Magnetic Rubber [new][experimental]", "Soft magnetizable strip that bends and pulls under the solved magnetic field more readily than soft iron.",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::MAGNETIC_RUBBER, 9000,0.36f,0,300,{1,0}, 0,0, {0.44f,0.48f,0.56f,1},
         0.34f, "Use strips, beams, or shell rectangles near bar magnets. This one is for visibly deforming under the field instead of only translating.", PresetQuickTab::FIELD_MEMORY, 0.02f, 0.92f, 0.84f, 0.02f},
        {"Topology Goo [new][experimental]", "Self-merging slime that wants to read more like a cohesive blob than a puddle at normal gravity, but still tears open under fast stress and heals back together when calm.",
         PresetCategory::FLUIDS, SpawnSolver::MPM, MPMMaterial::TOPO_GOO, 9800,0.38f,0,300,{1,0}, 0,0, {0.18f,0.78f,0.58f,1},
         0.34f, "Use medium blobs, rings, or shell circles. At normal gravity it should stay blob-like; fast pulls should still open holes and ribbons before it rounds back up.", PresetQuickTab::ADAPTIVE_MATTER, 0.04f, 0.28f, 0.92f, 0.04f},
        {"Oobleck Paste [new][experimental]", "A sharper shear-thickening, yield-stress material: it slumps when calm but should now jam much more obviously when struck or dragged hard.",
         PresetCategory::ELASTIC, SpawnSolver::MPM, MPMMaterial::OOBLECK, 8600,0.33f,0,300,{1,0}, 0,0, {0.76f,0.68f,0.46f,1},
         0.30f, "Use chunky pucks or squat piles. Slow pokes should still deform it, while quick hits and strong drag should make it feel noticeably firmer than an ordinary soft solid.", PresetQuickTab::ADAPTIVE_MATTER, 0.02f, 0.20f, 0.88f, 0.02f},
        {"Impact Memory Gel [new][experimental]", "Interaction-driven gel that now hardens faster under repeated hits and keeps dents or learned deformation patterns for longer before relaxing.",
         PresetCategory::ELASTIC, SpawnSolver::MPM, MPMMaterial::IMPACT_GEL, 9200,0.36f,0,300,{1,0}, 0,0, {0.66f,0.46f,0.88f,1},
         0.28f, "Use blobs, bars, or strips. Repeated taps, drops, or hammering should build a much more obvious hardened memory compared with a fresh piece.", PresetQuickTab::ADAPTIVE_MATTER, 0.02f, 0.18f, 0.90f, 0.03f},
        {"Replicator Goo [new][experimental]", "Topology goo retuned for the new bio field: it should bud into lobes, heal back together, and start reading more like a self-organizing soft colony than a plain puddle.",
         PresetCategory::FLUIDS, SpawnSolver::MPM, MPMMaterial::TOPO_GOO, 12400,0.37f,0,315,{1,0}, 0,0, {0.24f,0.82f,0.62f,1},
         0.42f, "Turn on Bio Field in Advanced, or use a bench that enables it. Medium blobs or shell circles should form budding fronts and re-merging pockets instead of only flattening.", PresetQuickTab::BIO_AUTOMATA, 0.06f, 0.34f, 0.92f, 0.05f},
        {"Mycelium Cell [new][experimental]", "Mushroom-like bio solid tuned for the reaction-diffusion pass: warm patches seed a living-looking field that can bias soft duplication, frilling, and spore-bright budding edges.",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::MUSHROOM, 7600,0.29f,0,325,{1,0}, 0,0, {0.76f,0.72f,0.56f,1},
         0.38f, "Use squat caps or bead clusters with Bio Field view on. This one is for self-organizing, mushroom-like lobe growth rather than hard fracture.", PresetQuickTab::BIO_AUTOMATA, 0.36f, 0.42f, 0.74f, 0.20f},
        {"Morph Crumb [new][experimental]", "Bread/crumb body driven by a morphogen field: pore regions can nucleate in patterned fronts so the loaf reads more cellular and less uniformly inflated.",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::CRUMB_LOAF, 8600,0.30f,0,330,{1,0}, 0,0, {0.95f,0.82f,0.56f,1},
         0.48f, "Use loaf-like blobs with Bio Field enabled. You should get patterned crumb pockets and split fronts instead of one even puff everywhere.", PresetQuickTab::BIO_AUTOMATA, 0.92f, 0.78f, 0.76f, 0.14f},
        {"Pattern Skin [new][experimental]", "Maillard skin tuned to show thin reaction fronts, blister bands, and self-organized browning islands from the new automata field.",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::MAILLARD, 9800,0.28f,0,335,{1,0}, 0,0, {0.96f,0.84f,0.54f,1},
         0.40f, "Use thin sheets or shell rectangles with Bio Field enabled. This is the easiest preset for reading traveling fronts and splitting islands on the surface.", PresetQuickTab::BIO_AUTOMATA, 0.30f, 0.70f, 0.62f, 0.08f},
        {"Morph Tissue [new][experimental]", "Differential-growth tissue that reads the bio field as a growth cue, then buckles and folds instead of only drifting along it.",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::MORPH_TISSUE, 11200,0.38f,2.2f,314,{1,0}, 0,0, {0.30f,0.76f,0.56f,1},
         0.46f, "Use medium domes or thick slabs with Bio Field enabled. This one is for lobe growth, buckling, and soft fold formation instead of simple melting.", PresetQuickTab::BIO_AUTOMATA, 0.18f, 0.36f, 0.82f, 0.10f},
        {"Root Weave [new][experimental]", "Anisotropic living strand that creeps along morphogen ridges, thickens into root-like bundles, and keeps more directional structure as it grows.",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::ROOT_WEAVE, 14800,0.31f,4.8f,308,{0,1}, 0,0, {0.40f,0.72f,0.30f,1},
         0.54f, "Use tall beams, brushy strips, or shell rectangles on ramps. This one is for directional tendrils and root-like arches rather than round budding cells.", PresetQuickTab::BIO_AUTOMATA, 0.12f, 0.24f, 0.92f, 0.05f},
        {"Cell Sheet [new][experimental]", "A smoothlife-like membrane proxy: active edges bud outward, wrinkle, and pinch into soft colonies when the bio field stays organized.",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::CELL_SHEET, 9200,0.40f,2.4f,314,{1,0}, 0,0, {0.92f,0.60f,0.70f,1},
         0.42f, "Use shell rectangles, flat domes, or thin disks with Bio Field enabled. The edge should frill and bud into lobes more than the center.", PresetQuickTab::BIO_AUTOMATA, 0.16f, 0.28f, 0.88f, 0.08f},
        {"Pulse Tissue [new][experimental]", "A more aggressive morph-tissue variant tuned to show breathing fronts, stronger lobe growth, and clearer buckling under the bio/automata coupling.",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::MORPH_TISSUE, 9800,0.39f,2.8f,320,{1,0}, 0,0, {0.38f,0.86f,0.62f,1},
         0.50f, "Use medium domes or squat slabs with Automata Test. This variant is meant to show pulsing and edge-led growth more clearly than the baseline Morph Tissue preset.", PresetQuickTab::BIO_AUTOMATA, 0.22f, 0.42f, 0.78f, 0.14f},
        {"Scout Root [new][experimental]", "A more reactive root-weave tuned to display longer exploratory tendrils and stronger air-guided steering before the bulk catches up.",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::ROOT_WEAVE, 13200,0.30f,5.2f,312,{0,1}, 0,0, {0.54f,0.82f,0.34f,1},
         0.52f, "Use narrow strips or tall beams near ramps with Bio Drive or Automata Drive visible. The tip should scout farther than the trunk.", PresetQuickTab::BIO_AUTOMATA, 0.16f, 0.30f, 0.90f, 0.08f},
        {"Regrowth Sheet [new][experimental]", "A hotter-running cell sheet variant for the fire/regrowth benches: the edge dies back more clearly under heat and then re-colonizes from cooler fronts.",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::CELL_SHEET, 8800,0.41f,2.6f,320,{1,0}, 0,0, {0.98f,0.70f,0.76f,1},
         0.46f, "Use thin slabs or shell rectangles, then burn one side. This variant is tuned to make the kill-and-regrow cycle easier to see.", PresetQuickTab::BIO_AUTOMATA, 0.22f, 0.36f, 0.76f, 0.14f},
        {"Ash Regrowth Tissue [new][experimental]", "Burns into fragile ash first, then uses that ash scaffold as a guide to regrow into a warped, not-quite-original body instead of simply snapping back.",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::ASH_REGROWTH, 9800,0.34f,2.2f,316,{0,1}, 0,0, {0.68f,0.78f,0.70f,1},
         0.50f, "Use medium slabs, pillars, or shell rectangles with Bio Field enabled. Burn one side: it should darken to ash, slump or shatter easily, then regrow back in a deformed way.", PresetQuickTab::BIO_AUTOMATA, 0.08f, 0.18f, 0.92f, 0.04f},
        {"Regrowth Wall Seed [new]", "Hybrid-lab pick: a chunky ash-regrowth body sized for the new wall bench, where burning should blacken, weaken, then grow back from cooler edges.",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::ASH_REGROWTH, 11600,0.34f,2.4f,314,{1,0}, 0,0, {0.82f,0.90f,0.82f,1},
         0.56f, "Use blocky walls or squat pillars. This copy is tuned for the Hybrid Regrowth Wall scene and should keep enough ash scaffold to make recovery legible.", PresetQuickTab::HYBRID_LABS, 0.10f, 0.20f, 0.92f, 0.05f},
        {"Bio Mine Cap [new][experimental]", "Hybrid-lab pick: a hotter-running living skin meant to sit over or beside pressure seeds so fire damage, regrowth, and blast scars can all read in one setup.",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::CELL_SHEET, 9400,0.40f,2.8f,320,{1,0}, 0,0, {0.96f,0.68f,0.78f,1},
         0.40f, "Use thin caps, membranes, or shell rectangles over compact bodies. This copy is for mixed fire-plus-regrowth experiments, not calm colony growth.", PresetQuickTab::HYBRID_LABS, 0.20f, 0.36f, 0.78f, 0.14f},
        {"Kiln Shell Blank [new]", "Hybrid-lab pick: a thicker glaze-clay body meant for the combined kiln / ordnance benches, where shell firing should still read before impact or blast damage takes over.",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::GLAZE_CLAY, 12200,0.22f,0,300,{1,0}, 0,0, {0.94f,0.80f,0.68f,1},
         0.50f, "Use bowls, tiles, or thick plugs. This one is sized for the Hybrid Kiln Process and Pressure Pottery scenes, where you want shell-setting before rupture.", PresetQuickTab::HYBRID_LABS, 0.38f, 0.92f, 0.78f, 0.04f},
        {"Soft HEAT Charge Seed [new][experimental]", "Hybrid-lab pick: a calmer sealed-charge bead for shaped-charge and layered-target testing without the full-strength room-filling blast of the heavier bomb presets.",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::SEALED_CHARGE, 26000,0.20f,0,300,{1,0}, 0,0, {0.90f,0.36f,0.16f,1},
         0.18f, "Use compact beads or short plugs in the Hybrid Soft HEAT Range. This copy is for local rupture and hot jet behavior, not giant demolition blasts.", PresetQuickTab::HYBRID_LABS, 1.28f, 1.18f, 0.72f, 0.02f},
        {"Ferro Splash Slurry [new][experimental]", "Hybrid-lab pick: a visibly magnetized ferro puddle meant for the combined magnetic splash bench, where field lines, attraction, and deformation are all easy to read.",
         PresetCategory::FLUIDS, SpawnSolver::MPM, MPMMaterial::FERRO_FLUID, 4400,0.33f,0,300,{1,0}, 0,0, {0.95f,0.55f,0.15f,1},
         0.28f, "Use shallow puddles or bead piles in the Hybrid Ferro Splash scene. Smaller pools still show spikes better than giant lakes.", PresetQuickTab::HYBRID_LABS, 0.00f, 0.72f, 0.78f, 0.06f},
        {"Armor Gel Pad [new][experimental]", "Hybrid-lab pick: a defense pad for the combined armor benches. It should stay readable as a bumper or sacrificial layer instead of behaving like a generic soft blob.",
         PresetCategory::ELASTIC, SpawnSolver::MPM, MPMMaterial::IMPACT_GEL, 10800,0.35f,0,300,{1,0}, 0,0, {0.72f,0.52f,0.92f,1},
         0.34f, "Use squat pads, slabs, or plugs in front of harder backers. This copy is for the Hybrid Oobleck Armor scene and repeated impact comparison.", PresetQuickTab::HYBRID_LABS, 0.02f, 0.16f, 0.92f, 0.02f},

        // --- ELASTIC ---
        {"Soft Jelly",     "Very wobbly, deforms easily",
         PresetCategory::ELASTIC, SpawnSolver::MPM, MPMMaterial::ELASTIC, 3000,0.35f,0,300,{1,0}, 0,0, {0.2f,0.8f,0.3f,1}},
        {"Rubber",         "Bouncy, holds shape under gravity",
         PresetCategory::ELASTIC, SpawnSolver::MPM, MPMMaterial::ELASTIC, 10000,0.4f,0,300,{1,0}, 0,0, {0.3f,0.6f,0.2f,1}},
        {"Stiff Elastic",  "Barely deforms, very rigid",
         PresetCategory::ELASTIC, SpawnSolver::MPM, MPMMaterial::ELASTIC, 30000,0.25f,0,300,{1,0}, 0,0, {0.15f,0.5f,0.15f,1}},
        {"Bouncy Ball",    "High restitution, near-incompressible",
         PresetCategory::ELASTIC, SpawnSolver::MPM, MPMMaterial::ELASTIC, 20000,0.45f,0,300,{1,0}, 0,0, {0.9f,0.3f,0.5f,1}},
        {"Soft Snow",      "Crumbles easily under load",
         PresetCategory::ELASTIC, SpawnSolver::MPM, MPMMaterial::SNOW, 2000,0.2f,0,300,{1,0}, 0,0, {0.9f,0.92f,0.98f,1}},
        {"Packed Snow",    "Holds shape, breaks on impact",
         PresetCategory::ELASTIC, SpawnSolver::MPM, MPMMaterial::SNOW, 8000,0.3f,0,300,{1,0}, 0,0, {0.85f,0.88f,0.95f,1}},
        {"Ice Chunks",     "Hard, shatters into pieces",
         PresetCategory::ELASTIC, SpawnSolver::MPM, MPMMaterial::SNOW, 25000,0.25f,0,300,{1,0}, 0,0, {0.7f,0.85f,0.95f,1}},

        // --- THERMAL ---
        {"Wax (meltable)", "Softens and deforms when heated",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::THERMAL, 5000,0.3f,0,300,{1,0}, 0,0, {0.85f,0.75f,0.5f,1}},
        {"Soft Clay",      "Deformable, softens more with heat",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::THERMAL, 3000,0.35f,0,300,{1,0}, 0,0, {0.6f,0.4f,0.25f,1}},
        {"Hot Metal",      "Starts hot (600K), softens with heat — will collapse if heated more",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::THERMAL, 30000,0.3f,0,600,{1,0}, 0,0, {0.7f,0.2f,0.1f,1}},
        {"Wet Clay",       "Hardens permanently above 400K (no cracks)",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::HARDEN, 5000,0.35f,0,300,{1,0}, 0,0, {0.55f,0.45f,0.35f,1}},
        {"Epoxy Resin",    "Cures into rigid solid when heated",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::HARDEN, 4000,0.3f,0,300,{1,0}, 0,0, {0.7f,0.65f,0.4f,1}},
        {"Ceramic Slip",   "Hardens and becomes stiff when fired",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::HARDEN, 6000,0.25f,0,300,{1,0}, 0,0, {0.8f,0.75f,0.7f,1}},
        {"Hot Glass [experimental]", "Viscous when hot, stiffens again as it cools",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::GLASS, 22000,0.22f,0,650,{1,0}, 0,0, {0.95f,0.55f,0.2f,1},
         0.42f, "Use a medium-hot blob. Too thin and it will slump before the re-hardening reads clearly."},
        {"Hot Glass Gob [new][experimental]", "A firmer hot-glass variant tuned for chunky gobs instead of runny sheets",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::GLASS, 42000,0.14f,0,620,{1,0}, 0,0, {0.98f,0.60f,0.22f,1},
         0.24f, "Best as a compact gob or bead. This variant is meant to stay more sculptural at the suggested size."},
        {"Glass Filament [new][experimental]", "First-pass codim-style hot glass that can be pulled into longer glowing filaments before it cools brittle",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::FILAMENT_GLASS, 26000,0.24f,2.5f,760,{1,0}, 0,0, {0.98f,0.78f,0.34f,1},
         0.28f, "Use a compact hot gob and pull it with Spring Drag. This one is for threads and necking, not stable panes.", PresetQuickTab::SHELL_CORE},
        {"Ice",            "Melts above 500K with latent heat plateau",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::PHASE, 15000,0.3f,0,100,{1,0}, 0,0, {0.3f,0.6f,0.95f,1}},
        {"Ice Block [new]", "A firmer phase-change ice tuned for blocky pieces that keep shape before melting",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::PHASE, 28000,0.22f,0,100,{1,0}, 0,0, {0.45f,0.72f,0.98f,1},
         0.26f, "Use a chunky cube or puck. It should stay blocky much longer than the original Ice preset."},
        {"Frozen Gel",     "Soft frozen blob, melts into goo",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::PHASE, 5000,0.35f,0,50,{1,0}, 0,0, {0.4f,0.8f,0.9f,1}},
        {"Butter",         "Soft, melts at low temperature",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::PHASE, 3000,0.4f,0,280,{1,0}, 0,0, {0.95f,0.85f,0.4f,1}},
        {"Terracotta",     "Cures then cracks into clean chunks on impact",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::CERAMIC, 8000,0.25f,0,300,{1,0}, 0,0, {0.8f,0.55f,0.35f,1}},
        {"Porcelain",      "Hard ceramic, shatters into fragments",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::CERAMIC, 12000,0.2f,0,300,{1,0}, 0,0, {0.9f,0.88f,0.85f,1}},
        {"Bloom Clay [new]", "Fires rigid, builds internal pressure, bursts into petal-like chunks",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::BLOOM, 10000,0.22f,0,300,{1,0}, 0,0, {0.88f,0.62f,0.44f,1},
         0.42f, "Use kiln-sized chunks or stout bowls. This one is best for harden-then-burst flower-like breakup.", PresetQuickTab::KILN_BAKE},
        {"Kiln Reed [new][experimental]", "Heat-cures hard, then self-splinters into hot fibers along the grain",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::SPLINTER, 11000,0.2f,5.5f,300,{0,1}, 0,0, {0.84f,0.68f,0.42f,1},
         0.46f, "Tall, grain-aligned columns show the harden-then-splinter behavior much better than tiny clumps.", PresetQuickTab::KILN_BAKE},
        {"Bread Dough [new][experimental]", "Soft dough that puffs from internal gas, sets a little, then tears open if overpressured",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::BREAD, 7200,0.31f,0,300,{1,0}, 0,0, {0.86f,0.74f,0.50f,1},
         0.56f, "Use a loaf-sized blob or thick puck. It should now stand up cold, then puff and keep a bread-like body longer before venting.", PresetQuickTab::KILN_BAKE},
        {"Puff Clay [new][experimental]", "Dense clay that cures hot, swells from trapped gas, and bursts into crumbly chunks",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::PUFF_CLAY, 9500,0.24f,0,300,{1,0}, 0,0, {0.82f,0.66f,0.48f,1},
         0.44f, "A squat kiln-sized lump makes the swelling and crumble easier to read than a thin strip.", PresetQuickTab::KILN_BAKE},
        {"Glaze Clay [new][experimental]", "Shell-first pottery blank: outer skin vitrifies and glazes while the core lags behind",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::GLAZE_CLAY, 11000,0.23f,0,300,{1,0}, 0,0, {0.92f,0.78f,0.64f,1},
         0.46f, "Use medium kiln slabs or bowls. This one needs enough thickness for the shell/core split to show up.", PresetQuickTab::SHELL_CORE},
        {"Crazing Tile [new][experimental]", "A stiffer glaze-clay variant tuned to craze and crack more than it drips",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::GLAZE_CLAY, 13500,0.18f,0,300,{1,0}, 0,0, {0.95f,0.80f,0.70f,1},
         0.40f, "Use thicker tiles or bowls. This one is for shell cracking and glaze crazing more than smooth flow.", PresetQuickTab::SHELL_CORE},
        {"Glaze Drip [new][experimental]", "Pottery blank with a hotter, runnier glaze shell that can droop and pool while the core stays ceramic",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::GLAZE_DRIP, 9000,0.27f,0,300,{1,0}, 0,0, {0.96f,0.82f,0.62f,1},
         0.48f, "Use bowls, tiles, or chunky ornaments. This variant is for visible shell glaze flow rather than clean shell cracking.", PresetQuickTab::SHELL_CORE},
        {"Crust Dough [new][experimental]", "Bread-like dough with a drying crust: outside sets first while the core keeps expanding",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::CRUST_DOUGH, 8200,0.30f,0,300,{1,0}, 0,0, {0.90f,0.76f,0.52f,1},
         0.60f, "Use a chunky loaf. The shell should now trap gas more reliably and stay bread-like after the rise instead of slumping flat.", PresetQuickTab::SHELL_CORE},
        {"Steam Bun [new][experimental]", "A springier shell/core dough that rises rounder, with softer skin and steamier internal lift than crust dough",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::STEAM_BUN, 7600,0.30f,0,300,{1,0}, 0,0, {0.96f,0.84f,0.58f,1},
         0.54f, "Round buns or squat domes work best. This one is tuned to keep a puffy body instead of becoming a flat crust shell.", PresetQuickTab::SHELL_CORE},
        {"Cheese Pull [new][experimental]", "Warm melt that softens into stretchy strands and necked filaments instead of immediately snapping apart",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::CHEESE_PULL, 6200,0.36f,1.8f,320,{1,0}, 0,0, {0.98f,0.86f,0.40f,1},
         0.26f, "Use a short warm blob or strip, then pull it apart with Spring Drag. It is tuned for strings more than clean puddling.", PresetQuickTab::SHELL_CORE},
        {"Pita Pocket [new][experimental]", "A flatter steam-bun variant tuned to balloon first, then vent and wrinkle into a pocket-like shell",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::STEAM_BUN, 6800,0.28f,0,300,{1,0}, 0,0, {0.94f,0.80f,0.50f,1},
         0.46f, "Use squat disks or flat buns. This one is meant to balloon outward before it stripes and vents.", PresetQuickTab::SHELL_CORE},
        {"Crumb Loaf [new][experimental]", "Bread-focused porous loaf: drying sets a scaffold, pores persist, and overpressure vents through localized crumb tears",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::CRUMB_LOAF, 8400,0.30f,0,300,{1,0}, 0,0, {0.95f,0.80f,0.54f,1},
         0.58f, "Use chunky boules or loafs. This one is for persistent crumb and vent seams, not just a temporary puff.", PresetQuickTab::POROUS_TEAR, 0.95f, 0.72f, 0.78f, 0.12f},
        {"Vent Crumb [new][experimental]", "Moisture-migration loaf: shell dries first, the wetter core vents through bigger channels, and the baked body keeps tunnel-like crumb instead of only round pores",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::VENT_CRUMB, 9800,0.28f,2.4f,300,{0,1}, 0,0, {0.97f,0.82f,0.58f,1},
         0.60f, "Use tall boules or loafs with real thickness. This one is for chimney-like vent channels and baked tunnel crumb, not just puff-and-crack.", PresetQuickTab::ADV_BAKE_KILN, 0.82f, 0.68f, 0.84f, 0.08f},
        {"Open Crumb [new][experimental]", "Bubble-coalescing loaf: pores merge into larger chambers, the shell stays set, and the baked body keeps more open alveoli instead of closing back down",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::OPEN_CRUMB, 11200,0.27f,2.0f,300,{0,1}, 0,0, {0.99f,0.84f,0.62f,1},
         0.60f, "Use thick boules, domes, or shell circles with some real volume. This one is for larger retained crumb chambers and stronger post-rise structure.", PresetQuickTab::ADV_BAKE_KILN, 0.88f, 0.60f, 0.88f, 0.08f},
        {"Bisque Clay [new][experimental]", "Porous pottery body: burns out moisture and pore-formers, sinters from the shell inward, then shrink-cracks like bisque ware",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::BISQUE, 11800,0.22f,0,300,{1,0}, 0,0, {0.90f,0.78f,0.68f,1},
         0.44f, "Use bowls, cups, or tiles with real thickness. This one needs enough body for porous bisque and shell shrink mismatch to show.", PresetQuickTab::POROUS_TEAR, 0.55f, 0.82f, 0.78f, 0.06f},
        {"Vitreous Clay [new][experimental]", "Dense vitrifying kiln body: shrinks and densifies harder than stoneware, with tighter warping and denser fired skin",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::VITREOUS_CLAY, 16800,0.18f,0,300,{1,0}, 0,0, {0.74f,0.66f,0.60f,1},
         0.46f, "Use thicker cups, mugs, or tiles. This one is for visible hard firing, tighter shrinkage, and denser vitrified body than bisque or stoneware.", PresetQuickTab::ADV_BAKE_KILN, 0.28f, 0.94f, 0.78f, 0.03f},
        {"Sinter-Lock Clay [new][experimental]", "Stronger kiln body that shell-sinters, shrinks, and then locks its fired form more aggressively instead of simply collapsing when hot",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::SINTER_LOCK, 18600,0.17f,0,300,{1,0}, 0,0, {0.68f,0.60f,0.56f,1},
         0.48f, "Use mugs, bowls, shell rectangles, or thick tiles. This one is for visible shrink-set and fired shape retention more than porous collapse.", PresetQuickTab::ADV_BAKE_KILN, 0.24f, 0.98f, 0.82f, 0.02f},
        {"Tear Skin [new][experimental]", "Thin heated skin or pastry-like sheet that dries, weakens into porous bands, and rips into long venting tears",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::TEAR_SKIN, 7600,0.30f,2.2f,300,{1,0}, 0,0, {0.96f,0.88f,0.68f,1},
         0.30f, "Use thin strips and skins, then heat and pull. This one is meant to localize tears instead of just melting or uniformly cracking.", PresetQuickTab::POROUS_TEAR, 0.75f, 0.70f, 0.82f, 0.10f},
        {"Laminated Pastry [new][experimental]", "Layered pastry sheet: steam separates layers, the shell browns, and the sheet flakes into delaminated ribbons",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::LAMINATED_PASTRY, 7200,0.29f,2.8f,300,{1,0}, 0,0, {0.98f,0.86f,0.58f,1},
         0.42f, "Use flat strips, folded sheets, or shell rectangles. This one is for flaky lift and layer peeling more than a round loaf rise.", PresetQuickTab::BIO_COOK, 0.82f, 0.68f, 0.82f, 0.10f},
        {"Stoneware [new][experimental]", "Denser pottery body that vitrifies and shrinks harder than bisque, with warp-and-craze firing instead of open pores",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::STONEWARE, 14500,0.20f,0,300,{1,0}, 0,0, {0.72f,0.62f,0.56f,1},
         0.44f, "Use thicker mugs, bowls, or tiles. This one is for shrinkage, hard firing, and denser craze patterns rather than porous bisque holes.", PresetQuickTab::SHELL_CORE, 0.34f, 0.88f, 0.76f, 0.04f},
        {"Blister Glaze [new][experimental]", "Reactive glaze shell: the surface fires glossy, traps volatiles into blister pockets, then vents pits or flakes while the core stays ceramic",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::BLISTER_GLAZE, 11200,0.24f,0,300,{1,0}, 0,0, {0.98f,0.84f,0.64f,1},
         0.40f, "Use medium-thick tiles, beads, or shell rectangles. This one is for blistered glaze skin and vent pits rather than clean dripping.", PresetQuickTab::ADV_BAKE_KILN, 0.62f, 0.84f, 0.80f, 0.05f},
        {"Quench Steel [new][experimental]", "Thermoelastic metal: heats soft, expands under constraint, then cools back harder and stress-prone",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::THERMO_METAL, 28000,0.28f,0,300,{1,0}, 0,0, {0.62f,0.64f,0.70f,1},
         0.38f, "Long bars and bridges are the best test. The effect reads when one side heats much more than the other.", PresetQuickTab::METAL_BURST},
        {"Spring Steel Strip [new][experimental]", "A slimmer thermoelastic steel tuned for bendy hot strips and stronger cool-set springback",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::THERMO_METAL, 22000,0.24f,0,300,{1,0}, 0,0, {0.70f,0.74f,0.82f,1},
         0.24f, "Thin strips and reeds are the best test. Try it in Stress Forge if you want the bend-and-recover read quickly.", PresetQuickTab::METAL_BURST},
        {"Tempering Blade [new][experimental]", "A heavier thermoelastic blade blank for uneven heating, bowing, and cool-set hardening tests",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::THERMO_METAL, 34000,0.26f,0,300,{1,0}, 0,0, {0.76f,0.80f,0.88f,1},
         0.30f, "Use long narrow blades or bars. It is tuned to show uneven-heat bowing more clearly than the chunkier quench steel preset.", PresetQuickTab::METAL_BURST},
        {"Maillard Slice [new][experimental]", "Cooking slab that dries and browns on the outside, with small steam blisters instead of a full bread rise",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::MAILLARD, 7800,0.30f,0,300,{1,0}, 0,0, {0.96f,0.84f,0.54f,1},
         0.34f, "Use medium slices, buns, or patties. This one is for surface browning and cooking-shell changes more than big expansion.", PresetQuickTab::BIO_COOK, 0.55f, 0.70f, 0.72f, 0.08f},
        {"Mushroom Cap [new][experimental]", "Porous cap-and-stem style bio solid that wilts, wrinkles, and vents a dusty spore plume when heated",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::MUSHROOM, 5200,0.32f,0,300,{0,1}, 0,0, {0.72f,0.68f,0.52f,1},
         0.32f, "Try squat domes or a cap-on-stem silhouette. It reads best when the cap has enough thickness to wilt before it tears.", PresetQuickTab::BIO_COOK, 0.65f, 0.45f, 0.82f, 0.20f},
        {"Spore Puff [new][experimental]", "A softer mushroom-like puff that sheds a stronger cool spore cloud and lifts more gently than it burns",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::MUSHROOM, 2600,0.36f,0,300,{1,0}, 0,0, {0.84f,0.80f,0.66f,1},
         0.22f, "Use small puffballs or domes. Heat it from below if you want the spore-like plume to read clearly.", PresetQuickTab::BIO_COOK, 1.15f, 0.30f, 0.88f, 0.36f},
        {"Memory Wax Bloom [new][experimental]", "Thermal wax with stronger cool-set recovery. It softens hot, deforms, then tries to pull back toward its original rest shape",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::MEMORY_WAX, 7000,0.34f,0,300,{1,0}, 0,0, {0.92f,0.76f,0.42f,1},
         0.26f, "Use rings, beams, or bent blobs. Heat, drag, then cool it to watch it recover instead of staying fully slumped.", PresetQuickTab::FIELD_MEMORY, 0.08f, 0.85f, 0.80f, 0.04f},
        {"Self-Heal Putty [new][experimental]", "A softer memory-wax cousin that stays deformed while warm but slowly rounds itself back up once it cools",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::MEMORY_WAX, 4200,0.38f,0,300,{1,0}, 0,0, {0.96f,0.82f,0.58f,1},
         0.22f, "Small blobs and bent worms are the best test. This one is for visible recovery rather than crisp structural snapback.", PresetQuickTab::FIELD_MEMORY, 0.03f, 0.80f, 0.72f, 0.02f},

        // --- FIRE ---
        {"Wood Log",       "Burns with fiber structure, collapses as it chars",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::BURNING, 12000,0.25f,2,300,{0,1}, 0,0, {0.5f,0.3f,0.15f,1},
         0.30f, "Medium logs keep a hot core and readable charring longer than thin sticks.", PresetQuickTab::ALL, 0.90f, 1.18f, 0.62f},
        {"Wood Log Firm [new]", "A sturdier burnable log that keeps a trunk-like silhouette longer before charring through",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::BURNING, 22000,0.20f,3.5f,300,{0,1}, 0,0, {0.46f,0.27f,0.13f,1},
         0.30f, "Tall short logs work best. This copy is tuned to read more like wood mass and less like a soft rope.", PresetQuickTab::ALL, 0.88f, 1.24f, 0.48f},
        {"Paper",          "Burns quickly, light ash",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::BURNING, 2000,0.3f,0,300,{1,0}, 0,0, {0.9f,0.85f,0.7f,1}},
        {"Paper Sheet [new]", "A firmer sheet-like paper variant that still burns fast without immediately flattening into a line",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::BURNING, 6500,0.18f,1.0f,300,{1,0}, 0,0, {0.94f,0.90f,0.78f,1},
         0.18f, "Use a thin but not giant strip. This copy is meant for readable sheet motion at the suggested size."},
        {"Coal",           "Hard, slow burn, intense heat",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::BURNING, 20000,0.2f,0,300,{1,0}, 0,0, {0.42f,0.30f,0.22f,1},
         0.24f, "Chunky briquettes or mounds work best. Coal is tuned to stay hot and keep feeding heat longer than wood.", PresetQuickTab::ALL, 0.78f, 1.55f, 0.18f},
        {"Charcoal Briquette [new]", "Dense glowing fuel block that catches slower but holds heat for a long time once lit",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::EMBER, 14000,0.18f,0,300,{1,0}, 0,0, {0.46f,0.34f,0.26f,1},
         0.24f, "Use chunky briquettes or a short pile. This one is meant for persistent red-hot burn and lingering heat.", PresetQuickTab::ALL, 0.82f, 1.58f, 0.14f},
        {"Dry Leaves",     "Very light, burns fast, floating embers",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::BURNING, 1000,0.35f,0.5f,300,{1,0}, 0,0, {0.6f,0.5f,0.2f,1}},
        {"Dry Leaves Mat [new]", "A slightly firmer leaf mat that still burns fast but doesn’t immediately pancake under its own weight",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::BURNING, 3600,0.18f,1.3f,300,{1,0}, 0,0, {0.66f,0.56f,0.22f,1},
         0.20f, "Try a small clump or mat. This copy is for more legible leafy collapse instead of instant ribboning."},
        {"Torch (pre-lit)","Already above ignition, burns on spawn",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::BURNING, 8000,0.25f,2,500,{0,1}, 0,0, {0.8f,0.4f,0.1f,1}},
        {"Sparkler",       "Intense sparks that float and glow for a long time",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::EMBER, 5000,0.3f,0,300,{1,0}, 0,0, {0.9f,0.6f,0.2f,1}},
        {"Hot Embers",     "Already hot, ignites immediately, long-lasting glow",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::EMBER, 3000,0.3f,0,500,{1,0}, 0,0, {1.0f,0.5f,0.15f,1},
         0.18f, "Small hot clusters glow for a long time and can relight nearby fuel.", PresetQuickTab::ALL, 1.0f, 1.25f, 0.75f},
        {"Magnesium",      "Bright white burn, very intense heat",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::EMBER, 8000,0.25f,0,300,{1,0}, 0,0, {0.85f,0.85f,0.9f,1}},
        {"Pitch Knot [new][experimental]", "Resinous wood knot that burns hot and smoky for a long time before finally collapsing",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::BURNING, 10000,0.24f,1.5f,300,{0,1}, 0,0, {0.32f,0.18f,0.10f,1},
         0.22f, "Use dense little knots or plugs. It is tuned for persistent smoky flame and a hotter core than plain wood.", PresetQuickTab::ALL, 1.05f, 1.48f, 0.30f},
        {"Firecracker Pellet [new][experimental]", "Heats like a shell, pressurizes, then vents and bursts into hot fragments",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::FIRECRACKER, 18000,0.22f,0,300,{1,0}, 0,0, {0.82f,0.16f,0.12f,1},
         0.16f, "Small pellets are best. Pack a few under a lid or block if you want the burst to read strongly.", PresetQuickTab::PRESSURE_POP},
        {"Sealed Charge [new][experimental]", "Airtight explosive bead with a much tighter shell/core hold: it should stay compact while pressure builds, then rupture faster and vent hotter than the porous firecracker variants",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::SEALED_CHARGE, 24000,0.20f,0,300,{1,0}, 0,0, {0.70f,0.16f,0.10f,1},
         0.18f, "Use beads, shell circles, or short plugs. This one is the cleaner airtight-burst benchmark if you want a sharper pop instead of a smoky vent.", PresetQuickTab::PRESSURE_POP, 1.55f, 1.70f, 0.60f, 0.03f},
        {"Smoke Firecracker [new][experimental]", "A smokier explosive pellet tuned for a longer fuse cloud before the burst kicks out",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::FIRECRACKER, 14000,0.18f,0,300,{1,0}, 0,0, {0.92f,0.22f,0.10f,1},
         0.18f, "Use a compact pellet or short chain. This copy is meant to read more like a smoky firecracker than a clean pop.", PresetQuickTab::PRESSURE_POP},
        {"Gunpowder [new][experimental]", "Fast-burning granular charge that builds pressure quickly and kicks nearby objects hard",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::FIRECRACKER, 9000,0.18f,0,300,{1,0}, 0,0, {0.50f,0.42f,0.32f,1},
         0.24f, "Make a compact mound or short fuse trail under a heavier object to see the strongest kick.", PresetQuickTab::PRESSURE_POP},
        {"Popping Resin [new][experimental]", "Chars on the outside, brews hot volatiles inside, then blisters and bursts into smoky fragments",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::REACTIVE_BURN, 15000,0.24f,0,300,{1,0}, 0,0, {0.56f,0.20f,0.12f,1},
         0.24f, "Use thumb-sized beads or short plugs in a confined space. The shell needs a few layers before the pop reads strongly.", PresetQuickTab::PRESSURE_POP},
        {"Popping Seeds [new][experimental]", "Tiny reactive kernels that toast, build pressure, and snap into little smoky pops",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::REACTIVE_BURN, 9000,0.20f,0,300,{1,0}, 0,0, {0.82f,0.72f,0.34f,1},
         0.12f, "Use lots of tiny beads or a shallow mound on a hot plate. This one is tuned for many little pops instead of one big rupture.", PresetQuickTab::PRESSURE_POP},
        {"Popcorn Charge [new][experimental]", "A puffier popping-seed variant tuned for bigger jumps and brighter little bursts under heat",
         PresetCategory::FIRE, SpawnSolver::MPM, MPMMaterial::REACTIVE_BURN, 7600,0.22f,0,300,{1,0}, 0,0, {0.92f,0.82f,0.40f,1},
         0.14f, "Use a shallow pan of tiny kernels. This copy is meant to kick upward harder and read more like popcorn than resin snaps.", PresetQuickTab::PRESSURE_POP},

        // --- STRUCTURAL (ANISO: heat → charring, fibers weaken) ---
        {"Soft Wood (H)",  "Chars when heated, fibers weaken. Heat to darken + soften",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::ANISO, 5000,0.3f,2,300,{1,0}, 0,0, {0.7f,0.45f,0.2f,1}},
        {"Soft Wood Beam [new]",  "A firmer softwood copy that keeps a plank-like body longer before it chars through",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::ANISO, 11000,0.22f,3.2f,300,{1,0}, 0,0, {0.76f,0.50f,0.24f,1},
         0.26f, "Use short beams or thick strips. This copy is for a more legible wood body at the suggested size.", PresetQuickTab::FIELD_MEMORY, 0.95f, 0.90f, 0.72f, 0.08f},
        {"Hard Wood (H)",  "Stiff wood. Heat → chars black, becomes brittle",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::ANISO, 20000,0.25f,4,300,{1,0}, 0,0, {0.5f,0.3f,0.1f,1}},
        {"Hard Wood Beam [new]",  "A stiffer hardwood copy tuned to hold beams and planks up without the old soft sag",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::ANISO, 28000,0.18f,5.2f,300,{1,0}, 0,0, {0.56f,0.34f,0.12f,1},
         0.28f, "Use beams, planks, or stout branches. This one is the easier wood benchmark if the original feels too slumpy.", PresetQuickTab::FIELD_MEMORY, 0.92f, 0.95f, 0.66f, 0.06f},
        {"Fiber (V)",      "Strong vertical fibers. Heat → fibers char + weaken",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::ANISO, 8000,0.3f,3,300,{0,1}, 0,0, {0.8f,0.5f,0.2f,1}},
        {"Rope",           "Flexible fibers. Heat weakens rapidly, snaps",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::ANISO, 3000,0.35f,5,300,{0,1}, 0,0, {0.65f,0.55f,0.35f,1}},

        // --- STRUCTURAL (FRACTURE: heat → lower threshold, more brittle) ---
        {"Brittle (V)",    "Breaks under strain. Heat makes it crack easier",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::BRITTLE, 15000,0.25f,3,300,{0,1}, 0,0, {0.55f,0.35f,0.15f,1}},
        {"Tough (V)",      "High threshold. Heat reduces it, enabling thermal cracking",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::TOUGH, 25000,0.3f,4,300,{0,1}, 0,0, {0.4f,0.25f,0.1f,1}},
        {"Brittle (H)",    "Horizontal fibers, snaps across grain",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::BRITTLE, 12000,0.25f,2,300,{1,0}, 0,0, {0.6f,0.4f,0.2f,1}},
        {"Glass [experimental]", "Very brittle, shatters on any impact",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::GLASS, 30000,0.2f,0,300,{1,0}, 0,0, {0.7f,0.75f,0.8f,1},
         0.34f, "Try thicker panes or chunky beads for now. Thin layers still tend to over-slump in this model."},
        {"Glass Bead [new][experimental]", "A firmer cold-glass copy tuned for beads, marbles, and small chunks that keep their silhouette better",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::GLASS, 62000,0.12f,0,300,{1,0}, 0,0, {0.78f,0.88f,0.96f,1},
         0.18f, "Use marble-sized beads or thick nuggets. This copy is the one to try when the original glass feels too slumpy."},
        {"Tempered Glass [experimental]", "Holds shape cold, softens in the kiln, re-hardens as it cools",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::GLASS, 36000,0.18f,0,300,{1,0}, 0,0, {0.78f,0.86f,0.92f,1},
         0.4f, "Use a thicker pane or chunky disk. Small thin drops still read too liquid in the current glass model."},
        {"Intumescent Block [new][experimental]","Very firm cold, softens and swells into porous foam when heated",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::FOAM, 28000,0.24f,0,300,{1,0}, 0,0, {0.78f,0.76f,0.64f,1}},

        // --- STRUCTURAL (COMPOSITE: cure + directional fracture) ---
        {"Hardwood Plank", "Cures when heated + directional fracture along grain",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::COMPOSITE, 10000,0.25f,3,300,{1,0}, 0,0, {0.55f,0.35f,0.2f,1}},
        {"Bamboo (V)",     "Vertical fibers, splits into splinters when stressed",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::COMPOSITE, 8000,0.3f,4,300,{0,1}, 0,0, {0.7f,0.6f,0.3f,1},
         0.5f, "Tall stalk-like shapes make the harden-and-crack transition much more visible than small bundles."},
        {"Bone",           "Very stiff, anisotropic fracture. Heat weakens joints",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::COMPOSITE, 15000,0.2f,3,300,{0,1}, 0,0, {0.85f,0.8f,0.7f,1},
         0.4f, "Medium-to-large rods work best so the anisotropic cracking has enough span to show up."},
        {"Bone Rod [new]", "A denser bone copy with less bounce and more rod-like stiffness under load",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::COMPOSITE, 26000,0.12f,4.5f,300,{0,1}, 0,0, {0.90f,0.84f,0.74f,1},
         0.28f, "Use medium rods or stubby bones. This copy is tuned to read less rubbery than the original Bone preset."},
        {"Carbon Fiber",   "Extremely stiff along fiber, weak cross-grain",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::COMPOSITE, 35000,0.15f,6,300,{1,0}, 0,0, {0.42f,0.45f,0.56f,1}},
        {"Ortho Beam [new][experimental]", "A benchmark-only orthotropic beam with very strong along-fiber stiffness and deliberately weak cross-grain bend response",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::ORTHO_BEND, 5600,0.22f,5.0f,300,{1,0}, 0,0, {0.78f,0.56f,0.26f,1},
         0.46f, "Use long thin cantilevers or beams. This one is meant to exaggerate directional stiffness so the anisotropic bend read is obvious.", PresetQuickTab::ALL, 1.0f, 1.0f, 0.82f, 0.08f},
        {"Ortho Tear Strap [new][experimental]", "A benchmark-only heat-cured orthotropic strap that should resist along-grain pull but split rapidly across the fiber",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::ORTHO_TEAR, 6800,0.22f,6.0f,300,{1,0}, 0,0, {0.86f,0.64f,0.34f,1},
         0.52f, "Use long heated straps or shell rectangles. This one is intentionally exaggerated to benchmark anisotropic tearing, not to look like a natural material.", PresetQuickTab::POROUS_TEAR, 0.96f, 1.0f, 0.70f, 0.08f},

        {"Binder Crumb [new][experimental]", "Moisture-plus-binder loaf: the matrix sets into a stronger crumb scaffold, so pores remain without the whole loaf immediately collapsing",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::BINDER_CRUMB, 10400,0.27f,2.2f,300,{0,1}, 0,0, {0.97f,0.84f,0.60f,1},
         0.62f, "Use thick boules or shell circles with 2.5D on if you want the scaffold to read clearly. This one is for stronger baked walls around the pores.", PresetQuickTab::MOISTURE_BINDER, 0.86f, 0.64f, 0.86f, 0.06f},
        {"Channel Crumb [new][experimental]", "Vent-aware loaf: moisture stays in the core longer, then opens channels and pockets without behaving like a full structural split",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::CHANNEL_CRUMB, 11200,0.26f,2.8f,300,{0,1}, 0,0, {0.98f,0.86f,0.64f,1},
         0.62f, "Use tall loafs or domes. This one is for visible vent tunnels and channels that still leave a loaf body instead of only a slit.", PresetQuickTab::MOISTURE_BINDER, 0.90f, 0.62f, 0.88f, 0.06f},
        {"Burnout Clay [new][experimental]", "Porous pottery body with moisture loss, pore-former burnout, and a firing binder that tries to keep the vessel connected while it densifies",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::BURNOUT_CLAY, 15200,0.20f,0,300,{1,0}, 0,0, {0.88f,0.76f,0.66f,1},
         0.48f, "Use bowls, cups, or shell rectangles. This one is for porous burnout and firmer fired walls, not just hot collapse or clean glaze flow.", PresetQuickTab::MOISTURE_BINDER, 0.40f, 0.90f, 0.82f, 0.03f},
        {"Vented Skin [new][experimental]", "Thin heated sheet that opens vents and slots before it fully tears, so venting reads differently from catastrophic fracture",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::VENTED_SKIN, 9200,0.24f,2.8f,300,{1,0}, 0,0, {0.96f,0.90f,0.72f,1},
         0.34f, "Use straps, shell rectangles, or thin sheets over a heater. This one is for vent holes and slots that keep some membrane body instead of instantly ripping apart.", PresetQuickTab::MOISTURE_BINDER, 0.62f, 0.66f, 0.84f, 0.04f},

        // --- THICKNESS-ENHANCED COPIES (for the 2.5D support toggle) ---
        {"Hot Glass [enhanced by thickness]", "A thicker-surrogate hot glass copy tuned for 2.5D support, so chunky gobs stay rounder before they flow",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::GLASS, 36000,0.16f,0,650,{1,0}, 0,0, {0.98f,0.62f,0.28f,1},
         0.38f, "Best with 2.5D Bake/Kiln Support on. Use chunky gobs or shell circles so the thickness surrogate has something to hold.", PresetQuickTab::THICKNESS_ENHANCED, 0.20f, 0.90f, 0.82f, 0.02f},
        {"Bread Dough [enhanced by thickness]", "Bread Dough retuned for 2.5D support so the rise holds a thicker loaf body instead of slumping flat right away",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::BREAD, 9800,0.28f,0,300,{1,0}, 0,0, {0.90f,0.78f,0.54f,1},
         0.62f, "Best with 2.5D Bake/Kiln Support on. Use a thick loaf or boule so the pseudo-depth can keep the crumb inflated longer.", PresetQuickTab::THICKNESS_ENHANCED, 0.92f, 0.76f, 0.86f, 0.08f},
        {"Crust Dough [enhanced by thickness]", "Crust dough with extra shell/body support for 2.5D tests, meant to keep a domed loaf longer while the crust cracks",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::CRUST_DOUGH, 10000,0.28f,0,300,{1,0}, 0,0, {0.94f,0.80f,0.56f,1},
         0.64f, "Best with 2.5D Bake/Kiln Support on. Thick loaves and shell circles should keep their dome longer than the original crust dough.", PresetQuickTab::THICKNESS_ENHANCED, 0.88f, 0.72f, 0.86f, 0.08f},
        {"Steam Bun [enhanced by thickness]", "Steam bun with stronger pseudo-thickness support for rounder buns and better retained lift in 2.5D mode",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::STEAM_BUN, 9400,0.28f,0,300,{1,0}, 0,0, {0.99f,0.86f,0.62f,1},
         0.58f, "Best with 2.5D Bake/Kiln Support on. Round buns or shell circles should stay puffier instead of venting into a flat shell too early.", PresetQuickTab::THICKNESS_ENHANCED, 0.84f, 0.68f, 0.88f, 0.08f},
        {"Crumb Loaf [enhanced by thickness]", "A thicker-surrogate crumb loaf tuned to keep larger cavities and a more lifted baked body in 2.5D",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::CRUMB_LOAF, 10800,0.27f,0,300,{1,0}, 0,0, {0.99f,0.84f,0.58f,1},
         0.62f, "Best with 2.5D Bake/Kiln Support on. Use a tall boule or loaf so the added pseudo-depth reads as retained crumb instead of a slit.", PresetQuickTab::THICKNESS_ENHANCED, 0.90f, 0.66f, 0.88f, 0.08f},
        {"Bisque Clay [enhanced by thickness]", "Bisque clay retuned for 2.5D body support so cups and tiles keep more fired volume before they slump",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::BISQUE, 14200,0.20f,0,300,{1,0}, 0,0, {0.94f,0.82f,0.72f,1},
         0.50f, "Best with 2.5D Bake/Kiln Support on. Use bowls, cups, or shell rectangles so the pseudo-thickness can help the fired body hold.", PresetQuickTab::THICKNESS_ENHANCED, 0.48f, 0.86f, 0.82f, 0.03f},
        {"Vitreous Clay [enhanced by thickness]", "A denser vitrifying clay copy meant to show the 2.5D shrink-and-lock support more clearly than the base version",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::VITREOUS_CLAY, 19800,0.16f,0,300,{1,0}, 0,0, {0.78f,0.70f,0.64f,1},
         0.50f, "Best with 2.5D Bake/Kiln Support on. Thick mugs, tiles, and shell rectangles should stay denser and firmer than the base vitreous clay.", PresetQuickTab::THICKNESS_ENHANCED, 0.24f, 0.98f, 0.82f, 0.02f},
        {"Glass [enhanced by thickness]", "Cold glass retuned for the 2.5D surrogate so beads, panes, and nuggets keep more body before they pancake",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::GLASS, 52000,0.14f,0,300,{1,0}, 0,0, {0.82f,0.90f,0.98f,1},
         0.30f, "Best with 2.5D Bake/Kiln Support on. Try shell circles, thick panes, or beads; this copy is the easiest cold-glass thickness benchmark.", PresetQuickTab::THICKNESS_ENHANCED, 0.08f, 0.90f, 0.86f, 0.01f},
        {"Tempered Glass [enhanced by thickness]", "Tempered glass with stronger 2.5D hold so thicker panes stay structural longer before the hot-soft path takes over",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::GLASS, 58000,0.12f,0,300,{1,0}, 0,0, {0.84f,0.92f,0.98f,1},
         0.42f, "Best with 2.5D Bake/Kiln Support on. Use chunky panes, thick disks, or shell rectangles so the thickness surrogate can resist slumping.", PresetQuickTab::THICKNESS_ENHANCED, 0.10f, 0.92f, 0.86f, 0.01f},
        {"Stoneware [enhanced by thickness]", "Stoneware with stronger pseudo-depth support, meant to keep a denser fired body and show shrink/craze without flattening",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::STONEWARE, 17600,0.18f,0,300,{1,0}, 0,0, {0.76f,0.66f,0.60f,1},
         0.48f, "Best with 2.5D Bake/Kiln Support on. Thick bowls, mugs, and shell rectangles should hold their ceramic body much better here.", PresetQuickTab::THICKNESS_ENHANCED, 0.28f, 0.92f, 0.82f, 0.02f},
        {"Tear Skin [enhanced by thickness]", "A thicker-surrogate tear skin copy so heated sheets keep more body and form longer tears before fully collapsing",
         PresetCategory::THERMAL, SpawnSolver::MPM, MPMMaterial::TEAR_SKIN, 10400,0.24f,2.6f,300,{1,0}, 0,0, {0.98f,0.90f,0.72f,1},
         0.34f, "Best with 2.5D Bake/Kiln Support on. Use thicker straps or shell rectangles if you want long tears instead of instant collapse.", PresetQuickTab::THICKNESS_ENHANCED, 0.70f, 0.70f, 0.84f, 0.06f},
        {"Ortho Tear Strap [enhanced by thickness]", "The orthotropic tear benchmark with extra 2.5D support so the strip keeps body long enough for the split direction to read clearly",
         PresetCategory::STRUCTURAL, SpawnSolver::MPM, MPMMaterial::ORTHO_TEAR, 9800,0.18f,7.2f,300,{1,0}, 0,0, {0.90f,0.68f,0.36f,1},
         0.58f, "Best with 2.5D Bake/Kiln Support on. Use long heated straps or shell rectangles when you want a clearer tear-path benchmark.", PresetQuickTab::THICKNESS_ENHANCED, 0.88f, 1.0f, 0.78f, 0.04f},
    };
    return presets;
}

// Batch record
struct BatchRecord {
    std::string label;
    std::string description;
    std::string techniques;
    std::string properties;
    std::string recommended_note;
    u32 sph_offset, sph_count;
    u32 mpm_offset, mpm_count;
    vec4 color;
    f32 recommended_size = 0.28f;
    f32 youngs_modulus = 0.0f;
    f32 poisson_ratio = 0.0f;
    f32 fiber_strength = 0.0f;
    f32 temperature = 300.0f;
    f32 gas_constant = 0.0f;
    f32 viscosity = 0.0f;
    f32 density_scale = 1.0f;
    f32 outgassing_scale = 1.0f;
    f32 heat_release_scale = 1.0f;
    f32 cooling_scale = 1.0f;
    f32 loft_scale = 1.0f;
    vec2 fiber_dir = vec2(1.0f, 0.0f);
    SpawnSolver solver = SpawnSolver::MPM;
    MPMMaterial mpm_type = MPMMaterial::ELASTIC;
    bool scene_authored = false;
};

// Creation menu state
struct CreationState {
    bool active = false;
    i32  preset_index = 0;
    i32  category = 0;          // Current tab
    i32  quick_tab = 0;         // Quick access tab for newer heat content
    SpawnShape shape = SpawnShape::CIRCLE;
    f32  size = 0.25f;
    f32  aspect = 1.0f;
    f32  shape_angle = 0.0f;
    f32  fiber_angle = 0.0f;    // Degrees, for customize panel
    vec2 preview_pos{0.0f};

    // Customized copy of selected preset (edited by user before placement)
    MaterialPreset custom;

    // Batch tracking
    std::vector<BatchRecord> batches;
    i32 highlighted_batch = -1;
    u32 batch_counter = 0;

    void select_preset(i32 idx) {
        preset_index = idx;
        custom = get_presets()[idx];
        category = static_cast<i32>(custom.category);
        // Derive fiber angle from preset direction
        fiber_angle = std::atan2(custom.fiber_dir.y, custom.fiber_dir.x) * 180.0f / 3.14159265f;
    }
};

// Generate a unique color for batch index
inline vec4 batch_color(u32 index) {
    f32 hue = glm::fract(index * 0.618033988f);
    f32 r = glm::clamp(glm::abs(hue * 6.0f - 3.0f) - 1.0f, 0.0f, 1.0f);
    f32 g = glm::clamp(2.0f - glm::abs(hue * 6.0f - 2.0f), 0.0f, 1.0f);
    f32 b = glm::clamp(2.0f - glm::abs(hue * 6.0f - 4.0f), 0.0f, 1.0f);
    return vec4(r * 0.8f + 0.2f, g * 0.8f + 0.2f, b * 0.8f + 0.2f, 1.0f);
}

const MaterialPreset* preset_for_material(SpawnSolver solver, MPMMaterial mpm_type);
vec4 default_material_color(SpawnSolver solver, MPMMaterial mpm_type);
std::string technique_summary(SpawnSolver solver, MPMMaterial mpm_type);
std::string property_summary(SpawnSolver solver, MPMMaterial mpm_type,
                             f32 youngs_modulus, f32 poisson_ratio,
                             f32 initial_temp, vec2 fiber_dir,
                             f32 density_scale = 1.0f);
BatchRecord make_batch_record(const char* label, const char* description,
                              SpawnSolver solver, MPMMaterial mpm_type, vec4 color,
                              f32 youngs_modulus, f32 poisson_ratio,
                              f32 initial_temp, vec2 fiber_dir,
                              f32 recommended_size, const char* recommended_note,
                              bool scene_authored = false);
void apply_sph_batch_properties(ParticleBuffer& particles, SPHSolver& sph,
                                u32 global_offset, u32 count,
                                MPMMaterial material, f32 initial_temp,
                                f32 gas_constant, f32 viscosity,
                                vec4 color,
                                vec4 thermal_coupling = vec4(1.0f));

void place_object(CreationState& state, ParticleBuffer& particles,
                  SPHSolver& sph, MPMSolver& mpm, UniformGrid& grid);

} // namespace ng
