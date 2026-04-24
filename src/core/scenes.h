#pragma once
#include "core/types.h"

namespace ng {
class ParticleBuffer;
class SPHSolver;
class MPMSolver;
class UniformGrid;
class SDFField;
struct CreationState;

enum class SceneID : u32 {
    DEFAULT=0, THERMAL_FURNACE=1, FRACTURE_TEST=2, MELTING=3,
    DAM_BREAK=4, STIFF_OBJECTS=5, HEAT_RAMP=6, FIRE_FORGE=7,
    CODIM_THREADS=8, EMPTY_BOX=9, WIND_TUNNEL=10, BOX_HEAT_IN_AIR=11, HEAT_NO_WALLS=12,
    OVEN_OPEN=13, POT_HEATER=14, OVEN_OPEN_WIND=15, POT_HEATER_WIND=16,
    CODIM_THREADS_COLD=17, ZERO_G_SOFT_LAB=18, THERMAL_BRIDGE=19, SPIRAL_METALS=20,
    THERMAL_BRIDGE_STRONG=21, THIN_PIPE=22, EMPTY_ZERO_G=23, FLOOR_ONLY=24,
    GLAZE_KILN=25, BAKE_OVEN=26, STRESS_FORGE=27, REACTIVE_HEARTH=28,
    GLAZE_RACK=29, STEAM_OVEN=30, SEED_ROASTER=31,
    CRAZING_SHELF=32, POCKET_OVEN=33, TEMPERING_BENCH=34, PRESSURE_PANTRY=35,
    ANISO_TEAR_BENCH=36, ANISO_BEND_BENCH=37, POROUS_BAKE_BENCH=38, POTTERY_BENCH=39,
    PASTRY_BENCH=40, STONEWARE_BENCH=41, ANISO_STRONG_BENCH=42,
    ADV_BAKE_BENCH=43, ADV_KILN_BENCH=44, OPEN_CRUMB_BENCH=45, SINTER_LOCK_BENCH=46,
    MOISTURE_BINDER_BENCH=47, BURNOUT_POTTERY_BENCH=48, VENTED_SKIN_BENCH=49,
    SPH_THERMAL_BENCH=50, OIL_OVER_WATER=51, FERRO_SPIKE_BENCH=52, MAGNETIC_BENCH=53,
    MAGNETIC_CLIMB_BENCH=54, MAGNETIC_FLOOR_BENCH=55, RIGID_MAGNETIC_FLOOR=56,
    MAG_CURSOR_UNIT=57, MAG_PERMANENT_POLE=58, MAG_SOFT_IRON_FIELD=59, MAG_SOFT_IRON_BODY=60,
    OOBLECK_IMPACTOR_BENCH=61, IMPACT_MEMORY_BENCH=62,
    BLAST_ARMOR_LANE=63, BREACH_CHAMBER=64, SPALL_PLATE_BENCH=65,
    OPEN_BLAST_RANGE_XL=66, BREACH_HALL_XL=67, SPALL_GALLERY_XL=68,
    BIO_REPLICATOR_BENCH=69, MYCELIUM_MORPH_BENCH=70,
    MORPHOGENESIS_BENCH=71, ROOT_GARDEN_BENCH=72, CELL_COLONY_BENCH=73,
    AUTOMATA_AIR_COUPLING_BENCH=74, AUTOMATA_FIRE_REGROWTH_BENCH=75,
    AUTOMATA_MAX_COUPLING_BENCH=76, ASH_REGROWTH_BENCH=77,
    FOOT_DEMO_BENCH=78,
    HYBRID_REGROWTH_WALL=79, HYBRID_KILN_PROCESS=80, HYBRID_SOFT_HEAT_RANGE=81,
    HYBRID_PRESSURE_POTTERY=82, HYBRID_FERRO_SPLASH=83, HYBRID_OOBLECK_ARMOR=84,
    THERMAL_VERIFY_SDF_JUNCTION=85, THERMAL_VERIFY_HOT_BLOCKS=86,
    THERMAL_VERIFY_CROSS_IGNITION=87, THERMAL_VERIFY_BRIDGE_WITNESS=88,
    THERMAL_VERIFY_IMPACT_RINGDOWN=89,
    FIX_TEST_BIO_HEAL=90, FIX_TEST_BLAST_COOLDOWN=91,
    FIX_TEST_COLLISION_HEAT=92, FIX_TEST_FERRO_DEMAG=93,
    FIX_TEST_SPH_EQUIL=94,
    HUGE_WEAPON_RANGE=95, HUGE_IMPACT_PLAYGROUND=96,
    // --- Showcase scenes for the latest physics additions ---
    SHOWCASE_FERRO_VARIANTS=97,  // 4 ferrofluid variants side-by-side under one magnet
    SHOWCASE_MEISSNER_FLOAT=98,  // cold superconductor block levitating over a magnet
    SHOWCASE_CURIE_DEMAG=99,     // Curie ferromagnet bar that loses M when heated
    SHOWCASE_SAND_CASTLE=100,    // Drucker-Prager sand angle-of-repose pile
    SHOWCASE_ION_CROSSFLOW=101,  // positive + negative ion clouds driven by ambient E
    SHOWCASE_PHASE_FRACTURE=102  // phase-field brittle tile under heavy-ball impact
};
constexpr int SCENE_COUNT = 103;
inline const char* scene_names[] = {
    "Default", "Thermal Furnace", "Fracture Test", "Melting",
    "Dam Break", "Stiff Objects", "Heat Ramp", "Fire & Forge",
    "Codim Threads", "Empty Box", "Wind Tunnel", "Box + Heat in Air", "Heat No Walls",
    "Open Oven", "Pot Heater", "Open Oven + Wind", "Pot Heater + Wind",
    "Codim Threads Cold", "Zero-G Soft Lab", "Thermal Bridge", "Spiral Metals",
    "Thermal Bridge Strong", "Thin Pipe", "Empty Zero-G", "Floor Only",
    "Glaze Kiln", "Bake Oven", "Stress Forge", "Reactive Hearth",
    "Glaze Rack", "Steam Oven", "Seed Roaster",
    "Crazing Shelf", "Pocket Oven", "Tempering Bench", "Pressure Pantry",
    "Aniso Tear Bench", "Aniso Bend Bench", "Porous Bake Bench", "Pottery Bench",
    "Pastry Bench", "Stoneware Bench", "Aniso Strong Bench",
    "Advanced Bake Bench", "Advanced Kiln Bench", "Open Crumb Bench", "Sinter Lock Bench",
    "Moisture Binder Bench", "Burnout Pottery Bench", "Vented Skin Bench",
    "SPH Thermal Bench", "Oil Over Water", "Ferro Spike Bench", "Magnetic Bench",
    "Magnetic Climb Bench", "Magnetic Floor Bench", "Rigid Magnetic Floor",
    "Mag Cursor Unit", "Mag Permanent Pole", "Mag Soft-Iron Field", "Mag Soft-Iron Body",
    "Oobleck Impactor Bench", "Impact Memory Bench",
    "Blast Armor Lane", "Breach Chamber", "Spall Plate Bench",
    "Open Blast Range XL", "Breach Hall XL", "Spall Gallery XL",
    "Bio Replicator Bench", "Mycelium Morph Bench",
    "Morphogenesis Bench", "Root Garden Bench", "Cell Colony Bench",
    "Automata Air Coupling Bench", "Automata Fire Regrowth Bench",
    "Automata Max Coupling Bench", "Ash Regrowth Bench",
    "Foot Demo Bench",
    "Hybrid: Regrowth Wall [new]",
    "Hybrid: Kiln Process [new]",
    "Hybrid: Soft HEAT Range [new]",
    "Hybrid: Pressure Pottery [new]",
    "Hybrid: Ferro Splash [new][experimental]",
    "Hybrid: Oobleck Armor [new][experimental]",
    "Thermal Verify: SDF Junction [new]",
    "Thermal Verify: Hot Blocks [new]",
    "Thermal Verify: Cross Ignition [new]",
    "Thermal Verify: Bridge Witness [new]",
    "Thermal Verify: Impact Ringdown [new]",
    "Fix Test: Bio Heal + Ash Brittle [new]",
    "Fix Test: Blast Cooldown [new]",
    "Fix Test: Collision Heat Rack [new]",
    "Fix Test: Ferro Demag Spikes [new]",
    "Fix Test: SPH Thermal Equilibrium [new]",
    "Huge Weapon Range [new]",
    "Huge Impact Playground [new]",
    "Showcase: Ferro Variants [new]",
    "Showcase: Meissner Float [new]",
    "Showcase: Curie Demag [new]",
    "Showcase: Sand Castle [new]",
    "Showcase: Ion Crossflow [new]",
    "Showcase: Phase Fracture [new]"
};

void load_scene(SceneID id, ParticleBuffer& particles, SPHSolver& sph,
                MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                CreationState* creation = nullptr);
}
