#pragma once

#include "core/types.h"
#include "gpu/compute_shader.h"
#include "gpu/buffer.h"
#include <vector>
#include <string>

namespace ng {

// GPU-resident 2D signed distance field.
// Stored as a Texture2D<float> where negative = inside solid.
// Rebuilt each frame from primitives, or modified interactively.
class SDFField {
public:
    enum PrimType : u32 { PRIM_BOX = 0, PRIM_CIRCLE = 1, PRIM_SEGMENT = 2 };
    enum class MagneticMode : u32 {
        NONE = 0,
        PERMANENT = 1,
        SOFT = 2
    };
    enum class MaterialPresetID : u32 {
        GLOBAL_DEFAULT = 0,
        SILVER_CONDUCTIVE = 1,
        BRASS_HEAT_SINK = 2,
        BRONZE_BALANCED = 3,
        ROSE_GOLD_LIGHT = 4,
        MAGNET_X = 5,
        MAGNET_Y = 6,
        SOFT_IRON = 7
    };

    struct MaterialPreset {
        const char* name = "Default";
        const char* summary = "Uses the global SDF solid settings.";
        u32 palette_code = 0; // 0 = use global UI palette, otherwise palette index + 1
        f32 conductivity_scale = 1.0f;
        f32 heat_capacity_scale = 1.0f;
        f32 contact_transfer_scale = 1.0f;
        f32 heat_loss_scale = 1.0f;
        MagneticMode magnetic_mode = MagneticMode::NONE;
        vec2 magnetic_dir = vec2(1.0f, 0.0f);
        f32 magnetic_strength = 0.0f;
        f32 magnetic_susceptibility = 0.0f;
    };

    struct ObjectRecord {
        u32 id = 0;
        PrimType type = PRIM_BOX;
        vec2 a = vec2(0.0f); // center for box/circle, endpoint A for segment
        vec2 b = vec2(0.0f); // half extents for box, endpoint B for segment
        f32 radius_or_thickness = 0.0f;
        MaterialPreset material{};
        std::string label;
        std::string summary;
        std::string techniques;
    };

    struct Config {
        ivec2 resolution = ivec2(512, 512);
        vec2  world_min  = vec2(-3.0f, -3.0f);
        vec2  world_max  = vec2(3.0f, 3.0f);
    };

    void init(const Config& config);

    // Clear to large positive distance (empty space)
    void clear();

    // Add a box to the SDF (negative distance inside)
    void add_box(vec2 center, vec2 half_extents,
                 MaterialPresetID preset = MaterialPresetID::GLOBAL_DEFAULT,
                 const char* label = nullptr, const char* summary = nullptr);

    // Add a circle
    void add_circle(vec2 center, f32 radius,
                    MaterialPresetID preset = MaterialPresetID::GLOBAL_DEFAULT,
                    const char* label = nullptr, const char* summary = nullptr);

    // Add a line segment (wall with given thickness)
    void add_segment(vec2 a, vec2 b, f32 thickness,
                     MaterialPresetID preset = MaterialPresetID::GLOBAL_DEFAULT,
                     const char* label = nullptr, const char* summary = nullptr);

    // Stamp user-drawn circle (for interactive drawing)
    void stamp_circle(vec2 center, f32 radius, bool add_solid);

    // Rebuild SDF from all accumulated primitives (dispatches compute)
    void rebuild();

    // Bind SDF texture for reading in shaders
    void bind_for_read(u32 unit = 0) const;
    void bind_props_for_read(u32 unit = 1) const;
    void bind_object_ids_for_read(u32 unit = 2) const;
    void bind_palette_for_read(u32 unit = 3) const;

    // Bind SDF texture as image for compute shader write
    void bind_for_write(u32 unit = 0) const;

    u32 texture() const { return sdf_texture_; }
    u32 props_texture() const { return props_texture_; }
    u32 object_id_texture() const { return object_id_texture_; }
    u32 palette_texture() const { return palette_texture_; }
    ivec2 resolution() const { return resolution_; }
    vec2 world_min() const { return world_min_; }
    vec2 world_max() const { return world_max_; }
    vec2 cell_size() const { return (world_max_ - world_min_) / vec2(resolution_); }
    const std::vector<ObjectRecord>& objects() const { return objects_; }
    const ObjectRecord* object_by_id(u32 id) const;
    bool set_object_material(u32 id, const MaterialPreset& material);
    static const MaterialPreset& material_preset(MaterialPresetID preset_id);

    static constexpr u32 BIND_SDF_IMAGE = 0; // image2D binding
    static constexpr u32 BIND_SDF_PROPS_IMAGE = 1;
    static constexpr u32 BIND_SDF_OBJECT_IDS_IMAGE = 2;
    static constexpr u32 BIND_SDF_PALETTE_IMAGE = 3;
    static constexpr u32 BIND_SDF_PRIMS = 17; // SSBO for primitive list

    struct GPUPrimitive {
        u32 type;
        u32 palette_code;
        f32 conductivity_scale;
        f32 heat_capacity_scale;
        f32 contact_transfer_scale;
        f32 heat_loss_scale;
        f32 params[7]; // Depends on type:
        // BOX: center.x, center.y, half.x, half.y, 0, 0, 0
        // CIRCLE: center.x, center.y, radius, 0, 0, 0, 0
        // SEGMENT: a.x, a.y, b.x, b.y, thickness, 0, 0
    };

private:
    ivec2 resolution_{0};
    vec2 world_min_{0.0f};
    vec2 world_max_{0.0f};

    u32 sdf_texture_ = 0;
    u32 props_texture_ = 0;
    u32 object_id_texture_ = 0;
    u32 palette_texture_ = 0;

    std::vector<GPUPrimitive> primitives_;
    std::vector<ObjectRecord> objects_;
    GPUBuffer prim_buffer_; // SSBO for primitives

    ComputeShader rebuild_shader_;
    ComputeShader stamp_shader_;
};

} // namespace ng
