#include "physics/sdf/sdf_field.h"
#include "core/log.h"

#include <glad/gl.h>

namespace ng {

namespace {

std::string default_object_techniques(const SDFField::MaterialPreset& preset) {
    std::string text = "Rigid SDF collider, Eulerian solid heat diffusion, per-object thermal scaling.";
    if (preset.magnetic_mode == SDFField::MagneticMode::PERMANENT) {
        text += " Permanent-magnet source for the real magnetic field solve.";
    } else if (preset.magnetic_mode == SDFField::MagneticMode::SOFT) {
        text += " Soft magnetic rigid object with induced magnetization in the real magnetic field solve.";
    }
    return text;
}

} // namespace

const SDFField::MaterialPreset& SDFField::material_preset(MaterialPresetID preset_id) {
    static const MaterialPreset presets[] = {
        { "Default", "Uses the global SDF solid settings.", 0u, 1.0f, 1.0f, 1.0f, 1.0f, MagneticMode::NONE, vec2(1.0f, 0.0f), 0.0f, 0.0f },
        { "Silver", "Highly conductive metal. Heat travels through it quickly.", 1u, 2.2f, 0.9f, 1.3f, 0.85f, MagneticMode::NONE, vec2(1.0f, 0.0f), 0.0f, 0.0f },
        { "Brass", "Good heat sink. Slower to heat up, but holds heat well.", 4u, 0.9f, 2.4f, 0.9f, 0.6f, MagneticMode::NONE, vec2(1.0f, 0.0f), 0.0f, 0.0f },
        { "Bronze", "Balanced warm metal for general rigid structures.", 3u, 1.15f, 1.4f, 1.0f, 0.8f, MagneticMode::NONE, vec2(1.0f, 0.0f), 0.0f, 0.0f },
        { "Rose Gold", "Decorative warm alloy with slightly softer thermal response.", 2u, 0.95f, 1.25f, 0.95f, 0.9f, MagneticMode::NONE, vec2(1.0f, 0.0f), 0.0f, 0.0f },
        { "Magnet X", "Permanent magnetized metal with horizontal remanence for the real magnetic field solve.", 3u, 1.1f, 1.3f, 1.0f, 0.78f, MagneticMode::PERMANENT, vec2(1.0f, 0.0f), 6.5f, 0.0f },
        { "Magnet Y", "Permanent magnetized metal with vertical remanence for the real magnetic field solve.", 2u, 1.1f, 1.3f, 1.0f, 0.78f, MagneticMode::PERMANENT, vec2(0.0f, 1.0f), 6.5f, 0.0f },
        { "Soft Iron", "Magnetically soft iron reference that becomes induced by nearby permanent magnets and concentrates the solved field.", 1u, 1.4f, 1.6f, 1.0f, 0.74f, MagneticMode::SOFT, vec2(1.0f, 0.0f), 0.0f, 5.8f }
    };
    return presets[static_cast<u32>(preset_id)];
}

const SDFField::ObjectRecord* SDFField::object_by_id(u32 id) const {
    if (id == 0 || id > objects_.size()) return nullptr;
    return &objects_[id - 1];
}

bool SDFField::set_object_material(u32 id, const MaterialPreset& material) {
    if (id == 0 || id > objects_.size() || id > primitives_.size()) return false;
    ObjectRecord& obj = objects_[id - 1];
    GPUPrimitive& prim = primitives_[id - 1];
    obj.material = material;
    obj.techniques = default_object_techniques(material);
    prim.palette_code = material.palette_code;
    prim.conductivity_scale = material.conductivity_scale;
    prim.heat_capacity_scale = material.heat_capacity_scale;
    prim.contact_transfer_scale = material.contact_transfer_scale;
    prim.heat_loss_scale = material.heat_loss_scale;
    rebuild();
    return true;
}

void SDFField::init(const Config& config) {
    resolution_ = config.resolution;
    world_min_ = config.world_min;
    world_max_ = config.world_max;

    if (sdf_texture_) glDeleteTextures(1, &sdf_texture_);
    if (props_texture_) glDeleteTextures(1, &props_texture_);
    if (object_id_texture_) glDeleteTextures(1, &object_id_texture_);
    if (palette_texture_) glDeleteTextures(1, &palette_texture_);

    // Create SDF texture (R32F)
    glCreateTextures(GL_TEXTURE_2D, 1, &sdf_texture_);
    glTextureStorage2D(sdf_texture_, 1, GL_R32F, resolution_.x, resolution_.y);
    glTextureParameteri(sdf_texture_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(sdf_texture_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(sdf_texture_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(sdf_texture_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glCreateTextures(GL_TEXTURE_2D, 1, &props_texture_);
    glTextureStorage2D(props_texture_, 1, GL_RGBA32F, resolution_.x, resolution_.y);
    glTextureParameteri(props_texture_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(props_texture_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(props_texture_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(props_texture_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glCreateTextures(GL_TEXTURE_2D, 1, &object_id_texture_);
    glTextureStorage2D(object_id_texture_, 1, GL_R32UI, resolution_.x, resolution_.y);
    glTextureParameteri(object_id_texture_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(object_id_texture_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(object_id_texture_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(object_id_texture_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glCreateTextures(GL_TEXTURE_2D, 1, &palette_texture_);
    glTextureStorage2D(palette_texture_, 1, GL_R32UI, resolution_.x, resolution_.y);
    glTextureParameteri(palette_texture_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(palette_texture_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(palette_texture_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(palette_texture_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Primitive buffer (generous initial capacity)
    prim_buffer_.create(256 * sizeof(GPUPrimitive));

    rebuild_shader_.load("shaders/physics/sdf_rebuild.comp");
    stamp_shader_.load("shaders/physics/sdf_stamp.comp");

    clear();

    LOG_INFO("SDFField: %dx%d, world [%.1f,%.1f]-[%.1f,%.1f]",
        resolution_.x, resolution_.y, world_min_.x, world_min_.y, world_max_.x, world_max_.y);
}

void SDFField::clear() {
    // Fill with large positive distance (empty space)
    float big = 100.0f;
    glClearTexImage(sdf_texture_, 0, GL_RED, GL_FLOAT, &big);
    float default_props[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    u32 zero = 0u;
    glClearTexImage(props_texture_, 0, GL_RGBA, GL_FLOAT, default_props);
    glClearTexImage(object_id_texture_, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glClearTexImage(palette_texture_, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    primitives_.clear();
    objects_.clear();
}

void SDFField::add_box(vec2 center, vec2 half_extents,
                       MaterialPresetID preset_id, const char* label, const char* summary) {
    const auto& preset = material_preset(preset_id);
    GPUPrimitive p{};
    p.type = PRIM_BOX;
    p.palette_code = preset.palette_code;
    p.conductivity_scale = preset.conductivity_scale;
    p.heat_capacity_scale = preset.heat_capacity_scale;
    p.contact_transfer_scale = preset.contact_transfer_scale;
    p.heat_loss_scale = preset.heat_loss_scale;
    p.params[0] = center.x; p.params[1] = center.y;
    p.params[2] = half_extents.x; p.params[3] = half_extents.y;
    primitives_.push_back(p);

    ObjectRecord obj{};
    obj.id = static_cast<u32>(objects_.size()) + 1u;
    obj.type = PRIM_BOX;
    obj.a = center;
    obj.b = half_extents;
    obj.material = preset;
    obj.label = label ? label : "SDF Box";
    obj.summary = summary ? summary : preset.summary;
    obj.techniques = default_object_techniques(preset);
    objects_.push_back(std::move(obj));
}

void SDFField::add_circle(vec2 center, f32 radius,
                          MaterialPresetID preset_id, const char* label, const char* summary) {
    const auto& preset = material_preset(preset_id);
    GPUPrimitive p{};
    p.type = PRIM_CIRCLE;
    p.palette_code = preset.palette_code;
    p.conductivity_scale = preset.conductivity_scale;
    p.heat_capacity_scale = preset.heat_capacity_scale;
    p.contact_transfer_scale = preset.contact_transfer_scale;
    p.heat_loss_scale = preset.heat_loss_scale;
    p.params[0] = center.x; p.params[1] = center.y;
    p.params[2] = radius;
    primitives_.push_back(p);

    ObjectRecord obj{};
    obj.id = static_cast<u32>(objects_.size()) + 1u;
    obj.type = PRIM_CIRCLE;
    obj.a = center;
    obj.radius_or_thickness = radius;
    obj.material = preset;
    obj.label = label ? label : "SDF Circle";
    obj.summary = summary ? summary : preset.summary;
    obj.techniques = default_object_techniques(preset);
    objects_.push_back(std::move(obj));
}

void SDFField::add_segment(vec2 a, vec2 b, f32 thickness,
                           MaterialPresetID preset_id, const char* label, const char* summary) {
    const auto& preset = material_preset(preset_id);
    GPUPrimitive p{};
    p.type = PRIM_SEGMENT;
    p.palette_code = preset.palette_code;
    p.conductivity_scale = preset.conductivity_scale;
    p.heat_capacity_scale = preset.heat_capacity_scale;
    p.contact_transfer_scale = preset.contact_transfer_scale;
    p.heat_loss_scale = preset.heat_loss_scale;
    p.params[0] = a.x; p.params[1] = a.y;
    p.params[2] = b.x; p.params[3] = b.y;
    p.params[4] = thickness;
    primitives_.push_back(p);

    ObjectRecord obj{};
    obj.id = static_cast<u32>(objects_.size()) + 1u;
    obj.type = PRIM_SEGMENT;
    obj.a = a;
    obj.b = b;
    obj.radius_or_thickness = thickness;
    obj.material = preset;
    obj.label = label ? label : "SDF Segment";
    obj.summary = summary ? summary : preset.summary;
    obj.techniques = default_object_techniques(preset);
    objects_.push_back(std::move(obj));
}

void SDFField::stamp_circle(vec2 center, f32 radius, bool add_solid) {
    bind_for_write(BIND_SDF_IMAGE);

    stamp_shader_.bind();
    stamp_shader_.set_vec2("u_center", center);
    stamp_shader_.set_float("u_radius", radius);
    stamp_shader_.set_int("u_add_solid", add_solid ? 1 : 0);
    stamp_shader_.set_vec2("u_world_min", world_min_);
    stamp_shader_.set_vec2("u_world_max", world_max_);
    stamp_shader_.set_ivec2("u_resolution", resolution_);
    stamp_shader_.dispatch_2d(static_cast<u32>(resolution_.x), static_cast<u32>(resolution_.y));
    ComputeShader::barrier_image();
}

void SDFField::rebuild() {
    if (primitives_.empty()) return;

    // Upload primitives
    size_t needed = primitives_.size() * sizeof(GPUPrimitive);
    if (needed > prim_buffer_.size()) {
        prim_buffer_.create(needed * 2); // Double capacity
    }
    prim_buffer_.upload(primitives_.data(), needed);
    prim_buffer_.bind_base(BIND_SDF_PRIMS);

    // Clear SDF
    float big = 100.0f;
    glClearTexImage(sdf_texture_, 0, GL_RED, GL_FLOAT, &big);
    float default_props[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    u32 zero = 0u;
    glClearTexImage(props_texture_, 0, GL_RGBA, GL_FLOAT, default_props);
    glClearTexImage(object_id_texture_, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glClearTexImage(palette_texture_, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);

    // Dispatch rebuild
    bind_for_write(BIND_SDF_IMAGE);
    glBindImageTexture(BIND_SDF_PROPS_IMAGE, props_texture_, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    glBindImageTexture(BIND_SDF_OBJECT_IDS_IMAGE, object_id_texture_, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    glBindImageTexture(BIND_SDF_PALETTE_IMAGE, palette_texture_, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    rebuild_shader_.bind();
    rebuild_shader_.set_ivec2("u_resolution", resolution_);
    rebuild_shader_.set_vec2("u_world_min", world_min_);
    rebuild_shader_.set_vec2("u_world_max", world_max_);
    rebuild_shader_.set_uint("u_prim_count", static_cast<u32>(primitives_.size()));
    rebuild_shader_.dispatch_2d(static_cast<u32>(resolution_.x), static_cast<u32>(resolution_.y));
    ComputeShader::barrier_image();
}

void SDFField::bind_for_read(u32 unit) const {
    glBindTextureUnit(unit, sdf_texture_);
}

void SDFField::bind_props_for_read(u32 unit) const {
    glBindTextureUnit(unit, props_texture_);
}

void SDFField::bind_object_ids_for_read(u32 unit) const {
    glBindTextureUnit(unit, object_id_texture_);
}

void SDFField::bind_palette_for_read(u32 unit) const {
    glBindTextureUnit(unit, palette_texture_);
}

void SDFField::bind_for_write(u32 unit) const {
    glBindImageTexture(unit, sdf_texture_, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
}

} // namespace ng
