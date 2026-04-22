#pragma once
#include "core/types.h"
#include "gpu/shader.h"

namespace ng {
class ParticleBuffer;
class Camera;

enum class SurfaceStyle : i32 {
    LIQUID = 0,
    GEL = 1,
    CLAY = 2,
    WAX = 3,
    PORCELAIN = 4,
    FIELD_MATTE = 5,
    CONTOUR = 6,
    SOFT_FILL = 7,
    THIN_CONTOUR = 8,
    INK_CONTOUR = 9
};

class MetaballRenderer {
public:
    void init(i32 width, i32 height);
    void splat(const ParticleBuffer& particles, const Camera& camera,
               u32 offset, u32 count, f32 kernel_radius);
    void render_surface(f32 threshold, SurfaceStyle style);
    bool enabled = false;
    bool keep_particles = false;
    f32 kernel_scale = 2.5f;
    f32 threshold = 0.5f;
    f32 edge_softness = 1.8f;
    f32 gloss = 0.65f;
    f32 rim = 0.45f;
    f32 opacity = 0.96f;
private:
    Shader splat_shader_;
    Shader surface_shader_;
    u32 fbo_ = 0, density_tex_ = 0, vao_ = 0;
    ivec2 tex_size_{0};
};
}
