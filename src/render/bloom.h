#pragma once

#include "core/types.h"
#include "gpu/shader.h"

namespace ng {

class ParticleBuffer;
class Camera;

// Two-pass Gaussian bloom: render bright particles to FBO, blur, add to screen
class BloomRenderer {
public:
    void init(i32 width, i32 height);

    // Render highlighted particles into bloom FBO (call between normal renders)
    void capture(const ParticleBuffer& particles, const Camera& camera,
                 u32 offset, u32 count, f32 point_size);

    // Blur the captured image and composite onto screen with additive blending
    void apply(f32 intensity = 0.4f);

private:
    Shader splat_shader_; // Golden glow splat
    Shader blur_shader_;
    u32 fbo_a_ = 0, fbo_b_ = 0;
    u32 tex_a_ = 0, tex_b_ = 0;
    u32 vao_ = 0;
    ivec2 size_{0};
    i32 saved_vp_[4] = {};
};

} // namespace ng
