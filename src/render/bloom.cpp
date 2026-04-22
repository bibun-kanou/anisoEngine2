#include "render/bloom.h"
#include "physics/common/particle_buffer.h"
#include "render/camera.h"
#include "core/log.h"

#include <glad/gl.h>

namespace ng {

void BloomRenderer::init(i32 width, i32 height) {
    size_ = ivec2(width / 4, height / 4);

    for (int i = 0; i < 2; i++) {
        u32& tex = (i == 0) ? tex_a_ : tex_b_;
        u32& fbo = (i == 0) ? fbo_a_ : fbo_b_;
        glCreateTextures(GL_TEXTURE_2D, 1, &tex);
        glTextureStorage2D(tex, 1, GL_RGBA16F, size_.x, size_.y);
        glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glCreateFramebuffers(1, &fbo);
        glNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, tex, 0);
    }

    splat_shader_.load("shaders/render/particle_draw.vert", "shaders/render/bloom_splat.frag");
    blur_shader_.load("shaders/render/sdf_draw.vert", "shaders/render/blur.frag");
    glCreateVertexArrays(1, &vao_);
}

void BloomRenderer::capture(const ParticleBuffer& particles, const Camera& camera,
                             u32 offset, u32 count, f32 point_size) {
    if (count == 0) return;

    // Save full-res viewport
    glGetIntegerv(GL_VIEWPORT, saved_vp_);

    // Render golden particles into FBO A at quarter res
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_a_);
    glViewport(0, 0, size_.x, size_.y);
    float clear[] = {0, 0, 0, 0};
    glClearNamedFramebufferfv(fbo_a_, GL_COLOR, 0, clear);

    // Scale point size for the smaller FBO
    f32 scale = static_cast<f32>(size_.y) / static_cast<f32>(saved_vp_[3]);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glEnable(GL_PROGRAM_POINT_SIZE);

    particles.bind_all();
    splat_shader_.bind();
    splat_shader_.set_mat4("u_view_proj", camera.view_proj());
    splat_shader_.set_float("u_point_size", point_size * scale);
    splat_shader_.set_int("u_offset", static_cast<i32>(offset));
    splat_shader_.set_vec3("u_glow_color", vec3(1.0f, 0.85f, 0.3f));

    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(count));
    glBindVertexArray(0);
    splat_shader_.unbind();

    // Restore
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(saved_vp_[0], saved_vp_[1], saved_vp_[2], saved_vp_[3]);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void BloomRenderer::apply(f32 intensity) {
    vec2 texel(1.0f / size_.x, 1.0f / size_.y);

    // Also clear FBO B before blur
    float clear[] = {0, 0, 0, 0};
    glClearNamedFramebufferfv(fbo_b_, GL_COLOR, 0, clear);

    // Gaussian blur: 3 iterations of H+V at quarter res
    glDisable(GL_BLEND); // Blur overwrites, not adds
    for (int iter = 0; iter < 3; iter++) {
        // Horizontal: A → B
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_b_);
        glViewport(0, 0, size_.x, size_.y);
        glBindTextureUnit(0, tex_a_);
        blur_shader_.bind();
        blur_shader_.set_int("u_tex", 0);
        blur_shader_.set_vec2("u_texel_size", texel);
        blur_shader_.set_int("u_horizontal", 1);
        blur_shader_.set_float("u_intensity", 1.0f);
        glBindVertexArray(vao_);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // Vertical: B → A
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_a_);
        glBindTextureUnit(0, tex_b_);
        blur_shader_.set_int("u_horizontal", 0);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    // Composite: add blurred texture onto main framebuffer at FULL resolution
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(saved_vp_[0], saved_vp_[1], saved_vp_[2], saved_vp_[3]);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE); // Additive

    glBindTextureUnit(0, tex_a_);
    blur_shader_.bind();
    blur_shader_.set_int("u_tex", 0);
    blur_shader_.set_vec2("u_texel_size", texel);
    blur_shader_.set_int("u_horizontal", -1); // Passthrough
    blur_shader_.set_float("u_intensity", intensity);

    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    blur_shader_.unbind();

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

} // namespace ng
