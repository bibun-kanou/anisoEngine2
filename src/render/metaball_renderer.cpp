#include "render/metaball_renderer.h"
#include "physics/common/particle_buffer.h"
#include "render/camera.h"
#include "core/log.h"

#include <glad/gl.h>

namespace ng {

void MetaballRenderer::init(i32 width, i32 height) {
    tex_size_ = ivec2(width / 2, height / 2);

    // Create density texture (RGBA16F at half resolution)
    glCreateTextures(GL_TEXTURE_2D, 1, &density_tex_);
    glTextureStorage2D(density_tex_, 1, GL_RGBA16F, tex_size_.x, tex_size_.y);
    glTextureParameteri(density_tex_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(density_tex_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(density_tex_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(density_tex_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Create FBO
    glCreateFramebuffers(1, &fbo_);
    glNamedFramebufferTexture(fbo_, GL_COLOR_ATTACHMENT0, density_tex_, 0);

    GLenum status = glCheckNamedFramebufferStatus(fbo_, GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        LOG_ERROR("Metaball FBO incomplete: 0x%X", status);
    }

    // Load shaders
    splat_shader_.load("shaders/render/metaball_splat.vert",
                       "shaders/render/metaball_splat.frag");
    surface_shader_.load("shaders/render/sdf_draw.vert",
                         "shaders/render/metaball_surface.frag");

    // Empty VAO for point / fullscreen draws
    glCreateVertexArrays(1, &vao_);
}

void MetaballRenderer::splat(const ParticleBuffer& particles, const Camera& camera,
                             u32 offset, u32 count, f32 kernel_radius) {
    if (count == 0) return;

    // Save current viewport
    GLint prev_viewport[4];
    glGetIntegerv(GL_VIEWPORT, prev_viewport);

    // Bind FBO and clear
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glViewport(0, 0, tex_size_.x, tex_size_.y);
    GLfloat clear_color[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    glClearNamedFramebufferfv(fbo_, GL_COLOR, 0, clear_color);

    // Additive blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Bind particle SSBOs
    particles.bind_all();

    // Draw splats
    splat_shader_.bind();
    splat_shader_.set_mat4("u_view_proj", camera.view_proj());
    splat_shader_.set_float("u_kernel_size", camera.zoom_level() * kernel_radius);
    splat_shader_.set_int("u_offset", static_cast<i32>(offset));

    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(count));
    glBindVertexArray(0);

    splat_shader_.unbind();

    // Restore viewport and blend mode
    glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3]);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void MetaballRenderer::render_surface(f32 threshold, SurfaceStyle style) {
    // Bind density texture to unit 1
    glBindTextureUnit(1, density_tex_);

    surface_shader_.bind();
    surface_shader_.set_int("u_density_tex", 1);
    surface_shader_.set_float("u_threshold", threshold);
    surface_shader_.set_int("u_style", static_cast<i32>(style));
    surface_shader_.set_float("u_edge_softness", edge_softness);
    surface_shader_.set_float("u_gloss", gloss);
    surface_shader_.set_float("u_rim_strength", rim);
    surface_shader_.set_float("u_opacity", opacity);
    surface_shader_.set_vec2("u_texel_size",
                             vec2(1.0f / static_cast<f32>(tex_size_.x),
                                  1.0f / static_cast<f32>(tex_size_.y)));

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Full-screen triangle
    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    surface_shader_.unbind();
    glBindTextureUnit(1, 0);
}

} // namespace ng
