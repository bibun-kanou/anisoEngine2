#include "render/particle_renderer.h"
#include "physics/common/particle_buffer.h"
#include "core/log.h"

#include <glad/gl.h>

namespace ng {

void ParticleRenderer::init() {
    shader_.load("shaders/render/particle_draw.vert", "shaders/render/particle_draw.frag");

    // Create empty VAO (all vertex data comes from SSBOs via gl_VertexID)
    glCreateVertexArrays(1, &vao_);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void ParticleRenderer::render(const ParticleBuffer& particles, const Camera& camera,
                               u32 offset, u32 count, f32 point_size) {
    if (count == 0) return;

    particles.bind_all();

    shader_.bind();
    shader_.set_mat4("u_view_proj", camera.view_proj());
    shader_.set_float("u_point_size", point_size);
    shader_.set_int("u_offset", static_cast<i32>(offset));

    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(count));
    glBindVertexArray(0);

    shader_.unbind();
}

} // namespace ng
