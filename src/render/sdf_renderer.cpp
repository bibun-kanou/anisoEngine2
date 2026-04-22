#include "render/sdf_renderer.h"
#include "physics/sdf/sdf_field.h"
#include "core/log.h"

#include <glad/gl.h>

namespace ng {

void SDFRenderer::init() {
    shader_.load("shaders/render/sdf_draw.vert", "shaders/render/sdf_draw.frag");
    glCreateVertexArrays(1, &vao_);
}

void SDFRenderer::render(const SDFField& sdf, const Camera& camera,
                          u32 smoke_tex, u32 air_temp_tex, u32 vel_tex, u32 bio_tex, u32 automata_tex,
                          f32 bio_view_gain, f32 automata_view_gain,
                          i32 air_vis_mode, i32 metal_palette,
                          f32 fire_temp_start, f32 fire_temp_range,
                          f32 fire_softness,
                          u32 selected_object_id) {
    shader_.bind();
    shader_.set_mat4("u_view_proj", camera.view_proj());
    shader_.set_vec2("u_sdf_world_min", sdf.world_min());
    shader_.set_vec2("u_sdf_world_max", sdf.world_max());
    shader_.set_int("u_metal_palette", metal_palette);
    shader_.set_float("u_fire_temp_start", fire_temp_start);
    shader_.set_float("u_fire_temp_range", fire_temp_range);
    shader_.set_float("u_fire_softness", fire_softness);
    shader_.set_int("u_selected_object_id", static_cast<i32>(selected_object_id));

    sdf.bind_for_read(0);
    shader_.set_int("u_sdf_tex", 0);
    sdf.bind_object_ids_for_read(1);
    shader_.set_int("u_sdf_object_id_tex", 1);
    sdf.bind_palette_for_read(5);
    shader_.set_int("u_sdf_palette_tex", 5);

    if (smoke_tex && air_temp_tex) {
        glBindTextureUnit(2, smoke_tex);
        glBindTextureUnit(3, air_temp_tex);
        glBindTextureUnit(4, vel_tex);
        glBindTextureUnit(6, bio_tex);
        glBindTextureUnit(7, automata_tex);
        shader_.set_int("u_smoke_tex", 2);
        shader_.set_int("u_air_temp_tex", 3);
        shader_.set_int("u_air_vel_tex", 4);
        shader_.set_int("u_air_bio_tex", 6);
        shader_.set_int("u_air_automata_tex", 7);
        shader_.set_float("u_air_bio_gain", bio_view_gain);
        shader_.set_float("u_air_automata_gain", automata_view_gain);
        shader_.set_int("u_show_air", air_vis_mode > 0 ? air_vis_mode : 1);
    } else {
        shader_.set_int("u_show_air", 0);
    }

    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    shader_.unbind();
}

} // namespace ng
