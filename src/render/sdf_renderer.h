#pragma once

#include "core/types.h"
#include "gpu/shader.h"
#include "render/camera.h"

namespace ng {

class SDFField;

class SDFRenderer {
public:
    void init();

    void render(const SDFField& sdf, const Camera& camera,
                u32 smoke_tex = 0, u32 air_temp_tex = 0, u32 vel_tex = 0, u32 bio_tex = 0, u32 automata_tex = 0,
                f32 bio_view_gain = 1.0f, f32 automata_view_gain = 1.0f,
                i32 air_vis_mode = 0, i32 metal_palette = 0,
                f32 fire_temp_start = 305.0f, f32 fire_temp_range = 400.0f,
                f32 fire_softness = 3.0f,
                u32 selected_object_id = 0);

private:
    Shader shader_;
    u32 vao_ = 0;
};

} // namespace ng
