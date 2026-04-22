#pragma once

#include "core/types.h"

namespace ng {

class Camera {
public:
    void set_viewport(i32 width, i32 height);
    void set_position(vec2 pos) { position_ = pos; }
    void set_zoom(f32 zoom) { zoom_ = zoom; }

    void pan(vec2 delta) { position_ += delta / zoom_; }
    void zoom(f32 factor) { zoom_ *= factor; zoom_ = glm::clamp(zoom_, 1.0f, 10000.0f); }

    mat4 view_proj() const;

    vec2 position() const { return position_; }
    f32 zoom_level() const { return zoom_; }
    vec2 screen_to_world(vec2 screen_pos) const;
    vec2 world_to_screen(vec2 world_pos) const;

private:
    vec2 position_ = vec2(0.0f);
    f32 zoom_ = 100.0f;  // Pixels per world unit
    i32 viewport_w_ = 1280;
    i32 viewport_h_ = 720;
};

} // namespace ng
