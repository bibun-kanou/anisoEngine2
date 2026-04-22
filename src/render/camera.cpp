#include "render/camera.h"
#include <glm/gtc/matrix_transform.hpp>

namespace ng {

void Camera::set_viewport(i32 width, i32 height) {
    viewport_w_ = width;
    viewport_h_ = height;
}

mat4 Camera::view_proj() const {
    f32 hw = static_cast<f32>(viewport_w_) / (2.0f * zoom_);
    f32 hh = static_cast<f32>(viewport_h_) / (2.0f * zoom_);

    return glm::ortho(
        position_.x - hw, position_.x + hw,
        position_.y - hh, position_.y + hh,
        -1.0f, 1.0f
    );
}

vec2 Camera::screen_to_world(vec2 screen_pos) const {
    f32 hw = static_cast<f32>(viewport_w_) / (2.0f * zoom_);
    f32 hh = static_cast<f32>(viewport_h_) / (2.0f * zoom_);

    vec2 ndc;
    ndc.x = (screen_pos.x / viewport_w_) * 2.0f - 1.0f;
    ndc.y = 1.0f - (screen_pos.y / viewport_h_) * 2.0f; // Flip Y

    return position_ + vec2(ndc.x * hw, ndc.y * hh);
}

vec2 Camera::world_to_screen(vec2 world_pos) const {
    f32 hw = static_cast<f32>(viewport_w_) / (2.0f * zoom_);
    f32 hh = static_cast<f32>(viewport_h_) / (2.0f * zoom_);
    vec2 rel = world_pos - position_;

    vec2 ndc;
    ndc.x = rel.x / hw;
    ndc.y = rel.y / hh;

    return vec2(
        (ndc.x * 0.5f + 0.5f) * viewport_w_,
        (0.5f - ndc.y * 0.5f) * viewport_h_);
}

} // namespace ng
