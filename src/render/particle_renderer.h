#pragma once

#include "core/types.h"
#include "gpu/shader.h"
#include "render/camera.h"

namespace ng {

class ParticleBuffer;

class ParticleRenderer {
public:
    void init();

    // Render particles as point sprites
    void render(const ParticleBuffer& particles, const Camera& camera,
                u32 offset, u32 count, f32 point_size = 3.0f);

private:
    Shader shader_;
    u32 vao_ = 0; // Empty VAO (vertex data comes from SSBOs)
};

} // namespace ng
