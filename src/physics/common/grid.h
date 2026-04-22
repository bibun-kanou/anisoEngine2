#pragma once

#include "core/types.h"
#include "gpu/buffer.h"

namespace ng {

// Uniform grid on GPU for MPM (and later Eulerian fluid).
// P2G uses atomic uint SSBOs; after grid update, float velocity SSBOs are filled.
class UniformGrid {
public:
    struct Config {
        ivec2 resolution = ivec2(192, 192);
        vec2  world_min  = vec2(-3.2f, -2.4f);
        vec2  world_max  = vec2(3.2f, 4.0f);
    };

    void init(const Config& config);

    void bind_for_p2g() const;   // Bind atomic uint buffers (mass, mom_x, mom_y)
    void bind_for_grid() const;  // Bind both atomic and velocity buffers
    void bind_for_g2p() const;   // Bind float velocity buffers

    f32 dx() const { return dx_; }
    ivec2 resolution() const { return resolution_; }
    vec2 world_min() const { return world_min_; }
    u32 total_cells() const { return static_cast<u32>(resolution_.x * resolution_.y); }

    // Grid SSBO binding points
    static constexpr u32 BIND_GRID_MASS   = 40; // uint (atomic P2G)
    static constexpr u32 BIND_GRID_MOM_X  = 41; // uint (atomic P2G)
    static constexpr u32 BIND_GRID_MOM_Y  = 42; // uint (atomic P2G)
    static constexpr u32 BIND_GRID_VEL_X  = 43; // float (after grid update)
    static constexpr u32 BIND_GRID_VEL_Y  = 44; // float (after grid update)
    static constexpr u32 BIND_GRID_TEMP   = 45; // float (thermal)
    static constexpr u32 BIND_GRID_TEMP2  = 46; // float (thermal diffusion output)
    static constexpr u32 BIND_GRID_TEMP_ATOMIC = 55; // int (atomic P2G temp scatter)
    static constexpr u32 BIND_GRID_CONTACT_TEMP = 56; // int (direct SPH->MPM contact heat)

    GPUBuffer& mass_buf()  { return mass_buf_; }
    const GPUBuffer& mass_buf() const { return mass_buf_; }
    GPUBuffer& mom_x_buf() { return mom_x_buf_; }
    GPUBuffer& mom_y_buf() { return mom_y_buf_; }
    GPUBuffer& vel_x_buf() { return vel_x_buf_; }
    GPUBuffer& vel_y_buf() { return vel_y_buf_; }
    GPUBuffer& temp_buf()  { return temp_buf_; }
    const GPUBuffer& temp_buf() const { return temp_buf_; }
    GPUBuffer& temp2_buf() { return temp2_buf_; }
    GPUBuffer& temp_atomic_buf() { return temp_atomic_buf_; }
    GPUBuffer& contact_temp_buf() { return contact_temp_buf_; }
    const GPUBuffer& contact_temp_buf() const { return contact_temp_buf_; }

private:
    ivec2 resolution_{0};
    vec2 world_min_{0.0f};
    vec2 world_max_{0.0f};
    f32 dx_ = 0.0f;

    GPUBuffer mass_buf_;   // uint[] for atomic add
    GPUBuffer mom_x_buf_;  // uint[]
    GPUBuffer mom_y_buf_;  // uint[]
    GPUBuffer vel_x_buf_;  // float[]
    GPUBuffer vel_y_buf_;  // float[]
    GPUBuffer temp_buf_;        // float[] (temperature)
    GPUBuffer temp2_buf_;       // float[] (temp diffusion scratch)
    GPUBuffer temp_atomic_buf_; // int[] (atomic P2G temp*mass scatter)
    GPUBuffer contact_temp_buf_; // int[] (atomic SPH->MPM contact heat scatter)
};

} // namespace ng
