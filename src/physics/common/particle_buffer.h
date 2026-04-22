#pragma once

#include "core/types.h"
#include "gpu/buffer.h"
#include <array>

namespace ng {

// Solver types that can own particles
enum class SolverType : u32 {
    NONE = 0,
    SPH  = 1,
    MPM  = 2,
    SPRING = 3,
    ARCANE = 4,
    RIGID_REF = 5, // Rigid body reference/boundary particles
};

// Particle flags (bit field)
namespace ParticleFlags {
    constexpr u32 ALIVE    = 1 << 0;
    constexpr u32 ACTIVE   = 1 << 1; // Can be temporarily deactivated
    constexpr u32 BOUNDARY = 1 << 2; // Boundary particle (for coupling)
}

// Index range owned by a solver
struct ParticleRange {
    u32 offset = 0;
    u32 count  = 0;
    u32 capacity = 0;

    u32 end() const { return offset + count; }
};

// SSBO binding points — shared between C++ and GLSL
namespace Binding {
    constexpr u32 POSITION     = 0;
    constexpr u32 VELOCITY     = 1;
    constexpr u32 MASS         = 2;
    constexpr u32 DENSITY      = 3;
    constexpr u32 DEFORM_GRAD  = 4;  // MPM: mat2 (4 floats)
    constexpr u32 AFFINE_MOM   = 5;  // MPM: mat2 (APIC C matrix)
    constexpr u32 TEMPERATURE  = 6;
    constexpr u32 MATERIAL_ID  = 7;
    constexpr u32 FLAGS        = 8;
    constexpr u32 COLOR        = 9;
    constexpr u32 PRESSURE     = 10; // SPH
    constexpr u32 CELL_INDEX   = 11; // Spatial hash cell
    constexpr u32 SORTED_INDEX = 12; // Sort result

    constexpr u32 SPRING_DATA  = 30; // Spring buffer
    constexpr u32 MATERIAL_DB  = 20; // Material database
}

// Unified SOA particle storage on GPU.
// All particle-like entities share a single large buffer array.
// Each solver owns a contiguous index range.
class ParticleBuffer {
public:
    struct Config {
        u32 max_particles = 500000;
        u32 sph_capacity  = 200000;
        u32 mpm_capacity  = 200000;
        u32 spring_capacity = 50000;
        u32 arcane_capacity = 50000;
    };

    void init(const Config& config);
    void destroy();

    // Bind all SSBOs to their binding points
    void bind_all() const;

    // Get range for a solver
    ParticleRange& range(SolverType type);
    const ParticleRange& range(SolverType type) const;

    // Allocate particles for a solver (within its range capacity)
    // Returns offset of first new particle, or UINT32_MAX on failure
    u32 allocate(SolverType type, u32 count);

    // Upload initial data for a batch of particles
    void upload_positions(u32 offset, const vec2* data, u32 count);
    void upload_velocities(u32 offset, const vec2* data, u32 count);
    void upload_masses(u32 offset, const f32* data, u32 count);
    void upload_colors(u32 offset, const vec4* data, u32 count);
    void upload_temperatures(u32 offset, const f32* data, u32 count);
    void upload_material_ids(u32 offset, const u32* data, u32 count);

    // Accessors
    u32 total_capacity() const { return total_capacity_; }
    u32 total_active() const;

    GPUBuffer& positions()    { return position_buf_; }
    GPUBuffer& velocities()   { return velocity_buf_; }
    GPUBuffer& masses()       { return mass_buf_; }
    GPUBuffer& densities()    { return density_buf_; }
    GPUBuffer& deform_grads() { return deform_grad_buf_; }
    GPUBuffer& affine_moms()  { return affine_mom_buf_; }
    GPUBuffer& temperatures() { return temperature_buf_; }
    GPUBuffer& material_ids() { return material_id_buf_; }
    GPUBuffer& flags()        { return flags_buf_; }
    GPUBuffer& colors()       { return color_buf_; }
    GPUBuffer& pressures()    { return pressure_buf_; }
    GPUBuffer& cell_indices() { return cell_index_buf_; }
    GPUBuffer& sorted_indices() { return sorted_index_buf_; }

private:
    u32 total_capacity_ = 0;

    // One range per solver type
    std::array<ParticleRange, 6> ranges_{}; // indexed by SolverType

    // SOA GPU buffers
    GPUBuffer position_buf_;
    GPUBuffer velocity_buf_;
    GPUBuffer mass_buf_;
    GPUBuffer density_buf_;
    GPUBuffer deform_grad_buf_;
    GPUBuffer affine_mom_buf_;
    GPUBuffer temperature_buf_;
    GPUBuffer material_id_buf_;
    GPUBuffer flags_buf_;
    GPUBuffer color_buf_;
    GPUBuffer pressure_buf_;
    GPUBuffer cell_index_buf_;
    GPUBuffer sorted_index_buf_;
};

} // namespace ng
