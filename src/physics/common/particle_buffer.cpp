#include "physics/common/particle_buffer.h"
#include "core/log.h"

namespace ng {

void ParticleBuffer::init(const Config& config) {
    total_capacity_ = config.max_particles;

    // Assign contiguous ranges
    u32 offset = 0;
    auto assign = [&](SolverType type, u32 cap) {
        auto& r = ranges_[static_cast<u32>(type)];
        r.offset = offset;
        r.count = 0; // Nothing allocated yet
        r.capacity = cap;
        offset += cap;
    };

    assign(SolverType::SPH,      config.sph_capacity);
    assign(SolverType::MPM,      config.mpm_capacity);
    assign(SolverType::SPRING,   config.spring_capacity);
    assign(SolverType::ARCANE,   config.arcane_capacity);

    if (offset > total_capacity_) {
        LOG_FATAL("Solver capacities (%u) exceed max_particles (%u)", offset, total_capacity_);
    }

    // Create all SOA buffers
    position_buf_.create(total_capacity_ * sizeof(vec2));
    velocity_buf_.create(total_capacity_ * sizeof(vec2));
    mass_buf_.create(total_capacity_ * sizeof(f32));
    density_buf_.create(total_capacity_ * sizeof(f32));
    deform_grad_buf_.create(config.mpm_capacity * sizeof(f32) * 4); // mat2 = 4 floats
    affine_mom_buf_.create(config.mpm_capacity * sizeof(f32) * 4);
    temperature_buf_.create(total_capacity_ * sizeof(f32));
    material_id_buf_.create(total_capacity_ * sizeof(u32));
    flags_buf_.create(total_capacity_ * sizeof(u32));
    color_buf_.create(total_capacity_ * sizeof(vec4));
    pressure_buf_.create(config.sph_capacity * sizeof(f32));
    cell_index_buf_.create(total_capacity_ * sizeof(u32));
    sorted_index_buf_.create(total_capacity_ * sizeof(u32));

    // Zero everything
    position_buf_.clear();
    velocity_buf_.clear();
    mass_buf_.clear();
    density_buf_.clear();
    deform_grad_buf_.clear();
    affine_mom_buf_.clear();
    temperature_buf_.clear();
    material_id_buf_.clear();
    flags_buf_.clear();
    color_buf_.clear();
    pressure_buf_.clear();
    cell_index_buf_.clear();
    sorted_index_buf_.clear();

    LOG_INFO("ParticleBuffer initialized: %u total capacity", total_capacity_);
    LOG_INFO("  SPH: %u, MPM: %u, Spring: %u, Arcane: %u",
        config.sph_capacity, config.mpm_capacity,
        config.spring_capacity, config.arcane_capacity);
}

void ParticleBuffer::destroy() {
    // GPUBuffer destructors handle cleanup
}

void ParticleBuffer::bind_all() const {
    position_buf_.bind_base(Binding::POSITION);
    velocity_buf_.bind_base(Binding::VELOCITY);
    mass_buf_.bind_base(Binding::MASS);
    density_buf_.bind_base(Binding::DENSITY);
    deform_grad_buf_.bind_base(Binding::DEFORM_GRAD);
    affine_mom_buf_.bind_base(Binding::AFFINE_MOM);
    temperature_buf_.bind_base(Binding::TEMPERATURE);
    material_id_buf_.bind_base(Binding::MATERIAL_ID);
    flags_buf_.bind_base(Binding::FLAGS);
    color_buf_.bind_base(Binding::COLOR);
    pressure_buf_.bind_base(Binding::PRESSURE);
    cell_index_buf_.bind_base(Binding::CELL_INDEX);
    sorted_index_buf_.bind_base(Binding::SORTED_INDEX);
}

ParticleRange& ParticleBuffer::range(SolverType type) {
    return ranges_[static_cast<u32>(type)];
}

const ParticleRange& ParticleBuffer::range(SolverType type) const {
    return ranges_[static_cast<u32>(type)];
}

u32 ParticleBuffer::allocate(SolverType type, u32 count) {
    auto& r = ranges_[static_cast<u32>(type)];
    if (r.count + count > r.capacity) {
        LOG_ERROR("Cannot allocate %u particles for solver %u: %u/%u used",
            count, static_cast<u32>(type), r.count, r.capacity);
        return UINT32_MAX;
    }
    u32 first = r.offset + r.count;
    r.count += count;
    return first;
}

u32 ParticleBuffer::total_active() const {
    u32 total = 0;
    for (auto& r : ranges_) total += r.count;
    return total;
}

void ParticleBuffer::upload_positions(u32 offset, const vec2* data, u32 count) {
    position_buf_.upload(data, count * sizeof(vec2), offset * sizeof(vec2));
}

void ParticleBuffer::upload_velocities(u32 offset, const vec2* data, u32 count) {
    velocity_buf_.upload(data, count * sizeof(vec2), offset * sizeof(vec2));
}

void ParticleBuffer::upload_masses(u32 offset, const f32* data, u32 count) {
    mass_buf_.upload(data, count * sizeof(f32), offset * sizeof(f32));
}

void ParticleBuffer::upload_colors(u32 offset, const vec4* data, u32 count) {
    color_buf_.upload(data, count * sizeof(vec4), offset * sizeof(vec4));
}

void ParticleBuffer::upload_temperatures(u32 offset, const f32* data, u32 count) {
    temperature_buf_.upload(data, count * sizeof(f32), offset * sizeof(f32));
}

void ParticleBuffer::upload_material_ids(u32 offset, const u32* data, u32 count) {
    material_id_buf_.upload(data, count * sizeof(u32), offset * sizeof(u32));
}

} // namespace ng
