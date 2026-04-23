#include "physics/xpbd/xpbd_field.h"

#include "physics/common/particle_buffer.h"
#include "core/log.h"

#include <glad/gl.h>
#include <algorithm>
#include <vector>

namespace ng {

namespace {
// SSBO bindings — chosen to not collide with magnetic (18-22) or
// electrostatic (23-24).
constexpr u32 kConstraintBinding   = 30u;
constexpr u32 kCorrectionXBinding  = 31u;
constexpr u32 kCorrectionYBinding  = 32u;
constexpr u32 kMaxParticles        = 500000u;
} // namespace

void XpbdField::init(const Config& config) {
    max_constraints_ = config.max_constraints;
    constraint_count_ = 0;

    constraint_buf_.create(static_cast<size_t>(max_constraints_) * sizeof(Constraint));
    correction_x_buf_.create(static_cast<size_t>(kMaxParticles) * sizeof(i32));
    correction_y_buf_.create(static_cast<size_t>(kMaxParticles) * sizeof(i32));
    correction_x_buf_.clear();
    correction_y_buf_.clear();

    solve_shader_.load("shaders/physics/xpbd_solve.comp");
    apply_shader_.load("shaders/physics/xpbd_apply.comp");

    LOG_INFO("XpbdField: capacity %u constraints", max_constraints_);
}

void XpbdField::clear_constraints() {
    constraint_count_ = 0;
}

bool XpbdField::append_chain_constraints(u32 start_global, u32 count,
                                         f32 rest_len, f32 compliance) {
    if (count < 2) return true;
    u32 n_new = count - 1;
    if (constraint_count_ + n_new > max_constraints_) {
        LOG_INFO("XpbdField: constraint capacity exhausted (%u + %u > %u)",
                 constraint_count_, n_new, max_constraints_);
        return false;
    }
    std::vector<Constraint> batch(n_new);
    for (u32 i = 0; i < n_new; ++i) {
        batch[i].p_a = start_global + i;
        batch[i].p_b = start_global + i + 1;
        batch[i].rest_len = rest_len;
        batch[i].compliance = compliance;
    }
    constraint_buf_.upload(batch.data(), n_new * sizeof(Constraint),
                           constraint_count_ * sizeof(Constraint));
    constraint_count_ += n_new;
    return true;
}

void XpbdField::step(ParticleBuffer& particles, f32 dt) {
    if (constraint_count_ == 0) return;
    if (dt <= 1e-6f) return;

    const auto& range = particles.range(SolverType::MPM);
    if (range.count == 0) return;

    particles.positions().bind_base(Binding::POSITION);
    constraint_buf_.bind_base(kConstraintBinding);
    correction_x_buf_.bind_base(kCorrectionXBinding);
    correction_y_buf_.bind_base(kCorrectionYBinding);

    const i32 iters = std::max(params_.iterations, 1);
    for (i32 it = 0; it < iters; ++it) {
        // Solve: parallel over constraints, accumulate per-particle deltas
        // into int fixed-point buffers via atomic adds.
        solve_shader_.bind();
        solve_shader_.set_uint("u_constraint_count", constraint_count_);
        solve_shader_.set_float("u_dt", dt);
        solve_shader_.dispatch_1d(constraint_count_);
        ComputeShader::barrier_ssbo();

        // Apply: per-particle pass that decodes the accumulator back into
        // a vec2 delta and adds it to positions, then clears the accum.
        apply_shader_.bind();
        apply_shader_.set_uint("u_particle_count", range.count);
        apply_shader_.set_uint("u_offset", range.offset);
        apply_shader_.dispatch_1d(range.count);
        ComputeShader::barrier_ssbo();
    }
}

} // namespace ng
