#pragma once

#include "core/types.h"
#include "gpu/compute_shader.h"
#include "gpu/buffer.h"

#include <vector>

namespace ng {

class ParticleBuffer;
class MPMSolver;

// Extended Position-Based Dynamics (XPBD) solver for inter-particle
// constraints — distance (rope/chain) and bending (cloth). Sits on top of
// the MPM particle buffer: particles tagged with a rope/cloth material get
// zero constitutive stress in MPM and instead have their positions
// corrected by this solver after each MPM step.
//
// Jacobi-style iteration (Macklin et al. "XPBD" 2016 simplified): each pass
// of xpbd_solve.comp iterates constraints in parallel, writes position
// deltas into a per-particle int fixed-point accumulator, and xpbd_apply.comp
// commits them. Repeated N times per frame for stability.
//
// Constraint storage is a GPU buffer of 16-byte XpbdConstraint records,
// pre-allocated at init to a generous capacity. Ropes / cloth patches
// append batches to it via allocate_constraints().
class XpbdField {
public:
    struct Config {
        u32 max_constraints = 200000u;
    };

    struct Params {
        i32 iterations = 8;                // Jacobi passes per frame
        f32 default_rope_compliance = 0.0f; // stiff by default (1/stiffness ~0)
    };

    // GPU-side constraint record. Compliance = 1/stiffness; 0 = rigid.
    struct Constraint {
        u32 p_a;
        u32 p_b;
        f32 rest_len;
        f32 compliance;
    };

    void init(const Config& config);
    Params& params() { return params_; }
    const Params& params() const { return params_; }

    void step(ParticleBuffer& particles, f32 dt);

    // Append a chain of distance constraints connecting consecutive
    // particles in [start_global, start_global + count). Returns false on
    // capacity exhaustion.
    bool append_chain_constraints(u32 start_global, u32 count,
                                  f32 rest_len, f32 compliance);

    u32 constraint_count() const { return constraint_count_; }
    void clear_constraints();

private:
    Params params_{};
    u32 max_constraints_ = 0;
    u32 constraint_count_ = 0;

    GPUBuffer constraint_buf_;
    // Per-particle fixed-point accumulators for position deltas. Sized to
    // the full particle capacity so we can index by global particle index.
    GPUBuffer correction_x_buf_;
    GPUBuffer correction_y_buf_;

    ComputeShader solve_shader_;
    ComputeShader apply_shader_;
};

} // namespace ng
