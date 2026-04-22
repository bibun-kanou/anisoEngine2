#pragma once

#include "core/types.h"

namespace ng {

struct TimeStep {
    f32 fixed_dt       = 1.0f / 120.0f; // Physics timestep (120 Hz)
    i32 max_substeps   = 4;             // Max physics steps per frame
    f32 accumulator    = 0.0f;
    f64 total_time     = 0.0;
    u64 frame_count    = 0;
    f32 frame_dt       = 0.0f;          // Actual frame delta (variable)
    f32 alpha          = 0.0f;          // Interpolation factor for rendering

    void advance(f32 dt) {
        frame_dt = dt;
        accumulator += dt;
        // Clamp to prevent spiral of death
        if (accumulator > fixed_dt * max_substeps)
            accumulator = fixed_dt * max_substeps;
        frame_count++;
        total_time += dt;
    }

    bool consume_step() {
        if (accumulator >= fixed_dt) {
            accumulator -= fixed_dt;
            alpha = accumulator / fixed_dt;
            return true;
        }
        return false;
    }
};

} // namespace ng
