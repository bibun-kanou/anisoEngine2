#pragma once

#include "core/types.h"

namespace ng {

class ParticleBuffer;
class SPHSolver;
class MPMSolver;
class UniformGrid;
class SDFField;
struct CreationState;

enum class FootControlFocus : u32 {
    ANKLE = 0,
    HEEL,
    BALL,
    ALL_TOES,
    HALLUX,
    TOE_2,
    TOE_3,
    TOE_4,
    PINKY,
    COUNT
};

struct FootControlInput {
    vec2 mouse_world = vec2(0.0f);
    vec2 mouse_delta_world = vec2(0.0f);
    f32 dt = 0.0f;
    f32 wheel_delta = 0.0f;
    bool lmb_down = false;
    bool lmb_pressed = false;
    bool rmb_down = false;
    bool shift_down = false;
    bool ctrl_down = false;
    bool cycle_prev_pressed = false;
    bool cycle_next_pressed = false;
    bool curl_down = false;
    bool straighten_down = false;
    bool contract_down = false;
    bool extend_down = false;
};

bool foot_demo_active();
void clear_foot_demo();
void load_foot_demo_scene(ParticleBuffer& particles, SPHSolver& sph,
                          MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                          CreationState* creation);
void update_foot_demo(const FootControlInput& input,
                      ParticleBuffer& particles, MPMSolver& mpm,
                      const SDFField* sdf = nullptr);

FootControlFocus foot_demo_focus();
void set_foot_demo_focus(FootControlFocus focus);
const char* foot_demo_focus_name(FootControlFocus focus);
const char* foot_demo_focus_name();
const char* foot_demo_mode_hint();
vec2 foot_demo_focus_point();
f32 foot_demo_focus_radius();

void foot_demo_apply_scene_defaults(bool& mpm_skin_enabled,
                                    int& mpm_surface_style,
                                    float& mpm_skin_threshold,
                                    float& mpm_skin_kernel);

} // namespace ng
