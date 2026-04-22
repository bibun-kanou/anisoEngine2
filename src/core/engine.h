#pragma once

#include "core/window.h"
#include "core/input.h"
#include "core/time_step.h"

namespace ng {

class Engine {
public:
    bool init(const WindowConfig& config = {});
    void run();

    Window& window() { return window_; }
    Input& input() { return input_; }
    TimeStep& time() { return time_step_; }

private:
    Window window_;
    Input input_;
    TimeStep time_step_;

    // Called each frame - override points for game logic
    void fixed_update(f32 dt);
    void render();
};

} // namespace ng
