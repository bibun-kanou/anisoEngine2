#include "core/engine.h"
#include "core/log.h"

#include <glad/gl.h>
#include <SDL.h>

namespace ng {

bool Engine::init(const WindowConfig& config) {
    if (!window_.init(config)) return false;
    LOG_INFO("Engine initialized");
    return true;
}

void Engine::run() {
    u64 prev_time = SDL_GetPerformanceCounter();
    u64 freq = SDL_GetPerformanceFrequency();

    while (!window_.should_close()) {
        // Compute frame delta
        u64 now = SDL_GetPerformanceCounter();
        f32 dt = static_cast<f32>(now - prev_time) / static_cast<f32>(freq);
        prev_time = now;

        // Input
        window_.poll_events();
        input_.update();

        // Fixed timestep physics loop
        time_step_.advance(dt);
        while (time_step_.consume_step()) {
            fixed_update(time_step_.fixed_dt);
        }

        // Render
        render();
        window_.swap();
    }
}

void Engine::fixed_update(f32 /*dt*/) {
    // TODO: physics world step, arcane update, damage resolve, AI tick
}

void Engine::render() {
    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // TODO: lighting, particle render, SDF render, post-process, UI
}

} // namespace ng
