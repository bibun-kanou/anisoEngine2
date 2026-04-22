#pragma once

#include "core/types.h"
#include <string>

struct SDL_Window;

namespace ng {

struct WindowConfig {
    std::string title = "Noita-Gish Engine";
    i32 width  = 1280;
    i32 height = 720;
    bool vsync = true;
};

class Window {
public:
    Window() = default;
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool init(const WindowConfig& config);
    void init_imgui();
    void shutdown_imgui();
    void swap();
    void poll_events();

    bool should_close() const { return should_close_; }
    i32 width() const { return width_; }
    i32 height() const { return height_; }
    f32 scroll_delta() const { return scroll_delta_; }
    SDL_Window* handle() const { return window_; }

private:
    SDL_Window* window_ = nullptr;
    void* gl_context_ = nullptr;
    i32 width_ = 0;
    i32 height_ = 0;
    f32 scroll_delta_ = 0.0f;
    bool should_close_ = false;
};

} // namespace ng
