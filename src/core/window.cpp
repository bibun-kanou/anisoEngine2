#include "core/window.h"
#include "core/log.h"

#include <glad/gl.h>
#include <SDL.h>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_opengl3.h>

namespace ng {

Window::~Window() {
    shutdown_imgui();
    if (gl_context_) SDL_GL_DeleteContext(gl_context_);
    if (window_) SDL_DestroyWindow(window_);
    SDL_Quit();
}

void Window::init_imgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForOpenGL(window_, gl_context_);
    ImGui_ImplOpenGL3_Init("#version 450");

    ImGuiStyle& style = ImGui::GetStyle();
    style.Alpha = 0.9f;
    style.WindowRounding = 6.0f;
    style.FrameRounding = 3.0f;
}

void Window::shutdown_imgui() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
}

bool Window::init(const WindowConfig& config) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        LOG_ERROR("SDL_Init failed: %s", SDL_GetError());
        return false;
    }

    // Request OpenGL 4.5 core
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0); // 2D engine, no depth buffer
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0);
#ifndef NDEBUG
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
#endif

    window_ = SDL_CreateWindow(
        config.title.c_str(),
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        config.width, config.height,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
    );
    if (!window_) {
        LOG_ERROR("SDL_CreateWindow failed: %s", SDL_GetError());
        return false;
    }

    gl_context_ = SDL_GL_CreateContext(window_);
    if (!gl_context_) {
        LOG_ERROR("SDL_GL_CreateContext failed: %s", SDL_GetError());
        return false;
    }

    // Load OpenGL functions via GLAD
    int version = gladLoadGL((GLADloadfunc)SDL_GL_GetProcAddress);
    if (!version) {
        LOG_ERROR("gladLoadGL failed");
        return false;
    }
    LOG_INFO("OpenGL %d.%d loaded", GLAD_VERSION_MAJOR(version), GLAD_VERSION_MINOR(version));
    LOG_INFO("Renderer: %s", glGetString(GL_RENDERER));
    LOG_INFO("Vendor: %s", glGetString(GL_VENDOR));

    // Check for compute shader support
    GLint max_compute_work_group_count[3];
    GLint max_compute_work_group_size[3];
    GLint max_compute_work_group_invocations;
    GLint max_ssbo_bindings;
    for (int i = 0; i < 3; i++) {
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, i, &max_compute_work_group_count[i]);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, i, &max_compute_work_group_size[i]);
    }
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &max_compute_work_group_invocations);
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &max_ssbo_bindings);

    LOG_INFO("Max compute work group size: %d x %d x %d",
        max_compute_work_group_size[0], max_compute_work_group_size[1], max_compute_work_group_size[2]);
    LOG_INFO("Max compute invocations: %d", max_compute_work_group_invocations);
    LOG_INFO("Max SSBO bindings: %d", max_ssbo_bindings);

#ifndef NDEBUG
    // Enable GL debug output
    if (GLAD_GL_KHR_debug) {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback([](GLenum source, GLenum type, GLuint id,
            GLenum severity, GLsizei /*length*/, const GLchar* message, const void* /*userParam*/) {
            if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;
            const char* sev_str = severity == GL_DEBUG_SEVERITY_HIGH ? "HIGH" :
                                  severity == GL_DEBUG_SEVERITY_MEDIUM ? "MED" : "LOW";
            LOG_WARN("GL [%s] type=0x%x id=%u: %s", sev_str, type, id, message);
        }, nullptr);
        // Suppress notification-level noise
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
    }
#endif

    SDL_GL_SetSwapInterval(config.vsync ? 1 : 0);

    width_ = config.width;
    height_ = config.height;

    return true;
}

void Window::swap() {
    SDL_GL_SwapWindow(window_);
}

void Window::poll_events() {
    scroll_delta_ = 0.0f;
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL2_ProcessEvent(&event);
        switch (event.type) {
        case SDL_QUIT:
            should_close_ = true;
            break;
        case SDL_WINDOWEVENT:
            if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                width_ = event.window.data1;
                height_ = event.window.data2;
                glViewport(0, 0, width_, height_);
            }
            break;
        case SDL_KEYDOWN:
            if (event.key.keysym.sym == SDLK_ESCAPE)
                should_close_ = true;
            break;
        case SDL_MOUSEWHEEL:
            scroll_delta_ += static_cast<f32>(event.wheel.y);
            break;
        }
    }
}

} // namespace ng
