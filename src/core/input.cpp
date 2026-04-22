#include "core/input.h"
#include <cstring>

namespace ng {

void Input::update() {
    // Rotate current → previous
    std::memcpy(prev_keys_, curr_keys_, SDL_NUM_SCANCODES);
    prev_mouse_pos_ = mouse_pos_;
    prev_mouse_buttons_ = mouse_buttons_;

    // Snapshot keyboard (copy out of SDL's live array)
    const u8* sdl_keys = SDL_GetKeyboardState(nullptr);
    std::memcpy(curr_keys_, sdl_keys, SDL_NUM_SCANCODES);

    // Snapshot mouse
    int mx, my;
    mouse_buttons_ = SDL_GetMouseState(&mx, &my);
    mouse_pos_ = vec2(static_cast<f32>(mx), static_cast<f32>(my));
}

} // namespace ng
