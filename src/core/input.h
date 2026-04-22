#pragma once

#include "core/types.h"
#include <SDL.h>

namespace ng {

class Input {
public:
    void update();

    bool key_down(SDL_Scancode key) const { return curr_keys_[key]; }
    bool key_pressed(SDL_Scancode key) const { return curr_keys_[key] && !prev_keys_[key]; }
    bool key_released(SDL_Scancode key) const { return !curr_keys_[key] && prev_keys_[key]; }

    vec2 mouse_pos() const { return mouse_pos_; }
    vec2 mouse_delta() const { return mouse_pos_ - prev_mouse_pos_; }
    bool mouse_down(int button) const { return (mouse_buttons_ & SDL_BUTTON(button)) != 0; }
    bool mouse_pressed(int button) const {
        u32 mask = SDL_BUTTON(button);
        return (mouse_buttons_ & mask) && !(prev_mouse_buttons_ & mask);
    }
    bool mouse_released(int button) const {
        u32 mask = SDL_BUTTON(button);
        return !(mouse_buttons_ & mask) && (prev_mouse_buttons_ & mask);
    }

private:
    // We keep our own copies because SDL_GetKeyboardState returns a live
    // pointer that mutates during SDL_PollEvent.  Comparing two snapshots
    // of that same pointer would never detect edges.
    u8 curr_keys_[SDL_NUM_SCANCODES] = {};
    u8 prev_keys_[SDL_NUM_SCANCODES] = {};
    vec2 mouse_pos_{0.0f};
    vec2 prev_mouse_pos_{0.0f};
    u32 mouse_buttons_ = 0;
    u32 prev_mouse_buttons_ = 0;
};

} // namespace ng
