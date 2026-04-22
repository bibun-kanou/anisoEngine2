#pragma once

#include "core/types.h"

namespace ng {

// Draw an interactive pipeline diagram using ImGui DrawList
// Shows all physics passes, their order, and connections
void draw_pipeline_viewer(bool* open);
void reset_pipeline_viewer_session();

} // namespace ng
