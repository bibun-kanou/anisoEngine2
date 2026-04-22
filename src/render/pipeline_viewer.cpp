#include "pipeline_viewer.h"
#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>

namespace ng {

// ── Colour palette ──────────────────────────────────────────────────────────
static const ImU32 COL_BG           = IM_COL32(22, 22, 30, 240);
static const ImU32 COL_GRID         = IM_COL32(38, 38, 52, 120);
static const ImU32 COL_BORDER       = IM_COL32(90, 90, 120, 180);

static const ImU32 COL_SPH          = IM_COL32(60, 140, 255, 255);
static const ImU32 COL_SPH_FILL     = IM_COL32(30,  70, 160, 180);
static const ImU32 COL_MPM          = IM_COL32(255, 160,  50, 255);
static const ImU32 COL_MPM_FILL     = IM_COL32(160,  80,  20, 180);
static const ImU32 COL_EULER        = IM_COL32(100, 220, 130, 255);
static const ImU32 COL_EULER_FILL   = IM_COL32( 40, 130,  60, 180);
static const ImU32 COL_SDF          = IM_COL32(200, 200, 200, 255);
static const ImU32 COL_SDF_FILL     = IM_COL32(100, 100, 110, 180);
static const ImU32 COL_COUPLING     = IM_COL32(255,  80, 200, 200);
static const ImU32 COL_THERMAL      = IM_COL32(255,  80,  60, 220);
static const ImU32 COL_VAPOR        = IM_COL32(180, 200, 255, 200);
static const ImU32 COL_FIELD        = IM_COL32(115, 230, 255, 230);
static const ImU32 COL_BIO          = IM_COL32(225, 205, 128, 230);
static const ImU32 COL_MEMORY       = IM_COL32(255, 214, 124, 230);
static const ImU32 COL_RENDER       = IM_COL32(200, 160, 255, 255);
static const ImU32 COL_RENDER_FILL  = IM_COL32(100,  60, 160, 180);
static const ImU32 COL_TEXT         = IM_COL32(230, 230, 240, 255);
static const ImU32 COL_TEXT_DIM     = IM_COL32(160, 160, 180, 200);
static const ImU32 COL_ARROW        = IM_COL32(200, 200, 220, 180);

// ── Layout constants ────────────────────────────────────────────────────────
static constexpr float NODE_W       = 168.0f;
static constexpr float NODE_H       = 38.0f;
static constexpr float SUB_W        = 158.0f;
static constexpr float SUB_H        = 30.0f;
static constexpr float COL_GAP      = 300.0f;
static constexpr float ROW_STEP     = 54.0f;
static constexpr float SUB_ROW      = 36.0f;
static constexpr float HEADER_H     = 26.0f;
static constexpr float CANVAS_W     = 1920.0f;
static constexpr float CANVAS_H     = 980.0f;

// ── Draw helpers ────────────────────────────────────────────────────────────
struct NodeBox {
    const char* id = "";
    ImVec2 pos;      // top-left corner (absolute)
    ImVec2 size;
    const char* label;
    const char* tooltip = "";
    const char* feature_key = nullptr;
    bool* toggle_ptr = nullptr;
    ImU32 fill;
    ImU32 border;
    bool  enabled;
    bool  hovered = false;
    bool  active = false;
    bool  toggle_hovered = false;
};

static std::unordered_map<std::string, ImVec2> g_node_offsets;
static ImVec2 g_canvas_origin = ImVec2(0.0f, 0.0f);
static ImVec2 g_view_pan = ImVec2(24.0f, 18.0f);
static float g_view_zoom = 1.0f;
static bool g_view_session_active = false;
static bool g_legend_open = true;
static const char* g_sidebar_hover_feature = nullptr;
static std::string g_dragging_node_id;

static constexpr const char* FEAT_SPH = "sph";
static constexpr const char* FEAT_SPH_CODIM = "sph_codim";
static constexpr const char* FEAT_SPH_SURFACE = "sph_surface";
static constexpr const char* FEAT_MPM = "mpm";
static constexpr const char* FEAT_MPM_THERMAL = "mpm_thermal";
static constexpr const char* FEAT_MPM_COMBUSTION = "mpm_combustion";
static constexpr const char* FEAT_MPM_FRACTURE = "mpm_fracture";
static constexpr const char* FEAT_MPM_PHASE = "mpm_phase";
static constexpr const char* FEAT_MPM_SHELL = "mpm_shell";
static constexpr const char* FEAT_MPM_PRESSURE = "mpm_pressure";
static constexpr const char* FEAT_MPM_FILAMENT = "mpm_filament";
static constexpr const char* FEAT_MPM_BIO = "mpm_bio";
static constexpr const char* FEAT_MPM_MEMORY = "mpm_memory";
static constexpr const char* FEAT_MPM_FIELD = "mpm_field";
static constexpr const char* FEAT_MPM_GSCALE = "mpm_gscale";
static constexpr const char* FEAT_EULER = "euler";
static constexpr const char* FEAT_EULER_VAPOR = "euler_vapor";
static constexpr const char* FEAT_EULER_THERMAL = "euler_thermal";
static constexpr const char* FEAT_EULER_LATENT = "euler_latent";
static constexpr const char* FEAT_SDF = "sdf";
static constexpr const char* FEAT_SDF_HEAT = "sdf_heat";
static constexpr const char* FEAT_COUPLE_SPH_MPM = "couple_sph_mpm";
static constexpr const char* FEAT_COUPLE_INJECT = "couple_inject";
static constexpr const char* FEAT_COUPLE_DRAG = "couple_drag";
static constexpr const char* FEAT_RENDER_BLOOM = "render_bloom";
static constexpr const char* FEAT_RENDER_METABALL = "render_metaball";

struct LinkHoverState {
    float best_dist = 10.0f;
    const char* tooltip = nullptr;
    const char* feature_key = nullptr;
    ImU32 color = 0;
    float thickness = 0.0f;
    ImVec2 pts[6]{};
    int pt_count = 0;
};

static bool feature_matches(const char* a, const char* b) {
    return a && b && std::strcmp(a, b) == 0;
}

static bool rect_clicked(ImVec2 minp, ImVec2 maxp) {
    return ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
           ImGui::IsMouseHoveringRect(minp, maxp, false) &&
           ImGui::IsMouseClicked(ImGuiMouseButton_Left);
}

void reset_pipeline_viewer_session() {
    g_node_offsets.clear();
    g_view_pan = ImVec2(24.0f, 18.0f);
    g_view_zoom = 1.0f;
    g_view_session_active = false;
    g_dragging_node_id.clear();
}

static ImVec2 view_pt(ImVec2 p) {
    return ImVec2(g_canvas_origin.x + g_view_pan.x + p.x * g_view_zoom,
                  g_canvas_origin.y + g_view_pan.y + p.y * g_view_zoom);
}

static ImVec2 view_size(ImVec2 s) {
    return ImVec2(s.x * g_view_zoom, s.y * g_view_zoom);
}

static ImVec2& offset_ref(const char* id) {
    return g_node_offsets[std::string(id)];
}

static NodeBox make_node(const char* id, ImVec2 base_pos, ImVec2 size,
                         const char* label, const char* tooltip,
                         ImU32 fill, ImU32 border, bool enabled) {
    ImVec2 off = offset_ref(id);
    NodeBox n;
    n.id = id;
    n.pos = view_pt(ImVec2(base_pos.x + off.x, base_pos.y + off.y));
    n.size = view_size(size);
    n.label = label;
    n.tooltip = tooltip;
    n.fill = fill;
    n.border = border;
    n.enabled = enabled;
    return n;
}

static ImVec2 node_center(const NodeBox& n) {
    return ImVec2(n.pos.x + n.size.x * 0.5f, n.pos.y + n.size.y * 0.5f);
}
static ImVec2 node_right(const NodeBox& n, float t = 0.5f) {
    return ImVec2(n.pos.x + n.size.x, n.pos.y + n.size.y * t);
}
static ImVec2 node_left(const NodeBox& n, float t = 0.5f) {
    return ImVec2(n.pos.x, n.pos.y + n.size.y * t);
}
static ImVec2 node_bottom(const NodeBox& n, float t = 0.5f) {
    return ImVec2(n.pos.x + n.size.x * t, n.pos.y + n.size.y);
}
static ImVec2 node_top(const NodeBox& n, float t = 0.5f) {
    return ImVec2(n.pos.x + n.size.x * t, n.pos.y);
}

static void draw_node(ImDrawList* dl, const NodeBox& n, const char* /*active_feature*/ = nullptr) {
    ImU32 fill = n.enabled ? n.fill : IM_COL32(50, 50, 50, 120);
    ImU32 bdr  = n.enabled ? n.border : IM_COL32(80, 80, 80, 120);
    ImU32 text = n.enabled ? COL_TEXT : COL_TEXT_DIM;
    bool glowy = n.hovered || n.active || n.toggle_hovered;
    if (glowy) {
        bdr = IM_COL32(255, 255, 255, 220);
        dl->AddRect(n.pos, ImVec2(n.pos.x + n.size.x, n.pos.y + n.size.y),
                    IM_COL32(255, 255, 255, 36), 8.0f, 0, 6.0f);
        dl->AddRect(n.pos, ImVec2(n.pos.x + n.size.x, n.pos.y + n.size.y),
                    IM_COL32(255, 255, 255, 80), 7.0f, 0, 3.0f);
    }

    dl->AddRectFilled(n.pos, ImVec2(n.pos.x + n.size.x, n.pos.y + n.size.y),
                      fill, 6.0f);
    dl->AddRect(n.pos, ImVec2(n.pos.x + n.size.x, n.pos.y + n.size.y),
                bdr, 6.0f, 0, 1.5f);

    // Centered text
    ImVec2 ts = ImGui::CalcTextSize(n.label);
    ImVec2 tp = ImVec2(n.pos.x + (n.size.x - ts.x) * 0.5f,
                       n.pos.y + (n.size.y - ts.y) * 0.5f);
    dl->AddText(tp, text, n.label);

    if (n.toggle_ptr) {
        float r = std::max(5.0f, 6.0f * g_view_zoom);
        ImVec2 c(n.pos.x + n.size.x - 12.0f * g_view_zoom, n.pos.y + 10.0f * g_view_zoom);
        ImU32 glow = *n.toggle_ptr ? n.border : IM_COL32(120, 120, 130, 200);
        if (n.toggle_hovered) {
            dl->AddCircle(c, r + 2.5f, IM_COL32(255, 255, 255, 80), 20, 4.0f);
        }
        dl->AddCircleFilled(c, r, *n.toggle_ptr ? glow : IM_COL32(60, 60, 68, 240), 20);
        dl->AddCircle(c, r, IM_COL32(255, 255, 255, 210), 20, 1.5f);
    }
}

static void interact_node(NodeBox& n) {
    ImGuiIO& io = ImGui::GetIO();
    bool window_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
    ImVec2 node_max(n.pos.x + n.size.x, n.pos.y + n.size.y);
    n.hovered = window_hovered && ImGui::IsMouseHoveringRect(n.pos, node_max, false);
    n.active = (g_dragging_node_id == n.id);
    n.toggle_hovered = false;

    if (n.toggle_ptr) {
        ImVec2 toggle_pos(n.pos.x + n.size.x - 24.0f * g_view_zoom, n.pos.y + 2.0f * g_view_zoom);
        ImVec2 toggle_size(20.0f * g_view_zoom, 20.0f * g_view_zoom);
        ImVec2 toggle_max(toggle_pos.x + toggle_size.x, toggle_pos.y + toggle_size.y);
        n.toggle_hovered = window_hovered && ImGui::IsMouseHoveringRect(toggle_pos, toggle_max, false);
        if (rect_clicked(toggle_pos, toggle_max)) {
            *n.toggle_ptr = !*n.toggle_ptr;
            g_dragging_node_id.clear();
        }
    }

    if (!n.toggle_hovered && n.hovered && rect_clicked(n.pos, node_max)) {
        g_dragging_node_id = n.id;
        n.active = true;
    }

    if (n.active && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        g_dragging_node_id.clear();
        n.active = false;
    }

    if (n.active && !n.toggle_hovered && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        ImVec2 delta = io.MouseDelta;
        ImVec2& off = offset_ref(n.id);
        float inv_zoom = (g_view_zoom > 1e-4f) ? (1.0f / g_view_zoom) : 1.0f;
        off.x += delta.x * inv_zoom;
        off.y += delta.y * inv_zoom;
        n.pos.x += delta.x;
        n.pos.y += delta.y;
    }
    if ((n.hovered || n.toggle_hovered) && n.tooltip && n.tooltip[0] != '\0') {
        ImGui::SetTooltip("%s", n.tooltip);
    }
}

static void draw_arrow_head(ImDrawList* dl, ImVec2 from, ImVec2 to, ImU32 col) {
    float angle = atan2f(to.y - from.y, to.x - from.x);
    float asz = 7.0f;
    ImVec2 a1(to.x - asz * cosf(angle - 0.45f), to.y - asz * sinf(angle - 0.45f));
    ImVec2 a2(to.x - asz * cosf(angle + 0.45f), to.y - asz * sinf(angle + 0.45f));
    dl->AddTriangleFilled(to, a1, a2, col);
}

static float distance_to_segment(ImVec2 p, ImVec2 a, ImVec2 b) {
    ImVec2 ab(b.x - a.x, b.y - a.y);
    ImVec2 ap(p.x - a.x, p.y - a.y);
    float ab_len2 = ab.x * ab.x + ab.y * ab.y;
    if (ab_len2 <= 1e-6f) {
        float dx = p.x - a.x;
        float dy = p.y - a.y;
        return std::sqrt(dx * dx + dy * dy);
    }
    float t = (ap.x * ab.x + ap.y * ab.y) / ab_len2;
    t = std::clamp(t, 0.0f, 1.0f);
    ImVec2 q(a.x + ab.x * t, a.y + ab.y * t);
    float dx = p.x - q.x;
    float dy = p.y - q.y;
    return std::sqrt(dx * dx + dy * dy);
}

static void consider_link_hover(LinkHoverState& hover, const char* tooltip,
                                ImVec2 a, ImVec2 b, ImVec2 mouse_pos) {
    if (!tooltip || tooltip[0] == '\0') return;
    float d = distance_to_segment(mouse_pos, a, b);
    if (d < hover.best_dist) {
        hover.best_dist = d;
        hover.tooltip = tooltip;
    }
}

static void draw_arrow(ImDrawList* dl, ImVec2 from, ImVec2 to, ImU32 col,
                       float lane_offset = 0.0f, float thickness = 1.5f,
                       LinkHoverState* hover = nullptr, const char* tooltip = nullptr,
                       ImVec2 mouse_pos = ImVec2(0.0f, 0.0f), const char* feature_key = nullptr) {
    float left = std::min(from.x, to.x);
    float right = std::max(from.x, to.x);
    float mid_x = (from.x + to.x) * 0.5f + lane_offset;
    mid_x = std::clamp(mid_x, left + 24.0f, right - 24.0f);

    float bend_pad = 22.0f * g_view_zoom;
    float exit_x = (to.x >= from.x) ? std::min(mid_x, from.x + bend_pad) : std::max(mid_x, from.x - bend_pad);
    float entry_x = (to.x >= from.x) ? std::max(mid_x, to.x - bend_pad) : std::min(mid_x, to.x + bend_pad);
    float detour_y = from.y;
    if (std::fabs(to.y - from.y) > 18.0f * g_view_zoom) {
        detour_y = (from.y + to.y) * 0.5f;
    }
    ImVec2 pts[6] = {
        from,
        ImVec2(exit_x, from.y),
        ImVec2(exit_x, detour_y),
        ImVec2(entry_x, detour_y),
        ImVec2(entry_x, to.y),
        to
    };

    float best_here = std::numeric_limits<float>::max();
    if (hover) {
        for (int i = 0; i < 5; ++i) {
            float d = distance_to_segment(mouse_pos, pts[i], pts[i + 1]);
            best_here = std::min(best_here, d);
        }
        if (tooltip && tooltip[0] != '\0' && best_here < hover->best_dist) {
            hover->best_dist = best_here;
            hover->tooltip = tooltip;
            hover->feature_key = feature_key;
            hover->color = col;
            hover->thickness = thickness;
            hover->pt_count = 6;
            for (int i = 0; i < 6; ++i) hover->pts[i] = pts[i];
        }
    }

    for (int i = 0; i < 5; ++i) {
        dl->AddLine(pts[i], pts[i + 1], col, thickness);
    }
    draw_arrow_head(dl, pts[4], pts[5], col);
}

// Straight vertical arrow (for sequential passes)
static void draw_varrow(ImDrawList* dl, ImVec2 from, ImVec2 to, ImU32 col, float thickness = 1.5f,
                        LinkHoverState* hover = nullptr, const char* tooltip = nullptr,
                        ImVec2 mouse_pos = ImVec2(0.0f, 0.0f), const char* feature_key = nullptr) {
    if (hover) {
        float d = distance_to_segment(mouse_pos, from, to);
        if (tooltip && tooltip[0] != '\0' && d < hover->best_dist) {
            hover->best_dist = d;
            hover->tooltip = tooltip;
            hover->feature_key = feature_key;
            hover->color = col;
            hover->thickness = thickness;
            hover->pt_count = 2;
            hover->pts[0] = from;
            hover->pts[1] = to;
        }
    }
    dl->AddLine(from, to, col, thickness);
    draw_arrow_head(dl, from, to, col);
}

static void draw_hovered_link(ImDrawList* dl, const LinkHoverState& hover) {
    if (!hover.tooltip || hover.pt_count < 2) return;
    ImU32 glow = IM_COL32(
        std::min(255, static_cast<int>(ImGui::ColorConvertU32ToFloat4(hover.color).x * 255.0f) + 20),
        std::min(255, static_cast<int>(ImGui::ColorConvertU32ToFloat4(hover.color).y * 255.0f) + 20),
        std::min(255, static_cast<int>(ImGui::ColorConvertU32ToFloat4(hover.color).z * 255.0f) + 20),
        72);
    for (int i = 0; i < hover.pt_count - 1; ++i) {
        dl->AddLine(hover.pts[i], hover.pts[i + 1], glow, hover.thickness + 8.0f);
        dl->AddLine(hover.pts[i], hover.pts[i + 1], IM_COL32(255, 255, 255, 58), hover.thickness + 4.0f);
        dl->AddLine(hover.pts[i], hover.pts[i + 1], IM_COL32(255, 255, 255, 235), hover.thickness + 1.7f);
    }
    draw_arrow_head(dl, hover.pts[hover.pt_count - 2], hover.pts[hover.pt_count - 1], IM_COL32(255, 255, 255, 235));
}

// Section header
static void draw_column_header(ImDrawList* dl, ImVec2 pos, float width, const char* label,
                                ImU32 col) {
    ImVec2 ts = ImGui::CalcTextSize(label);
    float x = pos.x + (width - ts.x) * 0.5f;
    dl->AddText(ImVec2(x, pos.y), col, label);
    dl->AddLine(ImVec2(pos.x, pos.y + HEADER_H - 4),
                ImVec2(pos.x + width, pos.y + HEADER_H - 4), col, 1.0f);
}

// ── Toggle state ────────────────────────────────────────────────────────────
struct PipelineToggles {
    bool sph_enabled     = true;
    bool sph_codim       = true;
    bool sph_surface_tension = true;
    bool mpm_enabled     = true;
    bool mpm_thermal     = true;
    bool mpm_fracture    = true;
    bool mpm_combustion  = true;
    bool mpm_phase       = true;
    bool mpm_aniso       = true;
    bool mpm_shell_core  = true;
    bool mpm_pressure    = true;
    bool mpm_filament    = true;
    bool mpm_bio         = true;
    bool mpm_memory      = true;
    bool mpm_field       = true;
    bool mpm_gravity_scale = false;
    bool euler_enabled   = true;
    bool euler_vapor     = true;
    bool euler_thermal_diffuse = true;
    bool euler_latent    = true;
    bool sdf_enabled     = true;
    bool sdf_heat        = true;
    bool coupling_sph_mpm = true;
    bool coupling_euler_inject = true;
    bool coupling_euler_drag   = true;
    bool render_bloom    = true;
    bool render_metaball = false;
};
static PipelineToggles g_toggles;

// ── Main draw function ──────────────────────────────────────────────────────
void draw_pipeline_viewer(bool* open)
{
    if (!g_view_session_active) {
        g_node_offsets.clear();
        g_view_pan = ImVec2(24.0f, 18.0f);
        g_view_zoom = 1.0f;
        g_view_session_active = true;
    }

    ImGuiIO& root_io = ImGui::GetIO();
    ImGui::SetNextWindowSizeConstraints(ImVec2(860.0f, 560.0f),
                                        ImVec2(root_io.DisplaySize.x * 0.96f, root_io.DisplaySize.y * 0.92f));
    ImGui::SetNextWindowSize(ImVec2(1100, 700), ImGuiCond_Appearing);
    if (!ImGui::Begin("Physics Pipeline", open)) {
        ImGui::End();
        if (open && !*open) {
            reset_pipeline_viewer_session();
        }
        return;
    }

    // ── Left sidebar: feature toggles ───────────────────────────────────
    ImGui::BeginChild("##toggles", ImVec2(250, 0), ImGuiChildFlags_Borders);
    ImGui::TextColored(ImVec4(1, 0.9f, 0.5f, 1), "Feature Toggles");
    ImGui::TextDisabled("Drag nodes in the diagram to untangle crossings.");
    if (ImGui::Button("Reset Node Positions")) {
        g_node_offsets.clear();
    }
    ImGui::Separator();
    const char* sidebar_hover = g_sidebar_hover_feature;
    auto draw_toggle = [&](const char* label, bool* value, const char* feature_key) {
        bool hi = feature_matches(sidebar_hover, feature_key);
        if (hi) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.96f, 0.76f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.28f, 0.22f, 0.16f, 0.95f));
            ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.36f, 0.28f, 0.18f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_CheckMark, ImVec4(1.0f, 0.95f, 0.82f, 1.0f));
        }
        ImGui::Checkbox(label, value);
        if (hi) {
            ImGui::PopStyleColor(4);
        }
    };

    ImGui::TextDisabled("SPH Fluid");
    draw_toggle("SPH Solver##t", &g_toggles.sph_enabled, FEAT_SPH);
    draw_toggle("Codim 1D/2D##t", &g_toggles.sph_codim, FEAT_SPH_CODIM);
    draw_toggle("Surface Tension##t", &g_toggles.sph_surface_tension, FEAT_SPH_SURFACE);
    ImGui::Spacing();

    ImGui::TextDisabled("MPM Solids");
    draw_toggle("MPM Solver##t", &g_toggles.mpm_enabled, FEAT_MPM);
    draw_toggle("Thermal System##t", &g_toggles.mpm_thermal, FEAT_MPM_THERMAL);
    draw_toggle("Fracture##t", &g_toggles.mpm_fracture, FEAT_MPM_FRACTURE);
    draw_toggle("Combustion##t", &g_toggles.mpm_combustion, FEAT_MPM_COMBUSTION);
    draw_toggle("Phase-field##t", &g_toggles.mpm_phase, FEAT_MPM_PHASE);
    draw_toggle("Anisotropic##t", &g_toggles.mpm_aniso, FEAT_MPM);
    draw_toggle("Gravity Scale##t", &g_toggles.mpm_gravity_scale, FEAT_MPM_GSCALE);
    ImGui::Spacing();

    ImGui::TextDisabled("Newer Techniques");
    draw_toggle("Shell / Core##t", &g_toggles.mpm_shell_core, FEAT_MPM_SHELL);
    draw_toggle("Pressure / Puff##t", &g_toggles.mpm_pressure, FEAT_MPM_PRESSURE);
    draw_toggle("Filament Pull##t", &g_toggles.mpm_filament, FEAT_MPM_FILAMENT);
    draw_toggle("Cooking / Bio##t", &g_toggles.mpm_bio, FEAT_MPM_BIO);
    draw_toggle("Memory / Recovery##t", &g_toggles.mpm_memory, FEAT_MPM_MEMORY);
    draw_toggle("Magnet / Field##t", &g_toggles.mpm_field, FEAT_MPM_FIELD);
    ImGui::Spacing();

    ImGui::TextDisabled("Eulerian Air");
    draw_toggle("Euler Solver##t", &g_toggles.euler_enabled, FEAT_EULER);
    draw_toggle("Vapor / Steam##t", &g_toggles.euler_vapor, FEAT_EULER_VAPOR);
    draw_toggle("Thermal Diffusion##t", &g_toggles.euler_thermal_diffuse, FEAT_EULER_THERMAL);
    draw_toggle("Latent Cooling##t", &g_toggles.euler_latent, FEAT_EULER_LATENT);
    ImGui::Spacing();

    ImGui::TextDisabled("Environment");
    draw_toggle("SDF Collision##t", &g_toggles.sdf_enabled, FEAT_SDF);
    draw_toggle("SDF Heat##t", &g_toggles.sdf_heat, FEAT_SDF_HEAT);
    ImGui::Spacing();

    ImGui::TextDisabled("Coupling");
    draw_toggle("SPH <-> MPM##t", &g_toggles.coupling_sph_mpm, FEAT_COUPLE_SPH_MPM);
    draw_toggle("Particle -> Air##t", &g_toggles.coupling_euler_inject, FEAT_COUPLE_INJECT);
    draw_toggle("Air -> Particle##t", &g_toggles.coupling_euler_drag, FEAT_COUPLE_DRAG);
    ImGui::Spacing();

    ImGui::TextDisabled("Rendering");
    draw_toggle("Bloom##t", &g_toggles.render_bloom, FEAT_RENDER_BLOOM);
    draw_toggle("Metaball##t", &g_toggles.render_metaball, FEAT_RENDER_METABALL);

    ImGui::EndChild();

    // ── Right pane: diagram ─────────────────────────────────────────────
    ImGui::SameLine();
    ImGui::BeginChild("##diagram", ImVec2(0, 0), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::TextDisabled("MMB pan | Wheel zoom | node moves reset when the viewer is reopened.");
    ImGui::Separator();

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 canvas_sz(std::max(avail.x, CANVAS_W), std::max(avail.y, CANVAS_H));
    ImGui::InvisibleButton("pipeline_canvas", canvas_sz);
    ImGui::SetItemAllowOverlap();
    ImVec2 canvas_pos = ImGui::GetItemRectMin();
    g_canvas_origin = canvas_pos;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImGuiIO& io = ImGui::GetIO();
    bool diagram_hovered = ImGui::IsItemHovered() ||
                           ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);

    if (diagram_hovered && ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
        g_view_pan.x += io.MouseDelta.x;
        g_view_pan.y += io.MouseDelta.y;
    }
    if (diagram_hovered && std::fabs(io.MouseWheel) > 1e-4f) {
        float prev_zoom = g_view_zoom;
        g_view_zoom = std::clamp(g_view_zoom * (1.0f + io.MouseWheel * 0.12f), 0.55f, 2.2f);
        if (std::fabs(g_view_zoom - prev_zoom) > 1e-4f) {
            ImVec2 mouse_local((io.MousePos.x - g_canvas_origin.x - g_view_pan.x) / prev_zoom,
                               (io.MousePos.y - g_canvas_origin.y - g_view_pan.y) / prev_zoom);
            g_view_pan.x = io.MousePos.x - g_canvas_origin.x - mouse_local.x * g_view_zoom;
            g_view_pan.y = io.MousePos.y - g_canvas_origin.y - mouse_local.y * g_view_zoom;
        }
    }
    LinkHoverState link_hover;
    ImVec2 mouse_pos = io.MousePos;

    // Background
    dl->AddRectFilled(canvas_pos,
                      ImVec2(canvas_pos.x + canvas_sz.x, canvas_pos.y + canvas_sz.y),
                      COL_BG);

    // Subtle grid
    for (float x = 0; x < canvas_sz.x; x += 40.0f)
        dl->AddLine(ImVec2(canvas_pos.x + x, canvas_pos.y),
                    ImVec2(canvas_pos.x + x, canvas_pos.y + canvas_sz.y), COL_GRID);
    for (float y = 0; y < canvas_sz.y; y += 40.0f)
        dl->AddLine(ImVec2(canvas_pos.x, canvas_pos.y + y),
                    ImVec2(canvas_pos.x + canvas_sz.x, canvas_pos.y + y), COL_GRID);

    // ── Column origins ──────────────────────────────────────────────────
    float x0 = 34.0f;
    float y0 = 16.0f;

    float col_sph   = x0;
    float col_mpm   = x0 + COL_GAP;
    float col_mpm_fx = x0 + 560.0f;
    float col_euler  = x0 + 870.0f;
    float col_air_fx = x0 + 1140.0f;
    float col_sdf    = x0 + 1470.0f;

    // ── Column headers ──────────────────────────────────────────────────
    draw_column_header(dl, view_pt(ImVec2(col_sph, y0)), NODE_W * g_view_zoom, "SPH FLUID", COL_SPH);
    draw_column_header(dl, view_pt(ImVec2(col_mpm, y0)), NODE_W * g_view_zoom, "MPM CORE", COL_MPM);
    draw_column_header(dl, view_pt(ImVec2(col_mpm_fx, y0)), SUB_W * g_view_zoom, "MPM TECHNIQUES", COL_THERMAL);
    draw_column_header(dl, view_pt(ImVec2(col_euler, y0)), NODE_W * g_view_zoom, "EULERIAN AIR", COL_EULER);
    draw_column_header(dl, view_pt(ImVec2(col_air_fx, y0)), SUB_W * g_view_zoom, "AIR / HEAT", COL_VAPOR);
    draw_column_header(dl, view_pt(ImVec2(col_sdf, y0)), NODE_W * g_view_zoom, "SDF / RENDER", COL_SDF);

    float row_start = y0 + HEADER_H + 8.0f;
    ImVec2 ns(NODE_W, NODE_H);
    ImVec2 sns(SUB_W, SUB_H);

    // ════════════════════════════════════════════════════════════════════
    //  SPH PIPELINE
    // ════════════════════════════════════════════════════════════════════
    NodeBox sph_hash  = make_node("sph_hash", ImVec2(col_sph, row_start + ROW_STEP * 0), ns,
                                  "Spatial Hash", "Neighborhood lookup for SPH density and force passes.",
                                  COL_SPH_FILL, COL_SPH, g_toggles.sph_enabled);
    NodeBox sph_codim = make_node("sph_codim", ImVec2(col_sph, row_start + ROW_STEP * 1), ns,
                                  "Codim Detect", "Adaptive 1D/2D kernels for threads, ropes, and sheets.",
                                  COL_SPH_FILL, COL_SPH, g_toggles.sph_enabled && g_toggles.sph_codim);
    NodeBox sph_dens  = make_node("sph_dens", ImVec2(col_sph, row_start + ROW_STEP * 2), ns,
                                  "Density", "WCSPH density solve with codim-aware support when enabled.",
                                  COL_SPH_FILL, COL_SPH, g_toggles.sph_enabled);
    NodeBox sph_force = make_node("sph_force", ImVec2(col_sph, row_start + ROW_STEP * 3), ns,
                                  "Force + Integrate", "Pressure, viscosity, drag tools, and particle integration.",
                                  COL_SPH_FILL, COL_SPH, g_toggles.sph_enabled);
    NodeBox sph_sdf   = make_node("sph_sdf", ImVec2(col_sph, row_start + ROW_STEP * 4), ns,
                                  "SDF Collision", "Ray-marched collision against the authored distance field.",
                                  COL_SPH_FILL, COL_SPH, g_toggles.sph_enabled && g_toggles.sdf_enabled);
    NodeBox sph_stens = make_node("sph_stens", ImVec2(col_sph, row_start + ROW_STEP * 5), ns,
                                  "Surface Tension", "CSF tension for droplets, films, and codim cleanup.",
                                  COL_SPH_FILL, COL_SPH, g_toggles.sph_enabled && g_toggles.sph_surface_tension);

    sph_hash.feature_key = FEAT_SPH;
    sph_codim.feature_key = FEAT_SPH_CODIM; sph_codim.toggle_ptr = &g_toggles.sph_codim;
    sph_dens.feature_key = FEAT_SPH;
    sph_force.feature_key = FEAT_SPH;
    sph_sdf.feature_key = FEAT_SDF;
    sph_stens.feature_key = FEAT_SPH_SURFACE; sph_stens.toggle_ptr = &g_toggles.sph_surface_tension;

    interact_node(sph_hash);
    interact_node(sph_codim);
    interact_node(sph_dens);
    interact_node(sph_force);
    interact_node(sph_sdf);
    interact_node(sph_stens);

    draw_node(dl, sph_hash, g_sidebar_hover_feature);
    draw_node(dl, sph_codim, g_sidebar_hover_feature);
    draw_node(dl, sph_dens, g_sidebar_hover_feature);
    draw_node(dl, sph_force, g_sidebar_hover_feature);
    draw_node(dl, sph_sdf, g_sidebar_hover_feature);
    draw_node(dl, sph_stens, g_sidebar_hover_feature);

    // SPH vertical arrows
    draw_varrow(dl, node_bottom(sph_hash),  node_top(sph_codim), COL_ARROW, 1.5f, &link_hover,
                "SPH neighbor data feeds codimensional detection.", mouse_pos, FEAT_SPH_CODIM);
    draw_varrow(dl, node_bottom(sph_codim), node_top(sph_dens),  COL_ARROW, 1.5f, &link_hover,
                "Codim classification influences the SPH density pass.", mouse_pos, FEAT_SPH_CODIM);
    draw_varrow(dl, node_bottom(sph_dens),  node_top(sph_force), COL_ARROW, 1.5f, &link_hover,
                "Density and pressure feed SPH force accumulation and integration.", mouse_pos, FEAT_SPH);
    draw_varrow(dl, node_bottom(sph_force), node_top(sph_sdf),   COL_ARROW, 1.5f, &link_hover,
                "Updated SPH positions are cleaned up against the SDF scene.", mouse_pos, FEAT_SDF);
    draw_varrow(dl, node_bottom(sph_sdf),   node_top(sph_stens), COL_ARROW, 1.5f, &link_hover,
                "Surface tension runs after collision cleanup for cleaner droplets and threads.", mouse_pos, FEAT_SPH_SURFACE);

    // ════════════════════════════════════════════════════════════════════
    //  MPM PIPELINE
    // ════════════════════════════════════════════════════════════════════
    NodeBox mpm_clear = make_node("mpm_clear", ImVec2(col_mpm, row_start + ROW_STEP * 0), ns,
                                  "Grid Clear", "Reset the MLS-MPM grid before particle scatter.",
                                  COL_MPM_FILL, COL_MPM, g_toggles.mpm_enabled);
    NodeBox mpm_p2g = make_node("mpm_p2g", ImVec2(col_mpm, row_start + ROW_STEP * 1), ns,
                                "P2G Scatter", "Scatter mass, momentum, APIC terms, and constitutive stress.",
                                COL_MPM_FILL, COL_MPM, g_toggles.mpm_enabled);
    NodeBox mpm_gridop = make_node("mpm_gridop", ImVec2(col_mpm, row_start + ROW_STEP * 2), ns,
                                   "Grid Op (BC)", "Gravity, walls, and tool forces on the grid.",
                                   COL_MPM_FILL, COL_MPM, g_toggles.mpm_enabled);
    NodeBox mpm_g2p = make_node("mpm_g2p", ImVec2(col_mpm, row_start + ROW_STEP * 3), ns,
                                "G2P Gather", "Sample updated grid velocity back to particles and advect them.",
                                COL_MPM_FILL, COL_MPM, g_toggles.mpm_enabled);
    NodeBox mpm_constitutive = make_node("mpm_constitutive", ImVec2(col_mpm_fx, row_start + SUB_ROW * 0), sns,
                                         "34 Materials", "Constitutive switchboard for fluids, woods, metals, glass, doughs, and more.",
                                         COL_MPM_FILL, COL_MPM, g_toggles.mpm_enabled);
    NodeBox mpm_thermal = make_node("mpm_thermal", ImVec2(col_mpm_fx, row_start + SUB_ROW * 1), sns,
                                    "Thermal + Burn", "Temperature-dependent stiffness, ignition, and heat release.",
                                    COL_MPM_FILL, COL_THERMAL, g_toggles.mpm_enabled && (g_toggles.mpm_thermal || g_toggles.mpm_combustion));
    NodeBox mpm_frac = make_node("mpm_frac", ImVec2(col_mpm_fx, row_start + SUB_ROW * 2), sns,
                                 "Fracture + Phase", "Damage, crack growth, melt bands, and phase softening.",
                                 COL_MPM_FILL, COL_MPM, g_toggles.mpm_enabled && (g_toggles.mpm_fracture || g_toggles.mpm_phase));
    NodeBox mpm_shell = make_node("mpm_shell", ImVec2(col_mpm_fx, row_start + SUB_ROW * 3), sns,
                                  "Shell / Core", "Surface-first glaze, crust, and shell-core mismatch behavior.",
                                  COL_MPM_FILL, COL_THERMAL, g_toggles.mpm_enabled && g_toggles.mpm_shell_core);
    NodeBox mpm_pressure = make_node("mpm_pressure", ImVec2(col_mpm_fx, row_start + SUB_ROW * 4), sns,
                                     "Pressure / Puff", "Gas growth, trapped expansion, puffing, and pressure rupture.",
                                     COL_MPM_FILL, COL_VAPOR, g_toggles.mpm_enabled && g_toggles.mpm_pressure);
    NodeBox mpm_filament = make_node("mpm_filament", ImVec2(col_mpm_fx, row_start + SUB_ROW * 5), sns,
                                     "Filament Pull", "Hot glass / cheese strand memory and stretch-aligned thread formation.",
                                     COL_MPM_FILL, COL_FIELD, g_toggles.mpm_enabled && g_toggles.mpm_filament);
    NodeBox mpm_bio = make_node("mpm_bio", ImVec2(col_mpm_fx, row_start + SUB_ROW * 6), sns,
                                "Cooking / Bio", "Maillard browning, mushroom collapse, and spore-like venting.",
                                COL_MPM_FILL, COL_BIO, g_toggles.mpm_enabled && g_toggles.mpm_bio);
    NodeBox mpm_memory = make_node("mpm_memory", ImVec2(col_mpm_fx, row_start + SUB_ROW * 7), sns,
                                   "Memory / Recovery", "Thermal memory wax and self-healing recovery paths.",
                                   COL_MPM_FILL, COL_MEMORY, g_toggles.mpm_enabled && g_toggles.mpm_memory);
    NodeBox mpm_field = make_node("mpm_field", ImVec2(col_mpm_fx, row_start + SUB_ROW * 8), sns,
                                  "Magnet / Ferro", "Field-driven forces for ferrofluid and related materials.",
                                  COL_MPM_FILL, COL_FIELD, g_toggles.mpm_enabled && g_toggles.mpm_field);
    NodeBox mpm_gscale = make_node("mpm_gscale", ImVec2(col_mpm_fx, row_start + SUB_ROW * 9), sns,
                                   "Gravity Scale", "Per-material physical scale for tiny and soft materials.",
                                   COL_MPM_FILL, COL_MPM, g_toggles.mpm_enabled && g_toggles.mpm_gravity_scale);
    NodeBox mpm_sdf = make_node("mpm_sdf", ImVec2(col_mpm, row_start + ROW_STEP * 4), ns,
                                "SDF Boundary", "Particle-level SDF collision cleanup after advection.",
                                COL_MPM_FILL, COL_MPM, g_toggles.mpm_enabled && g_toggles.sdf_enabled);

    mpm_clear.feature_key = FEAT_MPM;
    mpm_p2g.feature_key = FEAT_MPM;
    mpm_gridop.feature_key = FEAT_MPM;
    mpm_g2p.feature_key = FEAT_MPM;
    mpm_constitutive.feature_key = FEAT_MPM;
    mpm_thermal.feature_key = FEAT_MPM_THERMAL;
    mpm_frac.feature_key = FEAT_MPM_FRACTURE;
    mpm_shell.feature_key = FEAT_MPM_SHELL; mpm_shell.toggle_ptr = &g_toggles.mpm_shell_core;
    mpm_pressure.feature_key = FEAT_MPM_PRESSURE; mpm_pressure.toggle_ptr = &g_toggles.mpm_pressure;
    mpm_filament.feature_key = FEAT_MPM_FILAMENT; mpm_filament.toggle_ptr = &g_toggles.mpm_filament;
    mpm_bio.feature_key = FEAT_MPM_BIO; mpm_bio.toggle_ptr = &g_toggles.mpm_bio;
    mpm_memory.feature_key = FEAT_MPM_MEMORY; mpm_memory.toggle_ptr = &g_toggles.mpm_memory;
    mpm_field.feature_key = FEAT_MPM_FIELD; mpm_field.toggle_ptr = &g_toggles.mpm_field;
    mpm_gscale.feature_key = FEAT_MPM_GSCALE; mpm_gscale.toggle_ptr = &g_toggles.mpm_gravity_scale;
    mpm_sdf.feature_key = FEAT_SDF;

    interact_node(mpm_clear);
    interact_node(mpm_p2g);
    interact_node(mpm_gridop);
    interact_node(mpm_g2p);
    interact_node(mpm_constitutive);
    interact_node(mpm_thermal);
    interact_node(mpm_frac);
    interact_node(mpm_shell);
    interact_node(mpm_pressure);
    interact_node(mpm_filament);
    interact_node(mpm_bio);
    interact_node(mpm_memory);
    interact_node(mpm_field);
    interact_node(mpm_gscale);
    interact_node(mpm_sdf);

    draw_node(dl, mpm_clear, g_sidebar_hover_feature);
    draw_node(dl, mpm_p2g, g_sidebar_hover_feature);
    draw_node(dl, mpm_gridop, g_sidebar_hover_feature);
    draw_node(dl, mpm_g2p, g_sidebar_hover_feature);
    draw_node(dl, mpm_constitutive, g_sidebar_hover_feature);
    draw_node(dl, mpm_thermal, g_sidebar_hover_feature);
    draw_node(dl, mpm_frac, g_sidebar_hover_feature);
    draw_node(dl, mpm_shell, g_sidebar_hover_feature);
    draw_node(dl, mpm_pressure, g_sidebar_hover_feature);
    draw_node(dl, mpm_filament, g_sidebar_hover_feature);
    draw_node(dl, mpm_bio, g_sidebar_hover_feature);
    draw_node(dl, mpm_memory, g_sidebar_hover_feature);
    draw_node(dl, mpm_field, g_sidebar_hover_feature);
    draw_node(dl, mpm_gscale, g_sidebar_hover_feature);
    draw_node(dl, mpm_sdf, g_sidebar_hover_feature);

    // MPM vertical arrows
    draw_varrow(dl, node_bottom(mpm_clear),  node_top(mpm_p2g),    COL_ARROW, 1.5f, &link_hover,
                "A cleared grid is ready for particle-to-grid scatter.", mouse_pos, FEAT_MPM);
    draw_varrow(dl, node_bottom(mpm_p2g),    node_top(mpm_gridop), COL_ARROW, 1.5f, &link_hover,
                "Scattered mass and momentum become grid velocities and forces.", mouse_pos, FEAT_MPM);
    draw_varrow(dl, node_bottom(mpm_gridop), node_top(mpm_g2p),    COL_ARROW, 1.5f, &link_hover,
                "Grid operations feed the gather step back onto particles.", mouse_pos, FEAT_MPM);
    draw_varrow(dl, node_bottom(mpm_g2p),    node_top(mpm_sdf),    COL_ARROW, 1.5f, &link_hover,
                "Advected particles are pushed back out of rigid SDF boundaries.", mouse_pos, FEAT_SDF);

    // Sub-node connections (spread across a dedicated technique lane)
    draw_arrow(dl, node_right(mpm_p2g, 0.16f), node_left(mpm_constitutive), COL_MPM, -64.0f, 1.0f, &link_hover,
               "Constitutive branch: each particle chooses its material law during P2G.", mouse_pos);
    draw_arrow(dl, node_right(mpm_p2g, 0.28f), node_left(mpm_thermal), COL_THERMAL, -40.0f, 1.0f, &link_hover,
               "Thermal branch: temperature changes stiffness, ignition, and burn energy.", mouse_pos);
    draw_arrow(dl, node_right(mpm_p2g, 0.40f), node_left(mpm_frac), COL_MPM, -16.0f, 1.0f, &link_hover,
               "Fracture branch: damage and melt state alter the scattered stress.", mouse_pos);
    draw_arrow(dl, node_right(mpm_p2g, 0.52f), node_left(mpm_shell), COL_THERMAL, 8.0f, 1.0f, &link_hover,
               "Shell/core branch: surface exposure changes the outer layer faster than the interior.", mouse_pos);
    draw_arrow(dl, node_right(mpm_p2g, 0.64f), node_left(mpm_pressure), COL_VAPOR, 32.0f, 1.0f, &link_hover,
               "Pressure branch: trapped gas expands the matrix and can rupture it.", mouse_pos);
    draw_arrow(dl, node_right(mpm_p2g, 0.76f), node_left(mpm_filament), COL_FIELD, 56.0f, 1.0f, &link_hover,
               "Filament branch: extensional memory helps hot pulled materials form threads.", mouse_pos);
    draw_arrow(dl, node_right(mpm_p2g, 0.88f), node_left(mpm_bio), COL_BIO, 80.0f, 1.0f, &link_hover,
               "Bio/cooking branch: browning, wilting, and spore-like venting are evaluated here.", mouse_pos);
    draw_arrow(dl, node_right(mpm_g2p, 0.35f), node_left(mpm_memory), COL_MEMORY, -8.0f, 1.0f, &link_hover,
               "Recovery branch: cooldown can restore shape or stiffness after deformation.", mouse_pos);
    draw_arrow(dl, node_right(mpm_g2p, 0.58f), node_left(mpm_field), COL_FIELD, 28.0f, 1.0f, &link_hover,
               "Field branch: external brushes like the magnet modify particle motion.", mouse_pos);
    draw_arrow(dl, node_right(mpm_g2p, 0.82f), node_left(mpm_gscale), COL_MPM, 64.0f, 1.0f, &link_hover,
               "Per-material gravity scaling is applied after grid gather.", mouse_pos);

    // ════════════════════════════════════════════════════════════════════
    //  EULERIAN PIPELINE
    // ════════════════════════════════════════════════════════════════════
    NodeBox eu_inject = make_node("eu_inject", ImVec2(col_euler, row_start + ROW_STEP * 0), ns,
                                  "Inject (P->Air)", "Particles feed smoke, heat, and vapor into the Euler grid.",
                                  COL_EULER_FILL, COL_EULER, g_toggles.euler_enabled && g_toggles.coupling_euler_inject);
    NodeBox eu_forces = make_node("eu_forces", ImVec2(col_euler, row_start + ROW_STEP * 1), ns,
                                  "Forces + Buoy", "Buoyancy, vapor lift, and external forces.",
                                  COL_EULER_FILL, COL_EULER, g_toggles.euler_enabled);
    NodeBox eu_diffuse = make_node("eu_diffuse", ImVec2(col_euler, row_start + ROW_STEP * 2), ns,
                                   "Thermal Diffuse", "Air conduction, SDF heat exchange, and ambient cooling.",
                                   COL_EULER_FILL, COL_EULER, g_toggles.euler_enabled && g_toggles.euler_thermal_diffuse);
    NodeBox eu_div = make_node("eu_div", ImVec2(col_euler, row_start + ROW_STEP * 3), ns,
                               "Divergence", "Assemble divergence, including vapor expansion sources.",
                               COL_EULER_FILL, COL_EULER, g_toggles.euler_enabled);
    NodeBox eu_press = make_node("eu_press", ImVec2(col_euler, row_start + ROW_STEP * 4), ns,
                                 "Pressure (RB-GS)", "Pressure solve over the MAC grid.",
                                 COL_EULER_FILL, COL_EULER, g_toggles.euler_enabled);
    NodeBox eu_proj = make_node("eu_proj", ImVec2(col_euler, row_start + ROW_STEP * 5), ns,
                                "Project", "Subtract the pressure gradient to enforce incompressibility.",
                                COL_EULER_FILL, COL_EULER, g_toggles.euler_enabled);
    NodeBox eu_advect = make_node("eu_advect", ImVec2(col_euler, row_start + ROW_STEP * 6), ns,
                                  "Advect (SL)", "Semi-Lagrangian advection for air velocity, smoke, vapor, and heat.",
                                  COL_EULER_FILL, COL_EULER, g_toggles.euler_enabled);
    NodeBox eu_bc = make_node("eu_bc", ImVec2(col_euler, row_start + ROW_STEP * 7), ns,
                              "Enforce BC", "Clamp walls and keep SDF solids sealed in the grid.",
                              COL_EULER_FILL, COL_EULER, g_toggles.euler_enabled);
    NodeBox eu_drag = make_node("eu_drag", ImVec2(col_euler, row_start + ROW_STEP * 8), ns,
                                "Drag (Air->P)", "Feed air velocity, vapor carry, and loft back into particles.",
                                COL_EULER_FILL, COL_EULER, g_toggles.euler_enabled && g_toggles.coupling_euler_drag);

    NodeBox eu_vapor = make_node("eu_vapor", ImVec2(col_air_fx, row_start + SUB_ROW * 0.5f), sns,
                                 "Vapor / Steam", "Material-aware vapor emission, pressure source, and buoyancy.",
                                 COL_EULER_FILL, COL_VAPOR, g_toggles.euler_enabled && g_toggles.euler_vapor);
    NodeBox eu_latent = make_node("eu_latent", ImVec2(col_air_fx, row_start + SUB_ROW * 2.0f), sns,
                                  "Latent Cooling", "Trade direct heating for vapor generation so expansion costs heat.",
                                  COL_EULER_FILL, COL_THERMAL, g_toggles.euler_enabled && g_toggles.euler_latent);
    NodeBox eu_sdf_heat = make_node("eu_sdf_heat", ImVec2(col_air_fx, row_start + SUB_ROW * 3.5f), sns,
                                    "SDF Heat Sink", "Solid conduction, heat capacity, and dissipation into metal walls.",
                                    COL_EULER_FILL, COL_SDF, g_toggles.sdf_enabled && g_toggles.sdf_heat);
    NodeBox eu_loft = make_node("eu_loft", ImVec2(col_air_fx, row_start + SUB_ROW * 5.0f), sns,
                                "Air Carry / Loft", "Per-material carry response so smoke lifts some materials more than others.",
                                COL_EULER_FILL, COL_FIELD, g_toggles.euler_enabled && g_toggles.coupling_euler_drag);

    eu_inject.feature_key = FEAT_COUPLE_INJECT; eu_inject.toggle_ptr = &g_toggles.coupling_euler_inject;
    eu_forces.feature_key = FEAT_EULER;
    eu_diffuse.feature_key = FEAT_EULER_THERMAL; eu_diffuse.toggle_ptr = &g_toggles.euler_thermal_diffuse;
    eu_div.feature_key = FEAT_EULER;
    eu_press.feature_key = FEAT_EULER;
    eu_proj.feature_key = FEAT_EULER;
    eu_advect.feature_key = FEAT_EULER;
    eu_bc.feature_key = FEAT_EULER;
    eu_drag.feature_key = FEAT_COUPLE_DRAG; eu_drag.toggle_ptr = &g_toggles.coupling_euler_drag;
    eu_vapor.feature_key = FEAT_EULER_VAPOR; eu_vapor.toggle_ptr = &g_toggles.euler_vapor;
    eu_latent.feature_key = FEAT_EULER_LATENT; eu_latent.toggle_ptr = &g_toggles.euler_latent;
    eu_sdf_heat.feature_key = FEAT_SDF_HEAT; eu_sdf_heat.toggle_ptr = &g_toggles.sdf_heat;
    eu_loft.feature_key = FEAT_COUPLE_DRAG;

    interact_node(eu_inject);
    interact_node(eu_forces);
    interact_node(eu_diffuse);
    interact_node(eu_div);
    interact_node(eu_press);
    interact_node(eu_proj);
    interact_node(eu_advect);
    interact_node(eu_bc);
    interact_node(eu_drag);
    interact_node(eu_vapor);
    interact_node(eu_latent);
    interact_node(eu_sdf_heat);
    interact_node(eu_loft);

    draw_node(dl, eu_inject, g_sidebar_hover_feature);
    draw_node(dl, eu_forces, g_sidebar_hover_feature);
    draw_node(dl, eu_diffuse, g_sidebar_hover_feature);
    draw_node(dl, eu_div, g_sidebar_hover_feature);
    draw_node(dl, eu_press, g_sidebar_hover_feature);
    draw_node(dl, eu_proj, g_sidebar_hover_feature);
    draw_node(dl, eu_advect, g_sidebar_hover_feature);
    draw_node(dl, eu_bc, g_sidebar_hover_feature);
    draw_node(dl, eu_drag, g_sidebar_hover_feature);
    draw_node(dl, eu_vapor, g_sidebar_hover_feature);
    draw_node(dl, eu_latent, g_sidebar_hover_feature);
    draw_node(dl, eu_sdf_heat, g_sidebar_hover_feature);
    draw_node(dl, eu_loft, g_sidebar_hover_feature);

    // Euler vertical arrows
    draw_varrow(dl, node_bottom(eu_inject), node_top(eu_forces), COL_ARROW, 1.5f, &link_hover,
                "Injected smoke, heat, and vapor become Euler force sources.", mouse_pos, FEAT_COUPLE_INJECT);
    draw_varrow(dl, node_bottom(eu_forces), node_top(eu_diffuse), COL_ARROW, 1.5f, &link_hover,
                "Force accumulation feeds the thermal diffusion step.", mouse_pos, FEAT_EULER);
    draw_varrow(dl, node_bottom(eu_diffuse), node_top(eu_div), COL_ARROW, 1.5f, &link_hover,
                "Diffused heat and expansion sources contribute to divergence.", mouse_pos, FEAT_EULER_THERMAL);
    draw_varrow(dl, node_bottom(eu_div), node_top(eu_press), COL_ARROW, 1.5f, &link_hover,
                "Divergence is solved into pressure on the MAC grid.", mouse_pos, FEAT_EULER);
    draw_varrow(dl, node_bottom(eu_press), node_top(eu_proj), COL_ARROW, 1.5f, &link_hover,
                "Pressure is projected out of the velocity field.", mouse_pos, FEAT_EULER);
    draw_varrow(dl, node_bottom(eu_proj), node_top(eu_advect), COL_ARROW, 1.5f, &link_hover,
                "The divergence-free field is then advected.", mouse_pos, FEAT_EULER);
    draw_varrow(dl, node_bottom(eu_advect), node_top(eu_bc), COL_ARROW, 1.5f, &link_hover,
                "Advected air is clamped against domain and SDF boundaries.", mouse_pos, FEAT_EULER);
    draw_varrow(dl, node_bottom(eu_bc), node_top(eu_drag), COL_ARROW, 1.5f, &link_hover,
                "The finished air field is sampled back onto particles.", mouse_pos, FEAT_COUPLE_DRAG);

    // Air / heat technique connections
    draw_arrow(dl, node_right(eu_inject, 0.36f), node_left(eu_vapor), COL_VAPOR, -12.0f, 1.0f, &link_hover,
               "Material-aware vapor emission feeds the shared vapor field.", mouse_pos);
    draw_arrow(dl, node_right(eu_diffuse, 0.45f), node_left(eu_latent), COL_THERMAL, 18.0f, 1.0f, &link_hover,
               "Latent cooling reduces direct heating when vapor generation is strong.", mouse_pos);
    draw_arrow(dl, node_right(eu_diffuse, 0.74f), node_left(eu_sdf_heat), COL_SDF, -18.0f, 1.0f, &link_hover,
               "Air and SDF solids exchange heat through the solid thermal model.", mouse_pos);
    draw_arrow(dl, node_right(eu_drag, 0.52f), node_left(eu_loft), COL_FIELD, 18.0f, 1.0f, &link_hover,
               "Air carry is scaled per material so some batches float more than others.", mouse_pos);

    // ════════════════════════════════════════════════════════════════════
    //  SDF + RENDER COLUMN
    // ════════════════════════════════════════════════════════════════════
    NodeBox sdf_field = make_node("sdf_field", ImVec2(col_sdf, row_start + ROW_STEP * 0), ns,
                                  "SDF Field", "Scene objects, per-object properties, and rasterized distance field.",
                                  COL_SDF_FILL, COL_SDF, g_toggles.sdf_enabled);
    NodeBox sdf_sample = make_node("sdf_sample", ImVec2(col_sdf, row_start + ROW_STEP * 1), ns,
                                   "SDF Sample", "Lookup for collision normals, object IDs, and solid occupancy.",
                                   COL_SDF_FILL, COL_SDF, g_toggles.sdf_enabled);
    NodeBox sdf_render = make_node("sdf_render", ImVec2(col_sdf, row_start + ROW_STEP * 2), ns,
                                   "SDF Render", "Metal shading, heat tint, palette, and selection highlighting.",
                                   COL_SDF_FILL, COL_SDF, g_toggles.sdf_enabled);

    NodeBox ren_part = make_node("ren_part", ImVec2(col_sdf, row_start + ROW_STEP * 4), ns,
                                 "Particle Render", "SPH + MPM base draw with debug or batch colors.",
                                 COL_RENDER_FILL, COL_RENDER, true);
    NodeBox ren_meta = make_node("ren_meta", ImVec2(col_sdf, row_start + ROW_STEP * 5), ns,
                                 "Metaball", "Optional surface-style fluid rendering pass.",
                                 COL_RENDER_FILL, COL_RENDER, g_toggles.render_metaball);
    NodeBox ren_bloom = make_node("ren_bloom", ImVec2(col_sdf, row_start + ROW_STEP * 6), ns,
                                  "Bloom", "Hot particle glow and bloom composition.",
                                  COL_RENDER_FILL, COL_RENDER, g_toggles.render_bloom);
    NodeBox ren_euler = make_node("ren_euler", ImVec2(col_sdf, row_start + ROW_STEP * 7), ns,
                                  "Air Vis Overlay", "Smoke, fire, temperature, divergence, and other Euler debug views.",
                                  COL_RENDER_FILL, COL_RENDER, g_toggles.euler_enabled);
    NodeBox ren_comp = make_node("ren_comp", ImVec2(col_sdf, row_start + ROW_STEP * 8), ns,
                                 "Composite", "Final composition of SDF, particles, bloom, and air visuals.",
                                 COL_RENDER_FILL, COL_RENDER, true);

    sdf_field.feature_key = FEAT_SDF;
    sdf_sample.feature_key = FEAT_SDF;
    sdf_render.feature_key = FEAT_SDF;
    ren_part.feature_key = FEAT_MPM;
    ren_meta.feature_key = FEAT_RENDER_METABALL; ren_meta.toggle_ptr = &g_toggles.render_metaball;
    ren_bloom.feature_key = FEAT_RENDER_BLOOM; ren_bloom.toggle_ptr = &g_toggles.render_bloom;
    ren_euler.feature_key = FEAT_EULER;

    interact_node(sdf_field);
    interact_node(sdf_sample);
    interact_node(sdf_render);
    interact_node(ren_part);
    interact_node(ren_meta);
    interact_node(ren_bloom);
    interact_node(ren_euler);
    interact_node(ren_comp);

    bool any_node_hovered =
        sph_hash.hovered || sph_codim.hovered || sph_dens.hovered || sph_force.hovered || sph_sdf.hovered || sph_stens.hovered ||
        mpm_clear.hovered || mpm_p2g.hovered || mpm_gridop.hovered || mpm_g2p.hovered || mpm_constitutive.hovered ||
        mpm_thermal.hovered || mpm_frac.hovered || mpm_shell.hovered || mpm_pressure.hovered || mpm_filament.hovered ||
        mpm_bio.hovered || mpm_memory.hovered || mpm_field.hovered || mpm_gscale.hovered || mpm_sdf.hovered ||
        eu_inject.hovered || eu_forces.hovered || eu_diffuse.hovered || eu_div.hovered || eu_press.hovered ||
        eu_proj.hovered || eu_advect.hovered || eu_bc.hovered || eu_drag.hovered || eu_vapor.hovered || eu_latent.hovered ||
        eu_sdf_heat.hovered || eu_loft.hovered ||
        sdf_field.hovered || sdf_sample.hovered || sdf_render.hovered || ren_part.hovered || ren_meta.hovered ||
        ren_bloom.hovered || ren_euler.hovered || ren_comp.hovered;

    draw_node(dl, sdf_field, g_sidebar_hover_feature);
    draw_node(dl, sdf_sample, g_sidebar_hover_feature);
    draw_node(dl, sdf_render, g_sidebar_hover_feature);
    draw_node(dl, ren_part, g_sidebar_hover_feature);
    draw_node(dl, ren_meta, g_sidebar_hover_feature);
    draw_node(dl, ren_bloom, g_sidebar_hover_feature);
    draw_node(dl, ren_euler, g_sidebar_hover_feature);
    draw_node(dl, ren_comp, g_sidebar_hover_feature);

    // SDF vertical
    draw_varrow(dl, node_bottom(sdf_field),  node_top(sdf_sample), COL_ARROW, 1.5f, &link_hover,
                "The authored scene field is sampled for collision, ID lookup, and shading.", mouse_pos, FEAT_SDF);
    draw_varrow(dl, node_bottom(sdf_sample), node_top(sdf_render), COL_ARROW, 1.5f, &link_hover,
                "Sampled scene data feeds the final SDF render pass.", mouse_pos, FEAT_SDF);

    // Render vertical
    draw_varrow(dl, node_bottom(ren_part), node_top(ren_meta), COL_ARROW, 1.5f, &link_hover,
                "Particles feed the optional metaball pass.", mouse_pos, FEAT_RENDER_METABALL);
    draw_varrow(dl, node_bottom(ren_meta), node_top(ren_bloom), COL_ARROW, 1.5f, &link_hover,
                "Metaball output continues into bloom and hot-glow composition.", mouse_pos, FEAT_RENDER_BLOOM);
    draw_varrow(dl, node_bottom(ren_bloom), node_top(ren_euler), COL_ARROW, 1.5f, &link_hover,
                "Bloomed particles are composited with air overlays.", mouse_pos, FEAT_RENDER_BLOOM);
    draw_varrow(dl, node_bottom(ren_euler), node_top(ren_comp), COL_ARROW, 1.5f, &link_hover,
                "All render layers merge in the final composite.", mouse_pos, FEAT_EULER);

    // ════════════════════════════════════════════════════════════════════
    //  CROSS-COLUMN COUPLING ARROWS
    // ════════════════════════════════════════════════════════════════════

    // SDF -> SPH collision
    if (g_toggles.sdf_enabled && g_toggles.sph_enabled) {
        draw_arrow(dl, node_left(sdf_sample, 0.42f),
                   node_right(sph_sdf, 0.55f), COL_COUPLING, 18.0f, 2.0f, &link_hover,
                   "Scene SDF collision normals and distances are sampled by SPH.", mouse_pos, FEAT_SDF);
    }

    // SDF -> MPM grid_op boundary
    if (g_toggles.sdf_enabled && g_toggles.mpm_enabled) {
        draw_arrow(dl, node_left(sdf_sample, 0.62f),
                   node_right(mpm_sdf, 0.50f), COL_COUPLING, -12.0f, 2.0f, &link_hover,
                   "Scene SDF collision data constrains MPM particles and grid boundaries.", mouse_pos, FEAT_SDF);
    }

    // SDF -> Euler solid cells
    if (g_toggles.sdf_enabled && g_toggles.euler_enabled) {
        draw_arrow(dl, node_left(sdf_field, 0.38f),
                   node_right(eu_forces, 0.30f),
                   IM_COL32(200, 200, 200, 140), 28.0f, 1.5f, &link_hover,
                   "Rigid SDF cells mark solid occupancy in the Euler solve.", mouse_pos, FEAT_SDF);
    }
    if (g_toggles.sdf_enabled && g_toggles.sdf_heat) {
        draw_arrow(dl, node_left(sdf_field, 0.68f),
                   node_right(eu_sdf_heat, 0.55f),
                   COL_SDF, -12.0f, 1.3f, &link_hover,
                   "Per-object SDF thermal properties feed the solid heat-sink model.", mouse_pos, FEAT_SDF_HEAT);
    }

    // SPH <-> MPM coupling (via grid velocity sampling)
    if (g_toggles.coupling_sph_mpm && g_toggles.sph_enabled && g_toggles.mpm_enabled) {
        ImVec2 from = node_right(sph_force, 0.42f);
        ImVec2 to   = node_left(mpm_g2p, 0.36f);
        // Shift slightly up/down for bidirectional
        ImVec2 f1(from.x, from.y - 4);
        ImVec2 t1(to.x,   to.y   - 4);
        ImVec2 f2(to.x,   to.y   + 4);
        ImVec2 t2(node_right(sph_force, 0.70f).x, node_right(sph_force, 0.70f).y);
        draw_arrow(dl, f1, t1, COL_COUPLING, -28.0f, 2.0f, &link_hover,
                   "SPH samples MPM grid velocity for two-way particle coupling.", mouse_pos, FEAT_COUPLE_SPH_MPM);
        draw_arrow(dl, f2, t2, COL_COUPLING, 28.0f, 2.0f, &link_hover,
                   "MPM in turn feels SPH-driven grid motion through the shared velocity field.", mouse_pos, FEAT_COUPLE_SPH_MPM);
        // Label
        ImVec2 mid((f1.x + t1.x) * 0.5f, (f1.y + t1.y) * 0.5f - 12);
        dl->AddText(mid, COL_COUPLING, "grid vel sample");
    }

    // Particle -> Euler inject
    if (g_toggles.coupling_euler_inject && g_toggles.euler_enabled) {
        if (g_toggles.mpm_enabled) {
            draw_arrow(dl, node_right(mpm_g2p, 0.22f),
                       ImVec2(eu_inject.pos.x, eu_inject.pos.y + eu_inject.size.y * 0.32f),
                       COL_THERMAL, -18.0f, 1.5f, &link_hover,
                       "Hot / burning particles inject smoke, heat, and vapor into the air grid.", mouse_pos, FEAT_COUPLE_INJECT);
            if (g_toggles.mpm_pressure) {
                draw_arrow(dl, node_right(mpm_pressure, 0.52f),
                           node_left(eu_vapor, 0.70f),
                           COL_VAPOR, 46.0f, 1.2f, &link_hover,
                           "Pressure-generating materials contribute vapor and expansion sources to air.", mouse_pos, FEAT_MPM_PRESSURE);
            }
        }
    }

    // Euler drag -> particles
    if (g_toggles.coupling_euler_drag && g_toggles.euler_enabled) {
        if (g_toggles.mpm_enabled) {
            ImVec2 from = node_left(eu_drag, 0.62f);
            ImVec2 to   = ImVec2(mpm_g2p.pos.x + mpm_g2p.size.x,
                                 mpm_g2p.pos.y + mpm_g2p.size.y * 0.72f);
            draw_arrow(dl, from, to, COL_EULER, 18.0f, 1.5f, &link_hover,
                       "The finished air field pushes, carries, and cools particles.", mouse_pos, FEAT_COUPLE_DRAG);
            if (g_toggles.mpm_field) {
                draw_arrow(dl, node_left(eu_loft, 0.40f), node_right(mpm_field, 0.62f),
                           COL_FIELD, -42.0f, 1.2f, &link_hover,
                           "Per-material loft scales how strongly air and vapor carry each batch.", mouse_pos, FEAT_COUPLE_DRAG);
            }
        }
    }

    const char* active_feature = nullptr;
    auto absorb_feature = [&](const NodeBox& n) {
        if (!active_feature && (n.hovered || n.toggle_hovered) && n.feature_key) active_feature = n.feature_key;
    };
    absorb_feature(sph_hash); absorb_feature(sph_codim); absorb_feature(sph_dens); absorb_feature(sph_force);
    absorb_feature(sph_sdf); absorb_feature(sph_stens);
    absorb_feature(mpm_clear); absorb_feature(mpm_p2g); absorb_feature(mpm_gridop); absorb_feature(mpm_g2p);
    absorb_feature(mpm_constitutive); absorb_feature(mpm_thermal); absorb_feature(mpm_frac); absorb_feature(mpm_shell);
    absorb_feature(mpm_pressure); absorb_feature(mpm_filament); absorb_feature(mpm_bio); absorb_feature(mpm_memory);
    absorb_feature(mpm_field); absorb_feature(mpm_gscale); absorb_feature(mpm_sdf);
    absorb_feature(eu_inject); absorb_feature(eu_forces); absorb_feature(eu_diffuse); absorb_feature(eu_div);
    absorb_feature(eu_press); absorb_feature(eu_proj); absorb_feature(eu_advect); absorb_feature(eu_bc);
    absorb_feature(eu_drag); absorb_feature(eu_vapor); absorb_feature(eu_latent); absorb_feature(eu_sdf_heat); absorb_feature(eu_loft);
    absorb_feature(sdf_field); absorb_feature(sdf_sample); absorb_feature(sdf_render);
    absorb_feature(ren_part); absorb_feature(ren_meta); absorb_feature(ren_bloom); absorb_feature(ren_euler); absorb_feature(ren_comp);
    if (!active_feature && link_hover.feature_key) active_feature = link_hover.feature_key;
    g_sidebar_hover_feature = active_feature;

    draw_hovered_link(dl, link_hover);

    // Re-draw nodes on top so links don't scribble across them.
    draw_node(dl, sph_hash, active_feature);
    draw_node(dl, sph_codim, active_feature);
    draw_node(dl, sph_dens, active_feature);
    draw_node(dl, sph_force, active_feature);
    draw_node(dl, sph_sdf, active_feature);
    draw_node(dl, sph_stens, active_feature);
    draw_node(dl, mpm_clear, active_feature);
    draw_node(dl, mpm_p2g, active_feature);
    draw_node(dl, mpm_gridop, active_feature);
    draw_node(dl, mpm_g2p, active_feature);
    draw_node(dl, mpm_constitutive, active_feature);
    draw_node(dl, mpm_thermal, active_feature);
    draw_node(dl, mpm_frac, active_feature);
    draw_node(dl, mpm_shell, active_feature);
    draw_node(dl, mpm_pressure, active_feature);
    draw_node(dl, mpm_filament, active_feature);
    draw_node(dl, mpm_bio, active_feature);
    draw_node(dl, mpm_memory, active_feature);
    draw_node(dl, mpm_field, active_feature);
    draw_node(dl, mpm_gscale, active_feature);
    draw_node(dl, mpm_sdf, active_feature);
    draw_node(dl, eu_inject, active_feature);
    draw_node(dl, eu_forces, active_feature);
    draw_node(dl, eu_diffuse, active_feature);
    draw_node(dl, eu_div, active_feature);
    draw_node(dl, eu_press, active_feature);
    draw_node(dl, eu_proj, active_feature);
    draw_node(dl, eu_advect, active_feature);
    draw_node(dl, eu_bc, active_feature);
    draw_node(dl, eu_drag, active_feature);
    draw_node(dl, eu_vapor, active_feature);
    draw_node(dl, eu_latent, active_feature);
    draw_node(dl, eu_sdf_heat, active_feature);
    draw_node(dl, eu_loft, active_feature);
    draw_node(dl, sdf_field, active_feature);
    draw_node(dl, sdf_sample, active_feature);
    draw_node(dl, sdf_render, active_feature);
    draw_node(dl, ren_part, active_feature);
    draw_node(dl, ren_meta, active_feature);
    draw_node(dl, ren_bloom, active_feature);
    draw_node(dl, ren_euler, active_feature);
    draw_node(dl, ren_comp, active_feature);

    if (diagram_hovered && link_hover.tooltip && !any_node_hovered) {
        ImGui::SetTooltip("%s", link_hover.tooltip);
    }

    ImVec2 view_pos = ImGui::GetWindowPos();
    ImVec2 view_size = ImGui::GetWindowSize();
    const float pad = 12.0f;
    const float legend_panel_w = 604.0f;
    const float legend_panel_h = 92.0f;
    const float button_h = 24.0f;
    const float button_w = 104.0f;
    ImVec2 legend_button_pos(view_pos.x + pad, view_pos.y + view_size.y - button_h - pad);
    ImVec2 legend_panel_pos(view_pos.x + pad, legend_button_pos.y - legend_panel_h - 8.0f);

    ImVec2 legend_button_max(legend_button_pos.x + button_w, legend_button_pos.y + button_h);
    bool legend_button_hovered =
        ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
        ImGui::IsMouseHoveringRect(legend_button_pos, legend_button_max, false);
    if (rect_clicked(legend_button_pos, legend_button_max)) {
        g_legend_open = !g_legend_open;
    }
    dl->AddRectFilled(legend_button_pos, legend_button_max,
                      legend_button_hovered ? IM_COL32(64, 64, 78, 230) : IM_COL32(42, 42, 52, 220),
                      6.0f);
    dl->AddRect(legend_button_pos, legend_button_max,
                legend_button_hovered ? IM_COL32(255, 255, 255, 120) : IM_COL32(255, 255, 255, 46),
                6.0f, 0, 1.2f);
    const char* legend_button_label = g_legend_open ? "Hide Legend" : "Show Legend";
    ImVec2 legend_label_size = ImGui::CalcTextSize(legend_button_label);
    dl->AddText(ImVec2(legend_button_pos.x + (button_w - legend_label_size.x) * 0.5f,
                       legend_button_pos.y + (button_h - legend_label_size.y) * 0.5f - 1.0f),
                legend_button_hovered ? COL_TEXT : COL_TEXT_DIM, legend_button_label);

    if (g_legend_open) {
        ImVec2 legend_panel_max(legend_panel_pos.x + legend_panel_w, legend_panel_pos.y + legend_panel_h);
        dl->AddRectFilled(legend_panel_pos, legend_panel_max, IM_COL32(18, 18, 24, 226), 8.0f);
        dl->AddRect(legend_panel_pos, legend_panel_max, IM_COL32(255, 255, 255, 38), 8.0f, 0, 1.2f);

        float leg_x = legend_panel_pos.x + 10.0f;
        float leg_y = legend_panel_pos.y + 8.0f;
        dl->AddText(ImVec2(leg_x, leg_y), COL_TEXT_DIM, "LEGEND:");
        leg_y += 16.0f;

        auto legend_item = [&](float x, float y, ImU32 col, const char* label) {
            dl->AddRectFilled(ImVec2(x, y + 2), ImVec2(x + 22, y + 12), col, 3.0f);
            dl->AddText(ImVec2(x + 28, y - 1), COL_TEXT_DIM, label);
        };

        legend_item(leg_x,       leg_y, COL_SPH,      "SPH Fluid");
        legend_item(leg_x + 140, leg_y, COL_MPM,      "MPM Solid");
        legend_item(leg_x + 280, leg_y, COL_EULER,    "Eulerian Air");
        legend_item(leg_x + 420, leg_y, COL_SDF,      "SDF / Render");
        leg_y += 18.0f;
        legend_item(leg_x,       leg_y, COL_COUPLING, "Coupling");
        legend_item(leg_x + 140, leg_y, COL_THERMAL,  "Thermal");
        legend_item(leg_x + 280, leg_y, COL_VAPOR,    "Vapor / Puff");
        legend_item(leg_x + 420, leg_y, COL_FIELD,    "Field / Filament");
        leg_y += 18.0f;
        legend_item(leg_x,       leg_y, COL_BIO,      "Cooking / Bio");
        legend_item(leg_x + 140, leg_y, COL_MEMORY,   "Memory / Recovery");

        leg_y += 22.0f;
        dl->AddText(ImVec2(leg_x, leg_y), COL_TEXT_DIM,
            "5 MPM substeps | 2 SPH substeps | 30 pressure iterations | MAC staggered grid | drag nodes to fine-tune");
    }

    ImGui::EndChild();
    ImGui::End();
    if (open && !*open) {
        reset_pipeline_viewer_session();
    }
}

} // namespace ng
