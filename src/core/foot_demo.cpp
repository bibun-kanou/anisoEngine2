#include "core/foot_demo.h"

#include "core/creation_menu.h"
#include "core/log.h"
#include "physics/common/particle_buffer.h"
#include "physics/common/grid.h"
#include "physics/sph/sph_solver.h"
#include "physics/mpm/mpm_solver.h"
#include "physics/sdf/sdf_field.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace ng {

namespace {

constexpr f32 kFootSpacing = 0.0425f;
constexpr f32 kSkinBand = 0.070f;
constexpr f32 kPadBand = 0.265f;
constexpr i32 kBoneCount = 18;

enum class TissueLayer : u8 {
    CORE = 0,
    PAD = 1,
    SKIN = 2
};

struct BoneSegment {
    vec2 a = vec2(0.0f);
    vec2 b = vec2(0.0f);
    f32 bone_radius = 0.08f;
    f32 flesh_radius = 0.22f;
};

struct FootMorphology {
    f32 calf_length = 1.96f;
    f32 calf_radius = 0.46f;
    f32 heel_length = 0.62f;
    f32 heel_radius = 0.40f;
    f32 arch_length = 1.28f;
    f32 arch_radius = 0.34f;
    f32 forefoot_radius = 0.44f;
    std::array<f32, 5> toe_lengths{ 0.82f, 0.73f, 0.61f, 0.49f, 0.34f };
    std::array<f32, 5> toe_radii{ 0.235f, 0.185f, 0.165f, 0.145f, 0.118f };
    std::array<f32, 5> toe_base_x{ 1.18f, 1.10f, 1.02f, 0.95f, 0.88f };
    std::array<f32, 5> toe_base_y{ -0.16f, -0.085f, -0.01f, 0.072f, 0.145f };
    std::array<f32, 5> toe_rest_lift{ 0.22f, 0.06f, 0.01f, -0.05f, -0.12f };
    std::array<f32, 5> toe_rest_curl{ 0.04f, 0.08f, 0.12f, 0.17f, 0.24f };
};

struct FootPose {
    vec2 ankle = vec2(-3.05f, -0.22f);
    f32 foot_pitch = -0.04f;
    f32 calf_pitch = 1.55f;
    f32 heel_roll = 0.0f;
    f32 ball_lift = 0.0f;
    f32 plantar_contract = 0.0f;
    std::array<f32, 5> toe_curl{};
    std::array<f32, 5> toe_tip_curl{};
    std::array<f32, 5> toe_lift{};
};

struct AttachmentPoint {
    vec2 local = vec2(0.0f);
    u16 bone_index = 0;
    f32 weight = 0.0f;
    TissueLayer layer = TissueLayer::CORE;
};

struct FootDemoState {
    bool active = false;
    FootControlFocus focus = FootControlFocus::ANKLE;
    FootMorphology morph{};
    FootPose pose{};
    std::array<BoneSegment, kBoneCount> bones{};
    vec2 heel_point = vec2(0.0f);
    vec2 ball_point = vec2(0.0f);
    std::array<vec2, 5> toe_bases{};
    std::array<vec2, 5> toe_tips{};
    u32 mpm_offset = 0;
    u32 mpm_count = 0;
    f32 attachment_force = 118.0f;
    f32 attachment_damping = 20.0f;
    std::vector<AttachmentPoint> attachments;
    std::vector<vec2> scratch_targets;
    std::vector<f32> scratch_weights;
    std::vector<TissueLayer> scratch_layers;
    vec2 ankle_pin = vec2(0.0f);
    bool ankle_pinned = false;
};

struct FootObstacleField {
    bool valid = false;
    vec2 world_min = vec2(0.0f);
    ivec2 resolution = ivec2(0);
    f32 cell_size = 0.12f;
    f32 max_radius = 0.0f;
    std::vector<vec2> points;
    std::vector<f32> radii;
    std::vector<std::vector<u32>> cells;
};

FootDemoState g_state;

void push_scene_batch(CreationState* creation, BatchRecord&& batch) {
    if (!creation) return;
    creation->batches.push_back(std::move(batch));
    creation->batch_counter = static_cast<u32>(creation->batches.size());
}

vec2 dir_from_angle(f32 angle) {
    return vec2(std::cos(angle), std::sin(angle));
}

vec2 perp(vec2 v) {
    return vec2(-v.y, v.x);
}

f32 point_segment_param(vec2 p, vec2 a, vec2 b) {
    vec2 ab = b - a;
    f32 denom = glm::dot(ab, ab);
    if (denom <= 1e-6f) return 0.0f;
    return glm::clamp(glm::dot(p - a, ab) / denom, 0.0f, 1.0f);
}

vec2 closest_point_on_segment(vec2 p, vec2 a, vec2 b) {
    f32 t = point_segment_param(p, a, b);
    return glm::mix(a, b, t);
}

f32 sd_capsule(vec2 p, vec2 a, vec2 b, f32 radius) {
    return glm::length(p - closest_point_on_segment(p, a, b)) - radius;
}

f32 sd_box(vec2 p, vec2 center, vec2 half_extents) {
    vec2 q = glm::abs(p - center) - half_extents;
    vec2 outside = glm::max(q, vec2(0.0f));
    return glm::length(outside) + glm::min(glm::max(q.x, q.y), 0.0f);
}

vec2 foot_local_to_world(const FootPose& pose, vec2 local) {
    vec2 foot_dir = dir_from_angle(pose.foot_pitch);
    vec2 foot_up = perp(foot_dir);
    return pose.ankle + foot_dir * local.x + foot_up * local.y;
}

vec2 foot_world_to_local(const FootPose& pose, vec2 world) {
    vec2 foot_dir = dir_from_angle(pose.foot_pitch);
    vec2 foot_up = perp(foot_dir);
    vec2 rel = world - pose.ankle;
    return vec2(glm::dot(rel, foot_dir), glm::dot(rel, foot_up));
}

vec2 rotate_vec(vec2 v, f32 angle) {
    f32 c = std::cos(angle);
    f32 s = std::sin(angle);
    return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

f32 sd_local_ellipse(vec2 p_local, vec2 center_local, vec2 radii, f32 rotation = 0.0f) {
    vec2 q = rotate_vec(p_local - center_local, -rotation);
    vec2 safe_r = glm::max(radii, vec2(0.02f));
    vec2 n = q / safe_r;
    return (glm::length(n) - 1.0f) * glm::min(safe_r.x, safe_r.y);
}

void update_pose_cache() {
    const FootPose& pose = g_state.pose;
    const FootMorphology& morph = g_state.morph;
    std::array<BoneSegment, kBoneCount>& bones = g_state.bones;

    vec2 foot_dir = dir_from_angle(pose.foot_pitch);
    vec2 foot_up = perp(foot_dir);
    vec2 calf_dir = dir_from_angle(pose.calf_pitch + pose.foot_pitch * 0.18f);

    bones[0] = {
        pose.ankle + vec2(-0.02f, 0.04f),
        pose.ankle + calf_dir * morph.calf_length,
        0.12f,
        morph.calf_radius
    };

    vec2 heel_a = foot_local_to_world(pose, vec2(-0.06f, -0.02f));
    vec2 heel_b = foot_local_to_world(pose, vec2(-morph.heel_length, -0.14f - pose.heel_roll * 0.10f));
    bones[1] = { heel_a, heel_b, 0.12f, morph.heel_radius };

    vec2 arch_a = foot_local_to_world(pose, vec2(-0.04f, -0.05f));
    vec2 arch_b = foot_local_to_world(pose, vec2(morph.arch_length, -0.10f + pose.ball_lift * 0.04f - pose.plantar_contract * 0.05f));
    bones[2] = { arch_a, arch_b, 0.10f, morph.arch_radius };

    g_state.heel_point = heel_b - foot_up * 0.06f;

    i32 bone_idx = 3;
    vec2 ball_accum(0.0f);
    for (int toe = 0; toe < 5; ++toe) {
        vec2 toe_base = foot_local_to_world(pose, vec2(
            morph.toe_base_x[toe] - pose.plantar_contract * 0.06f,
            morph.toe_base_y[toe] + pose.ball_lift * 0.08f));
        vec2 met_start = foot_local_to_world(pose, vec2(0.48f, morph.toe_base_y[toe] * 0.52f));
        f32 toe_angle = pose.foot_pitch
            + morph.toe_rest_lift[toe] * 0.36f
            + pose.toe_lift[toe] * 0.48f
            - (morph.toe_rest_curl[toe] + pose.toe_curl[toe]) * 0.90f
            - pose.plantar_contract * 0.14f;
        vec2 toe_dir = dir_from_angle(toe_angle);
        f32 total_len = morph.toe_lengths[toe] * (1.0f - pose.plantar_contract * 0.14f);
        f32 prox_len = total_len * ((toe == 0) ? 0.58f : 0.56f);
        f32 dist_len = total_len - prox_len;
        vec2 prox_end = toe_base + toe_dir * prox_len;
        f32 tip_angle = toe_angle - pose.toe_tip_curl[toe] * 0.78f - pose.toe_curl[toe] * 0.18f;
        vec2 tip_end = prox_end + dir_from_angle(tip_angle) * dist_len;

        bones[bone_idx++] = { met_start, toe_base, morph.toe_radii[toe] * 0.48f, morph.toe_radii[toe] * 0.95f };
        bones[bone_idx++] = { toe_base, prox_end, morph.toe_radii[toe] * 0.44f, morph.toe_radii[toe] * 0.92f };
        bones[bone_idx++] = { prox_end, tip_end, morph.toe_radii[toe] * 0.34f, morph.toe_radii[toe] * 0.72f };

        g_state.toe_bases[toe] = toe_base;
        g_state.toe_tips[toe] = tip_end;
        ball_accum += toe_base;
    }
    g_state.ball_point = ball_accum / 5.0f;
}

AttachmentPoint make_attachment(vec2 p, TissueLayer layer) {
    f32 best_dist = std::numeric_limits<f32>::max();
    i32 best_idx = 0;
    vec2 best_local(0.0f);

    for (i32 i = 0; i < kBoneCount; ++i) {
        const BoneSegment& bone = g_state.bones[i];
        vec2 seg = bone.b - bone.a;
        f32 len = glm::length(seg);
        if (len <= 1e-5f) continue;
        vec2 tangent = seg / len;
        vec2 normal = perp(tangent);
        vec2 rel = p - bone.a;
        vec2 local(glm::dot(rel, tangent), glm::dot(rel, normal));
        vec2 closest = bone.a + tangent * glm::clamp(local.x, 0.0f, len);
        f32 dist = glm::length(p - closest);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
            best_local = local;
        }
    }

    f32 weight = 0.0f;
    switch (layer) {
    case TissueLayer::CORE: weight = 0.62f; break;
    case TissueLayer::PAD:  weight = 0.28f; break;
    case TissueLayer::SKIN: weight = 0.44f; break;
    }
    if (best_idx >= 3) weight += 0.06f;
    if (best_idx == 0) weight -= 0.10f;
    if (best_idx == 1 || best_idx == 2) weight += 0.03f;
    if (best_idx >= 15) weight -= 0.04f;
    weight = glm::clamp(weight, 0.18f, 0.96f);
    return { best_local, static_cast<u16>(best_idx), weight, layer };
}

void add_spawn_batch(CreationState* creation, ParticleBuffer& particles,
                     u32 before_count, const char* label, const char* description,
                     MPMMaterial material, vec4 color,
                     f32 youngs_modulus, f32 poisson_ratio, vec2 fiber_dir,
                     f32 recommended_size, const char* recommended_note) {
    if (!creation) return;
    u32 after_count = particles.range(SolverType::MPM).count;
    if (after_count <= before_count) return;
    BatchRecord batch = make_batch_record(
        label, description, SpawnSolver::MPM, material, color,
        youngs_modulus, poisson_ratio, 300.0f, fiber_dir,
        recommended_size, recommended_note, true);
    batch.properties = property_summary(batch.solver, batch.mpm_type,
                                        batch.youngs_modulus, batch.poisson_ratio,
                                        batch.temperature, batch.fiber_dir);
    batch.mpm_offset = particles.range(SolverType::MPM).offset + before_count;
    batch.mpm_count = after_count - before_count;
    push_scene_batch(creation, std::move(batch));
}

void build_foot_particles(std::vector<vec2>& core_positions,
                          std::vector<f32>& core_shell,
                          std::vector<AttachmentPoint>& core_attach,
                          std::vector<vec2>& pad_positions,
                          std::vector<f32>& pad_shell,
                          std::vector<AttachmentPoint>& pad_attach,
                          std::vector<vec2>& skin_positions,
                          std::vector<f32>& skin_shell,
                          std::vector<AttachmentPoint>& skin_attach) {
    update_pose_cache();

    vec2 bmin(std::numeric_limits<f32>::max());
    vec2 bmax(-std::numeric_limits<f32>::max());
    for (const BoneSegment& bone : g_state.bones) {
        vec2 r(bone.flesh_radius + 0.08f);
        bmin = glm::min(bmin, glm::min(bone.a, bone.b) - r);
        bmax = glm::max(bmax, glm::max(bone.a, bone.b) + r);
    }

    for (f32 y = bmin.y; y <= bmax.y; y += kFootSpacing) {
        for (f32 x = bmin.x; x <= bmax.x; x += kFootSpacing) {
            vec2 p(x, y);
            vec2 p_local = foot_world_to_local(g_state.pose, p);
            f32 best_sd = std::numeric_limits<f32>::max();
            i32 best_idx = 0;
            for (i32 i = 0; i < kBoneCount; ++i) {
                const BoneSegment& bone = g_state.bones[i];
                f32 sd = sd_capsule(p, bone.a, bone.b, bone.flesh_radius);
                if (sd < best_sd) {
                    best_sd = sd;
                    best_idx = i;
                }
            }
            f32 support_sd = best_sd;
            support_sd = glm::min(support_sd, sd_local_ellipse(p_local, vec2(-0.44f, -0.16f), vec2(0.58f, 0.34f), -0.08f));
            support_sd = glm::min(support_sd, sd_local_ellipse(p_local, vec2(0.52f, -0.02f), vec2(0.96f, 0.27f), -0.08f));
            support_sd = glm::min(support_sd, sd_local_ellipse(p_local, vec2(0.88f, 0.16f), vec2(0.76f, 0.25f), 0.02f));
            support_sd = glm::min(support_sd, sd_local_ellipse(p_local, vec2(1.26f, -0.06f), vec2(0.68f, 0.30f), -0.05f));
            support_sd = glm::min(support_sd, sd_local_ellipse(p_local, vec2(1.84f, -0.04f), vec2(0.34f, 0.17f), -0.06f));

            f32 pad_sd = sd_local_ellipse(p_local, vec2(-0.48f, -0.38f), vec2(0.66f, 0.30f), -0.05f);
            pad_sd = glm::min(pad_sd, sd_local_ellipse(p_local, vec2(0.06f, -0.29f), vec2(0.64f, 0.15f), -0.08f));
            pad_sd = glm::min(pad_sd, sd_local_ellipse(p_local, vec2(0.95f, -0.26f), vec2(0.82f, 0.20f), -0.02f));
            pad_sd = glm::min(pad_sd, sd_local_ellipse(p_local, vec2(1.70f, -0.20f), vec2(0.42f, 0.18f), -0.04f));
            for (int toe = 0; toe < 5; ++toe) {
                vec2 toe_base = g_state.toe_bases[toe];
                vec2 toe_tip = g_state.toe_tips[toe];
                f32 toe_pad_radius = g_state.morph.toe_radii[toe] * ((toe == 0) ? 1.05f : 0.88f);
                vec2 toe_start = glm::mix(toe_base, toe_tip, 0.20f);
                pad_sd = glm::min(pad_sd, sd_capsule(p, toe_start, toe_tip, toe_pad_radius));
            }
            // Soft interdigital webbing smooths the squeeze zone between adjacent toes
            // so trapped props feel compressed by a padded cleft instead of two hard pillars.
            for (int toe = 0; toe < 4; ++toe) {
                vec2 web_a = glm::mix(g_state.toe_bases[toe], g_state.toe_tips[toe], 0.16f);
                vec2 web_b = glm::mix(g_state.toe_bases[toe + 1], g_state.toe_tips[toe + 1], 0.14f);
                f32 web_r = glm::min(g_state.morph.toe_radii[toe], g_state.morph.toe_radii[toe + 1]) * 0.34f;
                pad_sd = glm::min(pad_sd, sd_capsule(p, web_a, web_b, web_r));

                vec2 inner_a = glm::mix(g_state.toe_bases[toe], g_state.toe_tips[toe], 0.28f);
                vec2 inner_b = glm::mix(g_state.toe_bases[toe + 1], g_state.toe_tips[toe + 1], 0.24f);
                f32 inner_r = web_r * 0.56f;
                pad_sd = glm::min(pad_sd, sd_capsule(p, inner_a, inner_b, inner_r));
            }

            f32 total_sd = glm::min(support_sd, pad_sd);
            if (total_sd > 0.0f) continue;

            f32 dist_to_surface = -total_sd;
            bool plantar_zone = p_local.y < 0.02f;
            bool in_pad = (pad_sd <= 0.02f) && plantar_zone;

            const BoneSegment& bone = g_state.bones[best_idx];
            vec2 closest = closest_point_on_segment(p, bone.a, bone.b);
            vec2 normal = p - closest;
            if (glm::dot(normal, normal) <= 1e-6f) {
                vec2 seg = bone.b - bone.a;
                normal = (glm::dot(seg, seg) > 1e-6f) ? perp(glm::normalize(seg)) : vec2(0.0f, 1.0f);
            } else {
                normal = glm::normalize(normal);
            }

            if (dist_to_surface <= kSkinBand) {
                vec2 shell_pos = p + normal * 0.014f;
                skin_positions.push_back(shell_pos);
                skin_shell.push_back(1.0f);
                AttachmentPoint shell_point = make_attachment(shell_pos, TissueLayer::SKIN);
                shell_point.weight = glm::clamp(shell_point.weight + 0.10f, 0.30f, 0.96f);
                skin_attach.push_back(shell_point);
            } else if (in_pad && dist_to_surface <= kPadBand) {
                vec2 pad_pos = p - normal * 0.008f;
                pad_positions.push_back(pad_pos);
                f32 pad_shell_seed = 1.0f - glm::smoothstep(kSkinBand, kPadBand, dist_to_surface);
                pad_shell.push_back(glm::clamp(pad_shell_seed * 0.55f + 0.20f, 0.0f, 1.0f));
                AttachmentPoint pad_point = make_attachment(pad_pos, TissueLayer::PAD);
                f32 pad_depth = 1.0f - glm::clamp(dist_to_surface / kPadBand, 0.0f, 1.0f);
                pad_point.weight = glm::clamp(pad_point.weight + pad_depth * 0.10f, 0.20f, 0.82f);
                pad_attach.push_back(pad_point);
            } else {
                core_positions.push_back(p - normal * 0.010f);
                f32 shell_seed = 1.0f - glm::smoothstep(kPadBand * 0.55f, kPadBand * 1.25f, dist_to_surface);
                core_shell.push_back(glm::clamp(shell_seed * 0.55f, 0.0f, 1.0f));
                AttachmentPoint core_point = make_attachment(p, TissueLayer::CORE);
                core_point.weight = glm::clamp(core_point.weight + (1.0f - glm::clamp(dist_to_surface / 0.42f, 0.0f, 1.0f)) * 0.10f, 0.26f, 0.96f);
                core_attach.push_back(core_point);
            }
        }
    }
}

vec2 attachment_target_world(const AttachmentPoint& a) {
    const BoneSegment& bone = g_state.bones[a.bone_index];
    vec2 seg = bone.b - bone.a;
    f32 len = glm::length(seg);
    vec2 tangent = (len > 1e-5f) ? (seg / len) : vec2(1.0f, 0.0f);
    vec2 normal = perp(tangent);
    return bone.a + tangent * a.local.x + normal * a.local.y;
}

vec2 attachment_outward_normal(const AttachmentPoint& a) {
    const BoneSegment& bone = g_state.bones[a.bone_index];
    vec2 seg = bone.b - bone.a;
    f32 len = glm::length(seg);
    vec2 tangent = (len > 1e-5f) ? (seg / len) : vec2(1.0f, 0.0f);
    vec2 normal = perp(tangent);
    if (std::abs(a.local.y) > 1e-4f) {
        normal *= glm::sign(a.local.y);
    }
    return glm::normalize(normal);
}

struct RepairTuning {
    f32 soft_radius = 0.0f;
    f32 hard_radius = 0.0f;
    f32 recover_rate = 0.0f;
    f32 snap_rate = 0.0f;
    f32 velocity_damping = 0.0f;
    f32 outward_bias = 0.0f;
};

RepairTuning repair_tuning(TissueLayer layer) {
    switch (layer) {
    case TissueLayer::CORE:
        return { 0.15f, 0.34f, 8.0f, 22.0f, 3.2f, 0.002f };
    case TissueLayer::PAD:
        return { 0.23f, 0.50f, 5.0f, 14.0f, 2.4f, 0.006f };
    case TissueLayer::SKIN:
    default:
        return { 0.19f, 0.42f, 6.0f, 16.0f, 2.8f, 0.004f };
    }
}

f32 contact_radius_for_layer(TissueLayer layer) {
    switch (layer) {
    case TissueLayer::CORE: return 0.040f;
    case TissueLayer::PAD:  return 0.072f;
    case TissueLayer::SKIN: return 0.060f;
    default: return 0.055f;
    }
}

f32 scene_skin_for_layer(TissueLayer layer) {
    switch (layer) {
    case TissueLayer::CORE: return 0.010f;
    case TissueLayer::PAD:  return 0.022f;
    case TissueLayer::SKIN: return 0.018f;
    default: return 0.016f;
    }
}

f32 base_drive_scale_for_layer(TissueLayer layer) {
    switch (layer) {
    case TissueLayer::CORE: return 1.00f;
    case TissueLayer::PAD:  return 0.82f;
    case TissueLayer::SKIN: return 0.72f;
    default: return 0.80f;
    }
}

f32 sample_sdf_union_distance(const SDFField* sdf, vec2 p) {
    if (!sdf) return std::numeric_limits<f32>::max();
    const auto& objects = sdf->objects();
    if (objects.empty()) return std::numeric_limits<f32>::max();

    f32 best = std::numeric_limits<f32>::max();
    for (const auto& obj : objects) {
        f32 d = std::numeric_limits<f32>::max();
        switch (obj.type) {
        case SDFField::PRIM_BOX:
            d = sd_box(p, obj.a, obj.b);
            break;
        case SDFField::PRIM_CIRCLE:
            d = glm::length(p - obj.a) - obj.radius_or_thickness;
            break;
        case SDFField::PRIM_SEGMENT:
            d = sd_capsule(p, obj.a, obj.b, obj.radius_or_thickness);
            break;
        default:
            break;
        }
        best = glm::min(best, d);
    }
    return best;
}

vec2 sample_sdf_union_normal(const SDFField* sdf, vec2 p) {
    f32 eps = sdf ? glm::max(glm::min(sdf->cell_size().x, sdf->cell_size().y), 0.0025f) : 0.005f;
    f32 dx = sample_sdf_union_distance(sdf, p + vec2(eps, 0.0f)) - sample_sdf_union_distance(sdf, p - vec2(eps, 0.0f));
    f32 dy = sample_sdf_union_distance(sdf, p + vec2(0.0f, eps)) - sample_sdf_union_distance(sdf, p - vec2(0.0f, eps));
    vec2 n(dx, dy);
    if (glm::dot(n, n) <= 1e-8f) return vec2(0.0f, 1.0f);
    return glm::normalize(n);
}

void add_obstacle_sample(FootObstacleField& field, vec2 p, f32 radius,
                         vec2& min_p, vec2& max_p) {
    field.points.push_back(p);
    field.radii.push_back(radius);
    field.max_radius = glm::max(field.max_radius, radius);
    min_p = glm::min(min_p, p - vec2(radius));
    max_p = glm::max(max_p, p + vec2(radius));
}

FootObstacleField build_foot_obstacle_field(ParticleBuffer& particles, const SDFField* sdf = nullptr) {
    FootObstacleField field;
    const ParticleRange& range = particles.range(SolverType::MPM);
    if (range.count == 0 || range.count <= g_state.mpm_count) {
        if (!sdf || sdf->objects().empty()) return field;
    }

    std::vector<vec2> all_positions;
    if (range.count > 0) {
        all_positions.resize(range.count);
        particles.positions().download(all_positions.data(), all_positions.size() * sizeof(vec2),
                                       static_cast<size_t>(range.offset) * sizeof(vec2));
    }

    vec2 min_p(std::numeric_limits<f32>::max());
    vec2 max_p(-std::numeric_limits<f32>::max());
    field.points.reserve(range.count);
    field.radii.reserve(range.count);
    for (u32 i = 0; i < range.count; ++i) {
        u32 global_idx = range.offset + i;
        bool is_foot = (global_idx >= g_state.mpm_offset) &&
                       (global_idx < g_state.mpm_offset + g_state.mpm_count);
        if (is_foot) continue;
        const vec2 p = all_positions[i];
        add_obstacle_sample(field, p, 0.058f, min_p, max_p);
    }

    if (sdf) {
        for (const auto& obj : sdf->objects()) {
            if (obj.label != "Street Floor") continue;
            if (obj.type == SDFField::PRIM_SEGMENT) {
                vec2 seg = obj.b - obj.a;
                f32 len = glm::length(seg);
                i32 samples = glm::max(2, static_cast<i32>(std::ceil(len / 0.09f)));
                f32 floor_radius = obj.radius_or_thickness + 0.14f;
                for (i32 s = 0; s <= samples; ++s) {
                    f32 t = static_cast<f32>(s) / static_cast<f32>(samples);
                    vec2 p = glm::mix(obj.a, obj.b, t);
                    add_obstacle_sample(field, p, floor_radius, min_p, max_p);
                }
            } else if (obj.type == SDFField::PRIM_BOX) {
                f32 floor_radius = glm::max(glm::min(obj.b.x, obj.b.y), 0.16f) + 0.12f;
                for (f32 x = obj.a.x - obj.b.x; x <= obj.a.x + obj.b.x; x += 0.09f) {
                    add_obstacle_sample(field, vec2(x, obj.a.y + obj.b.y), floor_radius, min_p, max_p);
                }
            }
        }
    }

    if (field.points.empty()) {
        return field;
    }

    vec2 margin(field.cell_size * 2.0f);
    field.world_min = min_p - margin;
    vec2 world_max = max_p + margin;
    vec2 extent = glm::max(world_max - field.world_min, vec2(field.cell_size));
    field.resolution = ivec2(glm::ceil(extent / field.cell_size));
    field.resolution.x = glm::max(field.resolution.x, 1);
    field.resolution.y = glm::max(field.resolution.y, 1);
    field.cells.assign(static_cast<size_t>(field.resolution.x * field.resolution.y), {});

    for (u32 i = 0; i < field.points.size(); ++i) {
        vec2 rel = (field.points[i] - field.world_min) / field.cell_size;
        ivec2 cell = ivec2(glm::clamp(rel, vec2(0.0f), vec2(field.resolution) - vec2(1.0f)));
        size_t bucket_idx = static_cast<size_t>(cell.x + cell.y * field.resolution.x);
        field.cells[bucket_idx].push_back(i);
    }

    field.valid = true;
    return field;
}

bool project_outside_sdf(const SDFField* sdf, vec2& p, f32 skin,
                         vec2* out_normal = nullptr, f32* out_strength = nullptr) {
    if (out_strength) *out_strength = 0.0f;
    if (!sdf) return false;
    f32 d = sample_sdf_union_distance(sdf, p);
    if (d >= skin) return false;
    vec2 n = sample_sdf_union_normal(sdf, p);
    p += n * (skin - d);
    if (out_normal) *out_normal = n;
    if (out_strength) *out_strength = glm::clamp((skin - d) / glm::max(skin, 1e-4f), 0.0f, 1.0f);
    return true;
}

bool project_outside_obstacle_field(const FootObstacleField* field, vec2& p, vec2 preferred_normal,
                                    f32 contact_radius, f32 skin,
                                    vec2* out_normal = nullptr, f32* out_strength = nullptr) {
    if (out_strength) *out_strength = 0.0f;
    if (!field || !field->valid || field->points.empty()) return false;

    f32 query_radius = contact_radius + field->max_radius + skin;
    vec2 rel = (p - field->world_min) / field->cell_size;
    ivec2 base = ivec2(glm::floor(rel));
    i32 reach = glm::max(1, static_cast<i32>(std::ceil(query_radius / field->cell_size)));

    vec2 accum_n(0.0f);
    f32 accum_w = 0.0f;
    f32 max_pen = 0.0f;
    for (i32 oy = -reach; oy <= reach; ++oy) {
        for (i32 ox = -reach; ox <= reach; ++ox) {
            ivec2 cell = base + ivec2(ox, oy);
            if (cell.x < 0 || cell.y < 0 ||
                cell.x >= field->resolution.x || cell.y >= field->resolution.y) {
                continue;
            }
            const auto& bucket = field->cells[static_cast<size_t>(cell.x + cell.y * field->resolution.x)];
            for (u32 point_idx : bucket) {
                vec2 delta = p - field->points[point_idx];
                f32 dist2 = glm::dot(delta, delta);
                if (dist2 > query_radius * query_radius) continue;
                f32 dist = std::sqrt(glm::max(dist2, 1e-8f));
                f32 effective_radius_here = contact_radius + field->radii[point_idx];
                f32 penetration = effective_radius_here - dist;
                if (penetration <= -skin) continue;
                vec2 n = (dist2 > 1e-8f) ? (delta / dist) : preferred_normal;
                f32 w = glm::clamp((penetration + skin) / glm::max(query_radius, 1e-4f), 0.0f, 1.0f);
                accum_n += n * (0.35f + 0.65f * w);
                accum_w += (0.35f + 0.65f * w);
                max_pen = glm::max(max_pen, penetration + skin);
            }
        }
    }

    if (max_pen <= 0.0f) return false;
    vec2 n = (accum_w > 1e-6f) ? glm::normalize(accum_n) : preferred_normal;
    p += n * max_pen;
    if (out_normal) *out_normal = n;
    if (out_strength) *out_strength = glm::clamp(max_pen / glm::max(query_radius, 1e-4f), 0.0f, 1.0f);
    return true;
}

vec2 contact_filtered_target(const AttachmentPoint& a, const SDFField* sdf,
                             const FootObstacleField* obstacle_field,
                             f32* out_drive_scale = nullptr) {
    vec2 target = attachment_target_world(a);
    vec2 preferred_normal = attachment_outward_normal(a);
    f32 drive_scale = base_drive_scale_for_layer(a.layer);
    f32 sdf_strength = 0.0f;
    f32 obstacle_strength = 0.0f;
    vec2 contact_n = preferred_normal;

    for (int iter = 0; iter < 2; ++iter) {
        if (project_outside_sdf(sdf, target, scene_skin_for_layer(a.layer), &contact_n, &sdf_strength)) {
            f32 hold = (a.layer == TissueLayer::CORE) ? 0.90f :
                       (a.layer == TissueLayer::PAD)  ? 0.58f : 0.48f;
            drive_scale *= glm::mix(1.0f, hold, sdf_strength);
        }
        if (project_outside_obstacle_field(obstacle_field, target, contact_n,
                                           contact_radius_for_layer(a.layer),
                                           scene_skin_for_layer(a.layer) * 0.75f,
                                           &contact_n, &obstacle_strength)) {
            f32 hold = (a.layer == TissueLayer::CORE) ? 0.92f :
                       (a.layer == TissueLayer::PAD)  ? 0.44f : 0.34f;
            drive_scale *= glm::mix(1.0f, hold, obstacle_strength);
        }
    }

    if (out_drive_scale) *out_drive_scale = glm::clamp(drive_scale, 0.18f, 1.0f);
    return target;
}

void resolve_scene_contacts(vec2& p, vec2& v, TissueLayer layer,
                            const SDFField* sdf, const FootObstacleField* obstacle_field,
                            vec2 preferred_normal) {
    for (int iter = 0; iter < 2; ++iter) {
        vec2 n = preferred_normal;
        f32 strength = 0.0f;
        if (project_outside_sdf(sdf, p, scene_skin_for_layer(layer), &n, &strength)) {
            f32 vn = glm::dot(v, n);
            if (vn < 0.0f) v -= 1.08f * vn * n;
            v *= glm::mix(1.0f, 0.92f, strength);
        }
        if (project_outside_obstacle_field(obstacle_field, p, n,
                                           contact_radius_for_layer(layer),
                                           scene_skin_for_layer(layer) * 0.65f,
                                           &n, &strength)) {
            f32 vn = glm::dot(v, n);
            if (vn < 0.0f) v -= 1.12f * vn * n;
            v *= glm::mix(1.0f, 0.88f, strength);
        }
    }
}

void apply_foot_repair(ParticleBuffer& particles, f32 dt, const SDFField* sdf,
                       const FootObstacleField* obstacle_field) {
    if (!g_state.active || g_state.mpm_count == 0 || g_state.attachments.size() < g_state.mpm_count) return;
    dt = glm::clamp(dt, 0.0f, 1.0f / 24.0f);
    if (dt <= 0.0f) return;

    std::vector<vec2> positions(g_state.mpm_count);
    std::vector<vec2> velocities(g_state.mpm_count);
    particles.positions().download(positions.data(), positions.size() * sizeof(vec2),
                                   static_cast<size_t>(g_state.mpm_offset) * sizeof(vec2));
    particles.velocities().download(velocities.data(), velocities.size() * sizeof(vec2),
                                    static_cast<size_t>(g_state.mpm_offset) * sizeof(vec2));

    bool changed = false;
    for (u32 i = 0; i < g_state.mpm_count; ++i) {
        const AttachmentPoint& a = g_state.attachments[i];
        RepairTuning tuning = repair_tuning(a.layer);
        vec2 preferred_normal = attachment_outward_normal(a);
        f32 drive_scale = 1.0f;
        vec2 target = contact_filtered_target(a, sdf, obstacle_field, &drive_scale);
        vec2 delta = target - positions[i];
        f32 dist = glm::length(delta);
        if (dist <= tuning.soft_radius * 0.52f) {
            resolve_scene_contacts(positions[i], velocities[i], a.layer, sdf, obstacle_field, preferred_normal);
            continue;
        }

        f32 weight_scale = glm::clamp((0.72f + a.weight * 0.70f) * drive_scale, 0.18f, 1.40f);
        f32 damage_t = glm::clamp((dist - tuning.soft_radius) /
                                  glm::max(tuning.hard_radius - tuning.soft_radius, 1e-4f),
                                  0.0f, 1.0f);
        f32 recover_alpha = 1.0f - std::exp(-(tuning.recover_rate + tuning.snap_rate * damage_t) * weight_scale * dt);
        if (dist > tuning.hard_radius * 1.55f) {
            recover_alpha = glm::max(recover_alpha, 0.58f);
        }

        positions[i] = glm::mix(positions[i], target, glm::clamp(recover_alpha, 0.0f, 0.96f));

        if (std::abs(a.local.y) > 1e-4f) {
            const BoneSegment& bone = g_state.bones[a.bone_index];
            vec2 seg = bone.b - bone.a;
            f32 len = glm::length(seg);
            vec2 tangent = (len > 1e-5f) ? (seg / len) : vec2(1.0f, 0.0f);
            vec2 normal = perp(tangent);
            vec2 rest_dir = normal * glm::sign(a.local.y);
            f32 bias_alpha = (1.0f - damage_t) * tuning.outward_bias * weight_scale;
            positions[i] += rest_dir * bias_alpha;
        }

        f32 vel_decay = std::exp(-(damage_t * (tuning.velocity_damping + tuning.snap_rate * 0.18f)) * dt);
        velocities[i] *= glm::clamp(vel_decay, 0.0f, 1.0f);
        if (damage_t > 1e-4f) {
            velocities[i] += glm::normalize(delta) * (dist * (0.16f + 0.44f * damage_t) * weight_scale);
        }

        resolve_scene_contacts(positions[i], velocities[i], a.layer, sdf, obstacle_field, preferred_normal);
        changed = true;
    }

    if (changed) {
        particles.upload_positions(g_state.mpm_offset, positions.data(), g_state.mpm_count);
        particles.upload_velocities(g_state.mpm_offset, velocities.data(), g_state.mpm_count);
    }
}

void upload_batch_color(ParticleBuffer& particles, u32 offset, u32 count, vec4 color) {
    if (count == 0) return;
    std::vector<vec4> colors(count, color);
    particles.upload_colors(offset, colors.data(), count);
}

void append_attachments(const std::vector<AttachmentPoint>& src) {
    g_state.attachments.insert(g_state.attachments.end(), src.begin(), src.end());
}

void clamp_pose() {
    g_state.pose.foot_pitch = glm::clamp(g_state.pose.foot_pitch, -0.95f, 0.55f);
    g_state.pose.calf_pitch = glm::clamp(g_state.pose.calf_pitch, 1.10f, 1.95f);
    g_state.pose.heel_roll = glm::clamp(g_state.pose.heel_roll, -0.55f, 0.55f);
    g_state.pose.ball_lift = glm::clamp(g_state.pose.ball_lift, -0.35f, 0.45f);
    g_state.pose.plantar_contract = glm::clamp(g_state.pose.plantar_contract, -0.35f, 0.60f);
    g_state.pose.ankle.x = glm::clamp(g_state.pose.ankle.x, -5.2f, 5.0f);
    g_state.pose.ankle.y = glm::clamp(g_state.pose.ankle.y, -0.85f, 2.8f);
    for (int i = 0; i < 5; ++i) {
        g_state.pose.toe_curl[i] = glm::clamp(g_state.pose.toe_curl[i], -0.35f, 1.25f);
        g_state.pose.toe_tip_curl[i] = glm::clamp(g_state.pose.toe_tip_curl[i], -0.25f, 1.30f);
        g_state.pose.toe_lift[i] = glm::clamp(g_state.pose.toe_lift[i], -0.45f, 0.60f);
    }
}

void apply_focus_delta(FootControlFocus focus, vec2 delta, bool ankle_locked) {
    FootPose& pose = g_state.pose;
    switch (focus) {
    case FootControlFocus::ANKLE:
        pose.ankle += delta * 0.95f;
        pose.foot_pitch += delta.y * 0.14f;
        pose.calf_pitch -= delta.x * 0.05f;
        break;
    case FootControlFocus::HEEL:
        if (!ankle_locked) pose.ankle += vec2(delta.x * 0.08f, delta.y * 0.18f);
        pose.heel_roll += delta.y * 1.35f;
        pose.foot_pitch += delta.y * 0.70f + delta.x * 0.10f;
        break;
    case FootControlFocus::BALL:
        if (!ankle_locked) pose.ankle += vec2(delta.x * 0.04f, delta.y * 0.05f);
        pose.ball_lift += delta.y * 1.25f;
        pose.foot_pitch += delta.y * 0.30f;
        break;
    case FootControlFocus::ALL_TOES:
        for (int i = 0; i < 5; ++i) {
            pose.toe_lift[i] += delta.y * 0.86f;
            pose.toe_curl[i] += (-delta.y + delta.x * 0.22f) * 1.18f;
            pose.toe_tip_curl[i] += (-delta.y + delta.x * 0.18f) * 1.32f;
        }
        break;
    default: {
        int idx = static_cast<int>(focus) - static_cast<int>(FootControlFocus::HALLUX);
        if (idx >= 0 && idx < 5) {
            pose.toe_lift[idx] += delta.y * 1.05f;
            pose.toe_curl[idx] += (-delta.y + delta.x * 0.18f) * 1.34f;
            pose.toe_tip_curl[idx] += (-delta.y + delta.x * 0.26f) * 1.56f;
        }
        break;
    }
    }
}

void apply_focus_key_delta(FootControlFocus focus, f32 curl_delta, f32 lift_delta) {
    FootPose& pose = g_state.pose;
    if (focus == FootControlFocus::ALL_TOES) {
        for (int i = 0; i < 5; ++i) {
            pose.toe_curl[i] += curl_delta;
            pose.toe_tip_curl[i] += curl_delta * 1.12f;
            pose.toe_lift[i] += lift_delta;
        }
        return;
    }
    int idx = static_cast<int>(focus) - static_cast<int>(FootControlFocus::HALLUX);
    if (idx >= 0 && idx < 5) {
        pose.toe_curl[idx] += curl_delta;
        pose.toe_tip_curl[idx] += curl_delta * 1.18f;
        pose.toe_lift[idx] += lift_delta;
    } else if (focus == FootControlFocus::HEEL) {
        pose.heel_roll += lift_delta * 0.8f;
    } else if (focus == FootControlFocus::BALL) {
        pose.ball_lift += lift_delta * 0.8f;
    }
}

void upload_attachment_targets(ParticleBuffer& particles, MPMSolver& mpm,
                               const SDFField* sdf,
                               const FootObstacleField* obstacle_field) {
    if (!g_state.active || g_state.mpm_count == 0 || g_state.attachments.empty()) {
        mpm.clear_kinematic_targets();
        return;
    }

    const ParticleRange& range = particles.range(SolverType::MPM);
    if (range.count == 0) {
        mpm.clear_kinematic_targets();
        return;
    }

    update_pose_cache();

    g_state.scratch_targets.assign(range.count, vec2(0.0f));
    g_state.scratch_weights.assign(range.count, 0.0f);

    u32 local_base = g_state.mpm_offset - range.offset;
    for (u32 i = 0; i < g_state.mpm_count && i < g_state.attachments.size(); ++i) {
        u32 local_idx = local_base + i;
        if (local_idx >= range.count) break;
        const AttachmentPoint& a = g_state.attachments[i];
        f32 drive_scale = 1.0f;
        g_state.scratch_targets[local_idx] = contact_filtered_target(a, sdf, obstacle_field, &drive_scale);
        g_state.scratch_weights[local_idx] = glm::clamp(a.weight * drive_scale, 0.0f, 1.0f);
    }

    mpm.set_kinematic_targets(g_state.scratch_targets, g_state.scratch_weights,
                              g_state.attachment_force, g_state.attachment_damping);
}

void spawn_prop_block(MPMSolver& mpm, ParticleBuffer& particles,
                      CreationState* creation, vec2 min_corner, vec2 max_corner,
                      f32 spacing, MPMMaterial material, vec4 color,
                      f32 E, f32 nu, const char* label, const char* description) {
    mpm.params().youngs_modulus = E;
    mpm.params().poisson_ratio = nu;
    mpm.params().fiber_strength = 1.2f;
    u32 before = particles.range(SolverType::MPM).count;
    mpm.spawn_block(particles, min_corner, max_corner, spacing, material, 300.0f);
    u32 count = particles.range(SolverType::MPM).count - before;
    upload_batch_color(particles, particles.range(SolverType::MPM).offset + before, count, color);
    add_spawn_batch(creation, particles, before, label, description, material, color,
                    E, nu, vec2(1.0f, 0.0f), 0.45f, "Use medium or large chunks so the giant foot can visibly squash them.");
}

void spawn_prop_circle(MPMSolver& mpm, ParticleBuffer& particles,
                       CreationState* creation, vec2 center, f32 radius,
                       f32 spacing, MPMMaterial material, vec4 color,
                       f32 E, f32 nu, const char* label, const char* description) {
    mpm.params().youngs_modulus = E;
    mpm.params().poisson_ratio = nu;
    mpm.params().fiber_strength = 1.1f;
    u32 before = particles.range(SolverType::MPM).count;
    mpm.spawn_circle(particles, center, radius, spacing, material, 300.0f);
    u32 count = particles.range(SolverType::MPM).count - before;
    upload_batch_color(particles, particles.range(SolverType::MPM).offset + before, count, color);
    add_spawn_batch(creation, particles, before, label, description, material, color,
                    E, nu, vec2(1.0f, 0.0f), radius * 0.9f, "Large round props make the toe-roll behaviour easiest to read.");
}

vec4 foot_demo_span_color(i32 span_hint, vec4 fallback) {
    switch (span_hint) {
    case 1: return vec4(0.28f, 0.54f, 0.94f, 1.0f); // thin / mostly one lane
    case 2: return vec4(0.36f, 0.88f, 0.72f, 1.0f); // two adjacent lanes
    case 3: return vec4(0.98f, 0.74f, 0.34f, 1.0f); // broad multi-lane bulk
    case 4: return vec4(0.96f, 0.92f, 0.58f, 1.0f); // effectively all lanes / floor-like
    default: return fallback;
    }
}

} // namespace

bool foot_demo_active() {
    return g_state.active;
}

void clear_foot_demo() {
    g_state = FootDemoState{};
}

void load_foot_demo_scene(ParticleBuffer& particles, SPHSolver& sph,
                          MPMSolver& mpm, UniformGrid& grid, SDFField& sdf,
                          CreationState* creation) {
    (void)sph;
    (void)grid;
    clear_foot_demo();
    g_state.active = true;
    g_state.focus = FootControlFocus::ANKLE;

    sdf.clear();
    sdf.add_segment(vec2(-6.0f, -1.65f), vec2(6.5f, -1.65f), 0.18f,
                    SDFField::MaterialPresetID::BRONZE_BALANCED,
                    "Street Floor",
                    "Wide rigid floor for the giant-foot interaction bench.");
    sdf.rebuild();

    mpm.params().enable_thermal = false;
    mpm.params().youngs_modulus = 22000.0f;
    mpm.params().poisson_ratio = 0.35f;
    mpm.params().fiber_strength = 1.8f;

    spawn_prop_block(mpm, particles, creation,
                     vec2(-0.15f, -1.48f), vec2(0.58f, -0.92f),
                     0.055f, MPMMaterial::FOAM, foot_demo_span_color(2, vec4(0.78f, 0.82f, 0.90f, 1.0f)),
                     9000.0f, 0.38f,
                     "Foam Parking Block [new]",
                     "Soft sacrificial block for reading heel and toe pressure. In the layer-band view, this is treated like a shallow two-lane object.");
    spawn_prop_block(mpm, particles, creation,
                     vec2(1.55f, -1.48f), vec2(2.35f, 1.00f),
                     0.055f, MPMMaterial::BRITTLE, foot_demo_span_color(3, vec4(0.78f, 0.74f, 0.66f, 1.0f)),
                     52000.0f, 0.24f,
                     "Brittle Tower [new]",
                     "Breakable masonry-style tower for giant-foot collapse tests. In the layer-band view, this reads as a deeper multi-lane bulk target.");
    spawn_prop_block(mpm, particles, creation,
                     vec2(3.25f, -1.34f), vec2(4.52f, -0.84f),
                     0.055f, MPMMaterial::TOUGH, foot_demo_span_color(2, vec4(0.46f, 0.22f, 0.18f, 1.0f)),
                     42000.0f, 0.31f,
                     "Car Body [new]",
                     "Tough proxy car body that should crumple under the forefoot. In the layer-band view, this is treated like a mid-depth two-lane object.");
    spawn_prop_block(mpm, particles, creation,
                     vec2(3.54f, -0.84f), vec2(4.18f, -0.48f),
                     0.055f, MPMMaterial::TOUGH, foot_demo_span_color(2, vec4(0.54f, 0.28f, 0.24f, 1.0f)),
                     40000.0f, 0.31f,
                     "Car Roof [new]",
                     "Secondary car roof shell for layered crushing. In the layer-band view, this is treated like another two-lane plate.");
    spawn_prop_circle(mpm, particles, creation,
                      vec2(3.56f, -1.36f), 0.19f, 0.052f,
                      MPMMaterial::ELASTIC, foot_demo_span_color(1, vec4(0.18f, 0.18f, 0.20f, 1.0f)),
                      18000.0f, 0.38f,
                      "Front Tire [new]",
                      "Elastic wheel proxy for toe-roll and smear tests. In the layer-band view, this is treated like a thin one-lane obstacle.");
    spawn_prop_circle(mpm, particles, creation,
                      vec2(4.22f, -1.36f), 0.19f, 0.052f,
                      MPMMaterial::ELASTIC, foot_demo_span_color(1, vec4(0.18f, 0.18f, 0.20f, 1.0f)),
                      18000.0f, 0.38f,
                      "Rear Tire [new]",
                      "Elastic wheel proxy for toe-roll and smear tests. In the layer-band view, this is treated like a thin one-lane obstacle.");

    std::vector<vec2> core_positions;
    std::vector<f32> core_shell;
    std::vector<AttachmentPoint> core_attach;
    std::vector<vec2> pad_positions;
    std::vector<f32> pad_shell;
    std::vector<AttachmentPoint> pad_attach;
    std::vector<vec2> skin_positions;
    std::vector<f32> skin_shell;
    std::vector<AttachmentPoint> skin_attach;
    build_foot_particles(core_positions, core_shell, core_attach,
                         pad_positions, pad_shell, pad_attach,
                         skin_positions, skin_shell, skin_attach);

    mpm.params().youngs_modulus = 64000.0f;
    mpm.params().poisson_ratio = 0.34f;
    mpm.params().fiber_strength = 2.3f;
    u32 core_before = particles.range(SolverType::MPM).count;
    mpm.spawn_points(particles, core_positions, core_shell, kFootSpacing,
                     MPMMaterial::TOUGH, 300.0f, vec2(1.0f, 0.0f), 1.08f);
    u32 core_count = particles.range(SolverType::MPM).count - core_before;
    upload_batch_color(particles, particles.range(SolverType::MPM).offset + core_before,
                       core_count, vec4(0.80f, 0.57f, 0.50f, 1.0f));
    add_spawn_batch(creation, particles, core_before,
                    "Deep Support Tissue [new][experimental]",
                    "Dense support tissue for the calf, arch, and toe shafts. This is the structural layer that keeps the giant foot from tearing apart under heavy presses.",
                    MPMMaterial::TOUGH, vec4(0.80f, 0.57f, 0.50f, 1.0f),
                    64000.0f, 0.34f, vec2(1.0f, 0.0f),
                    1.30f, "This core layer is authored as part of the giant foot and should stay fairly coherent at showcase size.");

    mpm.params().youngs_modulus = 26000.0f;
    mpm.params().poisson_ratio = 0.41f;
    mpm.params().fiber_strength = 1.5f;
    u32 pad_before = particles.range(SolverType::MPM).count;
    mpm.spawn_points(particles, pad_positions, pad_shell, kFootSpacing,
                     MPMMaterial::ELASTIC, 300.0f, vec2(1.0f, 0.0f), 1.00f);
    u32 pad_count = particles.range(SolverType::MPM).count - pad_before;
    upload_batch_color(particles, particles.range(SolverType::MPM).offset + pad_before,
                       pad_count, vec4(0.95f, 0.77f, 0.68f, 1.0f));
    add_spawn_batch(creation, particles, pad_before,
                    "Plantar Fat Pad [new][experimental]",
                    "Softer plantar and toe-pad layer. This is the cushion that should squash over props while staying attached to the deeper support tissue.",
                    MPMMaterial::ELASTIC, vec4(0.95f, 0.77f, 0.68f, 1.0f),
                    26000.0f, 0.41f, vec2(1.0f, 0.0f),
                    1.30f, "The plantar pads are already authored at the intended demo scale.");

    mpm.params().youngs_modulus = 52000.0f;
    mpm.params().poisson_ratio = 0.32f;
    mpm.params().fiber_strength = 2.7f;
    u32 skin_before = particles.range(SolverType::MPM).count;
    mpm.spawn_points(particles, skin_positions, skin_shell, kFootSpacing,
                     MPMMaterial::TOUGH, 300.0f, vec2(1.0f, 0.0f), 1.02f);
    u32 skin_count = particles.range(SolverType::MPM).count - skin_before;
    upload_batch_color(particles, particles.range(SolverType::MPM).offset + skin_before,
                       skin_count, vec4(0.98f, 0.90f, 0.86f, 1.0f));
    add_spawn_batch(creation, particles, skin_before,
                    "Foot Skin Shell [new][experimental]",
                    "Outer giant-foot skin shell. It is firmer than the pad layer so the silhouette and toe shapes stay readable even before the skin renderer is applied.",
                    MPMMaterial::TOUGH, vec4(0.98f, 0.90f, 0.86f, 1.0f),
                    52000.0f, 0.32f, vec2(1.0f, 0.0f),
                    1.30f, "This shell is authored as part of the giant foot itself.");

    g_state.mpm_offset = particles.range(SolverType::MPM).offset + core_before;
    g_state.mpm_count = core_count + pad_count + skin_count;
    g_state.attachments.reserve(g_state.mpm_count);
    append_attachments(core_attach);
    append_attachments(pad_attach);
    append_attachments(skin_attach);

    FootObstacleField obstacle_field = build_foot_obstacle_field(particles, &sdf);
    upload_attachment_targets(particles, mpm, &sdf, &obstacle_field);
    LOG_INFO("Foot demo scene loaded: %u foot particles", g_state.mpm_count);
}

void update_foot_demo(const FootControlInput& input,
                      ParticleBuffer& particles, MPMSolver& mpm,
                      const SDFField* sdf) {
    if (!g_state.active) {
        mpm.clear_kinematic_targets();
        return;
    }

    if (input.cycle_prev_pressed) {
        i32 idx = static_cast<i32>(g_state.focus);
        idx = (idx + static_cast<i32>(FootControlFocus::COUNT) - 1) % static_cast<i32>(FootControlFocus::COUNT);
        g_state.focus = static_cast<FootControlFocus>(idx);
    }
    if (input.cycle_next_pressed) {
        i32 idx = static_cast<i32>(g_state.focus);
        idx = (idx + 1) % static_cast<i32>(FootControlFocus::COUNT);
        g_state.focus = static_cast<FootControlFocus>(idx);
    }

    f32 dt = glm::clamp(input.dt, 0.0f, 0.05f);
    if (input.curl_down) apply_focus_key_delta(g_state.focus, 1.15f * dt, 0.0f);
    if (input.straighten_down) apply_focus_key_delta(g_state.focus, -1.15f * dt, 0.0f);
    if (std::abs(input.wheel_delta) > 1e-5f) apply_focus_key_delta(g_state.focus, input.wheel_delta * 0.09f, input.wheel_delta * 0.03f);
    if (input.contract_down) g_state.pose.plantar_contract += 0.72f * dt;
    if (input.extend_down) g_state.pose.plantar_contract -= 0.72f * dt;

    if (input.rmb_down && !g_state.ankle_pinned) {
        g_state.ankle_pin = g_state.pose.ankle;
        g_state.ankle_pinned = true;
    } else if (!input.rmb_down) {
        g_state.ankle_pinned = false;
    }

    if (input.lmb_down) {
        apply_focus_delta(g_state.focus, input.mouse_delta_world, g_state.ankle_pinned && g_state.focus != FootControlFocus::ANKLE);
    }

    if (g_state.ankle_pinned && g_state.focus != FootControlFocus::ANKLE) {
        g_state.pose.ankle = g_state.ankle_pin;
    }

    clamp_pose();
    FootObstacleField obstacle_field = build_foot_obstacle_field(particles, sdf);
    upload_attachment_targets(particles, mpm, sdf, &obstacle_field);
    apply_foot_repair(particles, dt, sdf, &obstacle_field);
}

FootControlFocus foot_demo_focus() {
    return g_state.focus;
}

void set_foot_demo_focus(FootControlFocus focus) {
    g_state.focus = focus;
}

const char* foot_demo_focus_name(FootControlFocus focus) {
    switch (focus) {
    case FootControlFocus::ANKLE: return "Ankle / Heel Root";
    case FootControlFocus::HEEL: return "Heel Pad";
    case FootControlFocus::BALL: return "Ball of Foot";
    case FootControlFocus::ALL_TOES: return "All Toes";
    case FootControlFocus::HALLUX: return "Big Toe";
    case FootControlFocus::TOE_2: return "Toe 2";
    case FootControlFocus::TOE_3: return "Toe 3";
    case FootControlFocus::TOE_4: return "Toe 4";
    case FootControlFocus::PINKY: return "Pinky Toe";
    default: return "Foot Focus";
    }
}

const char* foot_demo_focus_name() {
    return foot_demo_focus_name(g_state.focus);
}

const char* foot_demo_mode_hint() {
    return "LMB drag moves the focused anatomy. Hold RMB to pin the ankle. [ and ] cycle focus. Z/S curl and straighten. C/B contract and extend.";
}

vec2 foot_demo_focus_point() {
    if (!g_state.active) return vec2(0.0f);
    update_pose_cache();
    switch (g_state.focus) {
    case FootControlFocus::ANKLE: return g_state.pose.ankle;
    case FootControlFocus::HEEL: return g_state.heel_point;
    case FootControlFocus::BALL: return g_state.ball_point;
    case FootControlFocus::ALL_TOES:
        return (g_state.toe_tips[0] + g_state.toe_tips[1] + g_state.toe_tips[2] + g_state.toe_tips[3] + g_state.toe_tips[4]) / 5.0f;
    case FootControlFocus::HALLUX: return g_state.toe_tips[0];
    case FootControlFocus::TOE_2: return g_state.toe_tips[1];
    case FootControlFocus::TOE_3: return g_state.toe_tips[2];
    case FootControlFocus::TOE_4: return g_state.toe_tips[3];
    case FootControlFocus::PINKY: return g_state.toe_tips[4];
    default: return g_state.pose.ankle;
    }
}

f32 foot_demo_focus_radius() {
    switch (g_state.focus) {
    case FootControlFocus::ANKLE: return 0.22f;
    case FootControlFocus::HEEL: return 0.24f;
    case FootControlFocus::BALL: return 0.22f;
    case FootControlFocus::ALL_TOES: return 0.40f;
    default: return 0.16f;
    }
}

void foot_demo_apply_scene_defaults(bool& mpm_skin_enabled,
                                    int& mpm_surface_style,
                                    float& mpm_skin_threshold,
                                    float& mpm_skin_kernel) {
    mpm_skin_enabled = true;
    mpm_surface_style = 7; // Soft Fill
    mpm_skin_threshold = 0.34f;
    mpm_skin_kernel = 2.55f;
}

} // namespace ng
