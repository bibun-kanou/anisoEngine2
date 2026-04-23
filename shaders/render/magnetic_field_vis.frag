#version 450 core

in vec2 v_uv;
out vec4 frag_color;

layout(binding = 7) uniform sampler2D u_field_tex;

uniform int u_use_real_field;
uniform vec2 u_visible_world_min;
uniform vec2 u_visible_world_max;
uniform vec2 u_field_world_min;
uniform vec2 u_field_world_max;
uniform vec2 u_brush_pos;
uniform float u_brush_inner_radius;
uniform float u_brush_outer_radius;
uniform float u_brush_strength;
uniform float u_brush_spike_strength;
uniform float u_brush_spike_freq;
uniform float u_time;
uniform float u_overlay_alpha;
uniform float u_exposure;

vec3 hsv_to_rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

vec4 sample_real_field(vec2 world) {
    vec2 uv = (world - u_field_world_min) / (u_field_world_max - u_field_world_min);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return vec4(0.0);
    return texture(u_field_tex, uv);
}

vec4 sample_brush_field(vec2 world) {
    vec2 to_magnet = u_brush_pos - world;
    float dist = length(to_magnet);
    float inner_radius = max(u_brush_inner_radius, 1e-4);
    float outer_radius = max(u_brush_outer_radius, inner_radius);
    if (dist >= outer_radius || dist <= 1e-5 || abs(u_brush_strength) <= 1e-5) return vec4(0.0);

    vec2 shaped = vec2(to_magnet.x * 0.28,
                       sign(to_magnet.y) * max(abs(to_magnet.y), inner_radius * 0.22));
    vec2 dir = normalize(shaped);
    vec2 tangent = vec2(-dir.y, dir.x);
    float axial = dot(to_magnet, dir);
    float lateral = abs(dot(to_magnet, tangent));
    float axial_band = clamp((axial + inner_radius * 0.18) / (outer_radius + inner_radius * 0.18), 0.0, 1.0);
    float lateral_band = clamp(1.0 - lateral / (outer_radius * 0.92 + 1e-4), 0.0, 1.0);
    float brush_band = 1.0;
    if (dist > inner_radius && outer_radius > inner_radius + 1e-4) {
        float t = clamp((dist - inner_radius) / (outer_radius - inner_radius), 0.0, 1.0);
        brush_band = 1.0 - t * t * (3.0 - 2.0 * t);
    }

    float falloff = brush_band * axial_band * lateral_band * lateral_band;
    float stripe_phase = dot(world - u_brush_pos, tangent) * u_brush_spike_freq;
    float stripe = pow(clamp(0.5 + 0.5 * cos(stripe_phase), 0.0, 1.0), 5.0);
    float ridge_pull = -sin(stripe_phase);
    float spike_gain = clamp(u_brush_spike_strength, 0.0, 4.0);

    vec2 drive = dir * (u_brush_strength * falloff * (0.38 + 0.72 * stripe * spike_gain));
    drive += tangent * ridge_pull * (6.0 * spike_gain) * falloff;
    float mag = length(drive);
    return vec4(drive, mag, mag * mag);
}

void main() {
    vec2 world = mix(u_visible_world_min, u_visible_world_max, v_uv);
    vec4 field = (u_use_real_field != 0) ? sample_real_field(world) : sample_brush_field(world);
    // Apply exposure BEFORE the early-out so subtle far-field regions that
    // would otherwise round to zero still show up when exposure is cranked.
    float exposure = max(u_exposure, 0.001);
    vec2 H = field.xy * exposure;
    float mag = field.z * exposure;
    if (mag <= 1e-5) {
        frag_color = vec4(0.0);
        return;
    }

    vec2 dir = H / mag;
    vec2 tangent = vec2(-dir.y, dir.x);
    float hue = atan(dir.y, dir.x) / 6.2831853 + 0.5;
    float mag_norm = 1.0 - exp(-mag * 0.09);

    float lane = dot(world - u_brush_pos, tangent) * 13.0;
    float along = dot(world - u_brush_pos, dir) * 9.0;
    float streak = 0.5 + 0.5 * sin(lane + along * 0.25 - u_time * 1.3);
    streak = pow(streak, 2.8);
    float ripple = 0.5 + 0.5 * cos(along - u_time * 0.8 + mag * 0.05);
    float fil = clamp(streak * (0.55 + 0.45 * ripple), 0.0, 1.0);

    vec3 base = hsv_to_rgb(vec3(fract(hue), 0.78, 0.18 + 0.82 * pow(mag_norm, 0.58)));
    vec3 glow = mix(base * 0.45, base * 1.15, fil);
    float alpha = u_overlay_alpha * clamp(0.12 + 0.68 * pow(mag_norm, 0.75) + 0.18 * fil, 0.0, 0.95);
    frag_color = vec4(glow, alpha);
}
