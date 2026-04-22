#version 450 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_sdf_tex;
uniform usampler2D u_sdf_object_id_tex;
uniform usampler2D u_sdf_palette_tex;
uniform sampler2D u_smoke_tex;
uniform sampler2D u_air_temp_tex;
uniform sampler2D u_air_vel_tex;
uniform sampler2D u_air_bio_tex;
uniform sampler2D u_air_automata_tex;
uniform float u_air_bio_gain;
uniform float u_air_automata_gain;
uniform mat4 u_view_proj;
uniform vec2 u_sdf_world_min;
uniform vec2 u_sdf_world_max;
uniform vec2 u_viewport_size;
uniform int u_show_air;   // 0=off, 1=smoke+heat, 2=temperature, 3=smoke, 4=velocity, 5=vel+smoke, 6=curl, 7=divergence, 8=bio field, 9=automata field, 10=bio drive, 11=automata drive
uniform int u_metal_palette; // 0=silver, 1=rose gold, 2=bronze
uniform float u_fire_temp_start;
uniform float u_fire_temp_range;
uniform float u_fire_softness;
uniform int u_selected_object_id;

// Convert screen UV to world position using inverse view-proj
vec2 screen_to_world(vec2 uv) {
    vec4 clip = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
    // We need the inverse VP, but simpler: we'll pass camera params
    // For now use inverse directly
    mat4 inv_vp = inverse(u_view_proj);
    vec4 world = inv_vp * clip;
    return world.xy;
}

vec3 red_hot_metal(float t) {
    t = clamp(t, 0.0, 1.0);
    if (t < 0.25) return mix(vec3(0.18, 0.02, 0.02), vec3(0.45, 0.03, 0.02), t / 0.25);
    if (t < 0.55) return mix(vec3(0.45, 0.03, 0.02), vec3(0.78, 0.08, 0.03), (t - 0.25) / 0.30);
    if (t < 0.82) return mix(vec3(0.78, 0.08, 0.03), vec3(0.95, 0.28, 0.06), (t - 0.55) / 0.27);
    return mix(vec3(0.95, 0.28, 0.06), vec3(1.0, 0.72, 0.32), (t - 0.82) / 0.18);
}

vec3 metal_base_color(int palette, float depth) {
    vec3 light_c = vec3(0.66, 0.70, 0.76);
    vec3 dark_c = vec3(0.36, 0.40, 0.46);
    if (palette == 1) {
        light_c = vec3(0.76, 0.62, 0.58);
        dark_c = vec3(0.40, 0.27, 0.25);
    } else if (palette == 2) {
        light_c = vec3(0.72, 0.58, 0.42);
        dark_c = vec3(0.34, 0.22, 0.15);
    } else if (palette == 3) {
        light_c = vec3(0.79, 0.67, 0.34);
        dark_c = vec3(0.40, 0.29, 0.12);
    }
    return mix(light_c, dark_c, depth);
}

vec3 metal_rim_color(int palette) {
    if (palette == 1) return vec3(0.98, 0.86, 0.80);
    if (palette == 2) return vec3(0.95, 0.84, 0.68);
    if (palette == 3) return vec3(0.98, 0.90, 0.62);
    return vec3(0.92, 0.95, 1.0);
}

float flame_noise(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float smooth_flame_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = flame_noise(i);
    float b = flame_noise(i + vec2(1.0, 0.0));
    float c = flame_noise(i + vec2(0.0, 1.0));
    float d = flame_noise(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

vec3 drive_debug_color(vec2 grad, float gain, float scale) {
    vec2 drive = grad * gain;
    float mag = clamp(length(drive) * scale, 0.0, 1.0);
    vec2 dir = mag > 1e-5 ? normalize(drive) : vec2(0.0);
    return vec3(0.5 + 0.5 * dir.x, 0.5 + 0.5 * dir.y, mag);
}

void main() {
    // Map screen position to world position
    vec2 world_pos = screen_to_world(v_uv);

    // Map world position to SDF UV
    vec2 sdf_uv = (world_pos - u_sdf_world_min) / (u_sdf_world_max - u_sdf_world_min);

    if (sdf_uv.x < 0.0 || sdf_uv.x > 1.0 || sdf_uv.y < 0.0 || sdf_uv.y > 1.0) {
        // Outside SDF bounds: show dark border
        frag_color = vec4(0.01, 0.01, 0.02, 1.0);
        return;
    }

    float d = texture(u_sdf_tex, sdf_uv).r;
    uint object_id = texture(u_sdf_object_id_tex, sdf_uv).r;
    uint palette_code = texture(u_sdf_palette_tex, sdf_uv).r;
    int local_palette = (palette_code == 0u) ? u_metal_palette : int(palette_code - 1u);

    // Compute gradient for surface normal (for lighting)
    vec2 texel = 1.0 / vec2(textureSize(u_sdf_tex, 0));
    float dx = texture(u_sdf_tex, sdf_uv + vec2(texel.x, 0.0)).r
             - texture(u_sdf_tex, sdf_uv - vec2(texel.x, 0.0)).r;
    float dy = texture(u_sdf_tex, sdf_uv + vec2(0.0, texel.y)).r
             - texture(u_sdf_tex, sdf_uv - vec2(0.0, texel.y)).r;
    vec2 normal = normalize(vec2(dx, dy) + 1e-6);

    vec3 color;

    if (d < 0.0) {
        // Inside solid: brushed metal with selectable warm/cool palette
        float depth = clamp(-d * 10.0, 0.0, 1.0);
        vec3 surface_color = metal_base_color(local_palette, depth);

        // Cool top-right key light + a softer fill to feel more metallic.
        float key = max(dot(normal, normalize(vec2(0.55, 1.0))), 0.0);
        float fill = max(dot(normal, normalize(vec2(-0.75, 0.35))), 0.0);
        float light = 0.22 + 0.68 * key + 0.18 * fill;
        color = surface_color * light;

        // Polished rim highlight near surface
        float edge = smoothstep(-0.02, 0.0, d);
        color = mix(color, metal_rim_color(local_palette), edge * 0.45);

        if (u_selected_object_id > 0 && object_id == uint(u_selected_object_id)) {
            float highlight = 0.35 + 0.65 * smoothstep(-0.06, 0.0, d);
            vec3 select_c = vec3(1.0, 0.86, 0.34);
            color = mix(color, select_c, highlight * 0.45);
            color += select_c * 0.18 * highlight;
        }
    } else {
        // Outside solid: dark background with subtle distance fog
        float glow = exp(-d * 8.0) * 0.08; // Subtle glow near surfaces
        color = vec3(0.02, 0.02, 0.04) + vec3(0.15, 0.1, 0.05) * glow;

        // Grid lines for spatial reference
        vec2 grid = abs(fract(world_pos * 2.0) - 0.5);
        float grid_line = 1.0 - smoothstep(0.0, 0.03, min(grid.x, grid.y));
        color += vec3(0.03) * grid_line;
    }

    // Eulerian air visualization
    if (u_show_air > 0) {
        float smoke_val = texture(u_smoke_tex, sdf_uv).r;
        float air_t = texture(u_air_temp_tex, sdf_uv).r;
        // Smooth heat intensity (no hard boundary)
        float heat_raw = max(air_t - u_fire_temp_start, 0.0) / max(u_fire_temp_range, 1.0);
        float jitter = (smooth_flame_noise(world_pos * 6.0 + vec2(air_t * 0.004, air_t * 0.0025)) - 0.5) * 0.025;
        heat_raw = max(heat_raw + jitter * clamp(heat_raw * 1.4, 0.0, 1.0), 0.0);
        float heat = 1.0 - exp(-heat_raw * max(u_fire_softness, 0.01)); // Exponential ramp: smooth onset, saturates

        if (u_show_air == 1) {
            // Default: smoke + heat + fire
            // Smoke
            if (d >= 0.0 && smoke_val > 0.005) {
                float s = clamp(smoke_val, 0.0, 2.0);
                vec3 smoke_c = mix(vec3(0.2,0.15,0.12), vec3(0.5,0.45,0.4), s*0.5);
                color = mix(color, smoke_c, clamp(s * 0.5, 0.0, 0.7));
            }
            if (d < 0.0) {
                if (heat > 0.001) {
                    float metal_heat = clamp((air_t - 360.0) / 700.0, 0.0, 1.0);
                    vec3 hot_c = red_hot_metal(metal_heat);
                    float glow = smoothstep(0.01, 0.8, heat);
                    color = mix(color, hot_c, glow * 0.78);
                    color += hot_c * glow * 0.35;
                }
            } else if (heat > 0.001) {
                // Fire: smooth gradient, no hard edge
                vec3 fire_c;
                if (heat < 0.1) fire_c = vec3(0.4, 0.05, 0.0) * heat / 0.1;
                else if (heat < 0.3) fire_c = mix(vec3(0.5,0.15,0.0), vec3(0.9,0.4,0.0), (heat-0.1)/0.2);
                else if (heat < 0.6) fire_c = mix(vec3(0.9,0.4,0.0), vec3(1.0,0.7,0.1), (heat-0.3)/0.3);
                else fire_c = mix(vec3(1.0,0.7,0.1), vec3(1.0,0.95,0.5), (heat-0.6)/0.4);
                color += fire_c * heat * 1.5;
            }
        } else if (u_show_air == 2) {
            // Temperature field — always visible, even at ambient
            // Show any deviation from 300K as color
            float t_diff = air_t - 300.0; // Deviation from room temp
            if (abs(t_diff) > 1.0) {
                vec3 tc;
                if (d < 0.0 && t_diff >= 0.0) {
                    float hot = clamp(t_diff / 700.0, 0.0, 1.0);
                    tc = red_hot_metal(hot);
                } else if (t_diff < 0.0) {
                    // Cold: blue
                    float cold = clamp(-t_diff / 200.0, 0.0, 1.0);
                    tc = mix(vec3(0.1,0.1,0.2), vec3(0.0,0.2,0.8), cold);
                } else {
                    // Hot: yellow → orange → red
                    float hot = clamp(t_diff / 500.0, 0.0, 1.0);
                    if (hot < 0.3) tc = mix(vec3(0.2,0.2,0.0), vec3(0.8,0.6,0.0), hot/0.3);
                    else if (hot < 0.6) tc = mix(vec3(0.8,0.6,0.0), vec3(1.0,0.4,0.0), (hot-0.3)/0.3);
                    else tc = mix(vec3(1.0,0.4,0.0), vec3(1.0,0.1,0.0), (hot-0.6)/0.4);
                }
                color = mix(color, tc, clamp(abs(t_diff) / 100.0, 0.1, 0.7));
            }
        } else if (u_show_air == 3) {
            // Smoke density (greyscale)
            float s = clamp(smoke_val * 0.5, 0.0, 1.0);
            color = mix(color, vec3(s), 0.6);
        } else if (u_show_air == 4 || u_show_air == 5) {
            // Velocity field
            vec2 vel = texture(u_air_vel_tex, sdf_uv).rg;
            float speed = length(vel);
            if (speed > 0.001) {
                // Direction: hue, magnitude: brightness
                float angle = atan(vel.y, vel.x);
                float hue = (angle + 3.14159) / 6.28318;
                // HSV to RGB
                float h6 = hue * 6.0;
                float r = clamp(abs(h6 - 3.0) - 1.0, 0.0, 1.0);
                float g = clamp(2.0 - abs(h6 - 2.0), 0.0, 1.0);
                float b = clamp(2.0 - abs(h6 - 4.0), 0.0, 1.0);
                vec3 vel_color = vec3(r, g, b) * clamp(speed * 2.0, 0.0, 1.0);
                color = mix(color, vel_color, 0.6);
            }
            // Also show smoke in mode 5
            if (u_show_air == 5 && smoke_val > 0.005) {
                float s = clamp(smoke_val, 0.0, 2.0);
                vec3 smoke_c = mix(vec3(0.2,0.15,0.12), vec3(0.5,0.45,0.4), s*0.5);
                color = mix(color, smoke_c, clamp(s * 0.3, 0.0, 0.5));
            }
        } else if (u_show_air == 6) {
            // Curl/vorticity: compute from velocity texture
            vec2 texel = 1.0 / vec2(textureSize(u_air_vel_tex, 0));
            float vx_u = texture(u_air_vel_tex, sdf_uv + vec2(0, texel.y)).r;
            float vx_d = texture(u_air_vel_tex, sdf_uv - vec2(0, texel.y)).r;
            float vy_r = texture(u_air_vel_tex, sdf_uv + vec2(texel.x, 0)).g;
            float vy_l = texture(u_air_vel_tex, sdf_uv - vec2(texel.x, 0)).g;
            float curl_val = (vy_r - vy_l) - (vx_u - vx_d);
            // Blue=CW, Red=CCW
            float cn = clamp(curl_val * 20.0, -1.0, 1.0);
            vec3 curl_c;
            if (cn < 0.0) curl_c = mix(vec3(0.1), vec3(0.1, 0.3, 0.9), -cn);
            else curl_c = mix(vec3(0.1), vec3(0.9, 0.2, 0.1), cn);
            color = mix(color, curl_c, 0.6);
        } else if (u_show_air == 7) {
            // Divergence: compute from velocity texture
            vec2 texel = 1.0 / vec2(textureSize(u_air_vel_tex, 0));
            float vx_r = texture(u_air_vel_tex, sdf_uv + vec2(texel.x, 0)).r;
            float vx_l = texture(u_air_vel_tex, sdf_uv - vec2(texel.x, 0)).r;
            float vy_u = texture(u_air_vel_tex, sdf_uv + vec2(0, texel.y)).g;
            float vy_d = texture(u_air_vel_tex, sdf_uv - vec2(0, texel.y)).g;
            float div_val = (vx_r - vx_l) + (vy_u - vy_d);
            // Blue=converging, Red=diverging
            float dn = clamp(div_val * 30.0, -1.0, 1.0);
            vec3 div_c;
            if (dn < 0.0) div_c = mix(vec3(0.1), vec3(0.1, 0.3, 0.9), -dn);
            else div_c = mix(vec3(0.1), vec3(0.9, 0.2, 0.1), dn);
            color = mix(color, div_c, 0.6);
        } else if (u_show_air == 8) {
            float bio_raw = texture(u_air_bio_tex, sdf_uv).r;
            float bio_trace = clamp(pow(smoothstep(0.00008, 0.010, bio_raw * u_air_bio_gain), 0.72) * 0.46, 0.0, 1.0);
            float bio = clamp(pow(smoothstep(0.0007, 0.18, bio_raw * u_air_bio_gain), 0.46) * 1.92, 0.0, 1.0);
            if (bio > 0.001 || bio_trace > 0.001) {
                vec3 bio_c;
                if (bio < 0.33) bio_c = mix(vec3(0.04, 0.10, 0.20), vec3(0.10, 0.34, 0.78), bio / 0.33);
                else if (bio < 0.66) bio_c = mix(vec3(0.10, 0.34, 0.78), vec3(0.26, 0.84, 0.60), (bio - 0.33) / 0.33);
                else bio_c = mix(vec3(0.26, 0.84, 0.60), vec3(0.96, 0.94, 0.70), (bio - 0.66) / 0.34);
                float rim = smoothstep(0.18, 0.78, bio) - smoothstep(0.74, 1.0, bio);
                vec3 trace_c = mix(vec3(0.03, 0.08, 0.18), vec3(0.10, 0.40, 0.92), bio_trace);
                color = mix(color, trace_c, clamp(bio_trace * 0.34, 0.0, 0.26));
                color = mix(color, bio_c, clamp(bio * 0.82 + 0.20, 0.0, 0.92));
                color += bio_c * (bio * 0.10 + rim * 0.18);
            }
        } else if (u_show_air == 9) {
            float colony_raw = texture(u_air_automata_tex, sdf_uv).r;
            float colony = clamp(pow(smoothstep(0.004, 0.24, colony_raw * u_air_automata_gain), 0.62) * 1.38, 0.0, 1.0);
            if (colony > 0.001) {
                vec3 colony_c;
                if (colony < 0.25) colony_c = mix(vec3(0.04, 0.06, 0.12), vec3(0.18, 0.32, 0.78), colony / 0.25);
                else if (colony < 0.55) colony_c = mix(vec3(0.18, 0.32, 0.78), vec3(0.26, 0.82, 0.64), (colony - 0.25) / 0.30);
                else if (colony < 0.82) colony_c = mix(vec3(0.26, 0.82, 0.64), vec3(0.96, 0.86, 0.54), (colony - 0.55) / 0.27);
                else colony_c = mix(vec3(0.96, 0.86, 0.54), vec3(1.0, 0.96, 0.86), (colony - 0.82) / 0.18);
                float edge = smoothstep(0.18, 0.82, colony) - smoothstep(0.82, 0.98, colony);
                color = mix(color, colony_c, clamp(colony * 0.84 + 0.08, 0.0, 0.88));
                color += colony_c * edge * 0.24;
            }
        } else if (u_show_air == 10 || u_show_air == 11) {
            vec2 grad;
            vec3 drive_c;
            if (u_show_air == 10) {
                vec2 drive_texel = 1.0 / vec2(textureSize(u_air_bio_tex, 0));
                grad = vec2(
                    texture(u_air_bio_tex, sdf_uv + vec2(drive_texel.x, 0.0)).r - texture(u_air_bio_tex, sdf_uv - vec2(drive_texel.x, 0.0)).r,
                    texture(u_air_bio_tex, sdf_uv + vec2(0.0, drive_texel.y)).r - texture(u_air_bio_tex, sdf_uv - vec2(0.0, drive_texel.y)).r
                );
                drive_c = drive_debug_color(grad, u_air_bio_gain * 1.35, 22.0);
            } else {
                vec2 drive_texel = 1.0 / vec2(textureSize(u_air_automata_tex, 0));
                grad = vec2(
                    texture(u_air_automata_tex, sdf_uv + vec2(drive_texel.x, 0.0)).r - texture(u_air_automata_tex, sdf_uv - vec2(drive_texel.x, 0.0)).r,
                    texture(u_air_automata_tex, sdf_uv + vec2(0.0, drive_texel.y)).r - texture(u_air_automata_tex, sdf_uv - vec2(0.0, drive_texel.y)).r
                );
                drive_c = drive_debug_color(grad, u_air_automata_gain, 12.0);
            }
            float strength = clamp(drive_c.b, 0.0, 1.0);
            float mask = clamp(strength * 0.94 + 0.18, 0.0, 0.96);
            color = mix(color, drive_c, mask);
            color += drive_c * strength * 0.16 + vec3(0.08, 0.10, 0.16) * strength;
        }
    }

    frag_color = vec4(color, 1.0);
}
