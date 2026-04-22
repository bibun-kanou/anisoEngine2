// 2D Signed Distance Field primitives and operations

float sdf_circle(vec2 p, vec2 center, float radius) {
    return length(p - center) - radius;
}

float sdf_box(vec2 p, vec2 center, vec2 half_ext) {
    vec2 d = abs(p - center) - half_ext;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float sdf_segment(vec2 p, vec2 a, vec2 b, float thickness) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float t = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * t) - thickness;
}

// Boolean operations
float sdf_union(float a, float b) { return min(a, b); }
float sdf_intersect(float a, float b) { return max(a, b); }
float sdf_subtract(float a, float b) { return max(a, -b); }

// Smooth boolean (smooth minimum)
float sdf_smooth_union(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Gradient (surface normal) via central differences
vec2 sdf_gradient(sampler2D sdf_tex, vec2 uv, vec2 texel_size) {
    float dx = texture(sdf_tex, uv + vec2(texel_size.x, 0.0)).r
             - texture(sdf_tex, uv - vec2(texel_size.x, 0.0)).r;
    float dy = texture(sdf_tex, uv + vec2(0.0, texel_size.y)).r
             - texture(sdf_tex, uv - vec2(0.0, texel_size.y)).r;
    return normalize(vec2(dx, dy));
}
