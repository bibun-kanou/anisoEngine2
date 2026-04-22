#version 450 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_tex;
uniform vec2 u_texel_size;
uniform int u_horizontal; // 0=vertical, 1=horizontal (two-pass Gaussian)

// 9-tap Gaussian weights
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

uniform float u_intensity; // For final composite scaling

void main() {
    if (u_horizontal < 0) {
        // Passthrough: just sample and scale for final composite
        frag_color = texture(u_tex, v_uv) * u_intensity;
        return;
    }

    vec2 dir = u_horizontal == 1 ? vec2(u_texel_size.x, 0.0) : vec2(0.0, u_texel_size.y);

    vec4 result = texture(u_tex, v_uv) * weights[0];
    for (int i = 1; i < 5; i++) {
        result += texture(u_tex, v_uv + dir * float(i)) * weights[i];
        result += texture(u_tex, v_uv - dir * float(i)) * weights[i];
    }

    frag_color = result;
}
