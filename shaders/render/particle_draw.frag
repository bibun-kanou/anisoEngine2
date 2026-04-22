#version 450 core

in vec4 v_color;
out vec4 frag_color;

void main() {
    // Circular point sprite
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;

    // Soft edge
    float alpha = 1.0 - smoothstep(0.6, 1.0, r2);
    frag_color = vec4(v_color.rgb, v_color.a * alpha);
}
