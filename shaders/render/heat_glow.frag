#version 450 core

out vec4 frag_color;
uniform vec3 u_glow_color;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r = length(coord);
    if (r > 1.0) discard;

    // Soft radial falloff: warm center fading to transparent edge
    float alpha = (1.0 - r);
    alpha = alpha * alpha; // Quadratic
    alpha *= 0.25; // Keep it subtle/semi-transparent

    frag_color = vec4(u_glow_color * alpha, alpha);
}
