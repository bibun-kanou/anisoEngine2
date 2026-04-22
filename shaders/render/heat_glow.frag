#version 450 core

out vec4 frag_color;
uniform vec3 u_glow_color;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r = length(coord);
    if (r > 1.0) discard;

    // Soft radial falloff: warm center fading to transparent edge.
    // Bumped from 0.25 to 0.38 so collision-heat puffs read clearly without being
    // blown out; still keeps ember/hot-plate glow from over-saturating.
    float alpha = (1.0 - r);
    alpha = alpha * alpha; // Quadratic
    alpha *= 0.38;

    frag_color = vec4(u_glow_color * alpha, alpha);
}
