#version 450 core

out vec4 frag_color;
uniform vec3 u_glow_color;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;

    // Soft radial falloff for nice glow
    float alpha = (1.0 - sqrt(r2));
    alpha = alpha * alpha * alpha; // Cubic falloff — concentrated center

    frag_color = vec4(u_glow_color * alpha, alpha);
}
