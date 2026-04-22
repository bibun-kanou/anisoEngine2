#version 450 core
in vec4 v_color;
out vec4 frag_color;
void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;
    float w = (1.0 - sqrt(r2));
    w = w * w;
    frag_color = vec4(v_color.rgb * w, w);
}
