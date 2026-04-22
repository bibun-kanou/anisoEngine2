#version 450 core

uniform vec4 u_color;
uniform float u_time;

out vec4 frag_color;

void main() {
    // Dashed line effect: alternate on/off based on screen-space position
    float dash = sin(gl_FragCoord.x * 0.3 + gl_FragCoord.y * 0.3 + u_time * 5.0);
    if (dash < 0.0) discard;

    frag_color = u_color;
}
