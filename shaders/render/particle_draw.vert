#version 450 core

layout(std430, binding = 0) readonly buffer PositionBuffer {
    vec2 positions[];
};

layout(std430, binding = 9) readonly buffer ColorBuffer {
    vec4 colors[];
};

uniform mat4 u_view_proj;
uniform float u_point_size;
uniform int u_offset;

out vec4 v_color;

void main() {
    uint idx = uint(gl_VertexID + u_offset);
    vec2 pos = positions[idx];
    v_color = colors[idx];
    gl_Position = u_view_proj * vec4(pos, 0.0, 1.0);
    gl_PointSize = u_point_size;
}
