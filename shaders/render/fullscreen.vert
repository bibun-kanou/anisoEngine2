#version 450 core

// Full-screen triangle (no VBO needed, use gl_VertexID)
out vec2 v_uv;

void main() {
    // Generate full-screen triangle from vertex ID (0,1,2)
    v_uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(v_uv * 2.0 - 1.0, 0.0, 1.0);
}
