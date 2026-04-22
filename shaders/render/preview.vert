#version 450 core

// Generates vertices for several outline shapes using gl_VertexID.

uniform vec2 u_center;
uniform vec2 u_size;       // Shape half extents
uniform int u_shape;       // 0=circle, 1=rect, 2=beam, 3=triangle, 4=star
uniform float u_rotation;
uniform mat4 u_view_proj;

const int SEGMENTS = 64;

vec2 rot(vec2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
}

vec2 rect_vertex(int idx) {
    int side = idx / (SEGMENTS / 4);
    float t = float(idx % (SEGMENTS / 4)) / float(SEGMENTS / 4);
    vec2 corners[4] = vec2[4](
        vec2(-u_size.x, -u_size.y),
        vec2( u_size.x, -u_size.y),
        vec2( u_size.x,  u_size.y),
        vec2(-u_size.x,  u_size.y)
    );
    int next = (side + 1) % 4;
    return mix(corners[side], corners[next], t);
}

vec2 tri_corner(int i) {
    if (i == 0) return vec2(0.0, u_size.y);
    if (i == 1) return vec2(u_size.x, -0.5 * u_size.y);
    return vec2(-u_size.x, -0.5 * u_size.y);
}

vec2 triangle_vertex(int idx) {
    float ft = float(idx) / float(SEGMENTS) * 3.0;
    int seg = int(floor(ft));
    float t = fract(ft);
    vec2 a = tri_corner(seg % 3);
    vec2 b = tri_corner((seg + 1) % 3);
    return mix(a, b, t);
}

vec2 star_corner(int i) {
    float ang = -1.57079632679 + 0.62831853072 * float(i);
    float r = (i % 2 == 0) ? 1.0 : 0.42;
    return vec2(cos(ang) * u_size.x * r, sin(ang) * u_size.y * r);
}

vec2 star_vertex(int idx) {
    float ft = float(idx) / float(SEGMENTS) * 10.0;
    int seg = int(floor(ft));
    float t = fract(ft);
    vec2 a = star_corner(seg % 10);
    vec2 b = star_corner((seg + 1) % 10);
    return mix(a, b, t);
}

void main() {
    int idx = gl_VertexID;

    vec2 local;
    if (u_shape == 0) {
        float angle = float(idx) / float(SEGMENTS) * 6.28318530718;
        local = vec2(cos(angle) * u_size.x, sin(angle) * u_size.y);
    } else if (u_shape == 1 || u_shape == 2) {
        local = rect_vertex(idx);
    } else if (u_shape == 3) {
        local = triangle_vertex(idx);
    } else {
        local = star_vertex(idx);
    }

    vec2 pos = u_center + rot(local, u_rotation);
    gl_Position = u_view_proj * vec4(pos, 0.0, 1.0);
}
