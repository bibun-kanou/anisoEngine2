#version 450 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_density_tex;
uniform float u_threshold;
uniform vec2 u_texel_size;
uniform int u_style;
uniform float u_edge_softness;
uniform float u_gloss;
uniform float u_rim_strength;
uniform float u_opacity;

vec3 saturate_color(vec3 c, float amt) {
    float luma = dot(c, vec3(0.299, 0.587, 0.114));
    return mix(vec3(luma), c, amt);
}

vec3 density_color(vec2 uv) {
    vec4 s = texture(u_density_tex, uv);
    return s.rgb / max(s.a, 0.001);
}

float smooth_density(vec2 uv) {
    float d = texture(u_density_tex, uv).a * 0.34;
    d += (texture(u_density_tex, uv + vec2(u_texel_size.x, 0.0)).a +
          texture(u_density_tex, uv - vec2(u_texel_size.x, 0.0)).a +
          texture(u_density_tex, uv + vec2(0.0, u_texel_size.y)).a +
          texture(u_density_tex, uv - vec2(0.0, u_texel_size.y)).a) * 0.12;
    d += (texture(u_density_tex, uv + u_texel_size).a +
          texture(u_density_tex, uv + vec2(u_texel_size.x, -u_texel_size.y)).a +
          texture(u_density_tex, uv + vec2(-u_texel_size.x, u_texel_size.y)).a +
          texture(u_density_tex, uv - u_texel_size).a) * 0.045;
    return d;
}

vec3 smooth_fill_color(vec2 uv) {
    vec3 c = density_color(uv) * 0.34;
    c += (density_color(uv + vec2(u_texel_size.x, 0.0)) +
          density_color(uv - vec2(u_texel_size.x, 0.0)) +
          density_color(uv + vec2(0.0, u_texel_size.y)) +
          density_color(uv - vec2(0.0, u_texel_size.y))) * 0.12;
    c += (density_color(uv + u_texel_size) +
          density_color(uv + vec2(u_texel_size.x, -u_texel_size.y)) +
          density_color(uv + vec2(-u_texel_size.x, u_texel_size.y)) +
          density_color(uv - u_texel_size)) * 0.045;
    return c;
}

void main() {
    vec4 data = texture(u_density_tex, v_uv);
    float density = data.a;
    float filtered_density = smooth_density(v_uv);
    if (max(density, filtered_density) < u_threshold * 0.72) discard;

    vec3 color = data.rgb / max(density, 0.001);
    float dx = texture(u_density_tex, v_uv + vec2(u_texel_size.x, 0)).a
             - texture(u_density_tex, v_uv - vec2(u_texel_size.x, 0)).a;
    float dy = texture(u_density_tex, v_uv + vec2(0, u_texel_size.y)).a
             - texture(u_density_tex, v_uv - vec2(0, u_texel_size.y)).a;
    float sdx = smooth_density(v_uv + vec2(u_texel_size.x, 0.0))
              - smooth_density(v_uv - vec2(u_texel_size.x, 0.0));
    float sdy = smooth_density(v_uv + vec2(0.0, u_texel_size.y))
              - smooth_density(v_uv - vec2(0.0, u_texel_size.y));
    vec2 gn = normalize(vec2(dx, dy) + 1e-6);
    vec2 gn_soft = normalize(vec2(sdx, sdy) + 1e-6);
    float body = clamp((density - u_threshold) / max(u_threshold * max(u_edge_softness, 0.15), 0.001), 0.0, 1.0);
    float filtered_body = clamp((filtered_density - u_threshold) / max(u_threshold * max(u_edge_softness * 0.7, 0.12), 0.001), 0.0, 1.0);
    float cavity = clamp(length(vec2(dx, dy)) * 0.22, 0.0, 1.0);

    vec3 N = normalize(vec3(gn * 0.9, 0.45 + 0.55 * body));
    vec3 N_soft = normalize(vec3(gn_soft * 0.5, 0.72 + 0.28 * filtered_body));
    vec3 L = normalize(vec3(-0.35, 0.70, 0.62));
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 H = normalize(L + V);

    float ndl = max(dot(N, L), 0.0);
    float ndv = max(dot(N, V), 0.0);
    float ndl_soft = max(dot(N_soft, L), 0.0);
    float ndv_soft = max(dot(N_soft, V), 0.0);
    float rim = pow(1.0 - ndv, 1.6);
    float rim_soft = pow(1.0 - ndv_soft, 1.3);
    float spec = pow(max(dot(N, H), 0.0), mix(10.0, 64.0, clamp(u_gloss, 0.0, 1.0)));
    float spec_soft = pow(max(dot(N_soft, H), 0.0), mix(8.0, 28.0, clamp(u_gloss, 0.0, 1.0)));
    float contour_band = exp(-pow((density - u_threshold) / max(u_threshold * 0.16, 0.001), 2.0));
    float soft_contour_band = exp(-pow((filtered_density - u_threshold) / max(u_threshold * 0.10, 0.001), 2.0));
    float thin_contour_band = exp(-pow((filtered_density - u_threshold) / max(u_threshold * 0.055, 0.001), 2.0));

    vec3 lit = color;
    if (u_style == 0) {
        lit = saturate_color(color, 1.18);
        lit *= 0.30 + 0.88 * ndl;
        lit += vec3(0.95, 0.98, 1.0) * spec * (0.08 + 0.65 * u_gloss);
        lit += color * rim * (0.12 + 0.42 * u_rim_strength);
    } else if (u_style == 1) {
        vec3 subsurface = mix(vec3(0.95, 0.92, 0.82), color, 0.55);
        lit = mix(color * (0.34 + 0.72 * ndl), subsurface, 0.18 + 0.24 * rim);
        lit += vec3(1.0) * spec * (0.04 + 0.38 * u_gloss);
        lit += color * rim * (0.14 + 0.46 * u_rim_strength);
    } else if (u_style == 2) {
        lit = color * (0.42 + 0.52 * ndl);
        lit *= 1.0 - cavity * 0.24;
        lit += color * rim * (0.04 + 0.16 * u_rim_strength);
        lit += vec3(1.0) * spec * (0.01 + 0.10 * u_gloss);
    } else if (u_style == 3) {
        vec3 warm = mix(color, vec3(1.0, 0.90, 0.72), 0.18 + 0.10 * body);
        lit = warm * (0.36 + 0.68 * ndl);
        lit += vec3(1.0, 0.92, 0.82) * spec * (0.03 + 0.28 * u_gloss);
        lit += warm * rim * (0.08 + 0.26 * u_rim_strength);
    } else {
        vec3 porcelain = mix(color, vec3(0.98, 0.97, 0.95), 0.12 + 0.12 * body);
        lit = porcelain * (0.34 + 0.78 * ndl);
        lit += vec3(1.0) * spec * (0.06 + 0.48 * u_gloss);
        lit += vec3(1.0, 0.98, 0.96) * rim * (0.10 + 0.36 * u_rim_strength);
    }

    if (u_style == 5) {
        vec3 fill = saturate_color(smooth_fill_color(v_uv), 0.92);
        fill = mix(fill, vec3(dot(fill, vec3(0.299, 0.587, 0.114))), 0.16);
        lit = fill * (0.48 + 0.34 * ndl_soft);
        lit += vec3(1.0) * spec * (0.02 + 0.18 * u_gloss);
        lit = mix(lit, vec3(0.10, 0.12, 0.15), soft_contour_band * 0.32);
    } else if (u_style == 6) {
        vec3 fill = saturate_color(smooth_fill_color(v_uv), 1.05);
        lit = fill * (0.44 + 0.36 * ndl_soft);
        lit = mix(lit, vec3(0.05, 0.07, 0.10), soft_contour_band * 0.55);
        lit += fill * rim_soft * 0.06;
    } else if (u_style == 7) {
        vec3 fill = saturate_color(smooth_fill_color(v_uv), 0.96);
        float luma = dot(fill, vec3(0.299, 0.587, 0.114));
        fill = mix(fill, vec3(luma), 0.08);
        lit = fill * (0.82 + 0.10 * filtered_body);
        lit += fill * rim_soft * 0.04;
        lit += vec3(1.0) * spec_soft * (0.01 + 0.06 * u_gloss);
        lit = mix(lit, vec3(0.07, 0.09, 0.12), soft_contour_band * 0.26);
    } else if (u_style == 8) {
        vec3 fill = saturate_color(smooth_fill_color(v_uv), 1.00);
        lit = fill * (0.80 + 0.08 * filtered_body);
        lit = mix(lit, vec3(0.04, 0.05, 0.08), thin_contour_band * 0.82);
        lit += fill * rim_soft * 0.03;
    } else if (u_style == 9) {
        vec3 fill = saturate_color(smooth_fill_color(v_uv), 0.98);
        float luma = dot(fill, vec3(0.299, 0.587, 0.114));
        fill = mix(fill, vec3(luma), 0.14);
        vec3 paper_tint = mix(vec3(0.95, 0.96, 0.98), fill, 0.88);
        lit = paper_tint * (0.86 + 0.06 * filtered_body);
        lit = mix(lit, vec3(0.03, 0.04, 0.07), thin_contour_band * 0.94);
        lit += paper_tint * rim_soft * 0.02;
        lit += vec3(1.0) * spec_soft * (0.006 + 0.03 * u_gloss);
    }

    float alpha = smoothstep(u_threshold * 0.78, u_threshold * (1.05 + 0.55 * u_edge_softness), density);
    if (u_style >= 5) {
        alpha = smoothstep(u_threshold * 0.82, u_threshold * (1.02 + 0.38 * u_edge_softness), filtered_density);
    }
    frag_color = vec4(clamp(lit, 0.0, 1.8), alpha * clamp(u_opacity, 0.0, 1.0));
}
