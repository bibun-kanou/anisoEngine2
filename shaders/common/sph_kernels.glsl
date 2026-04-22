// SPH kernel functions (2D cubic spline)

#define PI 3.14159265358979323846

// Cubic spline kernel W(r, h) in 2D
// Normalized so integral over 2D = 1
float W_cubic(float r, float h) {
    float q = r / h;
    float sigma = 10.0 / (7.0 * PI * h * h); // 2D normalization
    if (q < 1.0) {
        return sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
    } else if (q < 2.0) {
        float t = 2.0 - q;
        return sigma * 0.25 * t * t * t;
    }
    return 0.0;
}

// Gradient of cubic spline kernel (scalar part: dW/dr)
// Full gradient = dW/dr * (r_vec / |r_vec|)
float dW_cubic(float r, float h) {
    float q = r / h;
    float sigma = 10.0 / (7.0 * PI * h * h);
    if (q < 0.001) return 0.0; // Avoid division by zero
    if (q < 1.0) {
        return sigma * (-3.0 * q + 2.25 * q * q) / h;
    } else if (q < 2.0) {
        float t = 2.0 - q;
        return sigma * (-0.75 * t * t) / h;
    }
    return 0.0;
}

// Spatial hash neighbor iteration helpers
uint hash_cell_2d(ivec2 cell, uint table_size) {
    uint h = uint(cell.x) * 73856093u ^ uint(cell.y) * 19349663u;
    return h & (table_size - 1u);
}

ivec2 pos_to_cell_2d(vec2 pos, float cell_size, vec2 world_min) {
    return ivec2(floor((pos - world_min) / cell_size));
}
