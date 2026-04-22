// MPM utilities: quadratic B-spline weights, 2D SVD, atomic float

// --- Quadratic B-spline (3-node support) ---
// N(x) for |x| in [0, 1.5), centered grid
// Returns weights for nodes at offsets 0, 1, 2 from base
vec3 bspline_weights(float fx) {
    // fx = fractional position within cell [0, 1)
    return vec3(
        0.5 * (1.5 - fx) * (1.5 - fx),
        0.75 - (fx - 1.0) * (fx - 1.0),
        0.5 * (fx - 0.5) * (fx - 0.5)
    );
}

// Atomic float add: uses NV extension (available on all NVIDIA GPUs)
// The P2G shader enables GL_NV_shader_atomic_float and declares buffers as float,
// so we can call atomicAdd(buffer[i], value) directly on float SSBOs.

// --- 2D SVD: F = U * Sigma * V^T ---
// Uses the analytic formula for 2x2 SVD
struct SVD2 {
    mat2 U;
    vec2 sigma;
    mat2 V;
};

SVD2 svd2x2(mat2 F) {
    SVD2 result;

    // Compute F^T * F
    mat2 FtF = transpose(F) * F;

    // Eigenvalues of symmetric 2x2: FtF = [[a, b], [b, d]]
    float a = FtF[0][0], b = FtF[1][0], d = FtF[1][1];
    float T = a + d;
    float D = a * d - b * b;
    float disc = sqrt(max(T * T * 0.25 - D, 0.0));
    float s1_sq = max(T * 0.5 + disc, 1e-12);
    float s2_sq = max(T * 0.5 - disc, 1e-12);

    result.sigma = vec2(sqrt(s1_sq), sqrt(s2_sq));

    // Eigenvectors of FtF for V
    if (abs(b) > 1e-10) {
        vec2 v1 = normalize(vec2(s1_sq - d, b));
        vec2 v2 = vec2(-v1.y, v1.x); // Orthogonal
        result.V = mat2(v1, v2);
    } else {
        result.V = mat2(1.0);
        if (a < d) {
            result.V = mat2(0, 1, -1, 0); // Swap if needed
            result.sigma = result.sigma.yx;
        }
    }

    // U = F * V * Sigma^-1
    vec2 inv_sig = vec2(
        result.sigma.x > 1e-8 ? 1.0 / result.sigma.x : 0.0,
        result.sigma.y > 1e-8 ? 1.0 / result.sigma.y : 0.0
    );
    result.U = F * result.V * mat2(inv_sig.x, 0, 0, inv_sig.y);

    // Ensure proper rotations (det = +1)
    if (determinant(result.U) < 0.0) {
        result.U[1] = -result.U[1]; // Flip second column
        result.sigma.y = -result.sigma.y;
    }
    if (determinant(result.V) < 0.0) {
        result.V[1] = -result.V[1];
        result.sigma.y = -result.sigma.y;
    }

    return result;
}

// --- 2D Polar decomposition: F = R * S ---
mat2 polar_R(mat2 F) {
    SVD2 s = svd2x2(F);
    return s.U * transpose(s.V);
}
