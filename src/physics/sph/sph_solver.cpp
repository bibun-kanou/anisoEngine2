#include "physics/sph/sph_solver.h"
#include "physics/sdf/sdf_field.h"
#include "physics/common/grid.h"
#include "physics/mpm/mpm_solver.h"
#include "physics/eulerian/euler_fluid.h"
#include "core/log.h"
#include <vector>
#include <cmath>
#include <glad/gl.h>

namespace ng {

void SPHSolver::init() {
    density_shader_.load("shaders/physics/sph_density.comp");
    codim_detect_shader_.load("shaders/physics/sph_codim_detect.comp");
    force_shader_.load("shaders/physics/sph_force.comp");
    contact_heat_shader_.load("shaders/physics/sph_contact_heat.comp");
    codim_buf_.create(300000 * sizeof(vec4)); // Max SPH capacity
    material_param_buf_.create(300000 * sizeof(vec4));
    thermal_coupling_buf_.create(300000 * sizeof(vec4));
    spring_anchor_buf_.create(300000 * sizeof(vec2));
    spring_weight_buf_.create(300000 * sizeof(f32));
    spring_anchor_buf_.clear();
    spring_weight_buf_.clear();
    material_param_buf_.clear();
    thermal_coupling_buf_.clear();
    LOG_INFO("SPH solver initialized (h=%.4f, rho0=%.1f, codim=%s)",
        params_.smoothing_radius, params_.rest_density,
        params_.codim_enabled ? "ON" : "OFF");
}

void SPHSolver::spawn_block(ParticleBuffer& particles, vec2 min, vec2 max, f32 spacing) {
    std::vector<vec2> positions;
    std::vector<vec2> velocities;
    std::vector<f32> masses;
    std::vector<vec4> colors;

    for (f32 y = min.y; y < max.y; y += spacing) {
        for (f32 x = min.x; x < max.x; x += spacing) {
            positions.push_back(vec2(x, y));
            velocities.push_back(vec2(0.0f));
            masses.push_back(params_.particle_mass);
            f32 t = (y - min.y) / (max.y - min.y + 0.001f);
            colors.push_back(vec4(0.1f + 0.15f * t, 0.3f + 0.2f * t, 0.6f + 0.3f * t, 1.0f));
        }
    }

    u32 count = static_cast<u32>(positions.size());
    u32 offset = particles.allocate(SolverType::SPH, count);
    if (offset == UINT32_MAX) return;

    particles.upload_positions(offset, positions.data(), count);
    particles.upload_velocities(offset, velocities.data(), count);
    particles.upload_masses(offset, masses.data(), count);
    particles.upload_colors(offset, colors.data(), count);
    std::vector<f32> temps(count, 300.0f);
    std::vector<u32> mat_ids(count, static_cast<u32>(MPMMaterial::SPH_WATER));
    std::vector<f32> thermal_state(count, 0.0f);
    std::vector<vec4> mat_params(count, vec4(params_.gas_constant, params_.viscosity, 1.0f, 0.0f));
    std::vector<vec4> thermal_params(count, default_thermal_coupling(MPMMaterial::SPH_WATER));
    particles.upload_temperatures(offset, temps.data(), count);
    particles.upload_material_ids(offset, mat_ids.data(), count);
    u32 local_offset = offset - particles.range(SolverType::SPH).offset;
    particles.pressures().upload(thermal_state.data(), count * sizeof(f32), local_offset * sizeof(f32));
    material_param_buf_.upload(mat_params.data(), count * sizeof(vec4), local_offset * sizeof(vec4));
    thermal_coupling_buf_.upload(thermal_params.data(), count * sizeof(vec4), local_offset * sizeof(vec4));

    particle_count_ = particles.range(SolverType::SPH).count;
    LOG_INFO("Spawned %u SPH particles (total: %u)", count, particle_count_);
}

void SPHSolver::spawn_circle(ParticleBuffer& particles, vec2 center, f32 radius, f32 spacing) {
    std::vector<vec2> positions;
    std::vector<vec2> velocities;
    std::vector<f32> masses;
    std::vector<vec4> colors;

    for (f32 y = center.y - radius; y < center.y + radius; y += spacing) {
        for (f32 x = center.x - radius; x < center.x + radius; x += spacing) {
            vec2 p(x, y);
            if (glm::length(p - center) < radius) {
                positions.push_back(p);
                velocities.push_back(vec2(0.0f));
                masses.push_back(params_.particle_mass);
                colors.push_back(vec4(0.2f, 0.5f, 0.9f, 1.0f));
            }
        }
    }

    u32 count = static_cast<u32>(positions.size());
    if (count == 0) return;
    u32 offset = particles.allocate(SolverType::SPH, count);
    if (offset == UINT32_MAX) return;

    particles.upload_positions(offset, positions.data(), count);
    particles.upload_velocities(offset, velocities.data(), count);
    particles.upload_masses(offset, masses.data(), count);
    particles.upload_colors(offset, colors.data(), count);
    std::vector<f32> temps(count, 300.0f);
    std::vector<u32> mat_ids(count, static_cast<u32>(MPMMaterial::SPH_WATER));
    std::vector<f32> thermal_state(count, 0.0f);
    std::vector<vec4> mat_params(count, vec4(params_.gas_constant, params_.viscosity, 1.0f, 0.0f));
    std::vector<vec4> thermal_params(count, default_thermal_coupling(MPMMaterial::SPH_WATER));
    particles.upload_temperatures(offset, temps.data(), count);
    particles.upload_material_ids(offset, mat_ids.data(), count);
    u32 local_offset = offset - particles.range(SolverType::SPH).offset;
    particles.pressures().upload(thermal_state.data(), count * sizeof(f32), local_offset * sizeof(f32));
    material_param_buf_.upload(mat_params.data(), count * sizeof(vec4), local_offset * sizeof(vec4));
    thermal_coupling_buf_.upload(thermal_params.data(), count * sizeof(vec4), local_offset * sizeof(vec4));

    particle_count_ = particles.range(SolverType::SPH).count;
}

void SPHSolver::spawn_points(ParticleBuffer& particles, const std::vector<vec2>& positions) {
    u32 count = static_cast<u32>(positions.size());
    if (count == 0) return;

    std::vector<vec2> velocities(count, vec2(0.0f));
    std::vector<f32> masses(count, params_.particle_mass);
    std::vector<vec4> colors(count, vec4(0.2f, 0.5f, 0.9f, 1.0f));

    u32 offset = particles.allocate(SolverType::SPH, count);
    if (offset == UINT32_MAX) return;

    particles.upload_positions(offset, positions.data(), count);
    particles.upload_velocities(offset, velocities.data(), count);
    particles.upload_masses(offset, masses.data(), count);
    particles.upload_colors(offset, colors.data(), count);
    std::vector<f32> temps(count, 300.0f);
    std::vector<u32> mat_ids(count, static_cast<u32>(MPMMaterial::SPH_WATER));
    std::vector<f32> thermal_state(count, 0.0f);
    std::vector<vec4> mat_params(count, vec4(params_.gas_constant, params_.viscosity, 1.0f, 0.0f));
    std::vector<vec4> thermal_params(count, default_thermal_coupling(MPMMaterial::SPH_WATER));
    particles.upload_temperatures(offset, temps.data(), count);
    particles.upload_material_ids(offset, mat_ids.data(), count);
    u32 local_offset = offset - particles.range(SolverType::SPH).offset;
    particles.pressures().upload(thermal_state.data(), count * sizeof(f32), local_offset * sizeof(f32));
    material_param_buf_.upload(mat_params.data(), count * sizeof(vec4), local_offset * sizeof(vec4));
    thermal_coupling_buf_.upload(thermal_params.data(), count * sizeof(vec4), local_offset * sizeof(vec4));

    particle_count_ = particles.range(SolverType::SPH).count;
}

void SPHSolver::begin_spring_drag(ParticleBuffer& particles, vec2 center, f32 radius, f32 falloff_radius) {
    const auto& range = particles.range(SolverType::SPH);
    if (range.count == 0 || radius <= 0.0f) {
        spring_drag_active_ = false;
        return;
    }

    if (falloff_radius < radius) falloff_radius = radius;

    spring_origin_ = center;

    std::vector<vec2> positions(range.count);
    std::vector<vec2> anchors(range.count, vec2(0.0f));
    std::vector<f32> weights(range.count, 0.0f);
    particles.positions().download(positions.data(), range.count * sizeof(vec2), range.offset * sizeof(vec2));

    bool any = false;
    for (u32 i = 0; i < range.count; ++i) {
        vec2 delta = positions[i] - center;
        f32 dist = glm::length(delta);
        if (dist >= falloff_radius) continue;
        if (dist <= radius || falloff_radius <= radius + 1e-4f) {
            weights[i] = 1.0f;
        } else {
            f32 t = (dist - radius) / (falloff_radius - radius);
            weights[i] = glm::clamp(1.0f - t, 0.0f, 1.0f);
        }
        anchors[i] = delta;
        any = any || weights[i] > 0.0f;
    }

    spring_anchor_buf_.clear();
    spring_weight_buf_.clear();
    if (any) {
        spring_anchor_buf_.upload(anchors.data(), range.count * sizeof(vec2));
        spring_weight_buf_.upload(weights.data(), range.count * sizeof(f32));
    }
    spring_drag_active_ = any;
}

void SPHSolver::end_spring_drag() {
    spring_anchor_buf_.clear();
    spring_weight_buf_.clear();
    spring_origin_ = vec2(0.0f);
    spring_drag_active_ = false;
}

void SPHSolver::step(ParticleBuffer& particles, SpatialHash& hash, f32 dt,
                      const SDFField* sdf, const MouseForce& mouse,
                      const UniformGrid* mpm_grid, const EulerianFluid* air) {
    if (particle_count_ == 0) return;

    // 2 substeps for CFL stability: with k=8, h=0.04, c_s=sqrt(8)=2.83,
    // CFL limit = 0.4*h/c_s = 0.0057s. At 120Hz, dt=0.0083, so 2 substeps
    // gives sub_dt=0.0042 < 0.0057 ✓.  Ray-march CCD handles collision.
    constexpr i32 SUBSTEPS = 2;
    f32 sub_dt = dt / static_cast<f32>(SUBSTEPS);
    for (i32 s = 0; s < SUBSTEPS; s++) {
        sub_step(particles, hash, sub_dt, sdf, mouse, mpm_grid, air);
    }
}

void SPHSolver::scatter_contact_heat(ParticleBuffer& particles, const UniformGrid& mpm_grid) {
    const auto& range = particles.range(SolverType::SPH);
    if (range.count == 0 || !params_.enable_thermal) return;

    particles.bind_all();
    thermal_coupling_buf_.bind_base(54);
    mpm_grid.contact_temp_buf().bind_base(UniformGrid::BIND_GRID_CONTACT_TEMP);

    contact_heat_shader_.bind();
    contact_heat_shader_.set_uint("u_offset", range.offset);
    contact_heat_shader_.set_uint("u_count", range.count);
    contact_heat_shader_.set_vec2("u_grid_origin", mpm_grid.world_min());
    contact_heat_shader_.set_ivec2("u_grid_res", mpm_grid.resolution());
    contact_heat_shader_.set_float("u_dx", mpm_grid.dx());
    contact_heat_shader_.set_float("u_ambient_temp", params_.ambient_temp);
    contact_heat_shader_.dispatch_1d(range.count);
    ComputeShader::barrier_ssbo();
}

void SPHSolver::sub_step(ParticleBuffer& particles, SpatialHash& hash, f32 dt,
                          const SDFField* sdf, const MouseForce& mouse,
                          const UniformGrid* mpm_grid, const EulerianFluid* air) {
    auto& range = particles.range(SolverType::SPH);

    // Rebuild spatial hash for current positions
    hash.build(particles, range.offset, range.count);

    particles.bind_all();
    hash.bind();
    codim_buf_.bind_base(16);
    spring_anchor_buf_.bind_base(17);
    spring_weight_buf_.bind_base(18);
    material_param_buf_.bind_base(55);
    thermal_coupling_buf_.bind_base(54);

    // Pass 0.5: Codimensional detection (if enabled)
    if (params_.codim_enabled) {
        codim_detect_shader_.bind();
        codim_detect_shader_.set_uint("u_offset", range.offset);
        codim_detect_shader_.set_uint("u_count", range.count);
        codim_detect_shader_.set_float("u_h", params_.smoothing_radius);
        codim_detect_shader_.set_uint("u_table_size", hash.table_size());
        codim_detect_shader_.set_float("u_cell_size", hash.cell_size());
        codim_detect_shader_.set_vec2("u_world_min", hash.world_min());
        codim_detect_shader_.set_float("u_codim_threshold", params_.codim_threshold);
        codim_detect_shader_.dispatch_1d(range.count);
        ComputeShader::barrier_ssbo();
    }

    // Pass 1: Density (with codim-adaptive kernel)
    density_shader_.bind();
    density_shader_.set_uint("u_offset", range.offset);
    density_shader_.set_uint("u_count", range.count);
    density_shader_.set_float("u_h", params_.smoothing_radius);
    density_shader_.set_uint("u_table_size", hash.table_size());
    density_shader_.set_float("u_cell_size", hash.cell_size());
    density_shader_.set_vec2("u_world_min", hash.world_min());
    density_shader_.set_int("u_use_codim", params_.codim_enabled ? 1 : 0);
    density_shader_.dispatch_1d(range.count);
    ComputeShader::barrier_ssbo();

    // Pass 2: Force + integrate + CCD collision
    force_shader_.bind();
    force_shader_.set_uint("u_offset", range.offset);
    force_shader_.set_uint("u_count", range.count);
    force_shader_.set_float("u_h", params_.smoothing_radius);
    force_shader_.set_float("u_rest_density", params_.rest_density);
    force_shader_.set_float("u_gas_constant", params_.gas_constant);
    force_shader_.set_float("u_viscosity", params_.viscosity);
    force_shader_.set_float("u_xsph", params_.xsph);
    force_shader_.set_float("u_dt", dt);
    force_shader_.set_vec2("u_gravity", params_.gravity);
    force_shader_.set_int("u_vis_mode", params_.vis_mode);
    force_shader_.set_int("u_keep_colors", params_.keep_colors ? 1 : 0);
    force_shader_.set_float("u_surface_tension", params_.surface_tension);
    force_shader_.set_int("u_enable_interfaces", params_.immiscible_interfaces ? 1 : 0);
    force_shader_.set_float("u_interface_repulsion", params_.interface_repulsion);
    force_shader_.set_float("u_interface_tension", params_.interface_tension);
    force_shader_.set_float("u_cross_mix", params_.cross_mix);
    force_shader_.set_float("u_cross_thermal_mix", params_.cross_thermal_mix);
    force_shader_.set_float("u_mpm_contact_push", params_.mpm_contact_push);
    force_shader_.set_float("u_mpm_contact_damping", params_.mpm_contact_damping);
    force_shader_.set_float("u_mpm_contact_recovery", params_.mpm_contact_recovery);
    force_shader_.set_int("u_enable_thermal", params_.enable_thermal ? 1 : 0);
    force_shader_.set_float("u_ambient_temp", params_.ambient_temp);
    force_shader_.set_vec2("u_heat_pos", params_.heat_source_pos);
    force_shader_.set_float("u_heat_radius", params_.heat_source_radius);
    force_shader_.set_float("u_heat_temp", params_.heat_source_temp);
    force_shader_.set_vec2("u_heat_gun_pos", params_.heat_gun_pos);
    force_shader_.set_float("u_heat_gun_radius", params_.heat_gun_radius);
    force_shader_.set_float("u_heat_gun_power", params_.heat_gun_power);
    force_shader_.set_float("u_particle_cooling", params_.particle_cooling_rate);
    force_shader_.set_uint("u_highlight_start", params_.highlight_start);
    force_shader_.set_uint("u_highlight_end", params_.highlight_end);
    force_shader_.set_float("u_time", params_.time);
    if (air && params_.enable_thermal) {
        glBindTextureUnit(2, air->temp_texture());
        force_shader_.set_int("u_air_temp_tex", 2);
        force_shader_.set_int("u_use_air_heat", 1);
        force_shader_.set_vec2("u_air_world_min", air->world_min());
        force_shader_.set_vec2("u_air_world_max", air->world_max());
    } else {
        force_shader_.set_int("u_use_air_heat", 0);
        force_shader_.set_vec2("u_air_world_min", vec2(0.0f));
        force_shader_.set_vec2("u_air_world_max", vec2(1.0f));
    }
    force_shader_.set_uint("u_table_size", hash.table_size());
    force_shader_.set_float("u_cell_size", hash.cell_size());
    force_shader_.set_vec2("u_world_min", hash.world_min());

    // Mouse force
    force_shader_.set_vec2("u_mouse_world", mouse.world_pos);
    force_shader_.set_float("u_mouse_radius", mouse.radius);
    force_shader_.set_float("u_mouse_inner_radius", mouse.inner_radius);
    force_shader_.set_float("u_mouse_force", mouse.force);
    force_shader_.set_vec2("u_mouse_dir", mouse.drag_dir);
    force_shader_.set_float("u_mouse_damping", mouse.damping);
    force_shader_.set_int("u_mouse_mode", mouse.mode);
    force_shader_.set_vec2("u_spring_origin", spring_origin_);

    // SDF collision
    if (sdf) {
        sdf->bind_for_read(0);
        force_shader_.set_int("u_sdf_tex", 0);
        force_shader_.set_int("u_use_sdf", 1);
        force_shader_.set_vec2("u_sdf_world_min", sdf->world_min());
        force_shader_.set_vec2("u_sdf_world_max", sdf->world_max());
    } else {
        force_shader_.set_int("u_use_sdf", 0);
    }
    force_shader_.set_vec2("u_bound_min", params_.bound_min);
    force_shader_.set_vec2("u_bound_max", params_.bound_max);

    if (mpm_grid) {
        mpm_grid->mass_buf().bind_base(UniformGrid::BIND_GRID_MASS);
        mpm_grid->temp_buf().bind_base(UniformGrid::BIND_GRID_TEMP);
        force_shader_.set_int("u_use_mpm_coupling", 1);
        force_shader_.set_vec2("u_mpm_grid_origin", mpm_grid->world_min());
        force_shader_.set_ivec2("u_mpm_grid_res", mpm_grid->resolution());
        force_shader_.set_float("u_mpm_dx", mpm_grid->dx());
    } else {
        force_shader_.set_int("u_use_mpm_coupling", 0);
        force_shader_.set_vec2("u_mpm_grid_origin", vec2(0.0f));
        force_shader_.set_ivec2("u_mpm_grid_res", ivec2(0));
        force_shader_.set_float("u_mpm_dx", 1.0f);
    }

    force_shader_.dispatch_1d(range.count);
    ComputeShader::barrier_ssbo();
}

} // namespace ng
