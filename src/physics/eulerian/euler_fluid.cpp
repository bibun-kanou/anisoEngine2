#include "physics/eulerian/euler_fluid.h"
#include "physics/common/particle_buffer.h"
#include "physics/sdf/sdf_field.h"
#include "core/log.h"

#include <glad/gl.h>
#include <cmath>
#include <algorithm>

namespace ng {

namespace {

inline f32 cpu_smoothstep(f32 edge0, f32 edge1, f32 x) {
    if (edge0 == edge1) return x < edge0 ? 0.0f : 1.0f;
    const f32 t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

struct SolverBudget {
    i32 iters;
    f32 sub_dt;
};

inline SolverBudget make_solver_budget(f32 total_dt, i32 min_iters, f32 max_sub_dt) {
    const f32 clamped_total = std::max(total_dt, 0.0f);
    i32 iters = min_iters;
    if (max_sub_dt > 1e-6f) {
        iters = std::max(min_iters, static_cast<i32>(std::ceil(clamped_total / max_sub_dt)));
    }
    const f32 sub_dt = (iters > 0) ? (clamped_total / static_cast<f32>(iters)) : 0.0f;
    return { iters, sub_dt };
}

} // namespace

void EulerianFluid::init(const Config& config) {
    config_ = config;
    resolution_ = config.resolution;
    world_min_ = config.world_min;
    world_max_ = config.world_max;
    dx_ = (world_max_.x - world_min_.x) / static_cast<f32>(resolution_.x);
    airtight_resolution_ = ivec2(std::max(resolution_.x / 4, 64), std::max(resolution_.y / 4, 64));
    airtight_dx_ = (world_max_.x - world_min_.x) / static_cast<f32>(airtight_resolution_.x);

    u32 n = static_cast<u32>(resolution_.x * resolution_.y);
    u32 n_airtight = static_cast<u32>(airtight_resolution_.x * airtight_resolution_.y);

    vel_x_buf_.create(n * sizeof(f32));
    vel_y_buf_.create(n * sizeof(f32));
    vel_x2_buf_.create(n * sizeof(f32));
    vel_y2_buf_.create(n * sizeof(f32));
    pressure_buf_.create(n * sizeof(f32));
    divergence_buf_.create(n * sizeof(f32));
    temp_buf_.create(n * sizeof(f32));
    temp2_buf_.create(n * sizeof(f32));
    smoke_buf_.create(n * sizeof(f32));
    smoke2_buf_.create(n * sizeof(f32));
    vapor_buf_.create(n * sizeof(f32));
    vapor2_buf_.create(n * sizeof(f32));
    vapor_source_buf_.create(n * sizeof(f32));
    bio_a_buf_.create(n * sizeof(f32));
    bio_a2_buf_.create(n * sizeof(f32));
    bio_b_buf_.create(n * sizeof(f32));
    bio_b2_buf_.create(n * sizeof(f32));
    bio_support_source_buf_.create(n * sizeof(f32));
    bio_support_buf_.create(n * sizeof(f32));
    bio_support2_buf_.create(n * sizeof(f32));
    bio_source_seed_buf_.create(n * sizeof(f32));
    bio_source_buf_.create(n * sizeof(f32));
    bio_source2_buf_.create(n * sizeof(f32));
    automata_buf_.create(n * sizeof(f32));
    automata2_buf_.create(n * sizeof(f32));
    airtight_occ_buf_.create(n_airtight * sizeof(u32));
    airtight_source_buf_.create(n_airtight * sizeof(u32));
    airtight_outside_buf_.create(n_airtight * sizeof(f32));
    airtight_outside2_buf_.create(n_airtight * sizeof(f32));
    airtight_pressure_buf_.create(n_airtight * sizeof(f32));
    airtight_pressure2_buf_.create(n_airtight * sizeof(f32));

    // Initialize temperature to ambient
    std::vector<f32> ambient(n, config.ambient_temp);
    temp_buf_.upload(ambient.data(), n * sizeof(f32));
    temp2_buf_.upload(ambient.data(), n * sizeof(f32));

    // Zero velocity, pressure, smoke
    vel_x_buf_.clear(); vel_y_buf_.clear();
    vel_x2_buf_.clear(); vel_y2_buf_.clear();
    pressure_buf_.clear(); divergence_buf_.clear();
    smoke_buf_.clear(); smoke2_buf_.clear();
    vapor_buf_.clear(); vapor2_buf_.clear();
    vapor_source_buf_.clear();
    std::vector<f32> bio_a_init(n, 1.0f);
    std::vector<f32> bio_b_init(n, 0.0f);
    bio_a_buf_.upload(bio_a_init.data(), n * sizeof(f32));
    bio_a2_buf_.upload(bio_a_init.data(), n * sizeof(f32));
    bio_b_buf_.upload(bio_b_init.data(), n * sizeof(f32));
    bio_b2_buf_.upload(bio_b_init.data(), n * sizeof(f32));
    bio_support_source_buf_.clear();
    bio_support_buf_.clear();
    bio_support2_buf_.clear();
    bio_source_seed_buf_.clear();
    bio_source_buf_.clear();
    bio_source2_buf_.clear();
    automata_buf_.upload(bio_b_init.data(), n * sizeof(f32));
    automata2_buf_.upload(bio_b_init.data(), n * sizeof(f32));
    airtight_occ_buf_.clear();
    airtight_source_buf_.clear();
    airtight_outside_buf_.clear();
    airtight_outside2_buf_.clear();
    airtight_pressure_buf_.clear();
    airtight_pressure2_buf_.clear();

    // Visualization textures
    glCreateTextures(GL_TEXTURE_2D, 1, &temp_tex_);
    glTextureStorage2D(temp_tex_, 1, GL_R32F, resolution_.x, resolution_.y);
    glTextureParameteri(temp_tex_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(temp_tex_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glCreateTextures(GL_TEXTURE_2D, 1, &smoke_tex_);
    glTextureStorage2D(smoke_tex_, 1, GL_R32F, resolution_.x, resolution_.y);
    glTextureParameteri(smoke_tex_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(smoke_tex_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glCreateTextures(GL_TEXTURE_2D, 1, &vel_tex_);
    glTextureStorage2D(vel_tex_, 1, GL_RG32F, resolution_.x, resolution_.y);
    glTextureParameteri(vel_tex_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(vel_tex_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glCreateTextures(GL_TEXTURE_2D, 1, &bio_a_tex_);
    glTextureStorage2D(bio_a_tex_, 1, GL_R32F, resolution_.x, resolution_.y);
    glTextureParameteri(bio_a_tex_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(bio_a_tex_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glCreateTextures(GL_TEXTURE_2D, 1, &bio_b_tex_);
    glTextureStorage2D(bio_b_tex_, 1, GL_R32F, resolution_.x, resolution_.y);
    glTextureParameteri(bio_b_tex_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(bio_b_tex_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glCreateTextures(GL_TEXTURE_2D, 1, &bio_source_tex_);
    glTextureStorage2D(bio_source_tex_, 1, GL_R32F, resolution_.x, resolution_.y);
    glTextureParameteri(bio_source_tex_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(bio_source_tex_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glCreateTextures(GL_TEXTURE_2D, 1, &bio_field_tex_);
    glTextureStorage2D(bio_field_tex_, 1, GL_R32F, resolution_.x, resolution_.y);
    glTextureParameteri(bio_field_tex_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(bio_field_tex_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glCreateTextures(GL_TEXTURE_2D, 1, &bio_support_tex_);
    glTextureStorage2D(bio_support_tex_, 1, GL_R32F, resolution_.x, resolution_.y);
    glTextureParameteri(bio_support_tex_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(bio_support_tex_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glCreateTextures(GL_TEXTURE_2D, 1, &automata_tex_);
    glTextureStorage2D(automata_tex_, 1, GL_R32F, resolution_.x, resolution_.y);
    glTextureParameteri(automata_tex_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(automata_tex_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glCreateTextures(GL_TEXTURE_2D, 1, &airtight_pressure_tex_);
    glTextureStorage2D(airtight_pressure_tex_, 1, GL_R32F, airtight_resolution_.x, airtight_resolution_.y);
    glTextureParameteri(airtight_pressure_tex_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(airtight_pressure_tex_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glCreateTextures(GL_TEXTURE_2D, 1, &airtight_outside_tex_);
    glTextureStorage2D(airtight_outside_tex_, 1, GL_R32F, airtight_resolution_.x, airtight_resolution_.y);
    glTextureParameteri(airtight_outside_tex_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(airtight_outside_tex_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    advect_shader_.load("shaders/physics/euler_advect.comp");
    forces_shader_.load("shaders/physics/euler_forces.comp");
    divergence_shader_.load("shaders/physics/euler_divergence.comp");
    pressure_shader_.load("shaders/physics/euler_pressure.comp");
    project_shader_.load("shaders/physics/euler_project.comp");
    inject_shader_.load("shaders/physics/euler_inject.comp");
    drag_shader_.load("shaders/physics/euler_drag.comp");
    heat_gun_shader_.load("shaders/physics/euler_heat_gun.comp");
    diffuse_temp_shader_.load("shaders/physics/euler_diffuse_temp.comp");
    reaction_diffuse_shader_.load("shaders/physics/euler_reaction_diffuse.comp");
    bio_support_shader_.load("shaders/physics/euler_bio_support.comp");
    bio_source_shader_.load("shaders/physics/euler_bio_source.comp");
    automata_shader_.load("shaders/physics/euler_smoothlife.comp");
    blow_shader_.load("shaders/physics/euler_blow.comp");
    blast_shader_.load("shaders/physics/euler_blast.comp");
    enforce_bc_shader_.load("shaders/physics/euler_enforce_bc.comp");
    airtight_rasterize_shader_.load("shaders/physics/euler_airtight_rasterize.comp");
    airtight_seed_shader_.load("shaders/physics/euler_airtight_seed.comp");
    airtight_propagate_shader_.load("shaders/physics/euler_airtight_propagate.comp");
    airtight_update_shader_.load("shaders/physics/euler_airtight_update.comp");
    airtight_smooth_shader_.load("shaders/physics/euler_airtight_smooth.comp");

    LOG_INFO("EulerianFluid: %dx%d, dx=%.4f, airtight=%dx%d, T_ambient=%.0fK",
        resolution_.x, resolution_.y, dx_,
        airtight_resolution_.x, airtight_resolution_.y,
        config.ambient_temp);
}

void EulerianFluid::bind_all() const {
    vel_x_buf_.bind_base(BIND_VEL_X);
    vel_y_buf_.bind_base(BIND_VEL_Y);
    vel_x2_buf_.bind_base(BIND_VEL_X2);
    vel_y2_buf_.bind_base(BIND_VEL_Y2);
    pressure_buf_.bind_base(BIND_PRESSURE);
    divergence_buf_.bind_base(BIND_DIVERGENCE);
    temp_buf_.bind_base(BIND_ETEMPERATURE);
    temp2_buf_.bind_base(BIND_ETEMP2);
    smoke_buf_.bind_base(BIND_SMOKE);
    smoke2_buf_.bind_base(BIND_SMOKE2);
    vapor_buf_.bind_base(BIND_VAPOR);
    vapor2_buf_.bind_base(BIND_VAPOR2);
    vapor_source_buf_.bind_base(BIND_VAPOR_SOURCE);
    bio_a_buf_.bind_base(BIND_BIO_A);
    bio_a2_buf_.bind_base(BIND_BIO_A2);
    bio_b_buf_.bind_base(BIND_BIO_B);
    bio_b2_buf_.bind_base(BIND_BIO_B2);
    bio_support_source_buf_.bind_base(BIND_BIO_SUPPORT_SOURCE);
    bio_support_buf_.bind_base(BIND_BIO_SUPPORT);
    bio_support2_buf_.bind_base(BIND_BIO_SUPPORT2);
    bio_source_seed_buf_.bind_base(BIND_BIO_SOURCE_SEED);
    bio_source_buf_.bind_base(BIND_BIO_SOURCE);
    bio_source2_buf_.bind_base(BIND_BIO_SOURCE2);
    automata_buf_.bind_base(BIND_AUTOMATA);
    automata2_buf_.bind_base(BIND_AUTOMATA2);
    airtight_occ_buf_.bind_base(BIND_AIR_OCCUPANCY);
    airtight_source_buf_.bind_base(BIND_AIR_SOURCE);
    airtight_outside_buf_.bind_base(BIND_AIR_OUTSIDE);
    airtight_outside2_buf_.bind_base(BIND_AIR_OUTSIDE2);
    airtight_pressure_buf_.bind_base(BIND_AIR_PRESSURE);
    airtight_pressure2_buf_.bind_base(BIND_AIR_PRESSURE2);
}

void EulerianFluid::step(f32 dt, const SDFField* sdf) {
    u32 n = static_cast<u32>(resolution_.x * resolution_.y);
    bind_all();

    // Step 1: Apply forces (gravity + buoyancy + SDF collision)
    if (sdf) sdf->bind_for_read(0);
    forces_shader_.bind();
    forces_shader_.set_ivec2("u_res", resolution_);
    forces_shader_.set_float("u_dt", dt);
    forces_shader_.set_float("u_dx", dx_);
    forces_shader_.set_float("u_ambient_temp", config_.ambient_temp);
    forces_shader_.set_float("u_buoyancy", config_.buoyancy_alpha);
    forces_shader_.set_float("u_vapor_buoyancy", config_.vapor_buoyancy);
    forces_shader_.set_float("u_viscosity", config_.air_viscosity);
    forces_shader_.set_int("u_use_sdf", sdf ? 1 : 0);
    forces_shader_.set_vec2("u_world_min", world_min_);
    forces_shader_.set_vec2("u_world_max", world_max_);
    if (sdf) forces_shader_.set_int("u_sdf_tex", 0);
    forces_shader_.dispatch_1d(n);
    ComputeShader::barrier_ssbo();

    // Step 1.5: Thermal diffusion (heat spreads through air)
    if (sdf) {
        sdf->bind_for_read(0);
        sdf->bind_props_for_read(1);
    }
    diffuse_temp_shader_.bind();
    diffuse_temp_shader_.set_ivec2("u_res", resolution_);
    diffuse_temp_shader_.set_float("u_dx", dx_);
    diffuse_temp_shader_.set_float("u_dt", dt);
    diffuse_temp_shader_.set_float("u_thermal_diffusivity", config_.thermal_diffusivity);
    diffuse_temp_shader_.set_float("u_solid_thermal_diffusivity", config_.solid_thermal_diffusivity);
    diffuse_temp_shader_.set_float("u_solid_heat_capacity", config_.solid_heat_capacity);
    diffuse_temp_shader_.set_float("u_solid_contact_transfer", config_.solid_contact_transfer);
    diffuse_temp_shader_.set_float("u_solid_heat_loss", config_.solid_heat_loss);
    diffuse_temp_shader_.set_float("u_ambient_temp", config_.ambient_temp);
    diffuse_temp_shader_.set_float("u_cooling_rate", config_.cooling_rate);
    diffuse_temp_shader_.set_float("u_combustion_hold", config_.combustion_hold);
    diffuse_temp_shader_.set_int("u_physically_based_heat", config_.physically_based_heat ? 1 : 0);
    diffuse_temp_shader_.set_int("u_use_sdf", sdf ? 1 : 0);
    diffuse_temp_shader_.set_vec2("u_world_min", world_min_);
    diffuse_temp_shader_.set_vec2("u_world_max", world_max_);
    if (sdf) {
        diffuse_temp_shader_.set_int("u_sdf_tex", 0);
        diffuse_temp_shader_.set_int("u_sdf_prop_tex", 1);
    }
    diffuse_temp_shader_.dispatch_1d(n);
    ComputeShader::barrier_ssbo();
    // Copy diffused temp back
    glCopyNamedBufferSubData(temp2_buf_.handle(), temp_buf_.handle(), 0, 0,
        static_cast<GLsizeiptr>(n * sizeof(f32)));
    ComputeShader::barrier_ssbo();

    if (config_.bio_enabled) {
        const SolverBudget source_budget = make_solver_budget(
            dt * 10.0f * std::max(config_.bio_pattern_speed, 0.05f), 3, 0.050f);
        const i32 source_iters = source_budget.iters;
        const f32 source_dt = source_budget.sub_dt;
        const f32 source_diffuse = 0.24f;
        const f32 source_decay = 0.30f;
        for (i32 iter = 0; iter < source_iters; ++iter) {
            bio_source_shader_.bind();
            bio_source_shader_.set_ivec2("u_res", resolution_);
            bio_source_shader_.set_float("u_dt", source_dt);
            bio_source_shader_.set_float("u_diffuse", source_diffuse);
            bio_source_shader_.set_float("u_decay", source_decay);
            bio_source_shader_.set_float("u_ambient_temp", config_.ambient_temp);
            bio_source_shader_.dispatch_1d(n);
            ComputeShader::barrier_ssbo();

            glCopyNamedBufferSubData(bio_source2_buf_.handle(), bio_source_buf_.handle(), 0, 0,
                static_cast<GLsizeiptr>(n * sizeof(f32)));
            ComputeShader::barrier_ssbo();
        }
    }

    if (config_.bio_enabled || config_.automata_enabled) {
        const i32 support_iters = (config_.bio_enabled && config_.automata_enabled) ? 5 : 4;
        const f32 support_decay = 0.88f;
        glCopyNamedBufferSubData(bio_support_source_buf_.handle(), bio_support_buf_.handle(), 0, 0,
            static_cast<GLsizeiptr>(n * sizeof(f32)));
        ComputeShader::barrier_ssbo();
        for (i32 iter = 0; iter < support_iters; ++iter) {
            bio_support_shader_.bind();
            bio_support_shader_.set_ivec2("u_res", resolution_);
            bio_support_shader_.set_float("u_decay", support_decay);
            bio_support_shader_.dispatch_1d(n);
            ComputeShader::barrier_ssbo();

            glCopyNamedBufferSubData(bio_support2_buf_.handle(), bio_support_buf_.handle(), 0, 0,
                static_cast<GLsizeiptr>(n * sizeof(f32)));
            ComputeShader::barrier_ssbo();
        }
    }

    if (config_.bio_enabled) {
        const SolverBudget bio_budget = make_solver_budget(
            dt * 16.0f * std::max(config_.bio_pattern_speed, 0.05f), 2, 0.060f);
        const i32 bio_iters = bio_budget.iters;
        const f32 rd_dt = bio_budget.sub_dt;
        if (rd_dt > 1e-6f) {
            for (i32 iter = 0; iter < bio_iters; ++iter) {
                reaction_diffuse_shader_.bind();
                reaction_diffuse_shader_.set_ivec2("u_res", resolution_);
                reaction_diffuse_shader_.set_float("u_dt", rd_dt);
                reaction_diffuse_shader_.set_float("u_diffuse_a", config_.bio_diffuse_a);
                reaction_diffuse_shader_.set_float("u_diffuse_b", config_.bio_diffuse_b);
                reaction_diffuse_shader_.set_float("u_feed", config_.bio_feed);
                reaction_diffuse_shader_.set_float("u_kill", config_.bio_kill);
                reaction_diffuse_shader_.set_float("u_ambient_temp", config_.ambient_temp);
                reaction_diffuse_shader_.dispatch_1d(n);
                ComputeShader::barrier_ssbo();

                glCopyNamedBufferSubData(bio_a2_buf_.handle(), bio_a_buf_.handle(), 0, 0,
                    static_cast<GLsizeiptr>(n * sizeof(f32)));
                glCopyNamedBufferSubData(bio_b2_buf_.handle(), bio_b_buf_.handle(), 0, 0,
                    static_cast<GLsizeiptr>(n * sizeof(f32)));
                ComputeShader::barrier_ssbo();
            }
        }
    }

    if (config_.automata_enabled) {
        const SolverBudget auto_budget = make_solver_budget(
            dt * 14.0f * std::max(config_.automata_pattern_speed, 0.05f), 2, 0.060f);
        const i32 auto_iters = auto_budget.iters;
        const f32 auto_dt = auto_budget.sub_dt;
        if (auto_dt > 1e-6f) {
            for (i32 iter = 0; iter < auto_iters; ++iter) {
                automata_shader_.bind();
                automata_shader_.set_ivec2("u_res", resolution_);
                automata_shader_.set_float("u_dt", auto_dt);
                automata_shader_.set_float("u_birth_lo", config_.automata_birth_lo);
                automata_shader_.set_float("u_birth_hi", config_.automata_birth_hi);
                automata_shader_.set_float("u_survive_lo", config_.automata_survive_lo);
                automata_shader_.set_float("u_survive_hi", config_.automata_survive_hi);
                automata_shader_.set_float("u_inner_radius", config_.automata_inner_radius);
                automata_shader_.set_float("u_outer_radius", config_.automata_outer_radius);
                automata_shader_.set_float("u_sigmoid", config_.automata_sigmoid);
                automata_shader_.set_float("u_ambient_temp", config_.ambient_temp);
                automata_shader_.dispatch_1d(n);
                ComputeShader::barrier_ssbo();

                glCopyNamedBufferSubData(automata2_buf_.handle(), automata_buf_.handle(), 0, 0,
                    static_cast<GLsizeiptr>(n * sizeof(f32)));
                ComputeShader::barrier_ssbo();
            }
        }
    }

    // Step 2: Compute divergence (SDF-aware)
    if (sdf) sdf->bind_for_read(0);
    divergence_shader_.bind();
    divergence_shader_.set_ivec2("u_res", resolution_);
    divergence_shader_.set_float("u_dx", dx_);
    divergence_shader_.set_float("u_vapor_pressure", config_.vapor_pressure);
    divergence_shader_.set_int("u_use_sdf", sdf ? 1 : 0);
    divergence_shader_.set_vec2("u_world_min", world_min_);
    divergence_shader_.set_vec2("u_world_max", world_max_);
    if (sdf) divergence_shader_.set_int("u_sdf_tex", 0);
    divergence_shader_.dispatch_1d(n);
    ComputeShader::barrier_ssbo();

    // Step 3: Pressure solve (Red-Black Gauss-Seidel — 20 iterations)
    pressure_buf_.clear();
    ComputeShader::barrier_ssbo();
    if (sdf) sdf->bind_for_read(0);
    pressure_shader_.bind();
    pressure_shader_.set_ivec2("u_res", resolution_);
    pressure_shader_.set_float("u_dx", dx_);
    pressure_shader_.set_int("u_use_sdf", sdf ? 1 : 0);
    pressure_shader_.set_vec2("u_world_min", world_min_);
    pressure_shader_.set_vec2("u_world_max", world_max_);
    if (sdf) pressure_shader_.set_int("u_sdf_tex", 0);
    for (int iter = 0; iter < 20; iter++) {
        pressure_shader_.set_int("u_color", 0); // Red cells
        pressure_shader_.dispatch_1d(n);
        ComputeShader::barrier_ssbo();
        pressure_shader_.set_int("u_color", 1); // Black cells
        pressure_shader_.dispatch_1d(n);
        ComputeShader::barrier_ssbo();
    }

    // Step 4: Project with SDF
    if (sdf) sdf->bind_for_read(0);
    project_shader_.bind();
    project_shader_.set_ivec2("u_res", resolution_);
    project_shader_.set_float("u_dx", dx_);
    project_shader_.set_int("u_use_sdf", sdf ? 1 : 0);
    project_shader_.set_vec2("u_world_min", world_min_);
    project_shader_.set_vec2("u_world_max", world_max_);
    if (sdf) project_shader_.set_int("u_sdf_tex", 0);
    project_shader_.dispatch_1d(n);
    ComputeShader::barrier_ssbo();

    // Step 5: Advect with SDF obstacle awareness
    if (sdf) sdf->bind_for_read(0);
    advect_shader_.bind();
    advect_shader_.set_ivec2("u_res", resolution_);
    advect_shader_.set_float("u_dt", dt);
    advect_shader_.set_float("u_dx", dx_);
    advect_shader_.set_float("u_smoke_decay", std::pow(config_.smoke_decay, dt));
    advect_shader_.set_float("u_vapor_decay", std::exp(-config_.vapor_decay * dt));
    advect_shader_.set_vec2("u_world_min", world_min_);
    advect_shader_.set_vec2("u_world_max", world_max_);
    advect_shader_.set_int("u_use_sdf", sdf ? 1 : 0);
    if (sdf) advect_shader_.set_int("u_sdf_tex", 0);
    advect_shader_.dispatch_1d(n);
    ComputeShader::barrier_ssbo();

    // Swap buffers
    glCopyNamedBufferSubData(vel_x2_buf_.handle(), vel_x_buf_.handle(), 0, 0, n * sizeof(f32));
    glCopyNamedBufferSubData(vel_y2_buf_.handle(), vel_y_buf_.handle(), 0, 0, n * sizeof(f32));
    glCopyNamedBufferSubData(temp2_buf_.handle(), temp_buf_.handle(), 0, 0, n * sizeof(f32));
    glCopyNamedBufferSubData(smoke2_buf_.handle(), smoke_buf_.handle(), 0, 0, n * sizeof(f32));
    glCopyNamedBufferSubData(vapor2_buf_.handle(), vapor_buf_.handle(), 0, 0, n * sizeof(f32));
    ComputeShader::barrier_ssbo();

    // Final BC enforcement: zero velocity inside solid (catches all residuals)
    if (sdf) {
        sdf->bind_for_read(0);
        bind_all();
        enforce_bc_shader_.bind();
        enforce_bc_shader_.set_ivec2("u_res", resolution_);
        enforce_bc_shader_.set_float("u_dx", dx_);
        enforce_bc_shader_.set_int("u_use_sdf", 1);
        enforce_bc_shader_.set_vec2("u_world_min", world_min_);
        enforce_bc_shader_.set_vec2("u_world_max", world_max_);
        enforce_bc_shader_.set_int("u_sdf_tex", 0);
        enforce_bc_shader_.dispatch_1d(n);
        ComputeShader::barrier_ssbo();
    }

    // Update visualization textures
    {
        const bool need_bio_fields =
            config_.bio_enabled ||
            visualization_mode_ == 8 || visualization_mode_ == 10 ||
            particle_visualization_mode_ == 11;
        const bool need_bio_composite = (visualization_mode_ == 8 || visualization_mode_ == 10);
        const bool need_support_field =
            config_.bio_enabled || config_.automata_enabled ||
            visualization_mode_ == 8 || visualization_mode_ == 9 ||
            visualization_mode_ == 10 || visualization_mode_ == 11 ||
            particle_visualization_mode_ == 10 ||
            particle_visualization_mode_ == 11 ||
            particle_visualization_mode_ == 12;
        const bool need_automata_field =
            config_.automata_enabled ||
            visualization_mode_ == 9 || visualization_mode_ == 11 ||
            particle_visualization_mode_ == 10 ||
            particle_visualization_mode_ == 12;
        const bool need_automata_gain =
            visualization_mode_ == 9 || visualization_mode_ == 11 ||
            particle_visualization_mode_ == 10;
        const bool need_velocity_field = (visualization_mode_ >= 4 && visualization_mode_ <= 7);
        std::vector<f32> data(n);
        temp_buf_.download(data.data(), n * sizeof(f32));
        glTextureSubImage2D(temp_tex_, 0, 0, 0, resolution_.x, resolution_.y, GL_RED, GL_FLOAT, data.data());
        smoke_buf_.download(data.data(), n * sizeof(f32));
        glTextureSubImage2D(smoke_tex_, 0, 0, 0, resolution_.x, resolution_.y, GL_RED, GL_FLOAT, data.data());
        if (need_bio_fields) {
            std::vector<f32> bio_b_data(n, 0.0f);
            std::vector<f32> bio_source_data(n, 0.0f);
            bio_a_buf_.download(data.data(), n * sizeof(f32));
            glTextureSubImage2D(bio_a_tex_, 0, 0, 0, resolution_.x, resolution_.y, GL_RED, GL_FLOAT, data.data());
            bio_b_buf_.download(bio_b_data.data(), n * sizeof(f32));
            glTextureSubImage2D(bio_b_tex_, 0, 0, 0, resolution_.x, resolution_.y, GL_RED, GL_FLOAT, bio_b_data.data());
            bio_source_buf_.download(bio_source_data.data(), n * sizeof(f32));
            glTextureSubImage2D(bio_source_tex_, 0, 0, 0, resolution_.x, resolution_.y, GL_RED, GL_FLOAT, bio_source_data.data());

            if (need_bio_composite) {
                std::vector<f32> bio_support_data(n, 0.0f);
                bio_support_buf_.download(bio_support_data.data(), n * sizeof(f32));
                f32 bio_peak = 0.0f;
                for (u32 i = 0; i < n; ++i) {
                    const f32 activator = std::clamp(bio_b_data[i], 0.0f, 1.0f);
                    const f32 scout = std::clamp(bio_source_data[i], 0.0f, 1.0f);
                    const f32 support = std::clamp(bio_support_data[i], 0.0f, 1.0f);
                    const f32 activator_vis = std::clamp(std::pow(cpu_smoothstep(0.002f, 0.18f, activator), 0.58f) * 1.55f, 0.0f, 1.0f);
                    const f32 scout_front = std::clamp(
                        cpu_smoothstep(0.006f, 0.085f, scout) *
                        (1.0f - cpu_smoothstep(0.16f, 0.34f, scout)),
                        0.0f, 1.0f);
                    const f32 scout_fill = std::clamp(std::pow(scout, 0.78f) * 0.72f, 0.0f, 1.0f);
                    const f32 scout_halo = std::clamp(cpu_smoothstep(0.0015f, 0.032f, scout) * 0.52f, 0.0f, 1.0f);
                    const f32 scout_trace = std::clamp(
                        std::pow(cpu_smoothstep(0.0002f, 0.012f, scout), 0.84f) *
                        (1.0f - cpu_smoothstep(0.12f, 0.42f, support)) * 0.78f,
                        0.0f, 1.0f);
                    const f32 support_fill = std::clamp(std::pow(cpu_smoothstep(0.003f, 0.22f, support), 0.72f) * 0.40f, 0.0f, 1.0f);
                    const f32 support_edge = std::clamp(
                        cpu_smoothstep(0.010f, 0.16f, support) *
                        (1.0f - cpu_smoothstep(0.28f, 0.66f, support)) * 0.30f,
                        0.0f, 1.0f);
                    data[i] = std::clamp(
                        std::max(activator_vis * 0.82f + scout_fill * 0.42f + support_fill * 0.22f,
                                 scout_front * 1.10f + scout_halo * 0.38f + support_edge * 0.28f)
                        + scout_fill * 0.60f + scout_halo * 0.18f + support_fill * 0.30f + scout_trace * 0.44f,
                        0.0f, 1.0f);
                    bio_peak = std::max(bio_peak, data[i]);
                }
                bio_field_view_gain_ = std::clamp(0.96f / std::max(bio_peak, 0.024f), 1.0f, 28.0f);
                glTextureSubImage2D(bio_field_tex_, 0, 0, 0, resolution_.x, resolution_.y, GL_RED, GL_FLOAT, data.data());
            }
        }

        if (need_support_field) {
            bio_support_buf_.download(data.data(), n * sizeof(f32));
            glTextureSubImage2D(bio_support_tex_, 0, 0, 0, resolution_.x, resolution_.y, GL_RED, GL_FLOAT, data.data());
        }

        if (need_automata_field) {
            automata_buf_.download(data.data(), n * sizeof(f32));
            if (need_automata_gain) {
                f32 automata_peak = 0.0f;
                for (u32 i = 0; i < n; ++i) automata_peak = std::max(automata_peak, data[i]);
                automata_view_gain_ = std::clamp(0.36f / std::max(automata_peak, 0.028f), 1.0f, 14.0f);
            }
            glTextureSubImage2D(automata_tex_, 0, 0, 0, resolution_.x, resolution_.y, GL_RED, GL_FLOAT, data.data());
        }

        if (need_velocity_field) {
            // MAC velocity → cell-center velocity for visualization
            // Average face values: u_cc = (u[i,j] + u[i+1,j])/2, v_cc = (v[i,j] + v[i,j+1])/2
            std::vector<f32> u_face(n), v_face(n);
            vel_x_buf_.download(u_face.data(), n * sizeof(f32));
            vel_y_buf_.download(v_face.data(), n * sizeof(f32));
            std::vector<f32> rg(n * 2);
            for (i32 j = 0; j < resolution_.y; j++) {
                for (i32 i = 0; i < resolution_.x; i++) {
                    u32 idx = static_cast<u32>(j * resolution_.x + i);
                    f32 u_cc = u_face[idx];
                    if (i + 1 < resolution_.x) u_cc = (u_face[idx] + u_face[idx + 1]) * 0.5f;
                    f32 v_cc = v_face[idx];
                    if (j + 1 < resolution_.y) v_cc = (v_face[idx] + v_face[idx + static_cast<u32>(resolution_.x)]) * 0.5f;
                    rg[idx * 2] = u_cc;
                    rg[idx * 2 + 1] = v_cc;
                }
            }
            glTextureSubImage2D(vel_tex_, 0, 0, 0, resolution_.x, resolution_.y, GL_RG, GL_FLOAT, rg.data());
        }
    }
}

void EulerianFluid::clear_particle_injection_sources() {
    vapor_source_buf_.clear();
    bio_support_source_buf_.clear();
    bio_source_seed_buf_.clear();
    ComputeShader::barrier_ssbo();
}

void EulerianFluid::inject_from_particles(const ParticleBuffer& particles,
                                            u32 offset, u32 count, f32 dt) {
    if (count == 0) return;
    particles.bind_all();
    bind_all();
    glBindTextureUnit(0, airtight_outside_tex_);

    inject_shader_.bind();
    inject_shader_.set_uint("u_offset", offset);
    inject_shader_.set_uint("u_count", count);
    inject_shader_.set_ivec2("u_res", resolution_);
    inject_shader_.set_float("u_dx", dx_);
    inject_shader_.set_float("u_dt", dt);
    inject_shader_.set_vec2("u_world_min", world_min_);
    inject_shader_.set_vec2("u_world_max", world_max_);
    inject_shader_.set_float("u_ambient_temp", config_.ambient_temp);
    inject_shader_.set_float("u_vapor_generation", config_.vapor_generation);
    inject_shader_.set_float("u_latent_cooling", config_.latent_cooling);
    inject_shader_.set_float("u_combustion_heat_boost", config_.combustion_heat_boost);
    inject_shader_.set_int("u_airtight_outside_tex", 0);
    inject_shader_.set_int("u_use_airtight_mask", 1);
    inject_shader_.set_int("u_use_bio", config_.bio_enabled ? 1 : 0);
    inject_shader_.set_float("u_bio_seed_strength", config_.bio_seed_strength);
    inject_shader_.set_int("u_use_automata", config_.automata_enabled ? 1 : 0);
    inject_shader_.set_float("u_automata_seed_strength", config_.automata_seed_strength);
    inject_shader_.dispatch_1d(count);
    ComputeShader::barrier_ssbo();
}

void EulerianFluid::update_airtight_from_particles(const ParticleBuffer& particles,
                                                    u32 offset, u32 count, f32 dt,
                                                    const SDFField* sdf) {
    u32 n = static_cast<u32>(airtight_resolution_.x * airtight_resolution_.y);
    if (n == 0) return;

    if (count == 0) {
        airtight_occ_buf_.clear();
        airtight_source_buf_.clear();
        airtight_outside_buf_.clear();
        airtight_outside2_buf_.clear();
        airtight_pressure_buf_.clear();
        airtight_pressure2_buf_.clear();
        ComputeShader::barrier_ssbo();
        std::vector<f32> ones(n, 1.0f);
        std::vector<f32> zeros(n, 0.0f);
        glTextureSubImage2D(airtight_outside_tex_, 0, 0, 0,
            airtight_resolution_.x, airtight_resolution_.y,
            GL_RED, GL_FLOAT, ones.data());
        glTextureSubImage2D(airtight_pressure_tex_, 0, 0, 0,
            airtight_resolution_.x, airtight_resolution_.y,
            GL_RED, GL_FLOAT, zeros.data());
        return;
    }

    particles.bind_all();
    bind_all();
    airtight_occ_buf_.clear();
    airtight_source_buf_.clear();
    ComputeShader::barrier_ssbo();

    if (sdf) sdf->bind_for_read(0);
    airtight_rasterize_shader_.bind();
    airtight_rasterize_shader_.set_uint("u_offset", offset);
    airtight_rasterize_shader_.set_uint("u_count", count);
    airtight_rasterize_shader_.set_ivec2("u_res", airtight_resolution_);
    airtight_rasterize_shader_.set_float("u_dx", airtight_dx_);
    airtight_rasterize_shader_.set_vec2("u_world_min", world_min_);
    airtight_rasterize_shader_.set_float("u_ambient_temp", config_.ambient_temp);
    airtight_rasterize_shader_.set_int("u_use_sdf", sdf ? 1 : 0);
    airtight_rasterize_shader_.set_vec2("u_world_max", world_max_);
    if (sdf) airtight_rasterize_shader_.set_int("u_sdf_tex", 0);
    airtight_rasterize_shader_.dispatch_1d(count);
    ComputeShader::barrier_ssbo();

    if (sdf) sdf->bind_for_read(0);
    airtight_seed_shader_.bind();
    airtight_seed_shader_.set_ivec2("u_res", airtight_resolution_);
    airtight_seed_shader_.set_vec2("u_world_min", world_min_);
    airtight_seed_shader_.set_vec2("u_world_max", world_max_);
    airtight_seed_shader_.set_int("u_use_sdf", sdf ? 1 : 0);
    if (sdf) airtight_seed_shader_.set_int("u_sdf_tex", 0);
    airtight_seed_shader_.dispatch_1d(n);
    ComputeShader::barrier_ssbo();

    const i32 outside_iters = std::max(airtight_resolution_.x, airtight_resolution_.y);
    for (i32 iter = 0; iter < outside_iters; ++iter) {
        if ((iter & 1) == 0) {
            airtight_outside_buf_.bind_base(BIND_AIR_OUTSIDE);
            airtight_outside2_buf_.bind_base(BIND_AIR_OUTSIDE2);
        } else {
            airtight_outside2_buf_.bind_base(BIND_AIR_OUTSIDE);
            airtight_outside_buf_.bind_base(BIND_AIR_OUTSIDE2);
        }
        if (sdf) sdf->bind_for_read(0);
        airtight_propagate_shader_.bind();
        airtight_propagate_shader_.set_ivec2("u_res", airtight_resolution_);
        airtight_propagate_shader_.set_vec2("u_world_min", world_min_);
        airtight_propagate_shader_.set_vec2("u_world_max", world_max_);
        airtight_propagate_shader_.set_int("u_use_sdf", sdf ? 1 : 0);
        if (sdf) airtight_propagate_shader_.set_int("u_sdf_tex", 0);
        airtight_propagate_shader_.dispatch_1d(n);
        ComputeShader::barrier_ssbo();
    }
    if ((outside_iters & 1) != 0) {
        glCopyNamedBufferSubData(airtight_outside2_buf_.handle(), airtight_outside_buf_.handle(), 0, 0,
            static_cast<GLsizeiptr>(n * sizeof(f32)));
        ComputeShader::barrier_ssbo();
    }
    airtight_outside_buf_.bind_base(BIND_AIR_OUTSIDE);
    airtight_outside2_buf_.bind_base(BIND_AIR_OUTSIDE2);
    airtight_pressure_buf_.bind_base(BIND_AIR_PRESSURE);
    airtight_pressure2_buf_.bind_base(BIND_AIR_PRESSURE2);

    airtight_update_shader_.bind();
    airtight_update_shader_.set_ivec2("u_res", airtight_resolution_);
    airtight_update_shader_.set_ivec2("u_air_res", resolution_);
    airtight_update_shader_.set_float("u_dx", airtight_dx_);
    airtight_update_shader_.set_float("u_dt", dt);
    airtight_update_shader_.set_float("u_ambient_temp", config_.ambient_temp);
    airtight_update_shader_.set_float("u_combustion_hold", config_.combustion_hold);
    airtight_update_shader_.set_vec2("u_world_min", world_min_);
    airtight_update_shader_.set_vec2("u_world_max", world_max_);
    airtight_update_shader_.dispatch_1d(n);
    ComputeShader::barrier_ssbo();

    // Five smoothing passes: enough to equalize pressure inside a pocket while
    // keeping the final result in the primary pressure buffer.
    for (i32 iter = 0; iter < 5; ++iter) {
        if ((iter & 1) == 0) {
            airtight_pressure2_buf_.bind_base(BIND_AIR_PRESSURE);
            airtight_pressure_buf_.bind_base(BIND_AIR_PRESSURE2);
        } else {
            airtight_pressure_buf_.bind_base(BIND_AIR_PRESSURE);
            airtight_pressure2_buf_.bind_base(BIND_AIR_PRESSURE2);
        }
        airtight_smooth_shader_.bind();
        airtight_smooth_shader_.set_ivec2("u_res", airtight_resolution_);
        airtight_smooth_shader_.dispatch_1d(n);
        ComputeShader::barrier_ssbo();
    }

    // Restore canonical bindings for the rest of the frame.
    airtight_outside_buf_.bind_base(BIND_AIR_OUTSIDE);
    airtight_outside2_buf_.bind_base(BIND_AIR_OUTSIDE2);
    airtight_pressure_buf_.bind_base(BIND_AIR_PRESSURE);
    airtight_pressure2_buf_.bind_base(BIND_AIR_PRESSURE2);

    std::vector<f32> cavity_data(n);
    airtight_outside_buf_.download(cavity_data.data(), n * sizeof(f32));
    glTextureSubImage2D(airtight_outside_tex_, 0, 0, 0,
        airtight_resolution_.x, airtight_resolution_.y,
        GL_RED, GL_FLOAT, cavity_data.data());
    airtight_pressure_buf_.download(cavity_data.data(), n * sizeof(f32));
    glTextureSubImage2D(airtight_pressure_tex_, 0, 0, 0,
        airtight_resolution_.x, airtight_resolution_.y,
        GL_RED, GL_FLOAT, cavity_data.data());
}

void EulerianFluid::apply_drag_to_particles(ParticleBuffer& particles,
                                              u32 offset, u32 count, f32 dt, f32 drag_coeff) {
    if (count == 0) return;
    particles.bind_all();
    bind_all();

    drag_shader_.bind();
    drag_shader_.set_uint("u_offset", offset);
    drag_shader_.set_uint("u_count", count);
    drag_shader_.set_ivec2("u_res", resolution_);
    drag_shader_.set_float("u_dx", dx_);
    drag_shader_.set_float("u_dt", dt);
    drag_shader_.set_float("u_drag", drag_coeff);
    drag_shader_.set_float("u_vapor_drag", config_.vapor_drag);
    drag_shader_.set_vec2("u_world_min", world_min_);
    drag_shader_.dispatch_1d(count);
    ComputeShader::barrier_ssbo();
}

void EulerianFluid::inject_heat_at(vec2 world_pos, f32 radius, f32 heat_power, f32 dt) {
    if (heat_power == 0.0f) return;
    u32 n = static_cast<u32>(resolution_.x * resolution_.y);
    bind_all();

    heat_gun_shader_.bind();
    heat_gun_shader_.set_ivec2("u_res", resolution_);
    heat_gun_shader_.set_float("u_dx", dx_);
    heat_gun_shader_.set_vec2("u_world_min", world_min_);
    heat_gun_shader_.set_vec2("u_gun_pos", world_pos);
    heat_gun_shader_.set_float("u_gun_radius", radius);
    heat_gun_shader_.set_float("u_gun_power", heat_power);
    heat_gun_shader_.set_float("u_dt", dt);
    heat_gun_shader_.dispatch_1d(n);
    ComputeShader::barrier_ssbo();
}

void EulerianFluid::blow_at(vec2 world_pos, vec2 direction, f32 radius, f32 strength, f32 dt) {
    if (strength == 0.0f) return;
    u32 n = static_cast<u32>(resolution_.x * resolution_.y);
    bind_all();

    blow_shader_.bind();
    blow_shader_.set_ivec2("u_res", resolution_);
    blow_shader_.set_float("u_dx", dx_);
    blow_shader_.set_vec2("u_world_min", world_min_);
    blow_shader_.set_vec2("u_pos", world_pos);
    blow_shader_.set_vec2("u_direction", direction);
    blow_shader_.set_float("u_radius", radius);
    blow_shader_.set_float("u_strength", strength);
    blow_shader_.set_float("u_dt", dt);
    blow_shader_.dispatch_1d(n);
    ComputeShader::barrier_ssbo();
}

void EulerianFluid::blast_at(vec2 world_pos, f32 inner_radius, f32 outer_radius,
                             f32 strength, f32 heat, f32 smoke, f32 divergence, f32 dt) {
    if (strength == 0.0f || outer_radius <= 1e-4f) return;
    u32 n = static_cast<u32>(resolution_.x * resolution_.y);
    bind_all();

    blast_shader_.bind();
    blast_shader_.set_ivec2("u_res", resolution_);
    blast_shader_.set_float("u_dx", dx_);
    blast_shader_.set_vec2("u_world_min", world_min_);
    blast_shader_.set_vec2("u_pos", world_pos);
    blast_shader_.set_float("u_inner_radius", glm::max(0.0f, inner_radius));
    blast_shader_.set_float("u_outer_radius", outer_radius);
    blast_shader_.set_float("u_strength", strength);
    blast_shader_.set_float("u_heat", heat);
    blast_shader_.set_float("u_smoke", smoke);
    blast_shader_.set_float("u_divergence", divergence);
    blast_shader_.set_float("u_dt", dt);
    blast_shader_.dispatch_1d(n);
    ComputeShader::barrier_ssbo();
}

void EulerianFluid::bind_for_vis(u32 unit) const {
    glBindTextureUnit(unit, smoke_tex_);
    glBindTextureUnit(unit + 1, temp_tex_);
}

} // namespace ng
