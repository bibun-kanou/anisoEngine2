#include "physics/common/grid.h"
#include "core/log.h"

namespace ng {

void UniformGrid::init(const Config& config) {
    resolution_ = config.resolution;
    world_min_ = config.world_min;
    world_max_ = config.world_max;
    dx_ = (world_max_.x - world_min_.x) / static_cast<f32>(resolution_.x);

    u32 n = total_cells();
    mass_buf_.create(n * sizeof(i32));  // int for atomic scatter
    mom_x_buf_.create(n * sizeof(i32));
    mom_y_buf_.create(n * sizeof(i32));
    vel_x_buf_.create(n * sizeof(f32));
    vel_y_buf_.create(n * sizeof(f32));
    temp_buf_.create(n * sizeof(f32));
    temp2_buf_.create(n * sizeof(f32));
    temp_atomic_buf_.create(n * sizeof(i32));
    contact_temp_buf_.create(n * sizeof(i32));

    LOG_INFO("UniformGrid: %dx%d, dx=%.4f, domain [%.1f,%.1f]-[%.1f,%.1f]",
        resolution_.x, resolution_.y, dx_,
        world_min_.x, world_min_.y, world_max_.x, world_max_.y);
}

void UniformGrid::bind_for_p2g() const {
    mass_buf_.bind_base(BIND_GRID_MASS);
    mom_x_buf_.bind_base(BIND_GRID_MOM_X);
    mom_y_buf_.bind_base(BIND_GRID_MOM_Y);
}

void UniformGrid::bind_for_grid() const {
    mass_buf_.bind_base(BIND_GRID_MASS);
    mom_x_buf_.bind_base(BIND_GRID_MOM_X);
    mom_y_buf_.bind_base(BIND_GRID_MOM_Y);
    vel_x_buf_.bind_base(BIND_GRID_VEL_X);
    vel_y_buf_.bind_base(BIND_GRID_VEL_Y);
    temp_buf_.bind_base(BIND_GRID_TEMP);
    temp2_buf_.bind_base(BIND_GRID_TEMP2);
    contact_temp_buf_.bind_base(BIND_GRID_CONTACT_TEMP);
}

void UniformGrid::bind_for_g2p() const {
    vel_x_buf_.bind_base(BIND_GRID_VEL_X);
    vel_y_buf_.bind_base(BIND_GRID_VEL_Y);
    temp_buf_.bind_base(BIND_GRID_TEMP);
    contact_temp_buf_.bind_base(BIND_GRID_CONTACT_TEMP);
}

} // namespace ng
