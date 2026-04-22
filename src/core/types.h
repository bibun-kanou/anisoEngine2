#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace ng {

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;

using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;
using ivec2 = glm::ivec2;
using uvec2 = glm::uvec2;
using mat2 = glm::mat2;
using mat3 = glm::mat3;
using mat4 = glm::mat4;

constexpr f32 PI = glm::pi<f32>();

} // namespace ng
