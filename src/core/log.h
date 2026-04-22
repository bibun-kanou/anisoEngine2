#pragma once

#include <cstdio>
#include <cstdlib>

#define LOG_INFO(fmt, ...)  do { std::printf("[INFO]  " fmt "\n", ##__VA_ARGS__); std::fflush(stdout); } while(0)
#define LOG_WARN(fmt, ...)  do { std::printf("[WARN]  " fmt "\n", ##__VA_ARGS__); std::fflush(stdout); } while(0)
#define LOG_ERROR(fmt, ...) do { std::fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__); std::fflush(stderr); } while(0)
#define LOG_FATAL(fmt, ...) do { std::fprintf(stderr, "[FATAL] " fmt "\n", ##__VA_ARGS__); std::abort(); } while(0)

#ifdef NDEBUG
#define LOG_DEBUG(fmt, ...) ((void)0)
#else
#define LOG_DEBUG(fmt, ...) std::printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
#endif
