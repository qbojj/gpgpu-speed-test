cmake_minimum_required(VERSION 3.8)
project(gpgpu-speed-test)

set(CMAKE_CXX_STANDARD 17)

set(KOMPUTE_OPT_LOG_LEVEL "Warn")
set(KOMPUTE_OPT_DISABLE_VK_DEBUG_LAYERS On)

add_subdirectory(kompute)

add_executable(gpgpu-speed-test main.cpp cpu_ops.cpp)
target_link_libraries(gpgpu-speed-test PUBLIC kompute::kompute)