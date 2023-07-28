cmake_minimum_required(VERSION 3.8)
project(gpgpu-speed-test)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(kompute)

add_executable(gpgpu-speed-test main.cpp)
target_link_libraries(gpgpu-speed-test PUBLIC kompute::kompute)