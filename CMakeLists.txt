project(rl-tools-example)
#set(RL_TOOLS_BACKEND_ENABLE_MKL ON) # if you have MKL installed (fastest on Intel)
#set(RL_TOOLS_BACKEND_ENABLE_OPENBLAS ON) # if you have OpenBLAS installed
#set(RL_TOOLS_BACKEND_ENABLE_ACCELERATE ON) # if you are on macOS (fastest on Apple Silicon)
add_subdirectory(external/rl_tools)

add_executable(my_pendulum src/main.cpp)
target_link_libraries(my_pendulum PRIVATE RLtools::RLtools)

# The following target disables evaluations and checkpointing during training to assess the training time
add_executable(my_pendulum_benchmark src/main.cpp)
target_compile_definitions(my_pendulum_benchmark PRIVATE BENCHMARK)
target_link_libraries(my_pendulum_benchmark PRIVATE RLtools::RLtools)



if(NOT MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(my_pendulum PRIVATE -Ofast)
    if(NOT APPLE)
        target_compile_options(my_pendulum PRIVATE -march=native)
    endif()
endif()