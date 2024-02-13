#set(RL_TOOLS_BACKEND_ENABLE_MKL ON) # if you have MKL installed (fastest on Intel)
#set(RL_TOOLS_BACKEND_ENABLE_OPENBLAS ON) # if you have OpenBLAS installed
#set(RL_TOOLS_BACKEND_ENABLE_ACCELERATE ON) # if you are on macOS (fastest on Apple Silicon)
set(RL_TOOLS_DISABLE_TARGETS ON)
add_subdirectory(external/rl_tools)

add_executable(my_pendulum
    src/main.cpp
)
#target_compile_definitions(my_pendulum PRIVATE BENCHMARK)
target_link_libraries(my_pendulum PRIVATE RLtools::RLtools)
#target_compile_options(my_pendulum PRIVATE -Ofast)
#target_compile_options(my_pendulum PRIVATE -march=native)



