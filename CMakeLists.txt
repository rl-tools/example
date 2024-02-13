#set(RL_TOOLS_BACKEND_ENABLE_MKL ON) # if you have MKL installed (fastest on Intel)
#set(RL_TOOLS_BACKEND_ENABLE_OPENBLAS ON) # if you have OpenBLAS installed
#set(RL_TOOLS_BACKEND_ENABLE_ACCELERATE ON) # if you are on macOS (fastest on Apple Silicon)
add_subdirectory(external/rl_tools)

add_executable(my_pendulum
    src/main.cpp
)
target_link_libraries(my_pendulum PRIVATE RLtools::RLtools)



