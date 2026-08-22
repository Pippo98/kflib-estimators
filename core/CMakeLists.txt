
add_library(
  core
  src/4_state_estimator.cpp
  src/5_state_estimator.cpp
)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)
target_link_libraries(core PUBLIC kflib)

add_executable(5_states_estimator executables/5_states_estimator/main.cpp)
target_link_libraries(5_states_estimator PRIVATE core)
