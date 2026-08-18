
add_library(
  core
  src/5_states_estimator.cpp
)

add_executable(5_states_estimator executables/5_states_estimator/main.cpp)
target_link_libraries(5_states_estimator PRIVATE kflib)
