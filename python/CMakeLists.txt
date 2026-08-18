option(BUILD_PYTHON_BINDINGS "Build nanobind Python bindings for FiveStateEstimator" ON)

if(NOT BUILD_PYTHON_BINDINGS)
  return()
endif()

if(CMAKE_VERSION VERSION_LESS 3.18)
  set(NB_DEV_MODULE Development)
else()
  set(NB_DEV_MODULE Development.Module)
endif()

find_package(Python 3.10 COMPONENTS Interpreter ${NB_DEV_MODULE})
if(NOT Python_FOUND)
  message(STATUS "Python bindings disabled: no suitable Python interpreter/development headers found")
  return()
endif()

# nanobind's own sources are third-party and not guaranteed clean under our
# strict warning flags; strip -Werror for the subdirectory that builds it.
string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

add_subdirectory(${CMAKE_SOURCE_DIR}/external/nanobind
                  ${CMAKE_BINARY_DIR}/external/nanobind)

nanobind_add_module(kflib_estimators bindings.cpp)
target_link_libraries(kflib_estimators PRIVATE core)
