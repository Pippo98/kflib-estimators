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

# Generate a .pyi stub (+ py.typed marker) from the freshly built module.
# Without this, the module is an opaque compiled binary with no static type
# information, so editors/LSPs (Pyright, Pylance, Jedi, ...) can't show
# argument names, types, or docstrings for it.
nanobind_add_stub(
  kflib_estimators_stub
  MODULE kflib_estimators
  OUTPUT kflib_estimators.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:kflib_estimators>
  DEPENDS kflib_estimators
  MARKER_FILE py.typed
)

# Installed as a standalone top-level module (no wrapping Python package),
# so it lands directly in site-packages as `import kflib_estimators`. Tagged
# with its own COMPONENT so `cmake --install --component kflib_estimators`
# (as scikit-build-core is configured to do) packages only this target,
# not the install() rules pulled in from Eigen/kflib/nanobind.
install(TARGETS kflib_estimators LIBRARY DESTINATION . COMPONENT kflib_estimators)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/kflib_estimators.pyi
  ${CMAKE_CURRENT_BINARY_DIR}/py.typed
  DESTINATION .
  COMPONENT kflib_estimators
)
