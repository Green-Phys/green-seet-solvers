project(green-seet-solvers-test)

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

option(EDLIB_BUILD_FIXTURE_TOOLS
       "Build the fixture-generator tool (only needed to regenerate test/data/*.h5)" OFF)
if(EDLIB_BUILD_FIXTURE_TOOLS)
    find_package(HDF5 COMPONENTS C REQUIRED)
    add_executable(make_fixtures tools/make_fixtures.cpp)
    if(TARGET hdf5::hdf5)
        target_link_libraries(make_fixtures PRIVATE hdf5::hdf5)
    else()
        target_link_libraries(make_fixtures PRIVATE ${HDF5_C_LIBRARIES})
        target_include_directories(make_fixtures PRIVATE ${HDF5_C_INCLUDE_DIRS})
    endif()
endif()

set(_data_input "${CMAKE_CURRENT_SOURCE_DIR}/data/input.h5")
set(_data_freq  "${CMAKE_CURRENT_SOURCE_DIR}/data/freq.h5")
set(_sim_out    "${CMAKE_CURRENT_BINARY_DIR}/sim.h5")

# Catch2 test calls seet::run() directly against the committed data;
# all three paths are baked in via compile definitions.
add_executable(solver_check_test solver_check_test.cpp)
target_compile_definitions(solver_check_test PRIVATE
    DATA_INPUT_PATH="${_data_input}"
    DATA_FREQ_PATH="${_data_freq}"
    SIM_H5_PATH="${_sim_out}")
target_link_libraries(solver_check_test PRIVATE
    Catch2::Catch2WithMain
    green_seet_ed_solver)

catch_discover_tests(solver_check_test)
