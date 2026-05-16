cmake_minimum_required(VERSION 3.18)
project(GreenSeetSolvers LANGUAGES C CXX
        VERSION 2.0)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(BLAS)
find_package(LAPACK)
find_package(Eigen3 REQUIRED)

include(GreenDeps)
include(EDLibDep)

option(USE_MPI "Use MPI for ED" OFF)
if(USE_MPI)
    find_package(MPI REQUIRED)
    add_compile_definitions(USE_MPI)
endif()

add_green_dependency(green-h5pp)
add_green_dependency(green-params)
add_edlib_dependency()

add_library(green_seet_solver STATIC seet_solver.cpp)
target_include_directories(green_seet_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(green_seet_solver PUBLIC
    EDLib::EDLib
    GREEN::H5PP
    GREEN::PARAMS
    Eigen3::Eigen
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES})

if(USE_MPI)
    target_link_libraries(green_seet_solver PUBLIC MPI::MPI_CXX)
endif()

add_executable(ed_solver Anderson.cpp)
target_link_libraries(ed_solver PRIVATE green_seet_solver)

install(TARGETS ed_solver DESTINATION bin)

option(Build_Tests "Build green-seet-solvers tests" OFF)
message(STATUS "Building tests ${Build_Tests}")
if(Build_Tests)
    enable_testing()
    add_subdirectory(test)
endif()
