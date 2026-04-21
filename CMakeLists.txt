cmake_minimum_required(VERSION 3.18)
project(tracker LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build unit tests" ON)
option(TRACKER_ENABLE_RUST_BRIDGE "Build the optional Rust safety helper library" OFF)
option(TRACKER_OPEN_SOURCE_TOOLCHAIN "Build with an open-source high-assurance toolchain profile" OFF)

set(TRACKER_BUILD_PROFILE "host" CACHE STRING "Compiler hardening profile")
set_property(CACHE TRACKER_BUILD_PROFILE PROPERTY STRINGS host high_assurance)

set(TRACKER_RUST_PROFILE "release" CACHE STRING "Cargo profile for tracker_rust_bridge")
set_property(CACHE TRACKER_RUST_PROFILE PROPERTY STRINGS release do178c)

if(NOT TRACKER_BUILD_PROFILE STREQUAL "host" AND
   NOT TRACKER_BUILD_PROFILE STREQUAL "high_assurance")
    message(FATAL_ERROR
        "TRACKER_BUILD_PROFILE must be one of: host, high_assurance"
    )
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TRACKER_GNU_LIKE OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(TRACKER_GNU_LIKE ON)
endif()

# ── Baseline compiler flags ───────────────────────────────────────
set(TRACKER_COMMON_COMPILE_OPTIONS)
if(TRACKER_GNU_LIKE)
    list(APPEND TRACKER_COMMON_COMPILE_OPTIONS
        -O2
        -Wall
        -Wextra
        -Werror
        -Wno-unused-parameter
    )

    if(TRACKER_BUILD_PROFILE STREQUAL "host")
        list(APPEND TRACKER_COMMON_COMPILE_OPTIONS
            -march=native
        )
    endif()
endif()

add_compile_options(${TRACKER_COMMON_COMPILE_OPTIONS})

if(TRACKER_BUILD_PROFILE STREQUAL "high_assurance" AND NOT TRACKER_OPEN_SOURCE_TOOLCHAIN)
    message(WARNING
        "TRACKER_BUILD_PROFILE=high_assurance selected without "
        "TRACKER_OPEN_SOURCE_TOOLCHAIN=ON. This enables hardening, but it does "
        "not constitute certification evidence."
    )
endif()

message(STATUS "Tracker build profile: ${TRACKER_BUILD_PROFILE}")
message(STATUS "Open-source toolchain mode: ${TRACKER_OPEN_SOURCE_TOOLCHAIN}")
message(STATUS "Rust bridge enabled: ${TRACKER_ENABLE_RUST_BRIDGE}")

# ── Dependencies ─────────────────────────────────────────────────
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d calib3d)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED)
endif()

# ── C++ core library (static, no dynamic alloc) ─────────────────
add_library(tracker_core STATIC
    cpp/src/association.cpp
    cpp/src/kalman_filter.cpp
    cpp/src/imm_filter.cpp
    cpp/src/tracker_state.cpp
)
set_target_properties(tracker_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tracker_core PUBLIC cpp/include)
target_link_libraries(tracker_core PUBLIC Eigen3::Eigen)
# Eigen: allow unaligned vectorized ops — avoids SIGSEGV when pybind11's
# CPython allocator doesn't honour alignof(T).  Zero perf impact on ≤10×10.
target_compile_definitions(tracker_core PUBLIC EIGEN_MAX_ALIGN_BYTES=0)

if(TRACKER_BUILD_PROFILE STREQUAL "high_assurance")
    target_compile_definitions(tracker_core PUBLIC TRACKER_HIGH_ASSURANCE=1)
endif()

set(TRACKER_CORE_HARDENING_OPTIONS)
if(TRACKER_GNU_LIKE)
    list(APPEND TRACKER_CORE_HARDENING_OPTIONS
        -fno-exceptions
        -fno-rtti
        -pedantic
        -Wshadow
        -Wconversion
        -Wsign-conversion
        -Wold-style-cast
        -Wzero-as-null-pointer-constant
    )
endif()
target_compile_options(tracker_core PRIVATE ${TRACKER_CORE_HARDENING_OPTIONS})

# ── GMC library (ORB + partial-affine, uses OpenCV) ──────────────
# OpenCV may throw internally, so -fexceptions is required here.
add_library(tracker_gmc STATIC cpp/src/gmc_estimator.cpp)
set_target_properties(tracker_gmc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tracker_gmc PUBLIC cpp/include)
target_link_libraries(tracker_gmc PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
target_compile_definitions(tracker_gmc PUBLIC EIGEN_MAX_ALIGN_BYTES=0)
if(TRACKER_GNU_LIKE)
    target_compile_options(tracker_gmc PRIVATE
        -fexceptions -fno-rtti
        -pedantic -Wshadow -Wconversion -Wsign-conversion
        -Wold-style-cast -Wzero-as-null-pointer-constant
    )
endif()

# ── C ABI bridge for association (Python ctypes-friendly) ─────────────────
add_library(tracker_association SHARED cpp/src/association_c_api.cpp)
target_include_directories(tracker_association PUBLIC cpp/include)
target_link_libraries(tracker_association PRIVATE tracker_core)
if(TRACKER_BUILD_PROFILE STREQUAL "high_assurance")
    target_compile_definitions(tracker_association PRIVATE TRACKER_HIGH_ASSURANCE=1)
endif()
if(TRACKER_GNU_LIKE)
    target_compile_options(tracker_association PRIVATE -fno-exceptions -fno-rtti)
endif()

# ── Pybind11 Python module ───────────────────────────────────────
pybind11_add_module(tracker_cpp cpp/binding/binding.cpp)
target_link_libraries(tracker_cpp PRIVATE tracker_core tracker_gmc)
if(TRACKER_BUILD_PROFILE STREQUAL "high_assurance")
    target_compile_definitions(tracker_cpp PRIVATE TRACKER_HIGH_ASSURANCE=1)
endif()
if(TRACKER_GNU_LIKE)
    target_compile_options(tracker_cpp PRIVATE -fexceptions -frtti)
endif()

# ── Optional Rust safety bridge ──────────────────────────────────
if(TRACKER_ENABLE_RUST_BRIDGE)
    set(TRACKER_CARGO_EXECUTABLE "" CACHE FILEPATH
        "Path to cargo executable for tracker_rust_bridge"
    )
    set(TRACKER_RUSTC_EXECUTABLE "" CACHE FILEPATH
        "Path to rustc executable for tracker_rust_bridge"
    )

    if(TRACKER_CARGO_EXECUTABLE)
        set(_tracker_cargo "${TRACKER_CARGO_EXECUTABLE}")
    elseif(DEFINED ENV{TRACKER_CARGO_BIN} AND
           NOT "$ENV{TRACKER_CARGO_BIN}" STREQUAL "")
        set(_tracker_cargo "$ENV{TRACKER_CARGO_BIN}")
    else()
        find_program(_tracker_cargo NAMES cargo REQUIRED)
    endif()

    if(TRACKER_RUSTC_EXECUTABLE)
        set(_tracker_rustc "${TRACKER_RUSTC_EXECUTABLE}")
    elseif(DEFINED ENV{TRACKER_RUSTC_BIN} AND
           NOT "$ENV{TRACKER_RUSTC_BIN}" STREQUAL "")
        set(_tracker_rustc "$ENV{TRACKER_RUSTC_BIN}")
    else()
        find_program(_tracker_rustc NAMES rustc REQUIRED)
    endif()

    set(_tracker_cargo_env
        "CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/cargo-target"
        "CARGO_INCREMENTAL=0"
        "RUSTC_WRAPPER="
        "RUSTC_WORKSPACE_WRAPPER="
        "RUSTC=${_tracker_rustc}"
    )

    # Map Cargo profile name to output directory name
    if(TRACKER_RUST_PROFILE STREQUAL "release")
        set(_cargo_out_dir "release")
    elseif(TRACKER_RUST_PROFILE STREQUAL "do178c")
        set(_cargo_out_dir "do178c")
    else()
        set(_cargo_out_dir "debug")
    endif()

    set(_rust_lib_path
        "${CMAKE_BINARY_DIR}/cargo-target/${_cargo_out_dir}/libtracker_safety.a")

    add_custom_command(
        OUTPUT "${_rust_lib_path}"
        COMMAND ${CMAKE_COMMAND} -E env
            ${_tracker_cargo_env}
            ${_tracker_cargo}
            build
            --config build.rustc-wrapper=""
            --config build.rustc-workspace-wrapper=""
            --manifest-path ${CMAKE_SOURCE_DIR}/rust/tracker_safety/Cargo.toml
            --profile ${TRACKER_RUST_PROFILE}
            --offline
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Building Rust safety bridge (${TRACKER_RUST_PROFILE})"
        DEPENDS ${CMAKE_SOURCE_DIR}/rust/tracker_safety/src/lib.rs
                ${CMAKE_SOURCE_DIR}/rust/tracker_safety/Cargo.toml
        VERBATIM
    )

    add_custom_target(tracker_rust_bridge DEPENDS "${_rust_lib_path}")

    add_library(tracker_safety_imported STATIC IMPORTED GLOBAL)
    set_target_properties(tracker_safety_imported PROPERTIES
        IMPORTED_LOCATION "${_rust_lib_path}"
    )
    add_dependencies(tracker_safety_imported tracker_rust_bridge)

    target_compile_definitions(tracker_core PUBLIC TRACKER_RUST_BRIDGE_ENABLED=1)
    target_link_libraries(tracker_core PUBLIC tracker_safety_imported)
endif()

# ── GoogleTest (optional, for dev builds) ────────────────────────
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()

    add_executable(test_kalman cpp/tests/test_kalman.cpp)
    target_link_libraries(test_kalman PRIVATE tracker_core GTest::gtest_main)
    if(TRACKER_GNU_LIKE)
        target_compile_options(test_kalman PRIVATE -fexceptions -frtti)
    endif()
    add_test(NAME KalmanTests COMMAND test_kalman)

    add_executable(test_coasting cpp/tests/test_coasting.cpp)
    target_link_libraries(test_coasting PRIVATE tracker_core GTest::gtest_main)
    if(TRACKER_GNU_LIKE)
        target_compile_options(test_coasting PRIVATE -fexceptions -frtti)
    endif()
    add_test(NAME CoastingTests COMMAND test_coasting)

    add_executable(test_imm cpp/tests/test_imm.cpp)
    target_link_libraries(test_imm PRIVATE tracker_core GTest::gtest_main)
    if(TRACKER_GNU_LIKE)
        target_compile_options(test_imm PRIVATE -fexceptions -frtti)
    endif()
    add_test(NAME IMMTests COMMAND test_imm)

    add_executable(test_association cpp/tests/test_association.cpp)
    target_link_libraries(test_association PRIVATE tracker_core GTest::gtest_main)
    if(TRACKER_GNU_LIKE)
        target_compile_options(test_association PRIVATE -fexceptions -frtti)
    endif()
    add_test(NAME AssociationTests COMMAND test_association)

    add_executable(test_gmc_estimator cpp/tests/test_gmc_estimator.cpp)
    target_link_libraries(test_gmc_estimator PRIVATE tracker_gmc GTest::gtest_main)
    if(TRACKER_GNU_LIKE)
        target_compile_options(test_gmc_estimator PRIVATE -fexceptions -frtti)
    endif()
    add_test(NAME GMCEstimatorTests COMMAND test_gmc_estimator)

    if(TRACKER_ENABLE_RUST_BRIDGE)
        add_executable(test_safety_bridge cpp/tests/test_safety_bridge.cpp)
        target_link_libraries(test_safety_bridge PRIVATE tracker_core GTest::gtest_main)
        if(TRACKER_GNU_LIKE)
            target_compile_options(test_safety_bridge PRIVATE -fexceptions -frtti)
        endif()
        add_test(NAME SafetyBridgeTests COMMAND test_safety_bridge)
    endif()
endif()
