#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build-high-assurance"
BUILD_TESTS="OFF"
ENABLE_RUST="ON"
TOOLCHAIN="gcc"

usage() {
    cat <<'EOF'
Usage: scripts/configure_open_toolchains.sh [options]

Options:
  --build-dir <dir>       CMake build directory (default: build-high-assurance)
  --toolchain <name>      gcc or clang (default: gcc)
  --with-tests            Keep C++ tests enabled
  --without-rust          Skip the optional Rust bridge target
  --help                  Show this help

Optional environment variables:
  TRACKER_CMAKE_BIN
  TRACKER_C_COMPILER_BIN
  TRACKER_CXX_COMPILER_BIN
  TRACKER_RUSTC_BIN
  TRACKER_CARGO_BIN
EOF
}

while (($# > 0)); do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --toolchain)
            TOOLCHAIN="$2"
            shift 2
            ;;
        --with-tests)
            BUILD_TESTS="ON"
            shift
            ;;
        --without-rust)
            ENABLE_RUST="OFF"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

command_path_or_empty() {
    local cmd="$1"
    command -v "$cmd" 2>/dev/null || true
}

pick_default_compilers() {
    case "${TOOLCHAIN}" in
        gcc)
            export TRACKER_C_COMPILER_BIN="${TRACKER_C_COMPILER_BIN:-$(command_path_or_empty gcc)}"
            if [[ -z "${TRACKER_C_COMPILER_BIN}" ]]; then
                export TRACKER_C_COMPILER_BIN="${TRACKER_C_COMPILER_BIN:-$(command_path_or_empty cc)}"
            fi
            export TRACKER_CXX_COMPILER_BIN="${TRACKER_CXX_COMPILER_BIN:-$(command_path_or_empty g++)}"
            if [[ -z "${TRACKER_CXX_COMPILER_BIN}" ]]; then
                export TRACKER_CXX_COMPILER_BIN="${TRACKER_CXX_COMPILER_BIN:-$(command_path_or_empty c++)}"
            fi
            TOOLCHAIN_FILE="${repo_root}/cmake/toolchains/opensource_gcc.cmake"
            ;;
        clang)
            export TRACKER_C_COMPILER_BIN="${TRACKER_C_COMPILER_BIN:-$(command_path_or_empty clang)}"
            export TRACKER_CXX_COMPILER_BIN="${TRACKER_CXX_COMPILER_BIN:-$(command_path_or_empty clang++)}"
            TOOLCHAIN_FILE="${repo_root}/cmake/toolchains/opensource_clang.cmake"
            ;;
        *)
            echo "[ERROR] Unsupported toolchain: ${TOOLCHAIN}" >&2
            exit 1
            ;;
    esac
}

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "[ERROR] Missing required environment variable: ${name}" >&2
        exit 1
    fi
}

pick_default_compilers
require_env TRACKER_C_COMPILER_BIN
require_env TRACKER_CXX_COMPILER_BIN

CMAKE_BIN="${TRACKER_CMAKE_BIN:-$(command_path_or_empty cmake)}"
if [[ -z "${CMAKE_BIN}" ]]; then
    echo "[ERROR] cmake not found. Set TRACKER_CMAKE_BIN to your CMake executable." >&2
    exit 1
fi

if [[ "${ENABLE_RUST}" == "ON" ]]; then
    export TRACKER_RUSTC_BIN="${TRACKER_RUSTC_BIN:-$(command_path_or_empty rustc)}"
    export TRACKER_CARGO_BIN="${TRACKER_CARGO_BIN:-$(command_path_or_empty cargo)}"
    require_env TRACKER_RUSTC_BIN
    require_env TRACKER_CARGO_BIN
fi

echo "[INFO] Configuring open-source high-assurance build in ${BUILD_DIR}"
echo "[INFO] Toolchain  : ${TOOLCHAIN}"
echo "[INFO] CMake      : ${CMAKE_BIN}"
echo "[INFO] C compiler : ${TRACKER_C_COMPILER_BIN}"
echo "[INFO] C++ compiler: ${TRACKER_CXX_COMPILER_BIN}"
if [[ "${ENABLE_RUST}" == "ON" ]]; then
    echo "[INFO] Rustc      : ${TRACKER_RUSTC_BIN}"
    echo "[INFO] Cargo      : ${TRACKER_CARGO_BIN}"
fi

"${CMAKE_BIN}" -S "${repo_root}" -B "${repo_root}/${BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DTRACKER_BUILD_PROFILE=high_assurance \
    -DTRACKER_OPEN_SOURCE_TOOLCHAIN=ON \
    -DTRACKER_ENABLE_RUST_BRIDGE="${ENABLE_RUST}" \
    -DTRACKER_RUST_PROFILE=do178c \
    -DBUILD_TESTS="${BUILD_TESTS}"

echo "[OK] Configuration complete."
echo "[NEXT] C++ build : ${CMAKE_BIN} --build ${repo_root}/${BUILD_DIR}"
if [[ "${ENABLE_RUST}" == "ON" ]]; then
    echo "[NEXT] Rust build: ${CMAKE_BIN} --build ${repo_root}/${BUILD_DIR} --target tracker_rust_bridge"
fi
