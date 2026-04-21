# Open-Source High-Assurance Toolchains

[< Back to Index](README.md)

## Purpose

This repository now supports an open-source build path for high-assurance deployments.
The goal is to use GCC or Clang for C/C++ and `rustc` for Rust without pretending that
certification comes from the repository alone.

What this layer adds:

- A `high_assurance` CMake profile that removes host-specific codegen flags such as `-march=native`
- Open-source CMake toolchain files for GCC and Clang
- An optional Rust safety bridge crate with a dedicated `do178c` Cargo profile
- A single setup script to configure the project in a repeatable way

What it does not do:

- Turn GCC, Clang, or `rustc` into ready-made certification evidence
- Replace project-specific certification evidence or verification artifacts
- Claim certification for the system by itself

## Supported Layout

The configuration layer accepts environment variables rather than hard-coded local paths:

```bash
export TRACKER_CMAKE_BIN=/usr/bin/cmake
export TRACKER_C_COMPILER_BIN=/usr/bin/gcc
export TRACKER_CXX_COMPILER_BIN=/usr/bin/g++
export TRACKER_RUSTC_BIN=/usr/bin/rustc
export TRACKER_CARGO_BIN=/usr/bin/cargo
```

This keeps the repository explicit and reproducible while staying fully open-source.

## Configure

```bash
bash scripts/configure_open_toolchains.sh
```

For a Clang-based build:

```bash
bash scripts/configure_open_toolchains.sh --toolchain clang
```

Even with these open-source compilers, this path is still hardening-oriented, not certification evidence.

## Build

```bash
cmake --build build-high-assurance
cmake --build build-high-assurance --target tracker_rust_bridge
```

The optional Rust target builds `rust/tracker_safety` with `cargo --offline` and the `do178c`
profile so we do not pull network dependencies into the high-assurance path. The build target
also clears wrapper settings through Cargo configuration and environment variables so host-side
cache wrappers do not leak into the controlled build path.

## Rust Bridge

The first Rust module is intentionally small. It implements a pure, allocation-free bounding-box
sanitizer as normal Rust logic and builds as both `rlib` and `staticlib`. This gives us a safe
place to start introducing verification-friendly Rust components without disturbing the existing
Python and C++ pipeline. The explicit C ABI layer can be added later once the runtime contract
and integration boundary are fixed.

## Protocol Status

The current repository state is intentionally conservative:

- `#![forbid(unsafe_code)]` is enabled in the Rust safety crate.
- `panic = "abort"` is enabled in the Rust release-oriented profiles.
- A full `#![no_std]` switch is not enabled in the default host build yet.

That last point is deliberate. The current tracker stack is still a hosted Linux pipeline with
Python, OpenCV, TensorRT, and C++ integration around the Rust helper crate. A production
`no_std` transition needs a dedicated target/runtime contract, panic strategy, and integration
boundary rather than a one-line crate attribute.

Verification tools such as Kani or Flux are also not wired into the default build. They are best
introduced per critical algorithm with explicit proof harnesses, review criteria, and CI gates.

## Certification Boundary

In practice, airborne software acceptance still depends on:

- The project plan, verification evidence, and configuration management
- The exact compiler version, switches, target, and runtime profile used for release
- Additional project-level evidence if you later choose to certify an airborne configuration

Use this repository layer as the integration spine, not as the certification claim itself.
