# MISRA C++ 2023 Traceability Matrix — Tracker Core

> **Scope**: `cpp/include/` and `cpp/src/` (tracker_core static library)
> **Standard**: MISRA C++ 2023 (based on C++17)
> **Baseline**: JSF AV C++ Rev C (fully compliant)

## Legend

| Disposition | Meaning |
|---|---|
| **C** | Compliant |
| **N/A** | Not applicable to this codebase |
| **D** | Deviation (documented rationale) |

---

## Rule Mapping

| MISRA Rule | Category | Description | Disposition | Evidence |
|---|---|---|---|---|
| **M0.1.1** | Required | No dead code | C | `-Wall -Wextra -Werror` with `-Wunused-parameter` suppressed only for interface stability |
| **M0.1.3** | Advisory | Unused variable | C | JSF AV single-exit pattern uses `static_cast<void>()` for intentional discards |
| **M0.3.1** | Required | Fixed-width integer types | C | `int32_t`, `uint8_t` throughout; `kalman_types.h` uses `int32_t` for dimension constants |
| **M2.2.1** | Required | Single responsibility per file | C | Each `.h`/`.cpp` pair has one class: `KalmanFilter`, `TrackerState` |
| **M2.7.1** | Required | `//` comment style only | C | All comments use `//` or `///` (Doxygen); no `/* */` blocks |
| **M5.0.1** | Required | No implicit conversions changing signedness | C | `-Wsign-conversion -Wconversion` enforced on tracker_core |
| **M5.2.1** | Required | No old-style casts | C | `-Wold-style-cast` enforced; all casts use `static_cast<>` |
| **M6.2.1** | Required | All variables initialized at declaration | C | All class members initialized in constructor or default member initializers |
| **M6.4.1** | Required | Enum class instead of plain enum | C | `tracker_state.h`: `enum class TrackState : uint8_t` |
| **M7.0.1** | Required | `noexcept` specification on non-throwing functions | C | All tracker_core public/private methods annotated `noexcept` |
| **M7.0.2** | Required | No exception specifications other than `noexcept` | C | `-fno-exceptions`; only `noexcept` used |
| **M7.1.1** | Required | `noexcept` on move operations | C | Rule-of-5 move constructor/assignment `noexcept = default` |
| **M7.1.2** | Advisory | Type aliases with `using` | C | `kalman_types.h` uses `using StateVec = ...` (no `typedef`) |
| **M8.2.1** | Required | Move semantics for Rule-of-5 | C | All five special members declared explicitly `= default` with `noexcept` |
| **M8.14.1** | Required | `[[nodiscard]]` on value-returning functions | C | `predict()`, `update()`, `get_state()`, `get_covariance()`, `is_initialized()` all `[[nodiscard]]` |
| **M9.3.1** | Required | No dynamic memory allocation | C | Eigen fixed-size matrices; `static_assert` verifies no heap; `-fno-exceptions` prevents `new` |
| **M9.6.1** | Required | No RTTI usage | C | `-fno-rtti` enforced on tracker_core |
| **M10.0.1** | Required | No `#define` for constants | C | All constants use `constexpr int32_t`; only `#define` is include guard and `TRACKER_ASSERT` |
| **M10.3.1** | Required | Include guards on all headers | C | `#ifndef / #define / #endif` pattern on all `.h` files |
| **M15.0.1** | Required | No `goto` | C | Zero `goto` statements |
| **M16.0.1** | Required | No `union` types | C | Zero `union` declarations |
| **M16.6.1** | Required | No pointer arithmetic | C | Zero raw pointer usage in tracker_core (Eigen handles memory) |
| **M19.0.1** | Required | No `<cstdio>` / `printf` in core | C | No I/O in tracker_core; only test/binding code uses output |
| **M21.2.1** | Required | No `reinterpret_cast` | C | Zero `reinterpret_cast` in tracker_core |
| **M27.0.1** | Advisory | Static analysis in CI | C | cppcheck + clang-tidy in `.github/workflows/ci.yml` |

---

## Deviations

| ID | Rule | Justification |
|---|---|---|
| **DEV-01** | M10.0.1 | `TRACKER_ASSERT` macro provides assert-like runtime checking without exception overhead. Cannot be expressed as `constexpr`. Defined once in `tracker_state.h`. |
| **DEV-02** | M0.1.1 | `-Wno-unused-parameter` applied globally to accommodate interface parameters that may be unused in some build configurations (e.g., `TRACKER_RUST_BRIDGE_ENABLED`). |

---

## Compiler Enforcement

The following flags on the `tracker_core` target enforce MISRA-adjacent rules at compile time:

```
-fno-exceptions        # M7.0.2, M9.3.1
-fno-rtti              # M9.6.1
-pedantic              # General conformance
-Wshadow               # M6.2.1
-Wconversion           # M5.0.1
-Wsign-conversion      # M5.0.1
-Wold-style-cast       # M5.2.1
-Wzero-as-null-pointer-constant  # M5.2.1
-Wall -Wextra -Werror  # Baseline quality
```

## Verification

- **Build**: `cmake -B build -DBUILD_TESTS=ON && cmake --build build` — zero warnings
- **C++ Tests**: `ctest --test-dir build` — 3/3 pass (KalmanTests, CoastingTests, SafetyBridgeTests)
- **Static Analysis**: cppcheck zero findings, clang-tidy zero bug-category findings
- **Valgrind**: 0 leaks, 0 reported errors
- **Rust Bridge Tests**: `cargo test` — 3/3 pass
- **Python Tests**: `pytest` — 31/31 pass
