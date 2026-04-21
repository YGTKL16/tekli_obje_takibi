---
description: "JSF AV C++ Rev C and MISRA C++ 2023 safety rules for tracker C++ core. Use when: writing or reviewing C++ code, adding new functions, modifying Kalman/IMM/association/GMC code."
applyTo: "cpp/**"
---

# C++ Safety Rules — JSF AV C++ Rev C + MISRA C++ 2023

## Memory

- No `new` / `delete` / `malloc` / `free` after construction
- All buffers pre-allocated in constructor, reused via index reset
- Fixed-size Eigen: `Matrix<double, 10, 10>`, never `MatrixXd`
- `EIGEN_MAX_ALIGN_BYTES=0` — pybind11 CPython allocator breaks Eigen alignment

## Error Handling

- No `throw` / `try` / `catch` in core modules (kalman, imm, tracker_state, association)
- Exception allowed ONLY in `tracker_gmc` (OpenCV may throw internally)
- All public functions must be `noexcept`
- Error states communicated via return values or state flags

## Type Safety

- No RTTI: no `dynamic_cast`, no `typeid`
- No implicit conversions between numeric types — use explicit casts
- All constants via `constexpr` named identifiers — no magic numbers
- Fixed-width types where width matters: `int32_t`, `double`

## Eigen Specifics

- Stack-only matrices: `StateVec` = `Matrix<double, 10, 1>`, `MeasVec` = `Matrix<double, 4, 1>`
- Numerical stability: use `S.llt().solve(v)` instead of `S.inverse() * v`
- Symmetry enforcement: `P = (P + P.transpose()) / 2.0` after every covariance update
- Positive-definiteness guard: eigenvalue floor if any eigenvalue < epsilon

## Style

- One class per file
- Header guards: `#ifndef TRACKER_<MODULE>_H_`
- Include order: own header → project headers → third-party → system
- `[[nodiscard]]` on functions returning error/status information
