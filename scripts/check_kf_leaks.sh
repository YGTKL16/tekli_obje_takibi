#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$(mktemp -d)"
BIN="$BUILD_DIR/test_heap_valgrind"
trap 'rm -rf "$BUILD_DIR"' EXIT

if ! command -v valgrind >/dev/null 2>&1; then
  echo "[ERROR] valgrind is required but not installed." >&2
  exit 1
fi

g++ -std=c++17 \
  -I"$PROJECT_ROOT/cpp/include" \
  -I"$PROJECT_ROOT/third_party/eigen-install/include/eigen3" \
  "$PROJECT_ROOT/cpp/tests/test_heap_valgrind.cpp" \
  "$PROJECT_ROOT/cpp/src/kalman_filter.cpp" \
  "$PROJECT_ROOT/cpp/src/tracker_state.cpp" \
  -o "$BIN"

OUTPUT="$(
  valgrind --tool=memcheck --leak-check=full --error-exitcode=1 "$BIN" 2>&1
)"
printf '%s\n' "$OUTPUT"

grep -q "All heap blocks were freed -- no leaks are possible" <<<"$OUTPUT"
grep -q "ERROR SUMMARY: 0 errors from 0 contexts" <<<"$OUTPUT"

echo "[OK] Leak check passed: zero leaks, zero valgrind errors."
