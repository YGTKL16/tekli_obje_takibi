// ---------------------------------------------------------------------------
// Standalone Valgrind heap-isolation test for tracker_core.
// No GoogleTest – pure tracker_core calls only.
// Build:  g++ -std=c++17 -I cpp/include test_heap_valgrind.cpp \
//              cpp/src/kalman_filter.cpp cpp/src/tracker_state.cpp \
//              -lEigen3 -o test_heap_valgrind
// Run:    valgrind --tool=memcheck ./test_heap_valgrind
// Expect: zero leaks and zero Valgrind errors; allocator startup traffic may
//         still appear in total heap usage.
// ---------------------------------------------------------------------------
#include "kalman_filter.h"
#include "tracker_state.h"
#include <cstdint>

int main() {
    // ── KalmanFilter exercise ───────────────────────────────────
    tracker::KalmanFilter kf;

    tracker::MeasVec z;
    z << 100.0F, 200.0F, 50.0F, 80.0F;

    static_cast<void>(kf.update(z));

    for (int32_t i = 0; i < 10; ++i) {
        static_cast<void>(kf.predict());
        z(0) += 1.0F;
        static_cast<void>(kf.update(z));
    }

    static_cast<void>(kf.get_state());
    static_cast<void>(kf.get_covariance());
    static_cast<void>(kf.is_initialized());

    kf.reset();
    static_cast<void>(kf.update(z));

    // ── TrackerState exercise ───────────────────────────────────
    tracker::TrackerState ts;
    ts.set_confidence_threshold(0.3F);
    ts.set_max_coast_frames(5);

    for (int32_t i = 0; i < 20; ++i) {
        float conf = (i % 3 == 0) ? 0.1F : 0.8F;
        static_cast<void>(ts.step(conf));
    }

    static_cast<void>(ts.state());
    static_cast<void>(ts.coast_count());
    ts.force_tracking();

    return 0;
}
