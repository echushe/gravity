// Minimal test helpers shared by the test programs.
//
// Each test is a standalone executable. CHECK* macros record failures instead
// of aborting, and test::run() prints PASS / FAIL / SKIP and returns the
// process exit code (0 = pass or skip, 1 = fail).
#pragma once

#include <cmath>
#include <cstdio>
#include <exception>

#include "gravity/gravity.h"

namespace test {

inline int failures = 0;

// Runs `body`, catching unexpected exceptions. Skips (exit code 0) when no
// CUDA device is available.
template <typename Fn>
int run(const char* name, Fn&& body) {
    if (gravity::device_count() == 0) {
        std::printf("[SKIP] %s: no CUDA device available\n", name);
        return 0;
    }
    try {
        body();
    } catch (const std::exception& e) {
        std::printf("  unexpected exception: %s\n", e.what());
        ++failures;
    }
    if (failures == 0) {
        std::printf("[PASS] %s\n", name);
        return 0;
    }
    std::printf("[FAIL] %s: %d check(s) failed\n", name, failures);
    return 1;
}

}  // namespace test

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, \
                        #cond);                                              \
            ++test::failures;                                                \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(actual, expected, tol)                                       \
    do {                                                                        \
        const double a_ = (actual), e_ = (expected);                            \
        if (!(std::fabs(a_ - e_) <= (tol))) {                                   \
            std::printf("  %s:%d: CHECK_NEAR(%s, %s) failed: %g vs %g\n",       \
                        __FILE__, __LINE__, #actual, #expected, a_, e_);        \
            ++test::failures;                                                   \
        }                                                                       \
    } while (0)

#define CHECK_THROWS(expr, ExceptionType)                                       \
    do {                                                                        \
        bool thrown_ = false;                                                   \
        try {                                                                   \
            (void)(expr);                                                       \
        } catch (const ExceptionType&) {                                        \
            thrown_ = true;                                                     \
        }                                                                       \
        if (!thrown_) {                                                         \
            std::printf("  %s:%d: CHECK_THROWS(%s, %s) failed\n", __FILE__,     \
                        __LINE__, #expr, #ExceptionType);                       \
            ++test::failures;                                                   \
        }                                                                       \
    } while (0)
