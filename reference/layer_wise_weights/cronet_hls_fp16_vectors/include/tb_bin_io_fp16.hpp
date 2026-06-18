#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifdef __SYNTHESIS__
#include <hls_half.h>
#else
// Vitis HLS C-sim normally provides hls_half.h. Keep this include outside
// synthesis too when compiling with Vitis. If your host compiler cannot find it,
// load FP16 files as uint16_t and convert/check in Python instead.
#include <hls_half.h>
#endif

template <typename T>
bool load_bin(const char* path, T* dst, std::size_t count) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "Failed to open %s\n", path);
        return false;
    }
    const std::size_t got = std::fread(dst, sizeof(T), count, f);
    std::fclose(f);
    if (got != count) {
        std::fprintf(stderr, "Read %zu elements from %s, expected %zu\n", got, path, count);
        return false;
    }
    return true;
}

template <typename T>
bool save_bin(const char* path, const T* src, std::size_t count) {
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "Failed to open %s\n", path);
        return false;
    }
    const std::size_t wrote = std::fwrite(src, sizeof(T), count, f);
    std::fclose(f);
    return wrote == count;
}

inline float fp16_bits_to_float(uint16_t bits) {
    half h;
    std::memcpy(&h, &bits, sizeof(uint16_t));
    return static_cast<float>(h);
}

inline uint16_t float_to_fp16_bits(float x) {
    half h = static_cast<half>(x);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(uint16_t));
    return bits;
}

inline bool compare_float_arrays(
    const float* got,
    const float* ref,
    std::size_t count,
    float abs_tol,
    float rel_tol,
    std::size_t max_print = 16
) {
    bool ok = true;
    std::size_t printed = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        const float denom = std::fmax(std::fabs(ref[i]), 1e-12f);
        const float rel = diff / denom;
        if (diff > max_abs) max_abs = diff;
        if (rel > max_rel) max_rel = rel;
        if (diff > abs_tol && rel > rel_tol) {
            ok = false;
            if (printed < max_print) {
                std::printf("mismatch[%zu]: got %.9e ref %.9e abs %.9e rel %.9e\n", i, got[i], ref[i], diff, rel);
                ++printed;
            }
        }
    }
    std::printf("compare: %s max_abs %.9e max_rel %.9e\n", ok ? "PASS" : "FAIL", max_abs, max_rel);
    return ok;
}

inline bool compare_half_arrays(
    const half* got,
    const half* ref,
    std::size_t count,
    float abs_tol,
    float rel_tol,
    std::size_t max_print = 16
) {
    bool ok = true;
    std::size_t printed = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        const float g = static_cast<float>(got[i]);
        const float r = static_cast<float>(ref[i]);
        const float diff = std::fabs(g - r);
        const float denom = std::fmax(std::fabs(r), 1e-12f);
        const float rel = diff / denom;
        if (diff > max_abs) max_abs = diff;
        if (rel > max_rel) max_rel = rel;
        if (diff > abs_tol && rel > rel_tol) {
            ok = false;
            if (printed < max_print) {
                std::printf("mismatch[%zu]: got %.9e ref %.9e abs %.9e rel %.9e\n", i, g, r, diff, rel);
                ++printed;
            }
        }
    }
    std::printf("compare_half: %s max_abs %.9e max_rel %.9e\n", ok ? "PASS" : "FAIL", max_abs, max_rel);
    return ok;
}
