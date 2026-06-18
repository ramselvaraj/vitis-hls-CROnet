#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace cronet {

inline std::string path_join(const std::string& a, const std::string& b) {
    if (a.empty()) {
        return b;
    }
    const char last = a[a.size() - 1];
    if (last == '/' || last == '\\') {
        return a + b;
    }
    return a + "/" + b;
}

template <typename T>
bool load_bin_exact(const std::string& path, std::vector<T>& dst, std::size_t count) {
    dst.assign(count, T());
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "ERROR: failed to open %s\n", path.c_str());
        return false;
    }
    const std::size_t got = std::fread(dst.data(), sizeof(T), count, f);
    std::fclose(f);
    if (got != count) {
        std::fprintf(stderr, "ERROR: read %lu elements from %s, expected %lu\n",
                     static_cast<unsigned long>(got),
                     path.c_str(),
                     static_cast<unsigned long>(count));
        return false;
    }
    return true;
}

inline float fp16_bits_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(bits & 0x8000u)) << 16;
    const std::uint32_t exp = (bits >> 10) & 0x1fu;
    const std::uint32_t mant = bits & 0x03ffu;

    union {
        std::uint32_t u;
        float f;
    } out;

    if (exp == 0) {
        if (mant == 0) {
            out.u = sign;
            return out.f;
        }

        float value = std::ldexp(static_cast<float>(mant), -24);
        return sign ? -value : value;
    }

    if (exp == 0x1fu) {
        out.u = sign | 0x7f800000u | (mant << 13);
        return out.f;
    }

    const std::uint32_t exp32 = exp + (127u - 15u);
    out.u = sign | (exp32 << 23) | (mant << 13);
    return out.f;
}

inline bool load_fp16_words_as_float(
    const std::string& path,
    std::vector<float>& dst,
    std::size_t count) {

    std::vector<std::uint16_t> words;
    if (!load_bin_exact(path, words, count)) {
        return false;
    }

    dst.assign(count, 0.0f);
    for (std::size_t i = 0; i < count; ++i) {
        dst[i] = fp16_bits_to_float(words[i]);
    }
    return true;
}

struct CompareStats {
    bool pass;
    float max_abs;
    float max_rel;
    std::size_t mismatches;
};

inline CompareStats compare_float_vectors(
    const std::vector<float>& got,
    const std::vector<float>& ref,
    float abs_tol,
    float rel_tol,
    const char* name,
    std::size_t max_print) {

    CompareStats stats;
    stats.pass = true;
    stats.max_abs = 0.0f;
    stats.max_rel = 0.0f;
    stats.mismatches = 0;

    if (got.size() != ref.size()) {
        std::fprintf(stderr, "FAIL %-28s size mismatch: got %lu ref %lu\n",
                     name,
                     static_cast<unsigned long>(got.size()),
                     static_cast<unsigned long>(ref.size()));
        stats.pass = false;
        stats.mismatches = 1;
        return stats;
    }

    for (std::size_t i = 0; i < got.size(); ++i) {
        const float g = got[i];
        const float r = ref[i];
        const float diff = std::fabs(g - r);
        const float denom = std::max(std::fabs(r), 1.0e-12f);
        const float rel = diff / denom;
        stats.max_abs = std::max(stats.max_abs, diff);
        stats.max_rel = std::max(stats.max_rel, rel);
        if (diff > abs_tol && rel > rel_tol) {
            if (stats.mismatches < max_print) {
                std::fprintf(stderr,
                             "  mismatch %-28s [%lu] got %.9e ref %.9e abs %.9e rel %.9e\n",
                             name,
                             static_cast<unsigned long>(i),
                             g,
                             r,
                             diff,
                             rel);
            }
            ++stats.mismatches;
            stats.pass = false;
        }
    }

    std::printf("%s %-28s max_abs %.9e max_rel %.9e mismatches %lu\n",
                stats.pass ? "PASS" : "FAIL",
                name,
                stats.max_abs,
                stats.max_rel,
                static_cast<unsigned long>(stats.mismatches));
    return stats;
}

} // namespace cronet
