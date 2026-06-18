#pragma once

#include <cstddef>
#include <cstdint>

#include "cronet_config.hpp"
#include "cronet_half_type.hpp"

#if !defined(CRONET_USE_NATIVE_FLOAT_HALF)
#include <ap_int.h>
#endif

namespace cronet {

static const int AXI_WORD_BITS = 512;
static const int FP16_BITS = 16;
static const int FP16_PER_AXI_WORD = AXI_WORD_BITS / FP16_BITS;
static const int BRANCH_IN_AXI_WORDS = (BRANCH_IN_COUNT + FP16_PER_AXI_WORD - 1) / FP16_PER_AXI_WORD;
static const int TRUNK_IN_AXI_WORDS = (TRUNK_IN_COUNT + FP16_PER_AXI_WORD - 1) / FP16_PER_AXI_WORD;
static const int FINAL_OUT_AXI_WORDS = (FINAL_OUT_COUNT + FP16_PER_AXI_WORD - 1) / FP16_PER_AXI_WORD;

#if defined(CRONET_USE_NATIVE_FLOAT_HALF)
struct axi_word_t {
    axi_word_t() : lane() {}
    axi_word_t(int) : lane() {}
    std::uint16_t lane[FP16_PER_AXI_WORD];
};
typedef std::uint64_t cycle_count_t;

inline float fp16_bits_to_float_local(std::uint16_t bits) {
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
        const float value = static_cast<float>(mant) * 5.9604644775390625e-8f;
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

inline std::uint16_t float_to_fp16_bits_local(float value) {
    union {
        float f;
        std::uint32_t u;
    } in;
    in.f = value;

    const std::uint32_t sign = (in.u >> 16) & 0x8000u;
    const std::uint32_t exp32 = (in.u >> 23) & 0xffu;
    const std::uint32_t mant32 = in.u & 0x7fffffu;

    if (exp32 == 0xffu) {
        if (mant32 == 0) {
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mant32 >> 13) | 1u);
    }

    int exp16 = static_cast<int>(exp32) - 127 + 15;
    if (exp16 >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }

    if (exp16 <= 0) {
        if (exp16 < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        std::uint32_t mant = mant32 | 0x800000u;
        const int shift = 14 - exp16;
        std::uint32_t half_mant = mant >> shift;
        const std::uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const std::uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) {
            ++half_mant;
        }
        return static_cast<std::uint16_t>(sign | half_mant);
    }

    std::uint32_t out = sign | (static_cast<std::uint32_t>(exp16) << 10) | (mant32 >> 13);
    const std::uint32_t round = mant32 & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (out & 1u))) {
        ++out;
    }
    return static_cast<std::uint16_t>(out);
}

inline fp16_t axi_get_lane(const axi_word_t& word, int lane) {
    return fp16_bits_to_float_local(word.lane[lane]);
}

inline void axi_set_lane(axi_word_t& word, int lane, fp16_t value) {
    word.lane[lane] = float_to_fp16_bits_local(value);
}
#else
typedef ap_uint<AXI_WORD_BITS> axi_word_t;
typedef ap_uint<64> cycle_count_t;

inline float pow2_int(int exp) {
    float scale = 1.0f;
    if (exp >= 0) {
        for (int i = 0; i < 32; ++i) {
            if (i < exp) {
                scale *= 2.0f;
            }
        }
    } else {
        const int neg_exp = -exp;
        for (int i = 0; i < 32; ++i) {
            if (i < neg_exp) {
                scale *= 0.5f;
            }
        }
    }
    return scale;
}

inline fp16_t fp16_from_bits(ap_uint<16> bits) {
    const unsigned int raw = bits.to_uint();
    const bool sign = (raw & 0x8000u) != 0;
    const unsigned int exp = (raw >> 10) & 0x1fu;
    const unsigned int mant = raw & 0x03ffu;

    float value = 0.0f;
    if (exp == 0) {
        value = static_cast<float>(mant) * 5.9604644775390625e-8f;
    } else if (exp == 0x1fu) {
        value = 3.4028234663852886e38f;
    } else {
        const float frac = 1.0f + static_cast<float>(mant) * 9.765625e-4f;
        value = frac * pow2_int(static_cast<int>(exp) - 15);
    }

    return fp16_t(sign ? -value : value);
}

inline ap_uint<16> fp16_to_bits(fp16_t value) {
    const float f = static_cast<float>(value);
    union {
        float f;
        std::uint32_t u;
    } in;
    in.f = f;

    const std::uint32_t sign = (in.u >> 16) & 0x8000u;
    const std::uint32_t exp32 = (in.u >> 23) & 0xffu;
    const std::uint32_t mant32 = in.u & 0x7fffffu;

    std::uint32_t out = sign;
    if (exp32 == 0xffu) {
        out |= 0x7c00u;
        if (mant32 != 0) {
            out |= (mant32 >> 13) | 1u;
        }
    } else {
        const int exp16 = static_cast<int>(exp32) - 127 + 15;
        if (exp16 >= 31) {
            out |= 0x7c00u;
        } else if (exp16 <= 0) {
            if (exp16 >= -10) {
                const std::uint32_t mant = mant32 | 0x800000u;
                const int shift = 14 - exp16;
                std::uint32_t half_mant = mant >> shift;
                const std::uint32_t round_bit = (mant >> (shift - 1)) & 1u;
                const std::uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
                if (round_bit && (sticky || (half_mant & 1u))) {
                    ++half_mant;
                }
                out |= half_mant;
            }
        } else {
            out |= (static_cast<std::uint32_t>(exp16) << 10) | (mant32 >> 13);
            const std::uint32_t round = mant32 & 0x1fffu;
            if (round > 0x1000u || (round == 0x1000u && (out & 1u))) {
                ++out;
            }
        }
    }

    return ap_uint<16>(out & 0xffffu);
}

inline fp16_t axi_get_lane(const axi_word_t& word, int lane) {
    const int lo = lane * FP16_BITS;
    const int hi = lo + FP16_BITS - 1;
    return fp16_from_bits(word.range(hi, lo));
}

inline void axi_set_lane(axi_word_t& word, int lane, fp16_t value) {
    const int lo = lane * FP16_BITS;
    const int hi = lo + FP16_BITS - 1;
    word.range(hi, lo) = fp16_to_bits(value);
}
#endif

} // namespace cronet
