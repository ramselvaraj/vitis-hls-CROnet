#include "cronet_config.hpp"
#include "cronet_half_type.hpp"
#include "cronet_axi.hpp"
#include "generated/cronet_weights_fp16_half.hpp"

#if defined(CRONET_USE_NATIVE_FLOAT_HALF)
#include <cmath>
namespace hls {
inline float expf(float x) {
    return std::exp(x);
}
inline float tanhf(float x) {
    return std::tanh(x);
}
} // namespace hls
#else
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_task.h>
#endif

namespace cronet {
namespace {

static const int BRANCH_CONV2_IC_PAR = 8;
static const int BRANCH_CONV2_IC_BLOCKS = BRANCH_C1_OC / BRANCH_CONV2_IC_PAR;
static const int TRUNK_CONV2_IC_PAR = 8;
static const int TRUNK_CONV2_IC_BLOCKS = TRUNK_C1_OC / TRUNK_CONV2_IC_PAR;
// Output-channel parallelism for trunk conv2: process TRUNK_OC_PAR output
// channels per pass. The input map reads are shared across the parallel OCs
// (same pixels, different weights), so this multiplies compute (DSP) without
// adding input read-port pressure -> ~TRUNK_OC_PAR x throughput.
static const int TRUNK_OC_PAR = 4;
static const int TRUNK_OC_BLOCKS = TRUNK_C2_OC / TRUNK_OC_PAR;

typedef fp16_t branch_c1_map_t[HISTORY][BRANCH_C1_OC][NELY][NELX];
typedef fp16_t branch_conv2_weight_ic8_t[BRANCH_C2_OC][BRANCH_CONV2_IC_BLOCKS][BRANCH_CONV2_IC_PAR][3][3];
typedef fp16_t trunk_c1_map_t[TRUNK_C1_OC][NODE_H][NODE_W][TRUNK_CONV1_OUT_W];
typedef fp16_t trunk_conv2_weight_ic8_t[TRUNK_C2_OC][TRUNK_CONV2_IC_BLOCKS][TRUNK_CONV2_IC_PAR][3][3][2];

fp16_t narrow(float x) {
    return fp16_t(x);
}

float widen(fp16_t x) {
    return static_cast<float>(x);
}

fp16_t h_add(fp16_t a, fp16_t b) {
    return narrow(widen(a) + widen(b));
}

fp16_t h_mul(fp16_t a, fp16_t b) {
    return narrow(widen(a) * widen(b));
}

fp16_t h_mac(fp16_t acc, fp16_t a, fp16_t b) {
    return h_add(acc, h_mul(a, b));
}

fp16_t silu_scalar(fp16_t x) {
    const float xf = widen(x);
    const fp16_t sig = narrow(1.0f / (1.0f + hls::expf(-xf)));
    return h_mul(x, sig);
}

fp16_t tanh_scalar(fp16_t x) {
    return narrow(hls::tanhf(widen(x)));
}

template<int COUNT>
void silu_vector_fixed(const fp16_t* in, fp16_t* out) {
    for (int i = 0; i < COUNT; ++i) {
        out[i] = silu_scalar(in[i]);
    }
}

void branch_conv1_silu_to_map_60x20(const fp16_t* input, branch_c1_map_t out) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=out cyclic factor=8 dim=2
    for (int n = 0; n < HISTORY; ++n) {
        for (int oc = 0; oc < BRANCH_C1_OC; ++oc) {
            for (int y = 0; y < NELY; ++y) {
                for (int x = 0; x < NELX; ++x) {
                    fp16_t acc = weights_fp16::branch_conv1_bias[oc];
                    for (int ky = 0; ky < 3; ++ky) {
                        const int iy = y + ky - 1;
                        if (iy < 0 || iy >= NELY) {
                            continue;
                        }
                        for (int kx = 0; kx < 3; ++kx) {
                            const int ix = x + kx - 1;
                            if (ix < 0 || ix >= NELX) {
                                continue;
                            }
                            const fp16_t a = input[idx_nchw(n, 0, iy, ix, 1, NELY, NELX)];
                            const fp16_t b = weights_fp16::branch_conv1_weight[idx4(oc, 0, ky, kx, 1, 3, 3)];
                            acc = h_mac(acc, a, b);
                        }
                    }
                    out[n][oc][y][x] = silu_scalar(acc);
                }
            }
        }
    }
}

void load_branch_conv2_weight_ic8(branch_conv2_weight_ic8_t weight) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=weight complete dim=3
    for (int oc = 0; oc < BRANCH_C2_OC; ++oc) {
        for (int icb = 0; icb < BRANCH_CONV2_IC_BLOCKS; ++icb) {
            for (int lane = 0; lane < BRANCH_CONV2_IC_PAR; ++lane) {
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        const int ic = icb * BRANCH_CONV2_IC_PAR + lane;
                        weight[oc][icb][lane][ky][kx] =
                            weights_fp16::branch_conv2_weight[idx4(oc, ic, ky, kx, BRANCH_C1_OC, 3, 3)];
                    }
                }
            }
        }
    }
}

fp16_t branch_conv2_pixel_ic8(
    const branch_c1_map_t input,
    const branch_conv2_weight_ic8_t weight,
    int n,
    int oc,
    int y,
    int x) {

#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=3

    float acc = widen(weights_fp16::branch_conv2_bias[oc]);
    const bool interior = y > 0 && y < (NELY - 1) && x > 0 && x < (NELX - 1);
    for (int icb = 0; icb < BRANCH_CONV2_IC_BLOCKS; ++icb) {
        float part[BRANCH_CONV2_IC_PAR];
#pragma HLS ARRAY_PARTITION variable=part complete dim=1
        for (int lane = 0; lane < BRANCH_CONV2_IC_PAR; ++lane) {
#pragma HLS UNROLL
            part[lane] = 0.0f;
        }

        for (int ky = 0; ky < 3; ++ky) {
            const int iy = y + ky - 1;
            if (interior || (iy >= 0 && iy < NELY)) {
                for (int kx = 0; kx < 3; ++kx) {
                    const int ix = x + kx - 1;
                    if (interior || (ix >= 0 && ix < NELX)) {
                        for (int lane = 0; lane < BRANCH_CONV2_IC_PAR; ++lane) {
#pragma HLS UNROLL
                            const int ic = icb * BRANCH_CONV2_IC_PAR + lane;
                            part[lane] += widen(input[n][ic][iy][ix]) * widen(weight[oc][icb][lane][ky][kx]);
                        }
                    }
                }
            }
        }

        const float sum01 = part[0] + part[1];
        const float sum23 = part[2] + part[3];
        const float sum45 = part[4] + part[5];
        const float sum67 = part[6] + part[7];
        acc += (sum01 + sum23) + (sum45 + sum67);
    }
    return narrow(acc);
}

void branch_conv2_silu_pool_aap_60x20_ic8(
    const branch_c1_map_t input,
    const branch_conv2_weight_ic8_t weight,
    fp16_t* output) {

#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=3

    const int out_h = NELY / 2;
    const int out_w = NELX / 2;
    const float scale = 1.0f / static_cast<float>(out_h * out_w);
    static const int POOL_PARTS = 8;

    for (int n = 0; n < HISTORY; ++n) {
        for (int oc = 0; oc < BRANCH_C2_OC; ++oc) {
            float psum[POOL_PARTS];
#pragma HLS ARRAY_PARTITION variable=psum complete dim=1
            for (int p = 0; p < POOL_PARTS; ++p) {
#pragma HLS UNROLL
                psum[p] = 0.0f;
            }
            for (int py = 0; py < out_h; ++py) {
                for (int px = 0; px < out_w; ++px) {
                    const int y0 = 2 * py;
                    const int x0 = 2 * px;
                    const fp16_t v0 = silu_scalar(branch_conv2_pixel_ic8(input, weight, n, oc, y0,     x0));
                    const fp16_t v1 = silu_scalar(branch_conv2_pixel_ic8(input, weight, n, oc, y0,     x0 + 1));
                    const fp16_t v2 = silu_scalar(branch_conv2_pixel_ic8(input, weight, n, oc, y0 + 1, x0));
                    const fp16_t v3 = silu_scalar(branch_conv2_pixel_ic8(input, weight, n, oc, y0 + 1, x0 + 1));
                    const fp16_t m01 = widen(v0) > widen(v1) ? v0 : v1;
                    const fp16_t m23 = widen(v2) > widen(v3) ? v2 : v3;
                    const fp16_t pooled = widen(m01) > widen(m23) ? m01 : m23;
                    psum[(py * out_w + px) & (POOL_PARTS - 1)] += widen(pooled);
                }
            }
            float sum = 0.0f;
            for (int p = 0; p < POOL_PARTS; ++p) {
#pragma HLS UNROLL
                sum += psum[p];
            }
            output[static_cast<int>(idx3(0, n, oc, HISTORY, BRANCH_C2_OC))] = narrow(sum * scale);
        }
    }
}

void trunk_conv1_silu_to_map_60x20(const fp16_t* input, trunk_c1_map_t out) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=out cyclic factor=8 dim=1
    for (int oc = 0; oc < TRUNK_C1_OC; ++oc) {
        for (int od = 0; od < NODE_H; ++od) {
            for (int oh = 0; oh < NODE_W; ++oh) {
                for (int ow = 0; ow < TRUNK_CONV1_OUT_W; ++ow) {
                    fp16_t acc = weights_fp16::trunk_conv1_bias[oc];
                    for (int kd = 0; kd < 3; ++kd) {
                        const int id = od + kd - 1;
                        if (id < 0 || id >= NODE_H) {
                            continue;
                        }
                        for (int kh = 0; kh < 3; ++kh) {
                            const int ih = oh + kh - 1;
                            if (ih < 0 || ih >= NODE_W) {
                                continue;
                            }
                            for (int kw = 0; kw < 2; ++kw) {
                                const int iw = ow + kw - 1;
                                if (iw < 0 || iw >= DOF) {
                                    continue;
                                }
                                const fp16_t a = input[idx_ncdhw(0, 0, id, ih, iw, 1, NODE_H, NODE_W, DOF)];
                                const fp16_t b = weights_fp16::trunk_conv1_weight[idx5(oc, 0, kd, kh, kw, 1, 3, 3, 2)];
                                acc = h_mac(acc, a, b);
                            }
                        }
                    }
                    out[oc][od][oh][ow] = silu_scalar(acc);
                }
            }
        }
    }
}

void load_trunk_conv2_weight_ic8(trunk_conv2_weight_ic8_t weight) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=weight complete dim=3
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=4 dim=1
    for (int oc = 0; oc < TRUNK_C2_OC; ++oc) {
        for (int icb = 0; icb < TRUNK_CONV2_IC_BLOCKS; ++icb) {
            for (int lane = 0; lane < TRUNK_CONV2_IC_PAR; ++lane) {
                for (int kd = 0; kd < 3; ++kd) {
                    for (int kh = 0; kh < 3; ++kh) {
                        for (int kw = 0; kw < 2; ++kw) {
                            const int ic = icb * TRUNK_CONV2_IC_PAR + lane;
                            weight[oc][icb][lane][kd][kh][kw] =
                                weights_fp16::trunk_conv2_weight[idx5(oc, ic, kd, kh, kw, TRUNK_C1_OC, 3, 3, 2)];
                        }
                    }
                }
            }
        }
    }
}

void trunk_conv2_silu_wsum_60x20_ic8(
    const trunk_c1_map_t input,
    const trunk_conv2_weight_ic8_t weight,
    fp16_t* output) {

#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=3
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=4 dim=1

    for (int ocb = 0; ocb < TRUNK_OC_BLOCKS; ++ocb) {
        for (int od = 0; od < NODE_H; ++od) {
            for (int oh = 0; oh < NODE_W; ++oh) {
                fp16_t width_silu[TRUNK_OC_PAR][TRUNK_CONV2_OUT_W];
#pragma HLS ARRAY_PARTITION variable=width_silu complete dim=0
                for (int ow = 0; ow < TRUNK_CONV2_OUT_W; ++ow) {
                    float acc[TRUNK_OC_PAR];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
                    for (int ocp = 0; ocp < TRUNK_OC_PAR; ++ocp) {
#pragma HLS UNROLL
                        acc[ocp] = widen(weights_fp16::trunk_conv2_bias[ocb * TRUNK_OC_PAR + ocp]);
                    }
                    for (int icb = 0; icb < TRUNK_CONV2_IC_BLOCKS; ++icb) {
                        float part[TRUNK_OC_PAR][TRUNK_CONV2_IC_PAR];
#pragma HLS ARRAY_PARTITION variable=part complete dim=0
                        for (int ocp = 0; ocp < TRUNK_OC_PAR; ++ocp) {
#pragma HLS UNROLL
                            for (int lane = 0; lane < TRUNK_CONV2_IC_PAR; ++lane) {
#pragma HLS UNROLL
                                part[ocp][lane] = 0.0f;
                            }
                        }

                        for (int kd = 0; kd < 3; ++kd) {
                            const int id = od + kd - 1;
                            if (id < 0 || id >= NODE_H) {
                                continue;
                            }
                            for (int kh = 0; kh < 3; ++kh) {
                                const int ih = oh + kh - 1;
                                if (ih < 0 || ih >= NODE_W) {
                                    continue;
                                }
                                for (int kw = 0; kw < 2; ++kw) {
                                    const int iw = ow + kw - 1;
                                    if (iw < 0 || iw >= TRUNK_CONV1_OUT_W) {
                                        continue;
                                    }
                                    for (int lane = 0; lane < TRUNK_CONV2_IC_PAR; ++lane) {
#pragma HLS UNROLL
                                        const int ic = icb * TRUNK_CONV2_IC_PAR + lane;
                                        const float in_val = widen(input[ic][id][ih][iw]);  // shared across OCs
                                        for (int ocp = 0; ocp < TRUNK_OC_PAR; ++ocp) {
#pragma HLS UNROLL
                                            const int oc = ocb * TRUNK_OC_PAR + ocp;
                                            part[ocp][lane] += in_val * widen(weight[oc][icb][lane][kd][kh][kw]);
                                        }
                                    }
                                }
                            }
                        }

                        for (int ocp = 0; ocp < TRUNK_OC_PAR; ++ocp) {
#pragma HLS UNROLL
                            const float sum01 = part[ocp][0] + part[ocp][1];
                            const float sum23 = part[ocp][2] + part[ocp][3];
                            const float sum45 = part[ocp][4] + part[ocp][5];
                            const float sum67 = part[ocp][6] + part[ocp][7];
                            acc[ocp] += (sum01 + sum23) + (sum45 + sum67);
                        }
                    }
                    for (int ocp = 0; ocp < TRUNK_OC_PAR; ++ocp) {
#pragma HLS UNROLL
                        width_silu[ocp][ow] = silu_scalar(narrow(acc[ocp]));
                    }
                }
                for (int ocp = 0; ocp < TRUNK_OC_PAR; ++ocp) {
#pragma HLS UNROLL
                    const int oc = ocb * TRUNK_OC_PAR + ocp;
                    const fp16_t sum01 = h_add(width_silu[ocp][0], width_silu[ocp][1]);
                    const fp16_t sum23 = h_add(width_silu[ocp][2], width_silu[ocp][3]);
                    output[idx3(oc, od, oh, NODE_H, NODE_W)] = h_add(sum01, sum23);
                }
            }
        }
    }
}

void adaptive_avg_pool3d_wsum_60x20(const fp16_t* input, fp16_t* output) {
#pragma HLS INLINE off
    static const int d_start[TRUNK_AAP_D] = {
        0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 14, 15, 16, 18, 19
    };
    static const int d_len[TRUNK_AAP_D] = {
        2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2
    };
    static const int h_start[TRUNK_AAP_H] = {
        0, 6, 12, 18, 24, 30, 36, 42, 48, 54
    };
    const fp16_t scale_d2 = narrow(1.0f / 56.0f);
    const fp16_t scale_d3 = narrow(1.0f / 84.0f);

    for (int oc = 0; oc < TRUNK_C2_OC; ++oc) {
        for (int od = 0; od < TRUNK_AAP_D; ++od) {
            const int ds = d_start[od];
            const int d_count = d_len[od];
            for (int oh = 0; oh < TRUNK_AAP_H; ++oh) {
                const int hs = h_start[oh];
                fp16_t depth_sum[3];
#pragma HLS ARRAY_PARTITION variable=depth_sum complete dim=1
                for (int dd = 0; dd < 3; ++dd) {
#pragma HLS UNROLL
                    depth_sum[dd] = fp16_t(0.0f);
                }

                for (int dd = 0; dd < 3; ++dd) {
#pragma HLS UNROLL
                    const int id = ds + dd;
                    if (dd < d_count) {
                        for (int hh = 0; hh < 7; ++hh) {
                            const int ih = hs + hh;
                            depth_sum[dd] = h_add(depth_sum[dd], input[idx3(oc, id, ih, NODE_H, NODE_W)]);
                        }
                    }
                }

                const fp16_t sum01 = h_add(depth_sum[0], depth_sum[1]);
                const fp16_t sum = d_count == 3 ? h_add(sum01, depth_sum[2]) : sum01;
                output[idx_ncdhw(0, oc, od, oh, 0, TRUNK_C2_OC, TRUNK_AAP_D, TRUNK_AAP_H, TRUNK_AAP_W)] =
                    h_mul(sum, d_count == 3 ? scale_d3 : scale_d2);
            }
        }
    }
}

void rnn_tanh_sequence(const fp16_t* input, fp16_t* rnn_last) {
    fp16_t h_prev[RNN_HIDDEN];
    fp16_t h_cur[RNN_HIDDEN];

    for (int h = 0; h < RNN_HIDDEN; ++h) {
        h_prev[h] = fp16_t(0.0f);
        h_cur[h] = fp16_t(0.0f);
    }

    for (int t = 0; t < HISTORY; ++t) {
        for (int h = 0; h < RNN_HIDDEN; ++h) {
            fp16_t acc = h_add(weights_fp16::branch_rnn_bias_ih_l0[h], weights_fp16::branch_rnn_bias_hh_l0[h]);
            for (int i = 0; i < BRANCH_C2_OC; ++i) {
                acc = h_mac(
                    acc,
                    input[idx3(0, t, i, HISTORY, BRANCH_C2_OC)],
                    weights_fp16::branch_rnn_weight_ih_l0[idx_linear_weight(h, i, BRANCH_C2_OC)]);
            }
            for (int hp = 0; hp < RNN_HIDDEN; ++hp) {
                acc = h_mac(
                    acc,
                    h_prev[hp],
                    weights_fp16::branch_rnn_weight_hh_l0[idx_linear_weight(h, hp, RNN_HIDDEN)]);
            }
            h_cur[h] = tanh_scalar(acc);
        }
        for (int h = 0; h < RNN_HIDDEN; ++h) {
            h_prev[h] = h_cur[h];
        }
    }

    for (int h = 0; h < RNN_HIDDEN; ++h) {
        rnn_last[h] = h_prev[h];
    }
}

// Short reduction (IN_COUNT small): HLS pipelines the outer loop at II=1 and
// unrolls the inner reduction into an adder tree. Accumulate in float and round
// once at the end (matches the PyTorch reference, removes per-tap fp16 rounding).
template<int OUT_COUNT, int IN_COUNT>
void linear_fixed(const fp16_t* input, const fp16_t* weight, const fp16_t* bias, fp16_t* output) {
    for (int out = 0; out < OUT_COUNT; ++out) {
        float acc = widen(bias[out]);
        for (int in = 0; in < IN_COUNT; ++in) {
            acc += widen(input[in]) * widen(weight[idx_linear_weight(out, in, IN_COUNT)]);
        }
        output[out] = narrow(acc);
    }
}

// Long reduction (e.g. trunk FC1 with IN_COUNT=4800): pipeline the inner
// reduction at II=1 by spreading the running sum across PARTS independent float
// accumulators (dependency distance = PARTS > float-add latency), then tree-add.
// This breaks the single-accumulator recurrence that forced II=4 on the fp16 path.
template<int OUT_COUNT, int IN_COUNT, int PARTS = 8>
void linear_partial(const fp16_t* input, const fp16_t* weight, const fp16_t* bias, fp16_t* output) {
#pragma HLS INLINE off
    for (int out = 0; out < OUT_COUNT; ++out) {
        float part[PARTS];
#pragma HLS ARRAY_PARTITION variable=part complete dim=1
        for (int p = 0; p < PARTS; ++p) {
#pragma HLS UNROLL
            part[p] = 0.0f;
        }
        for (int in = 0; in < IN_COUNT; ++in) {
#pragma HLS PIPELINE II=1
            const int p = in & (PARTS - 1);
            part[p] += widen(input[in]) * widen(weight[idx_linear_weight(out, in, IN_COUNT)]);
        }
        float acc = widen(bias[out]);
        for (int p = 0; p < PARTS; ++p) {
#pragma HLS UNROLL
            acc += part[p];
        }
        output[out] = narrow(acc);
    }
}

template<int COUNT, int WORD_COUNT>
void load_packed_fp16(const axi_word_t* input, fp16_t* output) {
#pragma HLS INLINE off
    for (int word_idx = 0; word_idx < WORD_COUNT; ++word_idx) {
        const axi_word_t packed = input[word_idx];
        for (int lane = 0; lane < FP16_PER_AXI_WORD; ++lane) {
#pragma HLS PIPELINE II=1
            const int i = word_idx * FP16_PER_AXI_WORD + lane;
            if (i < COUNT) {
                output[i] = axi_get_lane(packed, lane);
            }
        }
    }
}

template<int COUNT, int WORD_COUNT>
void store_packed_fp16(const fp16_t* input, axi_word_t* output) {
#pragma HLS INLINE off
    for (int word_idx = 0; word_idx < WORD_COUNT; ++word_idx) {
        axi_word_t packed = 0;
        for (int lane = 0; lane < FP16_PER_AXI_WORD; ++lane) {
#pragma HLS PIPELINE II=1
            const int i = word_idx * FP16_PER_AXI_WORD + lane;
            const fp16_t value = i < COUNT ? input[i] : fp16_t(0.0f);
            axi_set_lane(packed, lane, value);
        }
        output[word_idx] = packed;
    }
}

void compute_branch_fc2(const fp16_t* branch_reshaped, fp16_t* branch_fc2) {
#pragma HLS INLINE off
    static fp16_t branch_conv1_silu_map[HISTORY][BRANCH_C1_OC][NELY][NELX];
    static fp16_t branch_aap[BRANCH_AAP2D_OUT_COUNT];
    static fp16_t branch_rnn_last[RNN_HIDDEN];
    static fp16_t branch_fc1_pre[BRANCH_FC1_OUT_COUNT];
    static fp16_t branch_fc1_silu[BRANCH_FC1_OUT_COUNT];
    static fp16_t branch_conv2_weight_ic8[BRANCH_C2_OC][BRANCH_CONV2_IC_BLOCKS][BRANCH_CONV2_IC_PAR][3][3];

#pragma HLS ARRAY_PARTITION variable=branch_conv1_silu_map cyclic factor=8 dim=2
#pragma HLS BIND_STORAGE variable=branch_conv1_silu_map type=RAM_2P impl=URAM
#pragma HLS ARRAY_PARTITION variable=branch_conv2_weight_ic8 complete dim=3

    branch_conv1_silu_to_map_60x20(branch_reshaped, branch_conv1_silu_map);

    load_branch_conv2_weight_ic8(branch_conv2_weight_ic8);
    branch_conv2_silu_pool_aap_60x20_ic8(branch_conv1_silu_map, branch_conv2_weight_ic8, branch_aap);
    rnn_tanh_sequence(branch_aap, branch_rnn_last);

    linear_fixed<NEURONS, RNN_HIDDEN>(branch_rnn_last, weights_fp16::branch_fc1_weight, weights_fp16::branch_fc1_bias, branch_fc1_pre);
    silu_vector_fixed<BRANCH_FC1_OUT_COUNT>(branch_fc1_pre, branch_fc1_silu);
    linear_fixed<OUT_NEURONS, NEURONS>(branch_fc1_silu, weights_fp16::branch_fc2_weight, weights_fp16::branch_fc2_bias, branch_fc2);
}

void compute_trunk_fc2(const fp16_t* trunk_unsqueezed, fp16_t* trunk_fc2) {
#pragma HLS INLINE off
    static fp16_t trunk_conv1_silu_map[TRUNK_C1_OC][NODE_H][NODE_W][TRUNK_CONV1_OUT_W];
    static fp16_t trunk_conv2_wsum[TRUNK_CONV2_WSUM_COUNT];
    static fp16_t trunk_aap[TRUNK_AAP3D_OUT_COUNT];
    static fp16_t trunk_fc1_pre[TRUNK_FC1_OUT_COUNT];
    static fp16_t trunk_fc1_silu[TRUNK_FC1_OUT_COUNT];
    static fp16_t trunk_conv2_weight_ic8[TRUNK_C2_OC][TRUNK_CONV2_IC_BLOCKS][TRUNK_CONV2_IC_PAR][3][3][2];

#pragma HLS ARRAY_PARTITION variable=trunk_conv1_silu_map cyclic factor=8 dim=1
#pragma HLS BIND_STORAGE variable=trunk_conv1_silu_map type=RAM_2P impl=URAM
#pragma HLS ARRAY_PARTITION variable=trunk_conv2_weight_ic8 complete dim=3
#pragma HLS ARRAY_PARTITION variable=trunk_conv2_weight_ic8 cyclic factor=4 dim=1

    trunk_conv1_silu_to_map_60x20(trunk_unsqueezed, trunk_conv1_silu_map);

    load_trunk_conv2_weight_ic8(trunk_conv2_weight_ic8);
    trunk_conv2_silu_wsum_60x20_ic8(trunk_conv1_silu_map, trunk_conv2_weight_ic8, trunk_conv2_wsum);

    adaptive_avg_pool3d_wsum_60x20(trunk_conv2_wsum, trunk_aap);
    linear_partial<NEURONS, TRUNK_FLATTEN_COUNT>(trunk_aap, weights_fp16::trunk_fc1_weight, weights_fp16::trunk_fc1_bias, trunk_fc1_pre);
    silu_vector_fixed<TRUNK_FC1_OUT_COUNT>(trunk_fc1_pre, trunk_fc1_silu);
    linear_fixed<OUT_NEURONS, NEURONS>(trunk_fc1_silu, weights_fp16::trunk_fc2_weight, weights_fp16::trunk_fc2_bias, trunk_fc2);
}

void combine_branch_trunk_fc2(const fp16_t* branch_fc2, const fp16_t* trunk_fc2, fp16_t* final_output) {
#pragma HLS INLINE off
    for (int i = 0; i < FINAL_OUT_COUNT; ++i) {
        const fp16_t product = h_mul(branch_fc2[i], trunk_fc2[i]);
        final_output[i] = h_add(product, weights_fp16::out_bias[0]);
    }
}

void cronet_compute_from_local(
    const fp16_t* branch_reshaped,
    const fp16_t* trunk_unsqueezed,
    fp16_t* final_output) {

#pragma HLS INLINE off
#pragma HLS DATAFLOW

    fp16_t branch_fc2[BRANCH_FC2_OUT_COUNT];
    fp16_t trunk_fc2[TRUNK_FC2_OUT_COUNT];

    compute_branch_fc2(branch_reshaped, branch_fc2);
    compute_trunk_fc2(trunk_unsqueezed, trunk_fc2);
    combine_branch_trunk_fc2(branch_fc2, trunk_fc2, final_output);
}

#if defined(CRONET_ENABLE_HW_CORE_COUNTER) && !defined(CRONET_USE_NATIVE_FLOAT_HALF)
// In-fabric cycle counter via the HLS Task Library (hls::task). Validated on the
// VEK280 by vadd_task_kernel (PASS, 174 cyc, NO hang). The classic free-running
// tick_gen DATAFLOW process deadlocked because the dataflow done-detection cannot
// retire a never-terminating worker; an hls::task counter is ap_ctrl_none and
// free-runs independently, while the bounded dataflow I/O processes drive ap_done.
// M_AXI access stays in the regular dataflow processes (tasks may touch streams
// only). compute_proc and store_proc are plain dataflow procs (no heartbeats).

// dataflow proc: load DDR inputs -> START -> full CRONet compute -> stream outs.
void compute_proc(
    const axi_word_t* branch_input,
    const axi_word_t* trunk_input,
    hls::stream<bool>& start_out,
    hls::stream<fp16_t>& out_stream) {

#pragma HLS INLINE off
    static fp16_t branch_local[BRANCH_IN_COUNT];
    static fp16_t trunk_local[TRUNK_IN_COUNT];
    static fp16_t final_local[FINAL_OUT_COUNT];

    start_out.write(true);                       // counting starts here
    load_packed_fp16<BRANCH_IN_COUNT, BRANCH_IN_AXI_WORDS>(branch_input, branch_local);
    load_packed_fp16<TRUNK_IN_COUNT, TRUNK_IN_AXI_WORDS>(trunk_input, trunk_local);
    cronet_compute_from_local(branch_local, trunk_local, final_local);
    for (int i = 0; i < FINAL_OUT_COUNT; ++i) {
        out_stream.write(final_local[i]);
    }
}

// dataflow proc: drain the 2562 outputs into a buffer -> DDR store -> STOP.
// Draining the whole stream before the DDR burst keeps the STOP token's timing
// decoupled from the m_axi write schedule (the interleaved stream-store variant
// is what reported a bogus 1-cycle count).
void store_proc(
    hls::stream<fp16_t>& out_stream,
    axi_word_t* output,
    hls::stream<bool>& stop_out) {

#pragma HLS INLINE off
    static fp16_t final_local[FINAL_OUT_COUNT];
    for (int i = 0; i < FINAL_OUT_COUNT; ++i) {
        final_local[i] = out_stream.read();
    }
    store_packed_fp16<FINAL_OUT_COUNT, FINAL_OUT_AXI_WORDS>(final_local, output);
    stop_out.write(true);                        // counting stops here
}

// hls::task body: free-running PL-clock counter. No blocking read (read_nb) -> it
// fires every cycle. START resets t and arms; STOP latches t to count_out.
void tick_body(
    hls::stream<bool>& start_in,
    hls::stream<bool>& stop_in,
    hls::stream<cycle_count_t>& count_out) {

    static cycle_count_t t = 0;
    static bool running = false;
    bool v;
    if (start_in.read_nb(v)) { t = 0; running = true; }
    if (running) {
        if (stop_in.read_nb(v)) { count_out.write(t); running = false; }
        else { t++; }
    }
}

void read_count(
    hls::stream<cycle_count_t>& count_in,
    cycle_count_t* cycles_preload_done,
    cycle_count_t* cycles_first_send,
    cycle_count_t* cycles_last_send,
    cycle_count_t* cycles_first_recv,
    cycle_count_t* cycles_last_recv,
    cycle_count_t* cycles_compute_done,
    cycle_count_t* cycles_total) {

#pragma HLS INLINE off
    const cycle_count_t total = count_in.read();
    *cycles_preload_done = 0;
    *cycles_first_send = 0;
    *cycles_last_send = 0;
    *cycles_first_recv = 0;
    *cycles_last_recv = 0;
    *cycles_compute_done = 0;
    *cycles_total = total;
}
#endif

} // namespace
} // namespace cronet

extern "C" void cronet_kernel(
    const cronet::axi_word_t* branch_input,
    const cronet::axi_word_t* trunk_input,
    cronet::axi_word_t* output,
    cronet::cycle_count_t* cycles_preload_done,
    cronet::cycle_count_t* cycles_first_send,
    cronet::cycle_count_t* cycles_last_send,
    cronet::cycle_count_t* cycles_first_recv,
    cronet::cycle_count_t* cycles_last_recv,
    cronet::cycle_count_t* cycles_compute_done,
    cronet::cycle_count_t* cycles_total) {

#pragma HLS INTERFACE m_axi port=branch_input offset=slave bundle=gmem0 depth=375
#pragma HLS INTERFACE m_axi port=trunk_input offset=slave bundle=gmem1 depth=81
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2 depth=81
#pragma HLS INTERFACE s_axilite port=branch_input bundle=control
#pragma HLS INTERFACE s_axilite port=trunk_input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=cycles_preload_done bundle=control
#pragma HLS INTERFACE s_axilite port=cycles_first_send bundle=control
#pragma HLS INTERFACE s_axilite port=cycles_last_send bundle=control
#pragma HLS INTERFACE s_axilite port=cycles_first_recv bundle=control
#pragma HLS INTERFACE s_axilite port=cycles_last_recv bundle=control
#pragma HLS INTERFACE s_axilite port=cycles_compute_done bundle=control
#pragma HLS INTERFACE s_axilite port=cycles_total bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    using namespace cronet;

    static_assert(NELX == 60 && NELY == 20, "cronet_kernel is specialized for the 60x20 design");
    static_assert(BRANCH_IN_AXI_WORDS == 375, "unexpected branch input packing");
    static_assert(TRUNK_IN_AXI_WORDS == 81, "unexpected trunk input packing");
    static_assert(FINAL_OUT_AXI_WORDS == 81, "unexpected output packing");

#if defined(CRONET_ENABLE_HW_CORE_COUNTER) && !defined(CRONET_USE_NATIVE_FLOAT_HALF)
    // hls::task counter (validated on vadd_task). compute_proc/store_proc/read_count
    // are regular dataflow processes (do the M_AXI I/O and drive ap_done); t_tick is
    // a free-running hls::task that counts cycles (START resets, STOP latches). Task
    // instance order does not matter; the regular procs are ordered producer->consumer.
#pragma HLS DATAFLOW
    hls_thread_local hls::stream<fp16_t>        out_stream;
#pragma HLS STREAM variable=out_stream depth=64
    hls_thread_local hls::stream<bool>          start_s;
    hls_thread_local hls::stream<bool>          stop_s;
    hls_thread_local hls::stream<cycle_count_t> count_s;

    compute_proc(branch_input, trunk_input, start_s, out_stream);
    hls_thread_local hls::task t_tick(tick_body, start_s, stop_s, count_s);
    store_proc(out_stream, output, stop_s);
    read_count(count_s,
               cycles_preload_done,
               cycles_first_send,
               cycles_last_send,
               cycles_first_recv,
               cycles_last_recv,
               cycles_compute_done,
               cycles_total);
#else
    static fp16_t branch_local[BRANCH_IN_COUNT];
    static fp16_t trunk_local[TRUNK_IN_COUNT];
    static fp16_t final_local[FINAL_OUT_COUNT];

    load_packed_fp16<BRANCH_IN_COUNT, BRANCH_IN_AXI_WORDS>(branch_input, branch_local);
    load_packed_fp16<TRUNK_IN_COUNT, TRUNK_IN_AXI_WORDS>(trunk_input, trunk_local);

    cronet_compute_from_local(branch_local, trunk_local, final_local);

    store_packed_fp16<FINAL_OUT_COUNT, FINAL_OUT_AXI_WORDS>(final_local, output);
    *cycles_preload_done = 0;
    *cycles_first_send = 0;
    *cycles_last_send = 0;
    *cycles_first_recv = 0;
    *cycles_last_recv = 0;
    *cycles_compute_done = 0;
    *cycles_total = 0;
#endif
}
