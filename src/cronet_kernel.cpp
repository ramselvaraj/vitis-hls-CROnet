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
#endif

namespace cronet {
namespace {

static const int BRANCH_CONV2_IC_PAR = 4;
static const int BRANCH_CONV2_IC_BLOCKS = BRANCH_C1_OC / BRANCH_CONV2_IC_PAR;
static const int TRUNK_CONV2_IC_PAR = 8;
static const int TRUNK_CONV2_IC_BLOCKS = TRUNK_C1_OC / TRUNK_CONV2_IC_PAR;

typedef fp16_t branch_c1_map_t[HISTORY][BRANCH_C1_OC][NELY][NELX];
typedef fp16_t branch_conv2_weight_ic4_t[BRANCH_C2_OC][BRANCH_CONV2_IC_BLOCKS][BRANCH_CONV2_IC_PAR][3][3];
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

template<int N_COUNT, int IC_COUNT, int H_COUNT, int W_COUNT, int OC_COUNT>
void conv2d_same_3x3_fixed(
    const fp16_t* input,
    const fp16_t* weight,
    const fp16_t* bias,
    fp16_t* output) {

    for (int n = 0; n < N_COUNT; ++n) {
        for (int oc = 0; oc < OC_COUNT; ++oc) {
            for (int y = 0; y < H_COUNT; ++y) {
                for (int x = 0; x < W_COUNT; ++x) {
                    fp16_t acc = bias[oc];
                    for (int ic = 0; ic < IC_COUNT; ++ic) {
                        for (int ky = 0; ky < 3; ++ky) {
                            const int iy = y + ky - 1;
                            if (iy < 0 || iy >= H_COUNT) {
                                continue;
                            }
                            for (int kx = 0; kx < 3; ++kx) {
                                const int ix = x + kx - 1;
                                if (ix < 0 || ix >= W_COUNT) {
                                    continue;
                                }
                                const fp16_t a = input[idx_nchw(n, ic, iy, ix, IC_COUNT, H_COUNT, W_COUNT)];
                                const fp16_t b = weight[idx4(oc, ic, ky, kx, IC_COUNT, 3, 3)];
                                acc = h_mac(acc, a, b);
                            }
                        }
                    }
                    output[idx_nchw(n, oc, y, x, OC_COUNT, H_COUNT, W_COUNT)] = acc;
                }
            }
        }
    }
}

void branch_conv1_60x20(const fp16_t* input, fp16_t* output) {
    conv2d_same_3x3_fixed<HISTORY, 1, NELY, NELX, BRANCH_C1_OC>(
        input,
        weights_fp16::branch_conv1_weight,
        weights_fp16::branch_conv1_bias,
        output);
}

void silu_branch_conv1_to_map(const fp16_t* in, branch_c1_map_t out) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=out cyclic factor=4 dim=2
    for (int n = 0; n < HISTORY; ++n) {
        for (int c = 0; c < BRANCH_C1_OC; ++c) {
            for (int y = 0; y < NELY; ++y) {
                for (int x = 0; x < NELX; ++x) {
                    out[n][c][y][x] = silu_scalar(in[idx_nchw(n, c, y, x, BRANCH_C1_OC, NELY, NELX)]);
                }
            }
        }
    }
}

void load_branch_conv2_weight_ic4(branch_conv2_weight_ic4_t weight) {
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

void branch_conv2_60x20_ic4(
    const branch_c1_map_t input,
    const branch_conv2_weight_ic4_t weight,
    fp16_t* output) {

#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=3

    for (int n = 0; n < HISTORY; ++n) {
        for (int oc = 0; oc < BRANCH_C2_OC; ++oc) {
            for (int y = 0; y < NELY; ++y) {
                for (int x = 0; x < NELX; ++x) {
                    fp16_t acc = weights_fp16::branch_conv2_bias[oc];
                    for (int icb = 0; icb < BRANCH_CONV2_IC_BLOCKS; ++icb) {
                        fp16_t part[BRANCH_CONV2_IC_PAR];
#pragma HLS ARRAY_PARTITION variable=part complete dim=1
                        for (int lane = 0; lane < BRANCH_CONV2_IC_PAR; ++lane) {
#pragma HLS UNROLL
                            part[lane] = fp16_t(0.0f);
                        }

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
                                for (int lane = 0; lane < BRANCH_CONV2_IC_PAR; ++lane) {
#pragma HLS UNROLL
                                    const int ic = icb * BRANCH_CONV2_IC_PAR + lane;
                                    part[lane] = h_mac(part[lane], input[n][ic][iy][ix], weight[oc][icb][lane][ky][kx]);
                                }
                            }
                        }

                        const fp16_t sum01 = h_add(part[0], part[1]);
                        const fp16_t sum23 = h_add(part[2], part[3]);
                        acc = h_add(acc, h_add(sum01, sum23));
                    }
                    output[idx_nchw(n, oc, y, x, BRANCH_C2_OC, NELY, NELX)] = acc;
                }
            }
        }
    }
}

template<int N_COUNT, int IC_COUNT, int D_COUNT, int H_COUNT, int W_COUNT, int OC_COUNT>
void conv3d_same_3x3x2_fixed(
    const fp16_t* input,
    const fp16_t* weight,
    const fp16_t* bias,
    fp16_t* output) {

    const int OD_COUNT = D_COUNT;
    const int OH_COUNT = H_COUNT;
    const int OW_COUNT = W_COUNT + 1;
    for (int n = 0; n < N_COUNT; ++n) {
        for (int oc = 0; oc < OC_COUNT; ++oc) {
            for (int od = 0; od < OD_COUNT; ++od) {
                for (int oh = 0; oh < OH_COUNT; ++oh) {
                    for (int ow = 0; ow < OW_COUNT; ++ow) {
                        fp16_t acc = bias[oc];
                        for (int ic = 0; ic < IC_COUNT; ++ic) {
                            for (int kd = 0; kd < 3; ++kd) {
                                const int id = od + kd - 1;
                                if (id < 0 || id >= D_COUNT) {
                                    continue;
                                }
                                for (int kh = 0; kh < 3; ++kh) {
                                    const int ih = oh + kh - 1;
                                    if (ih < 0 || ih >= H_COUNT) {
                                        continue;
                                    }
                                    for (int kw = 0; kw < 2; ++kw) {
                                        const int iw = ow + kw - 1;
                                        if (iw < 0 || iw >= W_COUNT) {
                                            continue;
                                        }
                                        const fp16_t a = input[idx_ncdhw(n, ic, id, ih, iw, IC_COUNT, D_COUNT, H_COUNT, W_COUNT)];
                                        const fp16_t b = weight[idx5(oc, ic, kd, kh, kw, IC_COUNT, 3, 3, 2)];
                                        acc = h_mac(acc, a, b);
                                    }
                                }
                            }
                        }
                        output[idx_ncdhw(n, oc, od, oh, ow, OC_COUNT, OD_COUNT, OH_COUNT, OW_COUNT)] = acc;
                    }
                }
            }
        }
    }
}

void trunk_conv1_60x20(const fp16_t* input, fp16_t* output) {
    conv3d_same_3x3x2_fixed<1, 1, NODE_H, NODE_W, DOF, TRUNK_C1_OC>(
        input,
        weights_fp16::trunk_conv1_weight,
        weights_fp16::trunk_conv1_bias,
        output);
}

void silu_trunk_conv1_to_map(const fp16_t* in, trunk_c1_map_t out) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=out cyclic factor=8 dim=1
    for (int c = 0; c < TRUNK_C1_OC; ++c) {
        for (int d = 0; d < NODE_H; ++d) {
            for (int h = 0; h < NODE_W; ++h) {
                for (int w = 0; w < TRUNK_CONV1_OUT_W; ++w) {
                    out[c][d][h][w] = silu_scalar(in[idx_ncdhw(0, c, d, h, w, TRUNK_C1_OC, NODE_H, NODE_W, TRUNK_CONV1_OUT_W)]);
                }
            }
        }
    }
}

void load_trunk_conv2_weight_ic8(trunk_conv2_weight_ic8_t weight) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=weight complete dim=3
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

void trunk_conv2_60x20_ic8(
    const trunk_c1_map_t input,
    const trunk_conv2_weight_ic8_t weight,
    fp16_t* output) {

#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=3

    for (int oc = 0; oc < TRUNK_C2_OC; ++oc) {
        for (int od = 0; od < NODE_H; ++od) {
            for (int oh = 0; oh < NODE_W; ++oh) {
                for (int ow = 0; ow < TRUNK_CONV2_OUT_W; ++ow) {
                    fp16_t acc = weights_fp16::trunk_conv2_bias[oc];
                    for (int icb = 0; icb < TRUNK_CONV2_IC_BLOCKS; ++icb) {
                        fp16_t part[TRUNK_CONV2_IC_PAR];
#pragma HLS ARRAY_PARTITION variable=part complete dim=1
                        for (int lane = 0; lane < TRUNK_CONV2_IC_PAR; ++lane) {
#pragma HLS UNROLL
                            part[lane] = fp16_t(0.0f);
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
                                        part[lane] = h_mac(part[lane], input[ic][id][ih][iw], weight[oc][icb][lane][kd][kh][kw]);
                                    }
                                }
                            }
                        }

                        const fp16_t sum01 = h_add(part[0], part[1]);
                        const fp16_t sum23 = h_add(part[2], part[3]);
                        const fp16_t sum45 = h_add(part[4], part[5]);
                        const fp16_t sum67 = h_add(part[6], part[7]);
                        const fp16_t sum03 = h_add(sum01, sum23);
                        const fp16_t sum47 = h_add(sum45, sum67);
                        acc = h_add(acc, h_add(sum03, sum47));
                    }
                    output[idx_ncdhw(0, oc, od, oh, ow, TRUNK_C2_OC, NODE_H, NODE_W, TRUNK_CONV2_OUT_W)] = acc;
                }
            }
        }
    }
}

void maxpool2d_2x2_stride2(const fp16_t* input, fp16_t* output) {
    const int out_h = NELY / 2;
    const int out_w = NELX / 2;
    for (int n = 0; n < HISTORY; ++n) {
        for (int c = 0; c < BRANCH_C2_OC; ++c) {
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    const fp16_t v0 = input[idx_nchw(n, c, 2 * y,     2 * x,     BRANCH_C2_OC, NELY, NELX)];
                    const fp16_t v1 = input[idx_nchw(n, c, 2 * y,     2 * x + 1, BRANCH_C2_OC, NELY, NELX)];
                    const fp16_t v2 = input[idx_nchw(n, c, 2 * y + 1, 2 * x,     BRANCH_C2_OC, NELY, NELX)];
                    const fp16_t v3 = input[idx_nchw(n, c, 2 * y + 1, 2 * x + 1, BRANCH_C2_OC, NELY, NELX)];
                    const fp16_t m01 = widen(v0) > widen(v1) ? v0 : v1;
                    const fp16_t m23 = widen(v2) > widen(v3) ? v2 : v3;
                    output[idx_nchw(n, c, y, x, BRANCH_C2_OC, out_h, out_w)] = widen(m01) > widen(m23) ? m01 : m23;
                }
            }
        }
    }
}

void adaptive_avg_pool2d_to_1x1(const fp16_t* input, fp16_t* output) {
    const int h_count = NELY / 2;
    const int w_count = NELX / 2;
    const fp16_t scale = narrow(1.0f / static_cast<float>(h_count * w_count));
    for (int n = 0; n < HISTORY; ++n) {
        for (int c = 0; c < BRANCH_C2_OC; ++c) {
            fp16_t sum = fp16_t(0.0f);
            for (int y = 0; y < h_count; ++y) {
                for (int x = 0; x < w_count; ++x) {
                    sum = h_add(sum, input[idx_nchw(n, c, y, x, BRANCH_C2_OC, h_count, w_count)]);
                }
            }
            output[static_cast<int>(idx3(0, n, c, HISTORY, BRANCH_C2_OC))] = h_mul(sum, scale);
        }
    }
}

void adaptive_avg_pool3d_60x20(const fp16_t* input, fp16_t* output) {
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

    for (int c = 0; c < TRUNK_C2_OC; ++c) {
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
                            const fp16_t v0 = input[idx_ncdhw(0, c, id, ih, 0, TRUNK_C2_OC, NODE_H, NODE_W, TRUNK_CONV2_OUT_W)];
                            const fp16_t v1 = input[idx_ncdhw(0, c, id, ih, 1, TRUNK_C2_OC, NODE_H, NODE_W, TRUNK_CONV2_OUT_W)];
                            const fp16_t v2 = input[idx_ncdhw(0, c, id, ih, 2, TRUNK_C2_OC, NODE_H, NODE_W, TRUNK_CONV2_OUT_W)];
                            const fp16_t v3 = input[idx_ncdhw(0, c, id, ih, 3, TRUNK_C2_OC, NODE_H, NODE_W, TRUNK_CONV2_OUT_W)];
                            const fp16_t sum01 = h_add(v0, v1);
                            const fp16_t sum23 = h_add(v2, v3);
                            depth_sum[dd] = h_add(depth_sum[dd], h_add(sum01, sum23));
                        }
                    }
                }

                const fp16_t sum01 = h_add(depth_sum[0], depth_sum[1]);
                const fp16_t sum = d_count == 3 ? h_add(sum01, depth_sum[2]) : sum01;
                output[idx_ncdhw(0, c, od, oh, 0, TRUNK_C2_OC, TRUNK_AAP_D, TRUNK_AAP_H, TRUNK_AAP_W)] =
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

template<int OUT_COUNT, int IN_COUNT>
void linear_fixed(const fp16_t* input, const fp16_t* weight, const fp16_t* bias, fp16_t* output) {
    for (int out = 0; out < OUT_COUNT; ++out) {
        fp16_t acc = bias[out];
        for (int in = 0; in < IN_COUNT; ++in) {
            acc = h_mac(acc, input[in], weight[idx_linear_weight(out, in, IN_COUNT)]);
        }
        output[out] = acc;
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

void cronet_compute_from_local(
    const fp16_t* branch_reshaped,
    const fp16_t* trunk_unsqueezed,
    fp16_t* final_output) {

#pragma HLS INLINE off

    static fp16_t branch_conv1_pre[BRANCH_CONV1_OUT_COUNT];
    static fp16_t branch_conv1_silu_map[HISTORY][BRANCH_C1_OC][NELY][NELX];
    static fp16_t branch_conv2_pre[BRANCH_CONV2_OUT_COUNT];
    static fp16_t branch_conv2_silu[BRANCH_CONV2_OUT_COUNT];
    static fp16_t branch_maxpool[BRANCH_MAXPOOL_OUT_COUNT];
    static fp16_t branch_aap[BRANCH_AAP2D_OUT_COUNT];
    static fp16_t branch_rnn_last[RNN_HIDDEN];
    static fp16_t branch_fc1_pre[BRANCH_FC1_OUT_COUNT];
    static fp16_t branch_fc1_silu[BRANCH_FC1_OUT_COUNT];
    static fp16_t branch_fc2[BRANCH_FC2_OUT_COUNT];
    static fp16_t branch_conv2_weight_ic4[BRANCH_C2_OC][BRANCH_CONV2_IC_BLOCKS][BRANCH_CONV2_IC_PAR][3][3];

#pragma HLS ARRAY_PARTITION variable=branch_conv1_silu_map cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=branch_conv2_weight_ic4 complete dim=3

    static fp16_t trunk_conv1_pre[TRUNK_CONV1_OUT_COUNT];
    static fp16_t trunk_conv1_silu_map[TRUNK_C1_OC][NODE_H][NODE_W][TRUNK_CONV1_OUT_W];
    static fp16_t trunk_conv2_pre[TRUNK_CONV2_OUT_COUNT];
    static fp16_t trunk_conv2_silu[TRUNK_CONV2_OUT_COUNT];
    static fp16_t trunk_aap[TRUNK_AAP3D_OUT_COUNT];
    static fp16_t trunk_fc1_pre[TRUNK_FC1_OUT_COUNT];
    static fp16_t trunk_fc1_silu[TRUNK_FC1_OUT_COUNT];
    static fp16_t trunk_fc2[TRUNK_FC2_OUT_COUNT];
    static fp16_t trunk_conv2_weight_ic8[TRUNK_C2_OC][TRUNK_CONV2_IC_BLOCKS][TRUNK_CONV2_IC_PAR][3][3][2];

#pragma HLS ARRAY_PARTITION variable=trunk_conv1_silu_map cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=trunk_conv2_weight_ic8 complete dim=3

    branch_conv1_60x20(branch_reshaped, branch_conv1_pre);
    silu_branch_conv1_to_map(branch_conv1_pre, branch_conv1_silu_map);

    load_branch_conv2_weight_ic4(branch_conv2_weight_ic4);
    branch_conv2_60x20_ic4(branch_conv1_silu_map, branch_conv2_weight_ic4, branch_conv2_pre);
    silu_vector_fixed<BRANCH_CONV2_OUT_COUNT>(branch_conv2_pre, branch_conv2_silu);

    maxpool2d_2x2_stride2(branch_conv2_silu, branch_maxpool);
    adaptive_avg_pool2d_to_1x1(branch_maxpool, branch_aap);
    rnn_tanh_sequence(branch_aap, branch_rnn_last);

    linear_fixed<NEURONS, RNN_HIDDEN>(branch_rnn_last, weights_fp16::branch_fc1_weight, weights_fp16::branch_fc1_bias, branch_fc1_pre);
    silu_vector_fixed<BRANCH_FC1_OUT_COUNT>(branch_fc1_pre, branch_fc1_silu);
    linear_fixed<OUT_NEURONS, NEURONS>(branch_fc1_silu, weights_fp16::branch_fc2_weight, weights_fp16::branch_fc2_bias, branch_fc2);

    trunk_conv1_60x20(trunk_unsqueezed, trunk_conv1_pre);
    silu_trunk_conv1_to_map(trunk_conv1_pre, trunk_conv1_silu_map);

    load_trunk_conv2_weight_ic8(trunk_conv2_weight_ic8);
    trunk_conv2_60x20_ic8(trunk_conv1_silu_map, trunk_conv2_weight_ic8, trunk_conv2_pre);
    silu_vector_fixed<TRUNK_CONV2_OUT_COUNT>(trunk_conv2_pre, trunk_conv2_silu);

    adaptive_avg_pool3d_60x20(trunk_conv2_silu, trunk_aap);
    linear_fixed<NEURONS, TRUNK_FLATTEN_COUNT>(trunk_aap, weights_fp16::trunk_fc1_weight, weights_fp16::trunk_fc1_bias, trunk_fc1_pre);
    silu_vector_fixed<TRUNK_FC1_OUT_COUNT>(trunk_fc1_pre, trunk_fc1_silu);
    linear_fixed<OUT_NEURONS, NEURONS>(trunk_fc1_silu, weights_fp16::trunk_fc2_weight, weights_fp16::trunk_fc2_bias, trunk_fc2);

    for (int i = 0; i < FINAL_OUT_COUNT; ++i) {
        const fp16_t product = h_mul(branch_fc2[i], trunk_fc2[i]);
        final_output[i] = h_add(product, weights_fp16::out_bias[0]);
    }
}

#if defined(CRONET_ENABLE_HW_CORE_COUNTER) && !defined(CRONET_USE_NATIVE_FLOAT_HALF)
void compute_cycle_counter(
    hls::stream<bool>& started,
    hls::stream<bool>& stopped,
    cycle_count_t* profile_cycles) {

#pragma HLS INLINE off
    started.read();

    cycle_count_t cycles = 0;
    bool done = false;
    while (!done) {
#pragma HLS PIPELINE II=1
        bool token = false;
        if (stopped.read_nb(token)) {
            done = true;
        } else {
            ++cycles;
        }
    }

    profile_cycles[0] = cycles;
}

void run_compute_profiled(
    hls::stream<bool>& started,
    hls::stream<bool>& stopped,
    const fp16_t* branch_local,
    const fp16_t* trunk_local,
    fp16_t* final_output) {

#pragma HLS INLINE off
    started.write(true);
    cronet_compute_from_local(branch_local, trunk_local, final_output);
    stopped.write(true);
}
#endif

} // namespace
} // namespace cronet

extern "C" void cronet_kernel(
    const cronet::axi_word_t* branch_input,
    const cronet::axi_word_t* trunk_input,
    cronet::axi_word_t* output,
    cronet::cycle_count_t* profile_cycles) {

#pragma HLS INTERFACE m_axi port=branch_input offset=slave bundle=gmem0 depth=375
#pragma HLS INTERFACE m_axi port=trunk_input offset=slave bundle=gmem1 depth=81
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2 depth=81
#pragma HLS INTERFACE m_axi port=profile_cycles offset=slave bundle=gmem2 depth=1
#pragma HLS INTERFACE s_axilite port=branch_input bundle=control
#pragma HLS INTERFACE s_axilite port=trunk_input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=profile_cycles bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    using namespace cronet;

    static_assert(NELX == 60 && NELY == 20, "cronet_kernel is specialized for the 60x20 design");
    static_assert(BRANCH_IN_AXI_WORDS == 375, "unexpected branch input packing");
    static_assert(TRUNK_IN_AXI_WORDS == 81, "unexpected trunk input packing");
    static_assert(FINAL_OUT_AXI_WORDS == 81, "unexpected output packing");

    static fp16_t branch_local[BRANCH_IN_COUNT];
    static fp16_t trunk_local[TRUNK_IN_COUNT];
    static fp16_t final_local[FINAL_OUT_COUNT];

    load_packed_fp16<BRANCH_IN_COUNT, BRANCH_IN_AXI_WORDS>(branch_input, branch_local);
    load_packed_fp16<TRUNK_IN_COUNT, TRUNK_IN_AXI_WORDS>(trunk_input, trunk_local);

#if defined(CRONET_ENABLE_HW_CORE_COUNTER) && !defined(CRONET_USE_NATIVE_FLOAT_HALF)
    hls::stream<bool> counter_started;
    hls::stream<bool> counter_stopped;
#pragma HLS STREAM variable=counter_started depth=1
#pragma HLS STREAM variable=counter_stopped depth=1
    {
#pragma HLS DATAFLOW
        compute_cycle_counter(counter_started, counter_stopped, profile_cycles);
        run_compute_profiled(counter_started, counter_stopped, branch_local, trunk_local, final_local);
    }
#else
    cronet_compute_from_local(branch_local, trunk_local, final_local);
    profile_cycles[0] = 0;
#endif

    store_packed_fp16<FINAL_OUT_COUNT, FINAL_OUT_AXI_WORDS>(final_local, output);
}
