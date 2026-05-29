#include "cronet_config.hpp"
#include "cronet_half_type.hpp"
#include "generated/cronet_weights_fp16_half.hpp"

#include <hls_math.h>

namespace cronet {
namespace {

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

void silu_vector(const fp16_t* in, fp16_t* out, int count) {
    for (int i = 0; i < count; ++i) {
        out[i] = silu_scalar(in[i]);
    }
}

void conv2d_same_3x3(
    const fp16_t* input,
    int n_count,
    int ic_count,
    int h_count,
    int w_count,
    const fp16_t* weight,
    const fp16_t* bias,
    int oc_count,
    fp16_t* output) {

    for (int n = 0; n < n_count; ++n) {
        for (int oc = 0; oc < oc_count; ++oc) {
            for (int y = 0; y < h_count; ++y) {
                for (int x = 0; x < w_count; ++x) {
                    fp16_t acc = bias[oc];
                    for (int ic = 0; ic < ic_count; ++ic) {
                        for (int ky = 0; ky < 3; ++ky) {
                            const int iy = y + ky - 1;
                            if (iy < 0 || iy >= h_count) {
                                continue;
                            }
                            for (int kx = 0; kx < 3; ++kx) {
                                const int ix = x + kx - 1;
                                if (ix < 0 || ix >= w_count) {
                                    continue;
                                }
                                const fp16_t a = input[idx_nchw(n, ic, iy, ix, ic_count, h_count, w_count)];
                                const fp16_t b = weight[idx4(oc, ic, ky, kx, ic_count, 3, 3)];
                                acc = h_mac(acc, a, b);
                            }
                        }
                    }
                    output[idx_nchw(n, oc, y, x, oc_count, h_count, w_count)] = acc;
                }
            }
        }
    }
}

void conv3d_same_3x3x2(
    const fp16_t* input,
    int n_count,
    int ic_count,
    int d_count,
    int h_count,
    int w_count,
    const fp16_t* weight,
    const fp16_t* bias,
    int oc_count,
    fp16_t* output) {

    const int od_count = d_count;
    const int oh_count = h_count;
    const int ow_count = w_count + 1;
    for (int n = 0; n < n_count; ++n) {
        for (int oc = 0; oc < oc_count; ++oc) {
            for (int od = 0; od < od_count; ++od) {
                for (int oh = 0; oh < oh_count; ++oh) {
                    for (int ow = 0; ow < ow_count; ++ow) {
                        fp16_t acc = bias[oc];
                        for (int ic = 0; ic < ic_count; ++ic) {
                            for (int kd = 0; kd < 3; ++kd) {
                                const int id = od + kd - 1;
                                if (id < 0 || id >= d_count) {
                                    continue;
                                }
                                for (int kh = 0; kh < 3; ++kh) {
                                    const int ih = oh + kh - 1;
                                    if (ih < 0 || ih >= h_count) {
                                        continue;
                                    }
                                    for (int kw = 0; kw < 2; ++kw) {
                                        const int iw = ow + kw - 1;
                                        if (iw < 0 || iw >= w_count) {
                                            continue;
                                        }
                                        const fp16_t a = input[idx_ncdhw(n, ic, id, ih, iw, ic_count, d_count, h_count, w_count)];
                                        const fp16_t b = weight[idx5(oc, ic, kd, kh, kw, ic_count, 3, 3, 2)];
                                        acc = h_mac(acc, a, b);
                                    }
                                }
                            }
                        }
                        output[idx_ncdhw(n, oc, od, oh, ow, oc_count, od_count, oh_count, ow_count)] = acc;
                    }
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

int adaptive_start(int out_index, int in_size, int out_size) {
    return (out_index * in_size) / out_size;
}

int adaptive_end(int out_index, int in_size, int out_size) {
    return ((out_index + 1) * in_size + out_size - 1) / out_size;
}

void adaptive_avg_pool3d(const fp16_t* input, fp16_t* output) {
    for (int c = 0; c < TRUNK_C2_OC; ++c) {
        for (int od = 0; od < TRUNK_AAP_D; ++od) {
            const int ds = adaptive_start(od, NODE_H, TRUNK_AAP_D);
            const int de = adaptive_end(od, NODE_H, TRUNK_AAP_D);
            for (int oh = 0; oh < TRUNK_AAP_H; ++oh) {
                const int hs = adaptive_start(oh, NODE_W, TRUNK_AAP_H);
                const int he = adaptive_end(oh, NODE_W, TRUNK_AAP_H);
                fp16_t sum = fp16_t(0.0f);
                int count = 0;
                for (int id = ds; id < de; ++id) {
                    for (int ih = hs; ih < he; ++ih) {
                        for (int iw = 0; iw < TRUNK_CONV2_OUT_W; ++iw) {
                            sum = h_add(sum, input[idx_ncdhw(0, c, id, ih, iw, TRUNK_C2_OC, NODE_H, NODE_W, TRUNK_CONV2_OUT_W)]);
                            ++count;
                        }
                    }
                }
                output[idx_ncdhw(0, c, od, oh, 0, TRUNK_C2_OC, TRUNK_AAP_D, TRUNK_AAP_H, TRUNK_AAP_W)] =
                    h_mul(sum, narrow(1.0f / static_cast<float>(count)));
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

void linear(const fp16_t* input, const fp16_t* weight, const fp16_t* bias, int out_count, int in_count, fp16_t* output) {
    for (int out = 0; out < out_count; ++out) {
        fp16_t acc = bias[out];
        for (int in = 0; in < in_count; ++in) {
            acc = h_mac(acc, input[in], weight[idx_linear_weight(out, in, in_count)]);
        }
        output[out] = acc;
    }
}

} // namespace
} // namespace cronet

extern "C" void cronet_kernel(
    const cronet::fp16_t* branch_input,
    const cronet::fp16_t* trunk_input,
    cronet::fp16_t* output) {

#pragma HLS INTERFACE m_axi port=branch_input offset=slave bundle=gmem0 depth=12000
#pragma HLS INTERFACE m_axi port=trunk_input offset=slave bundle=gmem1 depth=2562
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=2562
#pragma HLS INTERFACE s_axilite port=branch_input bundle=control
#pragma HLS INTERFACE s_axilite port=trunk_input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    using namespace cronet;

    static fp16_t branch_reshaped[BRANCH_IN_COUNT];
    static fp16_t branch_conv1_pre[BRANCH_CONV1_OUT_COUNT];
    static fp16_t branch_conv1_silu[BRANCH_CONV1_OUT_COUNT];
    static fp16_t branch_conv2_pre[BRANCH_CONV2_OUT_COUNT];
    static fp16_t branch_conv2_silu[BRANCH_CONV2_OUT_COUNT];
    static fp16_t branch_maxpool[BRANCH_MAXPOOL_OUT_COUNT];
    static fp16_t branch_aap[BRANCH_AAP2D_OUT_COUNT];
    static fp16_t branch_rnn_last[RNN_HIDDEN];
    static fp16_t branch_fc1_pre[BRANCH_FC1_OUT_COUNT];
    static fp16_t branch_fc1_silu[BRANCH_FC1_OUT_COUNT];
    static fp16_t branch_fc2[BRANCH_FC2_OUT_COUNT];

    static fp16_t trunk_unsqueezed[TRUNK_IN_COUNT];
    static fp16_t trunk_conv1_pre[TRUNK_CONV1_OUT_COUNT];
    static fp16_t trunk_conv1_silu[TRUNK_CONV1_OUT_COUNT];
    static fp16_t trunk_conv2_pre[TRUNK_CONV2_OUT_COUNT];
    static fp16_t trunk_conv2_silu[TRUNK_CONV2_OUT_COUNT];
    static fp16_t trunk_aap[TRUNK_AAP3D_OUT_COUNT];
    static fp16_t trunk_fc1_pre[TRUNK_FC1_OUT_COUNT];
    static fp16_t trunk_fc1_silu[TRUNK_FC1_OUT_COUNT];
    static fp16_t trunk_fc2[TRUNK_FC2_OUT_COUNT];

    for (int i = 0; i < BRANCH_IN_COUNT; ++i) {
        branch_reshaped[i] = branch_input[i];
    }
    for (int i = 0; i < TRUNK_IN_COUNT; ++i) {
        trunk_unsqueezed[i] = trunk_input[i];
    }

    conv2d_same_3x3(
        branch_reshaped,
        HISTORY,
        1,
        NELY,
        NELX,
        weights_fp16::branch_conv1_weight,
        weights_fp16::branch_conv1_bias,
        BRANCH_C1_OC,
        branch_conv1_pre);
    silu_vector(branch_conv1_pre, branch_conv1_silu, BRANCH_CONV1_OUT_COUNT);

    conv2d_same_3x3(
        branch_conv1_silu,
        HISTORY,
        BRANCH_C1_OC,
        NELY,
        NELX,
        weights_fp16::branch_conv2_weight,
        weights_fp16::branch_conv2_bias,
        BRANCH_C2_OC,
        branch_conv2_pre);
    silu_vector(branch_conv2_pre, branch_conv2_silu, BRANCH_CONV2_OUT_COUNT);

    maxpool2d_2x2_stride2(branch_conv2_silu, branch_maxpool);
    adaptive_avg_pool2d_to_1x1(branch_maxpool, branch_aap);
    rnn_tanh_sequence(branch_aap, branch_rnn_last);

    linear(branch_rnn_last, weights_fp16::branch_fc1_weight, weights_fp16::branch_fc1_bias, NEURONS, RNN_HIDDEN, branch_fc1_pre);
    silu_vector(branch_fc1_pre, branch_fc1_silu, BRANCH_FC1_OUT_COUNT);
    linear(branch_fc1_silu, weights_fp16::branch_fc2_weight, weights_fp16::branch_fc2_bias, OUT_NEURONS, NEURONS, branch_fc2);

    conv3d_same_3x3x2(
        trunk_unsqueezed,
        1,
        1,
        NODE_H,
        NODE_W,
        DOF,
        weights_fp16::trunk_conv1_weight,
        weights_fp16::trunk_conv1_bias,
        TRUNK_C1_OC,
        trunk_conv1_pre);
    silu_vector(trunk_conv1_pre, trunk_conv1_silu, TRUNK_CONV1_OUT_COUNT);

    conv3d_same_3x3x2(
        trunk_conv1_silu,
        1,
        TRUNK_C1_OC,
        NODE_H,
        NODE_W,
        TRUNK_CONV1_OUT_W,
        weights_fp16::trunk_conv2_weight,
        weights_fp16::trunk_conv2_bias,
        TRUNK_C2_OC,
        trunk_conv2_pre);
    silu_vector(trunk_conv2_pre, trunk_conv2_silu, TRUNK_CONV2_OUT_COUNT);

    adaptive_avg_pool3d(trunk_conv2_silu, trunk_aap);
    linear(trunk_aap, weights_fp16::trunk_fc1_weight, weights_fp16::trunk_fc1_bias, NEURONS, TRUNK_FLATTEN_COUNT, trunk_fc1_pre);
    silu_vector(trunk_fc1_pre, trunk_fc1_silu, TRUNK_FC1_OUT_COUNT);
    linear(trunk_fc1_silu, weights_fp16::trunk_fc2_weight, weights_fp16::trunk_fc2_bias, OUT_NEURONS, NEURONS, trunk_fc2);

    for (int i = 0; i < FINAL_OUT_COUNT; ++i) {
        const fp16_t product = h_mul(branch_fc2[i], trunk_fc2[i]);
        output[i] = h_add(product, weights_fp16::out_bias[0]);
    }
}
