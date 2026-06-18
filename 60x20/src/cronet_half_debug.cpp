#include "cronet_half_debug.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "bin_io.hpp"
#include "generated/cronet_weights_fp16_half.hpp"

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
    const fp16_t sig = narrow(1.0f / (1.0f + static_cast<float>(std::exp(-xf))));
    return h_mul(x, sig);
}

fp16_t tanh_scalar(fp16_t x) {
    return narrow(static_cast<float>(std::tanh(widen(x))));
}

void silu_vector(const std::vector<fp16_t>& in, std::vector<fp16_t>& out) {
    out.assign(in.size(), fp16_t(0.0f));
    for (std::size_t i = 0; i < in.size(); ++i) {
        out[i] = silu_scalar(in[i]);
    }
}

void conv2d_same_3x3(
    const std::vector<fp16_t>& input,
    int N,
    int IC,
    int H,
    int W,
    const fp16_t* weight,
    const fp16_t* bias,
    int OC,
    std::vector<fp16_t>& output) {

    output.assign(static_cast<std::size_t>(N) * OC * H * W, fp16_t(0.0f));
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    fp16_t acc = bias[oc];
                    for (int ic = 0; ic < IC; ++ic) {
                        for (int ky = 0; ky < 3; ++ky) {
                            const int iy = y + ky - 1;
                            if (iy < 0 || iy >= H) {
                                continue;
                            }
                            for (int kx = 0; kx < 3; ++kx) {
                                const int ix = x + kx - 1;
                                if (ix < 0 || ix >= W) {
                                    continue;
                                }
                                const fp16_t a = input[idx_nchw(n, ic, iy, ix, IC, H, W)];
                                const fp16_t b = weight[idx4(oc, ic, ky, kx, IC, 3, 3)];
                                acc = h_mac(acc, a, b);
                            }
                        }
                    }
                    output[idx_nchw(n, oc, y, x, OC, H, W)] = acc;
                }
            }
        }
    }
}

void conv3d_same_3x3x2(
    const std::vector<fp16_t>& input,
    int N,
    int IC,
    int D,
    int H,
    int W,
    const fp16_t* weight,
    const fp16_t* bias,
    int OC,
    std::vector<fp16_t>& output) {

    const int OD = D;
    const int OH = H;
    const int OW = W + 1;
    output.assign(static_cast<std::size_t>(N) * OC * OD * OH * OW, fp16_t(0.0f));
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        fp16_t acc = bias[oc];
                        for (int ic = 0; ic < IC; ++ic) {
                            for (int kd = 0; kd < 3; ++kd) {
                                const int id = od + kd - 1;
                                if (id < 0 || id >= D) {
                                    continue;
                                }
                                for (int kh = 0; kh < 3; ++kh) {
                                    const int ih = oh + kh - 1;
                                    if (ih < 0 || ih >= H) {
                                        continue;
                                    }
                                    for (int kw = 0; kw < 2; ++kw) {
                                        const int iw = ow + kw - 1;
                                        if (iw < 0 || iw >= W) {
                                            continue;
                                        }
                                        const fp16_t a = input[idx_ncdhw(n, ic, id, ih, iw, IC, D, H, W)];
                                        const fp16_t b = weight[idx5(oc, ic, kd, kh, kw, IC, 3, 3, 2)];
                                        acc = h_mac(acc, a, b);
                                    }
                                }
                            }
                        }
                        output[idx_ncdhw(n, oc, od, oh, ow, OC, OD, OH, OW)] = acc;
                    }
                }
            }
        }
    }
}

void maxpool2d_2x2_stride2(
    const std::vector<fp16_t>& input,
    int N,
    int C,
    int H,
    int W,
    std::vector<fp16_t>& output) {

    const int OH = H / 2;
    const int OW = W / 2;
    output.assign(static_cast<std::size_t>(N) * C * OH * OW, fp16_t(0.0f));
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < OH; ++y) {
                for (int x = 0; x < OW; ++x) {
                    const fp16_t v0 = input[idx_nchw(n, c, 2 * y,     2 * x,     C, H, W)];
                    const fp16_t v1 = input[idx_nchw(n, c, 2 * y,     2 * x + 1, C, H, W)];
                    const fp16_t v2 = input[idx_nchw(n, c, 2 * y + 1, 2 * x,     C, H, W)];
                    const fp16_t v3 = input[idx_nchw(n, c, 2 * y + 1, 2 * x + 1, C, H, W)];
                    const fp16_t m01 = widen(v0) > widen(v1) ? v0 : v1;
                    const fp16_t m23 = widen(v2) > widen(v3) ? v2 : v3;
                    output[idx_nchw(n, c, y, x, C, OH, OW)] = widen(m01) > widen(m23) ? m01 : m23;
                }
            }
        }
    }
}

void adaptive_avg_pool2d_to_1x1(
    const std::vector<fp16_t>& input,
    int N,
    int C,
    int H,
    int W,
    std::vector<fp16_t>& output) {

    output.assign(static_cast<std::size_t>(N) * C, fp16_t(0.0f));
    const fp16_t scale = narrow(1.0f / static_cast<float>(H * W));
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            fp16_t sum = fp16_t(0.0f);
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    sum = h_add(sum, input[idx_nchw(n, c, y, x, C, H, W)]);
                }
            }
            output[static_cast<std::size_t>(n) * C + c] = h_mul(sum, scale);
        }
    }
}

int adaptive_start(int out_index, int in_size, int out_size) {
    return (out_index * in_size) / out_size;
}

int adaptive_end(int out_index, int in_size, int out_size) {
    return ((out_index + 1) * in_size + out_size - 1) / out_size;
}

void adaptive_avg_pool3d(
    const std::vector<fp16_t>& input,
    int N,
    int C,
    int D,
    int H,
    int W,
    int OD,
    int OH,
    int OW,
    std::vector<fp16_t>& output) {

    output.assign(static_cast<std::size_t>(N) * C * OD * OH * OW, fp16_t(0.0f));
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int od = 0; od < OD; ++od) {
                const int ds = adaptive_start(od, D, OD);
                const int de = adaptive_end(od, D, OD);
                for (int oh = 0; oh < OH; ++oh) {
                    const int hs = adaptive_start(oh, H, OH);
                    const int he = adaptive_end(oh, H, OH);
                    for (int ow = 0; ow < OW; ++ow) {
                        const int ws = adaptive_start(ow, W, OW);
                        const int we = adaptive_end(ow, W, OW);
                        fp16_t sum = fp16_t(0.0f);
                        int count = 0;
                        for (int id = ds; id < de; ++id) {
                            for (int ih = hs; ih < he; ++ih) {
                                for (int iw = ws; iw < we; ++iw) {
                                    sum = h_add(sum, input[idx_ncdhw(n, c, id, ih, iw, C, D, H, W)]);
                                    ++count;
                                }
                            }
                        }
                        output[idx_ncdhw(n, c, od, oh, ow, C, OD, OH, OW)] =
                            h_mul(sum, narrow(1.0f / static_cast<float>(count)));
                    }
                }
            }
        }
    }
}

void rnn_tanh_sequence(
    const std::vector<fp16_t>& input,
    ActivationsHalfDebug& act) {

    act.branch_rnn_all.assign(BRANCH_RNN_ALL_COUNT, fp16_t(0.0f));
    act.branch_rnn_last.assign(BRANCH_RNN_LAST_COUNT, fp16_t(0.0f));
    std::vector<fp16_t> h_prev(RNN_HIDDEN, fp16_t(0.0f));
    std::vector<fp16_t> h_cur(RNN_HIDDEN, fp16_t(0.0f));

    for (int t = 0; t < HISTORY; ++t) {
        act.branch_rnn_step[t].assign(RNN_HIDDEN, fp16_t(0.0f));
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
            act.branch_rnn_step[t][h] = h_cur[h];
            act.branch_rnn_all[idx3(0, t, h, HISTORY, RNN_HIDDEN)] = h_cur[h];
        }
        h_prev = h_cur;
    }

    std::copy(h_prev.begin(), h_prev.end(), act.branch_rnn_last.begin());
}

void linear(
    const std::vector<fp16_t>& input,
    const fp16_t* weight,
    const fp16_t* bias,
    int OUT,
    int IN,
    std::vector<fp16_t>& output) {

    output.assign(OUT, fp16_t(0.0f));
    for (int out = 0; out < OUT; ++out) {
        fp16_t acc = bias[out];
        for (int in = 0; in < IN; ++in) {
            acc = h_mac(acc, input[in], weight[idx_linear_weight(out, in, IN)]);
        }
        output[out] = acc;
    }
}

fp16_t word_to_half(std::uint16_t bits) {
#if defined(CRONET_USE_NATIVE_FLOAT_HALF)
    return fp16_t(fp16_bits_to_float(bits));
#else
    fp16_t value;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
#endif
}

bool load_fp16_words_as_half(const std::string& path, std::vector<fp16_t>& dst, std::size_t count) {
    std::vector<std::uint16_t> words;
    if (!load_bin_exact(path, words, count)) {
        return false;
    }
    dst.assign(count, fp16_t(0.0f));
    for (std::size_t i = 0; i < count; ++i) {
        dst[i] = word_to_half(words[i]);
    }
    return true;
}

} // namespace

bool load_case_inputs_fp16(
    const std::string& vector_root,
    const std::string& case_name,
    std::vector<fp16_t>& branch_input,
    std::vector<fp16_t>& trunk_input) {

    const std::string input_root =
        path_join(path_join(path_join(vector_root, "cases"), case_name), "inputs/fp16_words");

    bool ok = true;
    ok = load_fp16_words_as_half(path_join(input_root, "branch_input.bin"),
                                 branch_input,
                                 BRANCH_IN_COUNT) && ok;
    ok = load_fp16_words_as_half(path_join(input_root, "trunk_input.bin"),
                                 trunk_input,
                                 TRUNK_IN_COUNT) && ok;
    return ok;
}

void run_cronet_half_debug(
    const std::vector<fp16_t>& branch_input,
    const std::vector<fp16_t>& trunk_input,
    ActivationsHalfDebug& act) {

    act.branch_input_reshaped = branch_input;

    conv2d_same_3x3(
        act.branch_input_reshaped,
        HISTORY,
        1,
        NELY,
        NELX,
        weights_fp16::branch_conv1_weight,
        weights_fp16::branch_conv1_bias,
        BRANCH_C1_OC,
        act.branch_conv1_pre_silu);
    silu_vector(act.branch_conv1_pre_silu, act.branch_conv1_silu);

    conv2d_same_3x3(
        act.branch_conv1_silu,
        HISTORY,
        BRANCH_C1_OC,
        NELY,
        NELX,
        weights_fp16::branch_conv2_weight,
        weights_fp16::branch_conv2_bias,
        BRANCH_C2_OC,
        act.branch_conv2_pre_silu);
    silu_vector(act.branch_conv2_pre_silu, act.branch_conv2_silu);

    maxpool2d_2x2_stride2(
        act.branch_conv2_silu,
        HISTORY,
        BRANCH_C2_OC,
        NELY,
        NELX,
        act.branch_maxpool2d);

    adaptive_avg_pool2d_to_1x1(
        act.branch_maxpool2d,
        HISTORY,
        BRANCH_C2_OC,
        NELY / 2,
        NELX / 2,
        act.branch_aap2d);

    act.branch_rnn_input = act.branch_aap2d;
    rnn_tanh_sequence(act.branch_rnn_input, act);

    linear(
        act.branch_rnn_last,
        weights_fp16::branch_fc1_weight,
        weights_fp16::branch_fc1_bias,
        NEURONS,
        RNN_HIDDEN,
        act.branch_fc1_pre_silu);
    silu_vector(act.branch_fc1_pre_silu, act.branch_fc1_silu);

    linear(
        act.branch_fc1_silu,
        weights_fp16::branch_fc2_weight,
        weights_fp16::branch_fc2_bias,
        OUT_NEURONS,
        NEURONS,
        act.branch_fc2);
    act.branch_reshaped = act.branch_fc2;

    act.trunk_unsqueezed = trunk_input;

    conv3d_same_3x3x2(
        act.trunk_unsqueezed,
        1,
        1,
        NODE_H,
        NODE_W,
        DOF,
        weights_fp16::trunk_conv1_weight,
        weights_fp16::trunk_conv1_bias,
        TRUNK_C1_OC,
        act.trunk_conv1_pre_silu);
    silu_vector(act.trunk_conv1_pre_silu, act.trunk_conv1_silu);

    conv3d_same_3x3x2(
        act.trunk_conv1_silu,
        1,
        TRUNK_C1_OC,
        NODE_H,
        NODE_W,
        TRUNK_CONV1_OUT_W,
        weights_fp16::trunk_conv2_weight,
        weights_fp16::trunk_conv2_bias,
        TRUNK_C2_OC,
        act.trunk_conv2_pre_silu);
    silu_vector(act.trunk_conv2_pre_silu, act.trunk_conv2_silu);

    adaptive_avg_pool3d(
        act.trunk_conv2_silu,
        1,
        TRUNK_C2_OC,
        NODE_H,
        NODE_W,
        TRUNK_CONV2_OUT_W,
        TRUNK_AAP_D,
        TRUNK_AAP_H,
        TRUNK_AAP_W,
        act.trunk_aap3d);

    act.trunk_flatten = act.trunk_aap3d;

    linear(
        act.trunk_flatten,
        weights_fp16::trunk_fc1_weight,
        weights_fp16::trunk_fc1_bias,
        NEURONS,
        TRUNK_FLATTEN_COUNT,
        act.trunk_fc1_pre_silu);
    silu_vector(act.trunk_fc1_pre_silu, act.trunk_fc1_silu);

    linear(
        act.trunk_fc1_silu,
        weights_fp16::trunk_fc2_weight,
        weights_fp16::trunk_fc2_bias,
        OUT_NEURONS,
        NEURONS,
        act.trunk_fc2);
    act.trunk_reshaped = act.trunk_fc2;

    act.final_mul_pre_bias.assign(FINAL_OUT_COUNT, fp16_t(0.0f));
    act.final_output.assign(FINAL_OUT_COUNT, fp16_t(0.0f));
    for (int i = 0; i < FINAL_OUT_COUNT; ++i) {
        act.final_mul_pre_bias[i] = h_mul(act.branch_reshaped[i], act.trunk_reshaped[i]);
        act.final_output[i] = h_add(act.final_mul_pre_bias[i], weights_fp16::out_bias[0]);
    }
}

} // namespace cronet
