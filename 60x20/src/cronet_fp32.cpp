#include "cronet_fp32.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "bin_io.hpp"

namespace cronet {
namespace {

bool load_weight(const std::string& root, const char* name, std::size_t count, std::vector<float>& dst) {
    const std::string path = path_join(path_join(path_join(root, "weights"), "fp32"), name);
    return load_bin_exact(path, dst, count);
}

bool load_weight_fp16_as_float(const std::string& root, const char* name, std::size_t count, std::vector<float>& dst) {
    const std::string path = path_join(path_join(path_join(root, "weights"), "fp16_words"), name);
    return load_fp16_words_as_float(path, dst, count);
}

float silu_scalar(float x) {
    return x / (1.0f + static_cast<float>(std::exp(-x)));
}

void silu_vector(const std::vector<float>& in, std::vector<float>& out) {
    out.assign(in.size(), 0.0f);
    for (std::size_t i = 0; i < in.size(); ++i) {
        out[i] = silu_scalar(in[i]);
    }
}

void conv2d_same_3x3(
    const std::vector<float>& input,
    int N,
    int IC,
    int H,
    int W,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    int OC,
    std::vector<float>& output) {

    output.assign(static_cast<std::size_t>(N) * OC * H * W, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    float acc = bias[oc];
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
                                const float a = input[idx_nchw(n, ic, iy, ix, IC, H, W)];
                                const float b = weight[idx4(oc, ic, ky, kx, IC, 3, 3)];
                                acc += a * b;
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
    const std::vector<float>& input,
    int N,
    int IC,
    int D,
    int H,
    int W,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    int OC,
    std::vector<float>& output) {

    const int OD = D;
    const int OH = H;
    const int OW = W + 1;
    output.assign(static_cast<std::size_t>(N) * OC * OD * OH * OW, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        float acc = bias[oc];
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
                                        const float a = input[idx_ncdhw(n, ic, id, ih, iw, IC, D, H, W)];
                                        const float b = weight[idx5(oc, ic, kd, kh, kw, IC, 3, 3, 2)];
                                        acc += a * b;
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
    const std::vector<float>& input,
    int N,
    int C,
    int H,
    int W,
    std::vector<float>& output) {

    const int OH = H / 2;
    const int OW = W / 2;
    output.assign(static_cast<std::size_t>(N) * C * OH * OW, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < OH; ++y) {
                for (int x = 0; x < OW; ++x) {
                    const float v0 = input[idx_nchw(n, c, 2 * y,     2 * x,     C, H, W)];
                    const float v1 = input[idx_nchw(n, c, 2 * y,     2 * x + 1, C, H, W)];
                    const float v2 = input[idx_nchw(n, c, 2 * y + 1, 2 * x,     C, H, W)];
                    const float v3 = input[idx_nchw(n, c, 2 * y + 1, 2 * x + 1, C, H, W)];
                    output[idx_nchw(n, c, y, x, C, OH, OW)] = std::max(std::max(v0, v1), std::max(v2, v3));
                }
            }
        }
    }
}

void adaptive_avg_pool2d_to_1x1(
    const std::vector<float>& input,
    int N,
    int C,
    int H,
    int W,
    std::vector<float>& output) {

    output.assign(static_cast<std::size_t>(N) * C, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    sum += input[idx_nchw(n, c, y, x, C, H, W)];
                }
            }
            output[static_cast<std::size_t>(n) * C + c] = sum / static_cast<float>(H * W);
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
    const std::vector<float>& input,
    int N,
    int C,
    int D,
    int H,
    int W,
    int OD,
    int OH,
    int OW,
    std::vector<float>& output) {

    output.assign(static_cast<std::size_t>(N) * C * OD * OH * OW, 0.0f);
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
                        float sum = 0.0f;
                        int count = 0;
                        for (int id = ds; id < de; ++id) {
                            for (int ih = hs; ih < he; ++ih) {
                                for (int iw = ws; iw < we; ++iw) {
                                    sum += input[idx_ncdhw(n, c, id, ih, iw, C, D, H, W)];
                                    ++count;
                                }
                            }
                        }
                        output[idx_ncdhw(n, c, od, oh, ow, C, OD, OH, OW)] =
                            sum / static_cast<float>(count);
                    }
                }
            }
        }
    }
}

void rnn_tanh_sequence(
    const std::vector<float>& input,
    const std::vector<float>& weight_ih,
    const std::vector<float>& weight_hh,
    const std::vector<float>& bias_ih,
    const std::vector<float>& bias_hh,
    ActivationsFp32& act) {

    act.branch_rnn_all.assign(BRANCH_RNN_ALL_COUNT, 0.0f);
    act.branch_rnn_last.assign(BRANCH_RNN_LAST_COUNT, 0.0f);
    std::vector<float> h_prev(RNN_HIDDEN, 0.0f);
    std::vector<float> h_cur(RNN_HIDDEN, 0.0f);

    for (int t = 0; t < HISTORY; ++t) {
        act.branch_rnn_step[t].assign(RNN_HIDDEN, 0.0f);
        for (int h = 0; h < RNN_HIDDEN; ++h) {
            float acc = bias_ih[h] + bias_hh[h];
            for (int i = 0; i < BRANCH_C2_OC; ++i) {
                acc += input[idx3(0, t, i, HISTORY, BRANCH_C2_OC)] *
                       weight_ih[idx_linear_weight(h, i, BRANCH_C2_OC)];
            }
            for (int hp = 0; hp < RNN_HIDDEN; ++hp) {
                acc += h_prev[hp] * weight_hh[idx_linear_weight(h, hp, RNN_HIDDEN)];
            }
            h_cur[h] = static_cast<float>(std::tanh(acc));
            act.branch_rnn_step[t][h] = h_cur[h];
            act.branch_rnn_all[idx3(0, t, h, HISTORY, RNN_HIDDEN)] = h_cur[h];
        }
        h_prev = h_cur;
    }

    std::copy(h_prev.begin(), h_prev.end(), act.branch_rnn_last.begin());
}

void linear(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    int OUT,
    int IN,
    std::vector<float>& output) {

    output.assign(OUT, 0.0f);
    for (int out = 0; out < OUT; ++out) {
        float acc = bias[out];
        for (int in = 0; in < IN; ++in) {
            acc += input[in] * weight[idx_linear_weight(out, in, IN)];
        }
        output[out] = acc;
    }
}

} // namespace

bool load_weights_fp32(const std::string& vector_root, WeightsFp32& w) {
    bool ok = true;
    ok = load_weight(vector_root, "branch_conv1__weight.bin", BRANCH_C1_OC * 1 * 3 * 3, w.branch_conv1_weight) && ok;
    ok = load_weight(vector_root, "branch_conv1__bias.bin", BRANCH_C1_OC, w.branch_conv1_bias) && ok;
    ok = load_weight(vector_root, "branch_conv2__weight.bin", BRANCH_C2_OC * BRANCH_C1_OC * 3 * 3, w.branch_conv2_weight) && ok;
    ok = load_weight(vector_root, "branch_conv2__bias.bin", BRANCH_C2_OC, w.branch_conv2_bias) && ok;
    ok = load_weight(vector_root, "branch_rnn__weight_ih_l0.bin", RNN_HIDDEN * BRANCH_C2_OC, w.branch_rnn_weight_ih_l0) && ok;
    ok = load_weight(vector_root, "branch_rnn__weight_hh_l0.bin", RNN_HIDDEN * RNN_HIDDEN, w.branch_rnn_weight_hh_l0) && ok;
    ok = load_weight(vector_root, "branch_rnn__bias_ih_l0.bin", RNN_HIDDEN, w.branch_rnn_bias_ih_l0) && ok;
    ok = load_weight(vector_root, "branch_rnn__bias_hh_l0.bin", RNN_HIDDEN, w.branch_rnn_bias_hh_l0) && ok;
    ok = load_weight(vector_root, "branch_fc1__weight.bin", NEURONS * RNN_HIDDEN, w.branch_fc1_weight) && ok;
    ok = load_weight(vector_root, "branch_fc1__bias.bin", NEURONS, w.branch_fc1_bias) && ok;
    ok = load_weight(vector_root, "branch_fc2__weight.bin", OUT_NEURONS * NEURONS, w.branch_fc2_weight) && ok;
    ok = load_weight(vector_root, "branch_fc2__bias.bin", OUT_NEURONS, w.branch_fc2_bias) && ok;

    ok = load_weight(vector_root, "trunk_conv1__weight.bin", TRUNK_C1_OC * 1 * 3 * 3 * 2, w.trunk_conv1_weight) && ok;
    ok = load_weight(vector_root, "trunk_conv1__bias.bin", TRUNK_C1_OC, w.trunk_conv1_bias) && ok;
    ok = load_weight(vector_root, "trunk_conv2__weight.bin", TRUNK_C2_OC * TRUNK_C1_OC * 3 * 3 * 2, w.trunk_conv2_weight) && ok;
    ok = load_weight(vector_root, "trunk_conv2__bias.bin", TRUNK_C2_OC, w.trunk_conv2_bias) && ok;
    ok = load_weight(vector_root, "trunk_fc1__weight.bin", NEURONS * TRUNK_FLATTEN_COUNT, w.trunk_fc1_weight) && ok;
    ok = load_weight(vector_root, "trunk_fc1__bias.bin", NEURONS, w.trunk_fc1_bias) && ok;
    ok = load_weight(vector_root, "trunk_fc2__weight.bin", OUT_NEURONS * NEURONS, w.trunk_fc2_weight) && ok;
    ok = load_weight(vector_root, "trunk_fc2__bias.bin", OUT_NEURONS, w.trunk_fc2_bias) && ok;

    ok = load_weight(vector_root, "out_bias.bin", 1, w.out_bias) && ok;
    return ok;
}

bool load_weights_fp16_storage_as_float(const std::string& vector_root, WeightsFp32& w) {
    bool ok = true;
    ok = load_weight_fp16_as_float(vector_root, "branch_conv1__weight.bin", BRANCH_C1_OC * 1 * 3 * 3, w.branch_conv1_weight) && ok;
    ok = load_weight_fp16_as_float(vector_root, "branch_conv1__bias.bin", BRANCH_C1_OC, w.branch_conv1_bias) && ok;
    ok = load_weight_fp16_as_float(vector_root, "branch_conv2__weight.bin", BRANCH_C2_OC * BRANCH_C1_OC * 3 * 3, w.branch_conv2_weight) && ok;
    ok = load_weight_fp16_as_float(vector_root, "branch_conv2__bias.bin", BRANCH_C2_OC, w.branch_conv2_bias) && ok;
    ok = load_weight_fp16_as_float(vector_root, "branch_rnn__weight_ih_l0.bin", RNN_HIDDEN * BRANCH_C2_OC, w.branch_rnn_weight_ih_l0) && ok;
    ok = load_weight_fp16_as_float(vector_root, "branch_rnn__weight_hh_l0.bin", RNN_HIDDEN * RNN_HIDDEN, w.branch_rnn_weight_hh_l0) && ok;
    ok = load_weight_fp16_as_float(vector_root, "branch_rnn__bias_ih_l0.bin", RNN_HIDDEN, w.branch_rnn_bias_ih_l0) && ok;
    ok = load_weight_fp16_as_float(vector_root, "branch_rnn__bias_hh_l0.bin", RNN_HIDDEN, w.branch_rnn_bias_hh_l0) && ok;
    ok = load_weight_fp16_as_float(vector_root, "branch_fc1__weight.bin", NEURONS * RNN_HIDDEN, w.branch_fc1_weight) && ok;
    ok = load_weight_fp16_as_float(vector_root, "branch_fc1__bias.bin", NEURONS, w.branch_fc1_bias) && ok;
    ok = load_weight_fp16_as_float(vector_root, "branch_fc2__weight.bin", OUT_NEURONS * NEURONS, w.branch_fc2_weight) && ok;
    ok = load_weight_fp16_as_float(vector_root, "branch_fc2__bias.bin", OUT_NEURONS, w.branch_fc2_bias) && ok;

    ok = load_weight_fp16_as_float(vector_root, "trunk_conv1__weight.bin", TRUNK_C1_OC * 1 * 3 * 3 * 2, w.trunk_conv1_weight) && ok;
    ok = load_weight_fp16_as_float(vector_root, "trunk_conv1__bias.bin", TRUNK_C1_OC, w.trunk_conv1_bias) && ok;
    ok = load_weight_fp16_as_float(vector_root, "trunk_conv2__weight.bin", TRUNK_C2_OC * TRUNK_C1_OC * 3 * 3 * 2, w.trunk_conv2_weight) && ok;
    ok = load_weight_fp16_as_float(vector_root, "trunk_conv2__bias.bin", TRUNK_C2_OC, w.trunk_conv2_bias) && ok;
    ok = load_weight_fp16_as_float(vector_root, "trunk_fc1__weight.bin", NEURONS * TRUNK_FLATTEN_COUNT, w.trunk_fc1_weight) && ok;
    ok = load_weight_fp16_as_float(vector_root, "trunk_fc1__bias.bin", NEURONS, w.trunk_fc1_bias) && ok;
    ok = load_weight_fp16_as_float(vector_root, "trunk_fc2__weight.bin", OUT_NEURONS * NEURONS, w.trunk_fc2_weight) && ok;
    ok = load_weight_fp16_as_float(vector_root, "trunk_fc2__bias.bin", OUT_NEURONS, w.trunk_fc2_bias) && ok;

    ok = load_weight_fp16_as_float(vector_root, "out_bias.bin", 1, w.out_bias) && ok;
    return ok;
}

void run_cronet_fp32(
    const std::vector<float>& branch_input,
    const std::vector<float>& trunk_input,
    const WeightsFp32& w,
    ActivationsFp32& act) {

    act.branch_input_reshaped = branch_input;

    conv2d_same_3x3(
        act.branch_input_reshaped,
        HISTORY,
        1,
        NELY,
        NELX,
        w.branch_conv1_weight,
        w.branch_conv1_bias,
        BRANCH_C1_OC,
        act.branch_conv1_pre_silu);
    silu_vector(act.branch_conv1_pre_silu, act.branch_conv1_silu);

    conv2d_same_3x3(
        act.branch_conv1_silu,
        HISTORY,
        BRANCH_C1_OC,
        NELY,
        NELX,
        w.branch_conv2_weight,
        w.branch_conv2_bias,
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
    rnn_tanh_sequence(
        act.branch_rnn_input,
        w.branch_rnn_weight_ih_l0,
        w.branch_rnn_weight_hh_l0,
        w.branch_rnn_bias_ih_l0,
        w.branch_rnn_bias_hh_l0,
        act);

    linear(
        act.branch_rnn_last,
        w.branch_fc1_weight,
        w.branch_fc1_bias,
        NEURONS,
        RNN_HIDDEN,
        act.branch_fc1_pre_silu);
    silu_vector(act.branch_fc1_pre_silu, act.branch_fc1_silu);

    linear(
        act.branch_fc1_silu,
        w.branch_fc2_weight,
        w.branch_fc2_bias,
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
        w.trunk_conv1_weight,
        w.trunk_conv1_bias,
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
        w.trunk_conv2_weight,
        w.trunk_conv2_bias,
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
        w.trunk_fc1_weight,
        w.trunk_fc1_bias,
        NEURONS,
        TRUNK_FLATTEN_COUNT,
        act.trunk_fc1_pre_silu);
    silu_vector(act.trunk_fc1_pre_silu, act.trunk_fc1_silu);

    linear(
        act.trunk_fc1_silu,
        w.trunk_fc2_weight,
        w.trunk_fc2_bias,
        OUT_NEURONS,
        NEURONS,
        act.trunk_fc2);
    act.trunk_reshaped = act.trunk_fc2;

    act.final_mul_pre_bias.assign(FINAL_OUT_COUNT, 0.0f);
    act.final_output.assign(FINAL_OUT_COUNT, 0.0f);
    for (int i = 0; i < FINAL_OUT_COUNT; ++i) {
        act.final_mul_pre_bias[i] = act.branch_reshaped[i] * act.trunk_reshaped[i];
        act.final_output[i] = act.final_mul_pre_bias[i] + w.out_bias[0];
    }
}

} // namespace cronet
