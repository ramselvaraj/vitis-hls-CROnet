#pragma once

#include <cstddef>

namespace cronet {

static const int NELX = 30;
static const int NELY = 20;
static const int HISTORY = 10;
static const int NEURONS = 40;
static const int NODE_H = NELY + 1;
static const int NODE_W = NELX + 1;
static const int DOF = 2;
static const int OUT_NEURONS = NODE_H * NODE_W * DOF;

static const int BRANCH_C1_OC = 16;
static const int BRANCH_C2_OC = 32;
static const int TRUNK_C1_OC = 16;
static const int TRUNK_C2_OC = 32;
static const int RNN_HIDDEN = 64;
static const int TRUNK_AAP_D = 15;
static const int TRUNK_AAP_H = 10;
static const int TRUNK_AAP_W = 1;

static const int BRANCH_IN_COUNT = HISTORY * NELY * NELX;
static const int TRUNK_IN_COUNT = NODE_H * NODE_W * DOF;
static const int FINAL_OUT_COUNT = OUT_NEURONS;

static const int BRANCH_CONV1_OUT_COUNT = HISTORY * BRANCH_C1_OC * NELY * NELX;
static const int BRANCH_CONV2_OUT_COUNT = HISTORY * BRANCH_C2_OC * NELY * NELX;
static const int BRANCH_MAXPOOL_OUT_COUNT = HISTORY * BRANCH_C2_OC * (NELY / 2) * (NELX / 2);
static const int BRANCH_AAP2D_OUT_COUNT = HISTORY * BRANCH_C2_OC;
static const int BRANCH_RNN_INPUT_COUNT = HISTORY * BRANCH_C2_OC;
static const int BRANCH_RNN_ALL_COUNT = HISTORY * RNN_HIDDEN;
static const int BRANCH_RNN_LAST_COUNT = RNN_HIDDEN;
static const int BRANCH_FC1_OUT_COUNT = NEURONS;
static const int BRANCH_FC2_OUT_COUNT = OUT_NEURONS;

static const int TRUNK_CONV1_OUT_W = 3;
static const int TRUNK_CONV2_OUT_W = 4;
static const int TRUNK_CONV1_OUT_COUNT = TRUNK_C1_OC * NODE_H * NODE_W * TRUNK_CONV1_OUT_W;
static const int TRUNK_CONV2_OUT_COUNT = TRUNK_C2_OC * NODE_H * NODE_W * TRUNK_CONV2_OUT_W;
static const int TRUNK_CONV2_WSUM_COUNT = TRUNK_C2_OC * NODE_H * NODE_W;
static const int TRUNK_AAP3D_OUT_COUNT = TRUNK_C2_OC * TRUNK_AAP_D * TRUNK_AAP_H * TRUNK_AAP_W;
static const int TRUNK_FLATTEN_COUNT = TRUNK_AAP3D_OUT_COUNT;
static const int TRUNK_FC1_OUT_COUNT = NEURONS;
static const int TRUNK_FC2_OUT_COUNT = OUT_NEURONS;

inline std::size_t idx3(int a, int b, int c, int B, int C) {
    return (static_cast<std::size_t>(a) * B + b) * C + c;
}

inline std::size_t idx4(int a, int b, int c, int d, int B, int C, int D) {
    return ((static_cast<std::size_t>(a) * B + b) * C + c) * D + d;
}

inline std::size_t idx5(int a, int b, int c, int d, int e, int B, int C, int D, int E) {
    return (((static_cast<std::size_t>(a) * B + b) * C + c) * D + d) * E + e;
}

inline std::size_t idx_nchw(int n, int c, int h, int w, int C, int H, int W) {
    return idx4(n, c, h, w, C, H, W);
}

inline std::size_t idx_ncdhw(int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    return idx5(n, c, d, h, w, C, D, H, W);
}

inline std::size_t idx_linear_weight(int out, int in, int IN) {
    return static_cast<std::size_t>(out) * IN + in;
}

} // namespace cronet
