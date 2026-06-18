#pragma once
#include <cstddef>

// Generated CRONet HLS vector constants.
static constexpr int CRONET_NELX = 60;
static constexpr int CRONET_NELY = 20;
static constexpr int CRONET_HISTORY = 10;
static constexpr int CRONET_NEURONS = 40;
static constexpr int CRONET_OUT_NEURONS = 2562;

static constexpr int CRONET_BRANCH_IN_COUNT = CRONET_HISTORY * CRONET_NELY * CRONET_NELX;
static constexpr int CRONET_TRUNK_IN_COUNT = (CRONET_NELY + 1) * (CRONET_NELX + 1) * 2;
static constexpr int CRONET_FINAL_OUT_COUNT = CRONET_TRUNK_IN_COUNT;

static constexpr int BRANCH_CONV1_OUT_COUNT = CRONET_HISTORY * 16 * CRONET_NELY * CRONET_NELX;
static constexpr int BRANCH_CONV2_OUT_COUNT = CRONET_HISTORY * 32 * CRONET_NELY * CRONET_NELX;
static constexpr int BRANCH_MAXPOOL_OUT_COUNT = CRONET_HISTORY * 32 * (CRONET_NELY / 2) * (CRONET_NELX / 2);
static constexpr int BRANCH_AAP2D_OUT_COUNT = CRONET_HISTORY * 32;
static constexpr int BRANCH_RNN_ALL_COUNT = CRONET_HISTORY * 64;
static constexpr int BRANCH_RNN_LAST_COUNT = 64;
static constexpr int BRANCH_FC1_OUT_COUNT = CRONET_NEURONS;
static constexpr int BRANCH_FC2_OUT_COUNT = CRONET_OUT_NEURONS;

static constexpr int TRUNK_CONV1_OUT_COUNT = 16 * (CRONET_NELY + 1) * (CRONET_NELX + 1) * 3;
static constexpr int TRUNK_CONV2_OUT_COUNT = 32 * (CRONET_NELY + 1) * (CRONET_NELX + 1) * 4;
static constexpr int TRUNK_AAP3D_OUT_COUNT = 32 * 15 * 10 * 1;
static constexpr int TRUNK_FLATTEN_COUNT = TRUNK_AAP3D_OUT_COUNT;
static constexpr int TRUNK_FC1_OUT_COUNT = CRONET_NEURONS;
static constexpr int TRUNK_FC2_OUT_COUNT = CRONET_OUT_NEURONS;
