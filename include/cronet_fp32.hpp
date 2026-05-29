#pragma once

#include <string>
#include <vector>

#include "cronet_config.hpp"

namespace cronet {

struct WeightsFp32 {
    std::vector<float> branch_conv1_weight;
    std::vector<float> branch_conv1_bias;
    std::vector<float> branch_conv2_weight;
    std::vector<float> branch_conv2_bias;
    std::vector<float> branch_rnn_weight_ih_l0;
    std::vector<float> branch_rnn_weight_hh_l0;
    std::vector<float> branch_rnn_bias_ih_l0;
    std::vector<float> branch_rnn_bias_hh_l0;
    std::vector<float> branch_fc1_weight;
    std::vector<float> branch_fc1_bias;
    std::vector<float> branch_fc2_weight;
    std::vector<float> branch_fc2_bias;

    std::vector<float> trunk_conv1_weight;
    std::vector<float> trunk_conv1_bias;
    std::vector<float> trunk_conv2_weight;
    std::vector<float> trunk_conv2_bias;
    std::vector<float> trunk_fc1_weight;
    std::vector<float> trunk_fc1_bias;
    std::vector<float> trunk_fc2_weight;
    std::vector<float> trunk_fc2_bias;

    std::vector<float> out_bias;
};

struct ActivationsFp32 {
    std::vector<float> branch_input_reshaped;
    std::vector<float> branch_conv1_pre_silu;
    std::vector<float> branch_conv1_silu;
    std::vector<float> branch_conv2_pre_silu;
    std::vector<float> branch_conv2_silu;
    std::vector<float> branch_maxpool2d;
    std::vector<float> branch_aap2d;
    std::vector<float> branch_rnn_input;
    std::vector<float> branch_rnn_all;
    std::vector<float> branch_rnn_last;
    std::vector<float> branch_rnn_step[HISTORY];
    std::vector<float> branch_fc1_pre_silu;
    std::vector<float> branch_fc1_silu;
    std::vector<float> branch_fc2;
    std::vector<float> branch_reshaped;

    std::vector<float> trunk_unsqueezed;
    std::vector<float> trunk_conv1_pre_silu;
    std::vector<float> trunk_conv1_silu;
    std::vector<float> trunk_conv2_pre_silu;
    std::vector<float> trunk_conv2_silu;
    std::vector<float> trunk_aap3d;
    std::vector<float> trunk_flatten;
    std::vector<float> trunk_fc1_pre_silu;
    std::vector<float> trunk_fc1_silu;
    std::vector<float> trunk_fc2;
    std::vector<float> trunk_reshaped;

    std::vector<float> final_mul_pre_bias;
    std::vector<float> final_output;
};

bool load_weights_fp32(const std::string& vector_root, WeightsFp32& weights);

bool load_weights_fp16_storage_as_float(const std::string& vector_root, WeightsFp32& weights);

void run_cronet_fp32(
    const std::vector<float>& branch_input,
    const std::vector<float>& trunk_input,
    const WeightsFp32& weights,
    ActivationsFp32& act);

} // namespace cronet
