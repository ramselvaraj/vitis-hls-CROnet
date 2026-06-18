#pragma once

#include <string>
#include <vector>

#include "cronet_config.hpp"
#include "cronet_half_type.hpp"

namespace cronet {

struct ActivationsHalfDebug {
    std::vector<fp16_t> branch_input_reshaped;
    std::vector<fp16_t> branch_conv1_pre_silu;
    std::vector<fp16_t> branch_conv1_silu;
    std::vector<fp16_t> branch_conv2_pre_silu;
    std::vector<fp16_t> branch_conv2_silu;
    std::vector<fp16_t> branch_maxpool2d;
    std::vector<fp16_t> branch_aap2d;
    std::vector<fp16_t> branch_rnn_input;
    std::vector<fp16_t> branch_rnn_all;
    std::vector<fp16_t> branch_rnn_last;
    std::vector<fp16_t> branch_rnn_step[HISTORY];
    std::vector<fp16_t> branch_fc1_pre_silu;
    std::vector<fp16_t> branch_fc1_silu;
    std::vector<fp16_t> branch_fc2;
    std::vector<fp16_t> branch_reshaped;

    std::vector<fp16_t> trunk_unsqueezed;
    std::vector<fp16_t> trunk_conv1_pre_silu;
    std::vector<fp16_t> trunk_conv1_silu;
    std::vector<fp16_t> trunk_conv2_pre_silu;
    std::vector<fp16_t> trunk_conv2_silu;
    std::vector<fp16_t> trunk_aap3d;
    std::vector<fp16_t> trunk_flatten;
    std::vector<fp16_t> trunk_fc1_pre_silu;
    std::vector<fp16_t> trunk_fc1_silu;
    std::vector<fp16_t> trunk_fc2;
    std::vector<fp16_t> trunk_reshaped;

    std::vector<fp16_t> final_mul_pre_bias;
    std::vector<fp16_t> final_output;
};

bool load_case_inputs_fp16(
    const std::string& vector_root,
    const std::string& case_name,
    std::vector<fp16_t>& branch_input,
    std::vector<fp16_t>& trunk_input);

void run_cronet_half_debug(
    const std::vector<fp16_t>& branch_input,
    const std::vector<fp16_t>& trunk_input,
    ActivationsHalfDebug& act);

} // namespace cronet
