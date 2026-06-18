#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bin_io.hpp"
#include "cronet_fp32.hpp"

namespace {

struct TensorCheck {
    const char* name;
    const std::vector<float>* values;
    std::size_t count;
};

bool load_case_inputs(
    const std::string& vector_root,
    const std::string& case_name,
    std::vector<float>& branch_input,
    std::vector<float>& trunk_input) {

    const std::string input_root =
        cronet::path_join(cronet::path_join(cronet::path_join(vector_root, "cases"), case_name), "inputs/fp32");

    bool ok = true;
    ok = cronet::load_bin_exact(cronet::path_join(input_root, "branch_input.bin"),
                                branch_input,
                                cronet::BRANCH_IN_COUNT) && ok;
    ok = cronet::load_bin_exact(cronet::path_join(input_root, "trunk_input.bin"),
                                trunk_input,
                                cronet::TRUNK_IN_COUNT) && ok;
    return ok;
}

bool compare_tensor(
    const std::string& vector_root,
    const std::string& case_name,
    const TensorCheck& check,
    float abs_tol,
    float rel_tol) {

    std::vector<float> golden;
    const std::string golden_root =
        cronet::path_join(cronet::path_join(cronet::path_join(vector_root, "cases"), case_name), "golden/fp32");
    const std::string path = cronet::path_join(golden_root, std::string(check.name) + ".bin");
    if (!cronet::load_bin_exact(path, golden, check.count)) {
        return false;
    }
    const cronet::CompareStats stats =
        cronet::compare_float_vectors(*check.values, golden, abs_tol, rel_tol, check.name, 8);
    return stats.pass;
}

bool run_one_case(
    const std::string& vector_root,
    const std::string& case_name,
    const cronet::WeightsFp32& weights,
    float abs_tol,
    float rel_tol) {

    std::printf("\n=== CASE %s ===\n", case_name.c_str());

    std::vector<float> branch_input;
    std::vector<float> trunk_input;
    if (!load_case_inputs(vector_root, case_name, branch_input, trunk_input)) {
        return false;
    }

    cronet::ActivationsFp32 act;
    cronet::run_cronet_fp32(branch_input, trunk_input, weights, act);

    std::vector<TensorCheck> checks;
    checks.push_back({"branch_input_reshaped", &act.branch_input_reshaped, cronet::BRANCH_IN_COUNT});
    checks.push_back({"branch_conv1_pre_silu", &act.branch_conv1_pre_silu, cronet::BRANCH_CONV1_OUT_COUNT});
    checks.push_back({"branch_conv1_silu", &act.branch_conv1_silu, cronet::BRANCH_CONV1_OUT_COUNT});
    checks.push_back({"branch_conv2_pre_silu", &act.branch_conv2_pre_silu, cronet::BRANCH_CONV2_OUT_COUNT});
    checks.push_back({"branch_conv2_silu", &act.branch_conv2_silu, cronet::BRANCH_CONV2_OUT_COUNT});
    checks.push_back({"branch_maxpool2d", &act.branch_maxpool2d, cronet::BRANCH_MAXPOOL_OUT_COUNT});
    checks.push_back({"branch_aap2d", &act.branch_aap2d, cronet::BRANCH_AAP2D_OUT_COUNT});
    checks.push_back({"branch_rnn_input", &act.branch_rnn_input, cronet::BRANCH_RNN_INPUT_COUNT});

    const char* rnn_step_names[cronet::HISTORY] = {
        "branch_rnn_step_00",
        "branch_rnn_step_01",
        "branch_rnn_step_02",
        "branch_rnn_step_03",
        "branch_rnn_step_04",
        "branch_rnn_step_05",
        "branch_rnn_step_06",
        "branch_rnn_step_07",
        "branch_rnn_step_08",
        "branch_rnn_step_09"
    };
    for (int t = 0; t < cronet::HISTORY; ++t) {
        checks.push_back({rnn_step_names[t], &act.branch_rnn_step[t], cronet::BRANCH_RNN_LAST_COUNT});
    }

    checks.push_back({"branch_rnn_all", &act.branch_rnn_all, cronet::BRANCH_RNN_ALL_COUNT});
    checks.push_back({"branch_rnn_last", &act.branch_rnn_last, cronet::BRANCH_RNN_LAST_COUNT});
    checks.push_back({"branch_fc1_pre_silu", &act.branch_fc1_pre_silu, cronet::BRANCH_FC1_OUT_COUNT});
    checks.push_back({"branch_fc1_silu", &act.branch_fc1_silu, cronet::BRANCH_FC1_OUT_COUNT});
    checks.push_back({"branch_fc2", &act.branch_fc2, cronet::BRANCH_FC2_OUT_COUNT});
    checks.push_back({"branch_reshaped", &act.branch_reshaped, cronet::BRANCH_FC2_OUT_COUNT});

    checks.push_back({"trunk_unsqueezed", &act.trunk_unsqueezed, cronet::TRUNK_IN_COUNT});
    checks.push_back({"trunk_conv1_pre_silu", &act.trunk_conv1_pre_silu, cronet::TRUNK_CONV1_OUT_COUNT});
    checks.push_back({"trunk_conv1_silu", &act.trunk_conv1_silu, cronet::TRUNK_CONV1_OUT_COUNT});
    checks.push_back({"trunk_conv2_pre_silu", &act.trunk_conv2_pre_silu, cronet::TRUNK_CONV2_OUT_COUNT});
    checks.push_back({"trunk_conv2_silu", &act.trunk_conv2_silu, cronet::TRUNK_CONV2_OUT_COUNT});
    checks.push_back({"trunk_aap3d", &act.trunk_aap3d, cronet::TRUNK_AAP3D_OUT_COUNT});
    checks.push_back({"trunk_flatten", &act.trunk_flatten, cronet::TRUNK_FLATTEN_COUNT});
    checks.push_back({"trunk_fc1_pre_silu", &act.trunk_fc1_pre_silu, cronet::TRUNK_FC1_OUT_COUNT});
    checks.push_back({"trunk_fc1_silu", &act.trunk_fc1_silu, cronet::TRUNK_FC1_OUT_COUNT});
    checks.push_back({"trunk_fc2", &act.trunk_fc2, cronet::TRUNK_FC2_OUT_COUNT});
    checks.push_back({"trunk_reshaped", &act.trunk_reshaped, cronet::TRUNK_FC2_OUT_COUNT});

    checks.push_back({"final_mul_pre_bias", &act.final_mul_pre_bias, cronet::FINAL_OUT_COUNT});
    checks.push_back({"final_output", &act.final_output, cronet::FINAL_OUT_COUNT});

    bool ok = true;
    for (std::size_t i = 0; i < checks.size(); ++i) {
        ok = compare_tensor(vector_root, case_name, checks[i], abs_tol, rel_tol) && ok;
    }

    std::printf("CASE %s: %s\n", case_name.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s <vector_root> [all|sanity|ramp|sparse|random] [abs_tol] [rel_tol]\n",
                 argv0);
}

} // namespace

int main(int argc, char** argv) {
    const std::string vector_root = argc >= 2 ? argv[1] : "data/cronet_hls_fp16_vectors";
    const std::string selected_case = argc >= 3 ? argv[2] : "all";
    const float abs_tol = argc >= 4 ? static_cast<float>(std::atof(argv[3])) : 1.0e-5f;
    const float rel_tol = argc >= 5 ? static_cast<float>(std::atof(argv[4])) : 1.0e-5f;

    if (argc > 5) {
        print_usage(argv[0]);
        return 2;
    }

    cronet::WeightsFp32 weights;
    if (!cronet::load_weights_fp32(vector_root, weights)) {
        return 1;
    }

    std::vector<std::string> cases;
    if (selected_case == "all") {
        cases.push_back("sanity");
        cases.push_back("ramp");
        cases.push_back("sparse");
        cases.push_back("random");
    } else if (selected_case == "sanity" ||
               selected_case == "ramp" ||
               selected_case == "sparse" ||
               selected_case == "random") {
        cases.push_back(selected_case);
    } else {
        print_usage(argv[0]);
        return 2;
    }

    std::printf("CRONet FP32 layerwise testbench\n");
    std::printf("vector_root: %s\n", vector_root.c_str());
    std::printf("abs_tol: %.9e rel_tol: %.9e\n", abs_tol, rel_tol);

    bool ok = true;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        ok = run_one_case(vector_root, cases[i], weights, abs_tol, rel_tol) && ok;
    }

    if (ok) {
        std::printf("\nALL FP32 LAYERWISE TESTS PASS\n");
        return 0;
    }

    std::printf("\nFP32 LAYERWISE TESTS FAILED\n");
    return 1;
}
