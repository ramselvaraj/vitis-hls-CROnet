#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

#include "bin_io.hpp"
#include "cronet_config.hpp"
#include "cronet_half_type.hpp"

extern "C" void cronet_kernel(
    const cronet::fp16_t* branch_input,
    const cronet::fp16_t* trunk_input,
    cronet::fp16_t* output);

namespace {

bool run_one_case(
    const std::string& vector_root,
    const std::string& case_name,
    float abs_tol,
    float rel_tol) {

    const std::string case_root =
        cronet::path_join(cronet::path_join(vector_root, "cases"), case_name);
    const std::string input_root = cronet::path_join(case_root, "inputs/fp16_words");
    const std::string golden_root = cronet::path_join(case_root, "golden/fp16_io_f32");

    std::vector<float> branch_input;
    std::vector<float> trunk_input;
    std::vector<float> golden;
    std::vector<float> output(cronet::FINAL_OUT_COUNT, 0.0f);

    bool ok = true;
    ok = cronet::load_fp16_words_as_float(
             cronet::path_join(input_root, "branch_input.bin"),
             branch_input,
             cronet::BRANCH_IN_COUNT) && ok;
    ok = cronet::load_fp16_words_as_float(
             cronet::path_join(input_root, "trunk_input.bin"),
             trunk_input,
             cronet::TRUNK_IN_COUNT) && ok;
    ok = cronet::load_bin_exact(
             cronet::path_join(golden_root, "final_output.bin"),
             golden,
             cronet::FINAL_OUT_COUNT) && ok;
    if (!ok) {
        return false;
    }

    cronet_kernel(branch_input.data(), trunk_input.data(), output.data());

    const cronet::CompareStats stats =
        cronet::compare_float_vectors(output, golden, abs_tol, rel_tol, "final_output", 8);
    std::printf("CASE %s: %s\n", case_name.c_str(), stats.pass ? "PASS" : "FAIL");
    return stats.pass;
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
    const float abs_tol = argc >= 4 ? static_cast<float>(std::atof(argv[3])) : 2.0e-2f;
    const float rel_tol = argc >= 5 ? static_cast<float>(std::atof(argv[4])) : 2.0e-2f;

    if (argc > 5) {
        print_usage(argv[0]);
        return 2;
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

    std::printf("CRONet native kernel smoke test\n");
    std::printf("vector_root: %s\n", vector_root.c_str());
    std::printf("golden: fp16_io_f32\n");
    std::printf("abs_tol: %.9e rel_tol: %.9e\n", abs_tol, rel_tol);

    bool ok = true;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        ok = run_one_case(vector_root, cases[i], abs_tol, rel_tol) && ok;
    }

    if (ok) {
        std::printf("\nALL NATIVE KERNEL TESTS PASS\n");
        return 0;
    }

    std::printf("\nNATIVE KERNEL TESTS FAILED\n");
    return 1;
}
