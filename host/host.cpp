#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace {

constexpr std::size_t kBranchInputCount = 10 * 20 * 60;
constexpr std::size_t kTrunkInputCount = 21 * 61 * 2;
constexpr std::size_t kOutputCount = kTrunkInputCount;

std::string path_join(const std::string& a, const std::string& b) {
    if (a.empty()) {
        return b;
    }
    const char last = a[a.size() - 1];
    if (last == '/' || last == '\\') {
        return a + b;
    }
    return a + "/" + b;
}

template <typename T>
std::vector<T> load_bin(const std::string& path, std::size_t count) {
    std::vector<T> out(count);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("failed to open " + path);
    }
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (f.gcount() != static_cast<std::streamsize>(count * sizeof(T))) {
        throw std::runtime_error("short read from " + path);
    }
    return out;
}

float fp16_bits_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(bits & 0x8000u)) << 16;
    const std::uint32_t exp = (bits >> 10) & 0x1fu;
    const std::uint32_t mant = bits & 0x03ffu;

    union {
        std::uint32_t u;
        float f;
    } out;

    if (exp == 0) {
        if (mant == 0) {
            out.u = sign;
            return out.f;
        }
        const float value = std::ldexp(static_cast<float>(mant), -24);
        return sign ? -value : value;
    }

    if (exp == 0x1fu) {
        out.u = sign | 0x7f800000u | (mant << 13);
        return out.f;
    }

    const std::uint32_t exp32 = exp + (127u - 15u);
    out.u = sign | (exp32 << 23) | (mant << 13);
    return out.f;
}

bool compare_output(
    const std::vector<std::uint16_t>& got_words,
    const std::vector<float>& golden,
    float abs_tol,
    float rel_tol) {

    bool ok = true;
    std::size_t mismatches = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;

    for (std::size_t i = 0; i < got_words.size(); ++i) {
        const float got = fp16_bits_to_float(got_words[i]);
        const float ref = golden[i];
        const float diff = std::fabs(got - ref);
        const float denom = std::max(std::fabs(ref), 1.0e-12f);
        const float rel = diff / denom;
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, rel);
        if (diff > abs_tol && rel > rel_tol) {
            ok = false;
            if (mismatches < 16) {
                std::cerr << "mismatch[" << i << "]: got " << got
                          << " ref " << ref
                          << " abs " << diff
                          << " rel " << rel << "\n";
            }
            ++mismatches;
        }
    }

    std::cout << "compare final_output: " << (ok ? "PASS" : "FAIL")
              << " max_abs " << max_abs
              << " max_rel " << max_rel
              << " mismatches " << mismatches << "\n";
    return ok;
}

void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " <xclbin> [vector_root=data/cronet_hls_fp16_vectors] [case=sanity] [abs_tol=2e-2] [rel_tol=2e-2]\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 6) {
        usage(argv[0]);
        return 2;
    }

    const std::string xclbin_path = argv[1];
    const std::string vector_root = argc >= 3 ? argv[2] : "data/cronet_hls_fp16_vectors";
    const std::string case_name = argc >= 4 ? argv[3] : "sanity";
    const float abs_tol = argc >= 5 ? std::stof(argv[4]) : 2.0e-2f;
    const float rel_tol = argc >= 6 ? std::stof(argv[5]) : 2.0e-2f;

    const std::string case_root = path_join(path_join(vector_root, "cases"), case_name);
    const std::string input_root = path_join(case_root, "inputs/fp16_words");
    const std::string golden_root = path_join(case_root, "golden/fp16_torch_f32");

    try {
        std::cout << "CRONet sw_emu host\n";
        std::cout << "xclbin: " << xclbin_path << "\n";
        std::cout << "vector_root: " << vector_root << "\n";
        std::cout << "case: " << case_name << "\n";

        const auto branch_input = load_bin<std::uint16_t>(path_join(input_root, "branch_input.bin"), kBranchInputCount);
        const auto trunk_input = load_bin<std::uint16_t>(path_join(input_root, "trunk_input.bin"), kTrunkInputCount);
        const auto golden = load_bin<float>(path_join(golden_root, "final_output.bin"), kOutputCount);

        auto device = xrt::device(0);
        auto uuid = device.load_xclbin(xclbin_path);
        auto kernel = xrt::kernel(device, uuid, "cronet_kernel", xrt::kernel::cu_access_mode::exclusive);

        auto bo_branch = xrt::bo(device, kBranchInputCount * sizeof(std::uint16_t), kernel.group_id(0));
        auto bo_trunk = xrt::bo(device, kTrunkInputCount * sizeof(std::uint16_t), kernel.group_id(1));
        auto bo_output = xrt::bo(device, kOutputCount * sizeof(std::uint16_t), kernel.group_id(2));

        auto branch_map = bo_branch.map<std::uint16_t*>();
        auto trunk_map = bo_trunk.map<std::uint16_t*>();
        auto output_map = bo_output.map<std::uint16_t*>();

        std::copy(branch_input.begin(), branch_input.end(), branch_map);
        std::copy(trunk_input.begin(), trunk_input.end(), trunk_map);
        std::fill(output_map, output_map + kOutputCount, 0);

        bo_branch.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_trunk.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        std::cout << "Launching cronet_kernel...\n";
        auto run = kernel(bo_branch, bo_trunk, bo_output);
        run.wait();
        std::cout << "Kernel completed.\n";

        bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        std::vector<std::uint16_t> output_words(kOutputCount);
        std::copy(output_map, output_map + kOutputCount, output_words.begin());

        std::cout << "output[0]=" << fp16_bits_to_float(output_words[0])
                  << " output[1]=" << fp16_bits_to_float(output_words[1]) << "\n";

        const bool ok = compare_output(output_words, golden, abs_tol, rel_tol);
        std::cout << (ok ? "CRONET SW_EMU TEST PASS" : "CRONET SW_EMU TEST FAIL") << "\n";
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
