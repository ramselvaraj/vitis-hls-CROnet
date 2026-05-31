#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <dirent.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace {

constexpr std::size_t kBranchInputCount = 10 * 20 * 60;
constexpr std::size_t kTrunkInputCount = 21 * 61 * 2;
constexpr std::size_t kOutputCount = kTrunkInputCount;

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

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

bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string find_default_xclbin() {
    const char* candidates[] = {
        "a.xclbin",
        "binary_container_1.xclbin",
        "cronet_kernel.sw_emu.xclbin",
        "cronet_kernel.hw_emu.xclbin",
        "cronet_kernel.hw.xclbin"
    };

    for (const char* candidate : candidates) {
        std::ifstream f(candidate, std::ios::binary);
        if (f) {
            return candidate;
        }
    }

    DIR* dir = opendir(".");
    if (!dir) {
        throw std::runtime_error("failed to scan current directory for .xclbin");
    }

    std::string found;
    while (dirent* ent = readdir(dir)) {
        const std::string name = ent->d_name;
        if (ends_with(name, ".xclbin")) {
            found = name;
            break;
        }
    }
    closedir(dir);

    if (found.empty()) {
        throw std::runtime_error("no .xclbin argument provided and no .xclbin found in current directory");
    }
    return found;
}

double elapsed_ms(TimePoint start, TimePoint end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_timing_total(const std::string& name, double total_ms) {
    std::cout << "timing_ms " << std::left << std::setw(18) << name
              << " total " << std::fixed << std::setprecision(6) << total_ms << "\n";
}

void print_timing_iter(const std::string& name, double total_ms, std::size_t n_iter) {
    const double avg_ms = n_iter == 0 ? 0.0 : total_ms / static_cast<double>(n_iter);
    std::cout << "timing_ms " << std::left << std::setw(18) << name
              << " total " << std::fixed << std::setprecision(6) << total_ms
              << " avg " << avg_ms << "\n";
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
              << " <xclbin> [vector_root=data/cronet_hls_fp16_vectors] [case=sanity] [abs_tol=2e-2] [rel_tol=2e-2] [n_iter=1]\n";
    std::cerr << "       " << argv0
              << "    # auto-detects .xclbin in current directory, useful for packaged hw_emu\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 7) {
        usage(argv[0]);
        return 2;
    }

    const std::string vector_root = argc >= 3 ? argv[2] : "data/cronet_hls_fp16_vectors";
    const std::string case_name = argc >= 4 ? argv[3] : "sanity";
    const float abs_tol = argc >= 5 ? std::stof(argv[4]) : 2.0e-2f;
    const float rel_tol = argc >= 6 ? std::stof(argv[5]) : 2.0e-2f;
    const std::size_t n_iter = argc >= 7 ? static_cast<std::size_t>(std::stoul(argv[6])) : 1u;

    if (n_iter == 0) {
        std::cerr << "ERROR: n_iter must be >= 1\n";
        return 2;
    }

    try {
        const TimePoint t_total_start = Clock::now();
        const std::string xclbin_path = argc >= 2 ? argv[1] : find_default_xclbin();
        const std::string case_root = path_join(path_join(vector_root, "cases"), case_name);
        const std::string input_root = path_join(case_root, "inputs/fp16_words");
        const std::string golden_root = path_join(case_root, "golden/fp16_torch_f32");

        std::cout << "CRONet XRT host\n";
        std::cout << "xclbin: " << xclbin_path << "\n";
        std::cout << "vector_root: " << vector_root << "\n";
        std::cout << "case: " << case_name << "\n";
        std::cout << "iterations: " << n_iter << "\n";

        const TimePoint t_load_start = Clock::now();
        const auto branch_input = load_bin<std::uint16_t>(path_join(input_root, "branch_input.bin"), kBranchInputCount);
        const auto trunk_input = load_bin<std::uint16_t>(path_join(input_root, "trunk_input.bin"), kTrunkInputCount);
        const auto golden = load_bin<float>(path_join(golden_root, "final_output.bin"), kOutputCount);
        const TimePoint t_load_end = Clock::now();

        const TimePoint t_xrt_start = Clock::now();
        auto device = xrt::device(0);
        auto uuid = device.load_xclbin(xclbin_path);
        auto kernel = xrt::kernel(device, uuid, "cronet_kernel", xrt::kernel::cu_access_mode::exclusive);
        const TimePoint t_xrt_end = Clock::now();

        const TimePoint t_bo_start = Clock::now();
        auto bo_branch = xrt::bo(device, kBranchInputCount * sizeof(std::uint16_t), kernel.group_id(0));
        auto bo_trunk = xrt::bo(device, kTrunkInputCount * sizeof(std::uint16_t), kernel.group_id(1));
        auto bo_output = xrt::bo(device, kOutputCount * sizeof(std::uint16_t), kernel.group_id(2));

        auto branch_map = bo_branch.map<std::uint16_t*>();
        auto trunk_map = bo_trunk.map<std::uint16_t*>();
        auto output_map = bo_output.map<std::uint16_t*>();
        const TimePoint t_bo_end = Clock::now();

        const TimePoint t_host_copy_start = Clock::now();
        std::copy(branch_input.begin(), branch_input.end(), branch_map);
        std::copy(trunk_input.begin(), trunk_input.end(), trunk_map);
        std::fill(output_map, output_map + kOutputCount, 0);
        const TimePoint t_host_copy_end = Clock::now();

        const TimePoint t_output_init_start = Clock::now();
        bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        const TimePoint t_output_init_end = Clock::now();

        double h2d_sync_ms = 0.0;
        double kernel_run_ms = 0.0;
        double d2h_sync_ms = 0.0;

        std::cout << "Launching cronet_kernel for " << n_iter << " iteration(s)...\n";
        for (std::size_t iter = 0; iter < n_iter; ++iter) {
            const TimePoint t_h2d_start = Clock::now();
            bo_branch.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bo_trunk.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            const TimePoint t_h2d_end = Clock::now();
            h2d_sync_ms += elapsed_ms(t_h2d_start, t_h2d_end);

            const TimePoint t_kernel_start = Clock::now();
            auto run = kernel(bo_branch, bo_trunk, bo_output);
            run.wait();
            const TimePoint t_kernel_end = Clock::now();
            kernel_run_ms += elapsed_ms(t_kernel_start, t_kernel_end);

            const TimePoint t_d2h_start = Clock::now();
            bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const TimePoint t_d2h_end = Clock::now();
            d2h_sync_ms += elapsed_ms(t_d2h_start, t_d2h_end);
        }
        std::cout << "Kernel completed.\n";

        const TimePoint t_output_copy_start = Clock::now();
        std::vector<std::uint16_t> output_words(kOutputCount);
        std::copy(output_map, output_map + kOutputCount, output_words.begin());
        const TimePoint t_output_copy_end = Clock::now();

        std::cout << "output[0]=" << fp16_bits_to_float(output_words[0])
                  << " output[1]=" << fp16_bits_to_float(output_words[1]) << "\n";

        const TimePoint t_compare_start = Clock::now();
        const bool ok = compare_output(output_words, golden, abs_tol, rel_tol);
        const TimePoint t_compare_end = Clock::now();

        const TimePoint t_total_end = Clock::now();
        const double file_load_ms = elapsed_ms(t_load_start, t_load_end);
        const double xrt_setup_ms = elapsed_ms(t_xrt_start, t_xrt_end);
        const double bo_setup_ms = elapsed_ms(t_bo_start, t_bo_end);
        const double host_copy_ms = elapsed_ms(t_host_copy_start, t_host_copy_end);
        const double output_init_ms = elapsed_ms(t_output_init_start, t_output_init_end);
        const double output_copy_ms = elapsed_ms(t_output_copy_start, t_output_copy_end);
        const double compare_ms = elapsed_ms(t_compare_start, t_compare_end);
        const double inference_io_ms = h2d_sync_ms + kernel_run_ms + d2h_sync_ms;
        const double total_host_ms = elapsed_ms(t_total_start, t_total_end);

        std::cout << "timing_iterations " << n_iter << "\n";
        print_timing_total("file_load", file_load_ms);
        print_timing_total("xrt_setup", xrt_setup_ms);
        print_timing_total("bo_setup", bo_setup_ms);
        print_timing_total("host_copy", host_copy_ms);
        print_timing_total("output_init_sync", output_init_ms);
        print_timing_iter("h2d_sync", h2d_sync_ms, n_iter);
        print_timing_iter("kernel_run", kernel_run_ms, n_iter);
        print_timing_iter("d2h_sync", d2h_sync_ms, n_iter);
        print_timing_iter("inference_io", inference_io_ms, n_iter);
        print_timing_total("output_copy", output_copy_ms);
        print_timing_total("compare", compare_ms);
        print_timing_total("total_host", total_host_ms);

        std::cout << (ok ? "CRONET XRT TEST PASS" : "CRONET XRT TEST FAIL") << "\n";
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
