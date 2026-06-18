# CRONet HLS Codex Porting Summary

Last updated: 2026-06-02.

This file summarizes the current CRONet Vitis HLS implementation state so a new
Codex client can continue without rediscovering the project history.

## Repository And Sync Model

Primary local repo:

```text
C:\ramse\CRONet_hls\vitis-hls-CROnet
```

Mirror repo also used for GitHub branch handoff:

```text
C:\ramse\CRONet_hls\git\CRONet_HLS
```

Important workflow detail:

- The Vitis Linux machine is updated by one-way WinSCP sync from the Windows repo.
- The Linux machine does not pull from GitHub for normal iteration.
- Files ignored by Git can still exist on Linux after WinSCP sync, which is intentional for data/weights.
- Before asking the user to run Linux commands, assume the Windows repo must be synced first.

Latest pushed commits at the time this file was written:

```text
vitis-hls-CROnet main:          f822f70 Stabilize CRONet hw handoff build
CRONet_HLS vitis-cronet-hls:    287ccc1 Stabilize CRONet hw handoff build
```

There may be newer local changes after those commits.

## Current Goal

The active implementation is a clean-room Vitis HLS CRONet kernel for the
`60x20` problem size, with the smaller paper sizes planned as future separate
specializations.

The final implementation target is:

- HLS/PL inference using IEEE FP16.
- Weights compiled into the kernel.
- Runtime DDR traffic only for packed branch input, packed trunk input, packed
  final output, and one profiling counter word.
- Fully functional Vitis software emulation, hardware emulation, hardware build,
  and board handoff package.
- Board execution compares final output against golden values and reports timing
  over `n_iter` kernel runs.

## Model Shape

Current specialization:

```text
NELX = 60
NELY = 20
HISTORY = 10
DOF = 2
```

Runtime tensor sizes:

```text
branch input: [1, 10, 20, 60] = 12000 FP16 values
trunk input:  [1, 21, 61, 2]  = 2562 FP16 values
output:       [1, 21, 61, 2]  = 2562 FP16 values
```

Current Vitis kernel ABI packs these FP16 values into 512-bit AXI words:

```text
branch input: 375 x 512-bit words
trunk input:   81 x 512-bit words
output:        81 x 512-bit words
profile:        1 x 64-bit cycle counter word
```

The final equation remains:

```text
output[i] = branch_fc2[i] * trunk_fc2[i] + out_bias[0]
```

## Precision

Current implementation precision:

```text
FP16 in HLS/PL
```

The referenced AIE paper reports BF16, so comparisons should be described as:

```text
AIE BF16 paper target vs this PL/HLS FP16 implementation
```

Do not claim it is BF16 unless the datatype is explicitly rebuilt.

## Golden Data And Test Cases

Board/runtime test cases available:

```text
sanity
ramp
sparse
random
```

Each case has:

```text
data/cronet_hls_fp16_vectors/cases/<case>/inputs/fp16_words/branch_input.bin
data/cronet_hls_fp16_vectors/cases/<case>/inputs/fp16_words/trunk_input.bin
data/cronet_hls_fp16_vectors/cases/<case>/golden/fp16_torch_f32/final_output.bin
```

The board-side comparison checks only final output against the golden final
output for the selected case. It does not check every layer on board.

Layerwise golden checks exist for HLS C simulation / debug testbench flows, not
for the packaged board run.

## Current Stable Kernel Design

Current source file:

```text
src/cronet_kernel.cpp
```

Stable board candidate currently uses:

```text
BRANCH_CONV2_IC_PAR = 4
TRUNK_CONV2_IC_PAR = 8
fused trunk pool+FC1: not used
adaptive_avg_pool3d_60x20: used as separate layer
trunk_fc1: normal linear_fixed after trunk_aap
top-level DDR ABI: ap_uint<512> packed FP16 words
profile_cycles: one m_axi output word written by an in-kernel counter
```

The trunk fused pool+FC1 experiment was reverted for the stable handoff build.

Current expected source signals:

```bash
grep -n "TRUNK_CONV2_IC_PAR = \|trunk_conv2_60x20_ic[48]\|trunk_pool_fc1\|adaptive_avg_pool3d_60x20\|trunk_aap\|load_packed_fp16\|compute_cycle_counter\|profile_cycles" src/cronet_kernel.cpp
```

Expected:

```text
TRUNK_CONV2_IC_PAR = 8
trunk_conv2_60x20_ic8
adaptive_avg_pool3d_60x20
trunk_aap
load_packed_fp16
compute_cycle_counter
profile_cycles
```

Expected not to see:

```text
trunk_pool_fc1
```

## Optimization History

Baseline was a straightforward generic implementation with very high HLS
latency. The main optimizations were:

1. Specialized generic loops to fixed `60x20` constants.
2. Added branch conv2 IC-parallel optimized function:

```text
branch_conv2_60x20_ic4
```

3. Added trunk conv2 IC-parallel optimized function:

```text
trunk_conv2_60x20_ic8
```

4. Tried an adaptive-pool rewrite with reduction trees. It did not materially
   reduce latency.
5. Tried fused trunk adaptive-pool + trunk FC1:

```text
trunk_pool_fc1_60x20_p4
```

This gave strong HLS latency improvement but caused high BRAM/congestion and
hardware timing issues. It was removed from the stable board candidate.

6. Tried backing trunk conv2 down to IC4 to help timing:

```text
trunk_conv2_60x20_ic4
```

This made trunk conv2 extremely slow again, so it was rejected.

Stable selected point:

```text
trunk_conv2_60x20_ic8 + separate adaptive pool + separate trunk FC1
```

## Known Measured Results

The final stable hardware package was built successfully.

Reported HLS estimates for the stable board candidate:

```text
BRAM utilization: about 61%
DSP utilization: about 15%
max HLS latency: about 10.128 ms
```

Routed hardware clock:

```text
requested kernel clock: 250 MHz
freqhz compile flag: --freqhz=250000000
freqhz link flag:    --freqhz=250000000:cronet_kernel_1
```

Routed timing passed. Reported timing margins from the routed timing report:

```text
setup worst slack: +0.527 ns
hold worst slack:  +0.038 ns
PW worst slack:    +0.194 ns
```

Interpretation:

- 250 MHz is the stable handoff clock.
- There is some setup headroom.
- Hold slack is tight, so do not casually raise the clock for delivery.
- 275 MHz could be tested later as an experiment, not as the stable handoff.

## Why DSP Utilization Is Low

DSP utilization is low because the design is not a fully parallel MAC array.
Most arithmetic is pipelined and reused over cycles. Also, using more DSPs
requires more parallel memory reads, which requires more array partitioning or
replication. In this design, BRAM/banking/routing pressure is a stronger limiter
than raw DSP availability.

Short version:

```text
DSP low = arithmetic units are not saturated
BRAM/routing = actual limiting resources
```

## Vitis Version Rules

Use Vitis 2024 for software emulation:

```bash
source env_source.sh 24
export BASE_PLATFORM=/mnt/vault0/tools/Vitis_2024.1/Vitis/2024.1/base_platforms/xilinx_vek280_base_202410_1/xilinx_vek280_base_202410_1.xpfm
```

Use Vitis 2025 for HLS hardware compile, hardware emulation, hardware build,
and packaging:

```bash
source env_source.sh 25
export BASE_PLATFORM=/mnt/vault0/tools/Vitis_2025.2/2025.2/Vitis/base_platforms/xilinx_vek280_base_202520_1/xilinx_vek280_base_202520_1.xpfm
```

This split matters. The user has repeatedly confirmed:

```text
sw_emu -> 2024
hw/hw_emu/hw package -> 2025
```

## Important Makefile Features

The Makefile has a hardware clock knob:

```text
KERNEL_FREQ ?= 250000000
KERNEL_CU ?= cronet_kernel_1
```

For `TARGET=hw`, compile uses:

```text
--freqhz=250000000
```

and link uses:

```text
--freqhz=250000000:cronet_kernel_1
```

There is also a quick parser check target:

```bash
make BASE_PLATFORM="$BASE_PLATFORM" TARGET=hw vitis-hw-preflight
```

Expected output includes:

```text
compile clock flag: --freqhz=250000000
link clock flag:    --freqhz=250000000:cronet_kernel_1
Vitis hw link option parser accepted the clock/config syntax.
```

If Linux says `No rule to make target 'vitis-hw-preflight'`, then WinSCP did
not sync the latest Makefile.

## Native Local Test

Windows/native `g++` test target:

```bash
make kernel-native-test NATIVE_CXX=/usr/bin/g++
```

If running directly on Windows without `make`:

```powershell
g++ -std=c++11 -O2 -Wall -Wextra -Iinclude -DCRONET_USE_NATIVE_FLOAT_HALF tests/tb_kernel_native.cpp src/cronet_kernel.cpp -o build/tb_kernel_native.exe
.\build\tb_kernel_native.exe data/cronet_hls_fp16_vectors all
```

Expected:

```text
CASE sanity: PASS
CASE ramp: PASS
CASE sparse: PASS
CASE random: PASS
ALL NATIVE KERNEL TESTS PASS
```

Native test uses float as the stand-in for HLS `half`; it catches indexing and
final-output behavior drift, but it does not prove true Vitis half behavior or
hardware timing.

## Software Emulation Commands

After WinSCP sync to Linux:

```bash
cd ~/vitis-hls-CROnet
source env_source.sh 24
export BASE_PLATFORM=/mnt/vault0/tools/Vitis_2024.1/Vitis/2024.1/base_platforms/xilinx_vek280_base_202410_1/xilinx_vek280_base_202410_1.xpfm

make vitis-clean
make BASE_PLATFORM="$BASE_PLATFORM" vitis-sw-emu-all
```

Expected:

```text
CRONET XRT TEST PASS
```

for all cases.

## Hardware Compile And Package Commands

Use Vitis 2025:

```bash
cd ~/vitis-hls-CROnet
source env_source.sh 25
export BASE_PLATFORM=/mnt/vault0/tools/Vitis_2025.2/2025.2/Vitis/base_platforms/xilinx_vek280_base_202520_1/xilinx_vek280_base_202520_1.xpfm
```

Quick parser/preflight:

```bash
make BASE_PLATFORM="$BASE_PLATFORM" TARGET=hw vitis-hw-preflight
```

HLS kernel compile only:

```bash
make vitis-clean
make BASE_PLATFORM="$BASE_PLATFORM" TARGET=hw vitis-kernel
```

Full hardware package:

```bash
make BASE_PLATFORM="$BASE_PLATFORM" TARGET=hw vitis-package
```

Verify package:

```bash
ls -lh build/package.hw/sd_card.img
ls -lh build/package.hw/sd_card/a.xclbin build/package.hw/sd_card/host.exe build/package.hw/sd_card/run_hw_board.sh
```

## Board Handoff

Prepare handoff folder:

```bash
bash scripts/prepare_board_handoff.sh
ls -lht build/board_handoff | head
```

The handoff folder contains:

```text
sd_card.img
sd_card/
README_BOARD_RUN.md
BUILD_INFO.txt
SHA256SUMS.txt
reports/
docs/
```

The `sd_card/` payload is slimmed to keep only the files required for board
execution:

```text
a.xclbin
host.exe
run_hw_board.sh
data/cronet_hls_fp16_vectors/manifest.json
branch_input.bin / trunk_input.bin / final_output.bin for each case
```

Board-owner command sheet:

```text
docs/BOARD_OWNER_RUN_COMMANDS.md
```

## Board Run Commands

After booting the VEK280 from `sd_card.img`, find the SD mount:

```bash
find /run/media /mnt -name run_hw_board.sh 2>/dev/null
```

Enter that directory, usually:

```bash
cd /run/media/mmcblk0p1
chmod +x host.exe run_hw_board.sh 2>/dev/null || true
```

Single sanity run:

```bash
sh ./run_hw_board.sh sanity
```

Timed run with 100 iterations:

```bash
sh ./run_hw_board.sh sanity 2e-2 2e-2 100
```

All cases with 100 iterations:

```bash
for case in sanity ramp sparse random; do
  echo "=== $case ==="
  sh ./run_hw_board.sh "$case" 2e-2 2e-2 100
done
```

Argument order:

```text
run_hw_board.sh <case> <abs_tol> <rel_tol> <n_iter>
```

Expected output includes:

```text
compare final_output: PASS ...
timing_iterations 100
kernel_core_cycles ...
kernel_core_ms_at_250mhz ...
timing_ms kernel_run    total ... avg ...
timing_ms inference_io  total ... avg ...
CRONET XRT TEST PASS
```

Timing field meanings:

```text
kernel_core_cycles: cycles measured inside HLS after packed input DDR loads and before packed output DDR stores
kernel_core_ms_at_250mhz: kernel_core_cycles converted at the 250 MHz handoff clock
kernel_run avg:   average XRT kernel execution time
inference_io avg: average H2D sync + kernel execution + D2H sync time
```

## Report Extraction Commands

HLS latency:

```bash
grep -n -E "Latency|cronet_kernel|branch_conv2|trunk_conv2|adaptive|linear|VITIS_LOOP| sec| ms| us" \
  _x/reports/cronet_kernel.hw/hls_reports/cronet_kernel_csynth.rpt | head -80
```

HLS utilization:

```bash
grep -n -E "Total|Available|Utilization \(%\)|BRAM_18K|DSP|FF|LUT|URAM" \
  _x/reports/cronet_kernel.hw/hls_reports/cronet_kernel_csynth.rpt
```

Clock flags:

```bash
grep -Rni "freqhz\|kernel_frequency\|clock" \
  build/cronet_kernel.hw.xo.compile_summary \
  build/cronet_kernel.hw.xsa.link_summary \
  build/v++.package_summary \
  _x/logs 2>/dev/null | head -80
```

Routed timing slack:

```bash
grep -n -E "WNS|TNS|Slack|Requirement|Data Path Delay|Timing constraints" \
  _x/reports/link/imp/impl_1_route_report_timing_summary_0.rpt | head -80
```

## Mistakes To Avoid

Do not reintroduce the fused trunk pool+FC1 path for stable handoff without
careful BRAM/timing validation. It was fast in HLS but caused board-build risk.

Do not switch trunk conv2 to IC4 for stable handoff. It reduced parallelism but
made trunk conv2 latency very large.

Do not use `--freqhz=250MHz` for the Makefile clock path. Vitis 2025 rejected
or mishandled clock formatting during link in this flow. Use raw Hz:

```text
250000000
250000000:cronet_kernel_1
```

Do not run full hardware package just to inspect HLS latency or utilization.
Use:

```bash
make BASE_PLATFORM="$BASE_PLATFORM" TARGET=hw vitis-kernel
```

Do not assume GitHub pull on Linux. The Linux machine is normally synced by
WinSCP from Windows.

When analyzing pasted reports, search only for the needed lines with `rg` or
`grep`. Do not read huge reports top to bottom.

## Current Open Follow-ups

Future optimizations should be done one layer at a time:

1. Keep stable handoff build as baseline.
2. Try safer branch-side pooling/FC optimizations before touching trunk conv2.
3. If trying more DSP use, partition weights/activations carefully and check
   BRAM/routing after each step.
4. Consider separate clean specializations for the smaller paper sizes:

```text
30x10
30x20
60x20
```

5. Do not optimize all sizes at once.
