# vitis-hls-CROnet

Clean-room CRONet HLS implementation workspace for the 60x20 FP16 1:1 spec.

This repository is intentionally separate from the old `archive/` implementation. The source of truth is the exported vector package:

```text
data/cronet_hls_fp16_vectors
```

The first checkpoints are native C++ layerwise models. They prove shapes, flattening order, indexing, layer formulas, and FP16 payload interpretation before moving into Vitis HLS.

## Data setup

The vector package must live at:

```text
data/cronet_hls_fp16_vectors
```

That folder is ignored by git on purpose. A GitHub clone will contain [data/README.md](data/README.md), but not the 102 MB binary vector package itself.

Before running on the Vitis Linux machine, copy the ignored folder into the clone:

```bash
mkdir -p data
cp -a /path/to/cronet_hls_fp16_vectors data/
test -f data/cronet_hls_fp16_vectors/manifest.json
```

Expected check result: the `test` command prints nothing and exits with code `0`.

## Vitis environment

On the Vitis Linux machine, source the 2024.1 VEK280 environment before running HLS:

```bash
source env_source.sh advent_rigel_2024_1
```

Short aliases also work:

```bash
source env_source.sh 24
source env_source.sh vitis_2024_1
```

Expected output includes paths for `vitis_hls`, `vitis`, `v++`, and `vivado`. If one of those prints nothing, the corresponding tool was not found in the sourced environment.

## Current checkpoints

Build and run the FP32 layerwise testbench from this directory:

```bash
g++ -std=c++11 -O2 -Wall -Wextra -Iinclude src/cronet_fp32.cpp tests/tb_fp32_layerwise.cpp -o build/tb_fp32_layerwise
./build/tb_fp32_layerwise data/cronet_hls_fp16_vectors all
```

On Windows PowerShell with the local MinGW compiler:

```powershell
New-Item -ItemType Directory -Force build | Out-Null
g++ -std=c++11 -O2 -Wall -Wextra -Iinclude src\cronet_fp32.cpp tests\tb_fp32_layerwise.cpp -o build\tb_fp32_layerwise.exe
.\build\tb_fp32_layerwise.exe data\cronet_hls_fp16_vectors all
```

Expected final output:

```text
CASE sanity: PASS
CASE ramp: PASS
CASE sparse: PASS
CASE random: PASS
ALL FP32 LAYERWISE TESTS PASS
```

Then run the FP16-storage / FP32-accumulation checkpoint:

```bash
g++ -std=c++11 -O2 -Wall -Wextra -Iinclude src/cronet_fp32.cpp tests/tb_fp16_storage_f32acc.cpp -o build/tb_fp16_storage_f32acc
./build/tb_fp16_storage_f32acc data/cronet_hls_fp16_vectors all
```

On Windows PowerShell:

```powershell
g++ -std=c++11 -O2 -Wall -Wextra -Iinclude src\cronet_fp32.cpp tests\tb_fp16_storage_f32acc.cpp -o build\tb_fp16_storage_f32acc.exe
.\build\tb_fp16_storage_f32acc.exe data\cronet_hls_fp16_vectors all
```

Expected final output:

```text
CASE sanity: PASS
CASE ramp: PASS
CASE sparse: PASS
CASE random: PASS
ALL FP16-STORAGE FP32-ACC LAYERWISE TESTS PASS
```

If a mismatch occurs, the testbench prints the tensor name, first mismatching indices, and max absolute/relative error.

## Staged implementation flow

1. FP32 debug path against `golden/fp32`.
2. FP16 storage with float accumulation against `golden/fp16_io_f32`.
3. True HLS `half` compute/storage against `golden/fp16_torch` or `golden/fp16_torch_f32`.
4. Convert validated FP16 weights into static HLS ROM headers.
5. Wrap the network as a Vitis PL kernel.
6. Run Vitis in order: `sw_emu`, then `hw_emu`, then `hw`.

Vitis builds are not expected to run on this authoring machine. They should be run on the Vitis-capable machine after each handoff.

## Generate FP16 ROM header

After FP16 storage is validated, generate the static weight header:

```bash
python3 scripts/gen_fp16_weight_header.py data/cronet_hls_fp16_vectors
```

Expected output:

```text
wrote include/generated/cronet_weights_fp16_words.hpp
```

The generated arrays store raw IEEE binary16 payloads as `std::uint16_t`. The Vitis HLS kernel should keep these as ROMs and convert them to `half` at compute boundaries.

Generate the direct HLS `half` ROM header used by the half debug checkpoint:

```bash
python3 scripts/gen_fp16_half_header.py data/cronet_hls_fp16_vectors
```

Expected output:

```text
wrote include/generated/cronet_weights_fp16_half.hpp
```

## HLS half debug checkpoint

This checkpoint is meant for the Vitis machine. It uses `hls_half.h`, stores named layer outputs as `half`, and compares them against `golden/fp16_torch_f32`.

From the repository root on the Vitis machine:

```bash
export CRONET_VECTOR_ROOT=data/cronet_hls_fp16_vectors
export CRONET_CASE=all
export CRONET_ABS_TOL=2e-2
export CRONET_REL_TOL=2e-2
vitis_hls -f vitis/half_layerwise_csim.tcl
```

Expected final output:

```text
CASE sanity: PASS
CASE ramp: PASS
CASE sparse: PASS
CASE random: PASS
ALL HLS HALF LAYERWISE TESTS PASS
```

On this authoring machine, the same file can be syntax-smoked with a float-backed stub:

```powershell
g++ -std=c++11 -O2 -Wall -Wextra -DCRONET_USE_NATIVE_FLOAT_HALF -DCRONET_HALF_DEBUG_SINGLE_TRANSLATION_UNIT -Iinclude tests\tb_half_layerwise.cpp -o build\tb_half_layerwise_native_stub.exe
.\build\tb_half_layerwise_native_stub.exe data\cronet_hls_fp16_vectors all
```

That smoke test does not prove real HLS half behavior; it only catches C++ integration mistakes before handoff.

## First Vitis sw_emu Kernel Check

The first Vitis kernel wrapper is correctness-first. It uses AXI memory ports for branch input, trunk input, and final output. Weights are compiled into the kernel through generated FP16 ROM headers.

On the Vitis Linux machine:

```bash
cd ~/vitis-hls-CROnet
source env_source.sh 24
test -f data/cronet_hls_fp16_vectors/manifest.json && echo "vectors ready"
make vitis-check
make vitis-sw-emu CASE=sanity
```

Expected high-level output:

```text
vectors ready
VITIS_HOME=...
BASE_PLATFORM=...
TARGET=sw_emu
VECTORS=data/cronet_hls_fp16_vectors
...
CRONet XRT host
case: sanity
Launching cronet_kernel...
Kernel completed.
compare final_output: PASS ...
CRONET XRT TEST PASS
```

After `sanity` passes, run the other cases one at a time:

```bash
make vitis-sw-emu CASE=ramp
make vitis-sw-emu CASE=sparse
make vitis-sw-emu CASE=random
```

Or run all four cases:

```bash
make vitis-sw-emu-all
```

The first `sw_emu` build may be slow because Vitis has to compile and link the kernel. Re-running a different case should reuse the built xclbin unless sources changed.

## First Vitis hw_emu Check

After all `sw_emu` cases pass, build and package `hw_emu`:

```bash
cd ~/vitis-hls-CROnet
source env_source.sh 24
make vitis-hw-emu-build
```

Expected high-level flow:

```text
v++ -c -t hw_emu ...
v++ -l -t hw_emu ... -o build/cronet_kernel.hw_emu.xsa
v++ -p -t hw_emu ... --package.sd_file host.exe --package.sd_file ../scripts/run_hw_board.sh --package.sd_file ../data ...
```

This creates:

```text
build/package.hw_emu/launch_hw_emu.sh
build/package.hw_emu/sd_card.img
```

Then launch the emulator and run the packaged host:

```bash
make vitis-hw-emu-run
```

Expected host ending:

```text
CRONet XRT host
case: sanity
Launching cronet_kernel...
Kernel completed.
compare final_output: PASS ...
CRONET XRT TEST PASS
```

`hw_emu` is much slower than `sw_emu`; start with the default `sanity` case only.

To run a specific case manually inside the packaged emulation directory:

```bash
cd build/package.hw_emu
./launch_hw_emu.sh -run-app "./run_hw_board.sh sanity"
```

## First Vitis hw Package

After `hw_emu` passes, build the hardware package:

```bash
cd ~/vitis-hls-CROnet
source env_source.sh 24
make vitis-hw-build
```

Expected package artifact:

```text
build/package.hw/sd_card.img
```

This target builds the image only. Booting/running on the board is the next handoff step.

After booting the VEK280 from the generated SD image, run on the board:

```bash
cd /run/media/mmcblk0p1
./run_hw_board.sh sanity
```

The board runner accepts optional tolerances and an iteration count:

```bash
./run_hw_board.sh sanity 2e-2 2e-2 100
```

The fourth argument runs the same input case for `n_iter` kernel launches and
prints total/average timing for host-to-device sync, kernel execution,
device-to-host sync, and the combined input-to-output path.

Expected board-side ending:

```text
CRONet XRT host
case: sanity
Launching cronet_kernel...
Kernel completed.
compare final_output: PASS ...
timing_iterations ...
timing_ms kernel_run ...
timing_ms inference_io ...
CRONET XRT TEST PASS
```

The script also accepts the same case names:

```bash
./run_hw_board.sh ramp
./run_hw_board.sh sparse
./run_hw_board.sh random
```
