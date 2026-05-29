# vitis-hls-CROnet

Clean-room CRONet HLS implementation workspace for the 60x20 FP16 1:1 spec.

This repository is intentionally separate from the old `archive/` implementation. The source of truth is the exported vector package:

```text
../layer_wise_weights/cronet_hls_fp16_vectors
```

The first checkpoints are native C++ layerwise models. They prove shapes, flattening order, indexing, layer formulas, and FP16 payload interpretation before moving into Vitis HLS.

## Current checkpoints

Build and run the FP32 layerwise testbench from this directory:

```bash
g++ -std=c++11 -O2 -Wall -Wextra -Iinclude src/cronet_fp32.cpp tests/tb_fp32_layerwise.cpp -o build/tb_fp32_layerwise
./build/tb_fp32_layerwise ../layer_wise_weights/cronet_hls_fp16_vectors all
```

On Windows PowerShell with the local MinGW compiler:

```powershell
New-Item -ItemType Directory -Force build | Out-Null
g++ -std=c++11 -O2 -Wall -Wextra -Iinclude src\cronet_fp32.cpp tests\tb_fp32_layerwise.cpp -o build\tb_fp32_layerwise.exe
.\build\tb_fp32_layerwise.exe ..\layer_wise_weights\cronet_hls_fp16_vectors all
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
./build/tb_fp16_storage_f32acc ../layer_wise_weights/cronet_hls_fp16_vectors all
```

On Windows PowerShell:

```powershell
g++ -std=c++11 -O2 -Wall -Wextra -Iinclude src\cronet_fp32.cpp tests\tb_fp16_storage_f32acc.cpp -o build\tb_fp16_storage_f32acc.exe
.\build\tb_fp16_storage_f32acc.exe ..\layer_wise_weights\cronet_hls_fp16_vectors all
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
python3 scripts/gen_fp16_weight_header.py ../layer_wise_weights/cronet_hls_fp16_vectors
```

Expected output:

```text
wrote include/generated/cronet_weights_fp16_words.hpp
```

The generated arrays store raw IEEE binary16 payloads as `std::uint16_t`. The Vitis HLS kernel should keep these as ROMs and convert them to `half` at compute boundaries.

Generate the direct HLS `half` ROM header used by the half debug checkpoint:

```bash
python3 scripts/gen_fp16_half_header.py ../layer_wise_weights/cronet_hls_fp16_vectors
```

Expected output:

```text
wrote include/generated/cronet_weights_fp16_half.hpp
```

## HLS half debug checkpoint

This checkpoint is meant for the Vitis machine. It uses `hls_half.h`, stores named layer outputs as `half`, and compares them against `golden/fp16_torch_f32`.

From the repository root on the Vitis machine:

```bash
export CRONET_VECTOR_ROOT=../layer_wise_weights/cronet_hls_fp16_vectors
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
g++ -std=c++11 -O2 -Wall -Wextra -DCRONET_USE_NATIVE_FLOAT_HALF -Iinclude src\cronet_half_debug.cpp tests\tb_half_layerwise.cpp -o build\tb_half_layerwise_native_stub.exe
.\build\tb_half_layerwise_native_stub.exe ..\layer_wise_weights\cronet_hls_fp16_vectors all
```

That smoke test does not prove real HLS half behavior; it only catches C++ integration mistakes before handoff.
