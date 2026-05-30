# CRONet HLS Hardware Utilization and Timing Summary

This note summarizes the first correctness-first hardware package for the
60x20 CRONet FP16 Vitis HLS implementation.

## Build Context

| Item | Value |
|---|---|
| Target board | AMD Versal VEK280 |
| Vitis version used for successful hardware package | Vitis 2025.2 |
| Platform | `xilinx_vek280_base_202520_1.xpfm` |
| Kernel | `cronet_kernel` |
| Precision | PL/HLS IEEE FP16 / binary16 |
| Model size | `nelx=60`, `nely=20` |
| Output tensor | `[1,21,61,2] = 2562` FP16 values |
| Package artifact | `build/package.hw/sd_card.img` |

Precision comparison note: the AIE paper reports BF16 on Versal AIE-ML. This
HLS build is FP16 because the current exported HLS vector package is FP16. Any
comparison should therefore be labeled `AIE BF16 vs PL/HLS FP16` unless both
implementations are rebuilt using the same datatype.

## Package Artifact Check

The self-contained hardware package contained:

| Artifact | Observed size |
|---|---:|
| `build/package.hw/sd_card.img` | 3.0 GB |
| `build/package.hw/sd_card/a.xclbin` | 14 MB |
| `build/package.hw/sd_card/host.exe` | 80 KB |
| `build/package.hw/sd_card/run_hw_board.sh` | 1.2 KB |

The packaged SD payload also included:

- `data/cronet_hls_fp16_vectors/manifest.json`
- the full local vector package used by the host-side golden comparison
- `run_hw_board.sh` for board-side execution

## Routed Timing Result

The routed implementation timing report stated:

```text
All user specified timing constraints are met.
```

Key visible timing checks from the routed timing summary:

| Check | Failing endpoints | Worst slack | Total violation |
|---|---:|---:|---:|
| Setup | 0 | 0.000 ns | 0.000 ns |
| Hold | 0 | 0.003 ns | 0.000 ns |
| Pulse width | 0 | 0.171 ns | 0.000 ns |

Interpretation:

- Timing is closed.
- Setup slack is exactly at the edge in the copied report excerpt.
- This is acceptable for a first correctness-first build, but it leaves no setup
  timing margin for later optimization until the design is rechecked.

## HLS Resource Estimate

The HLS synthesis estimate reported the following resource usage:

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| BRAM_18K | 391 | 1200 | 32% |
| DSP | 386 | 1312 | 29% |
| FF | 77,860 | 1,041,408 | 7% |
| LUT | 168,169 | 520,704 | 32% |
| URAM | 135 | 264 | 51% |

Interpretation:

- URAM is the tightest resource class because weights are stored on chip.
- DSP and LUT still have meaningful headroom for optimization.
- The estimate is consistent with the core requirement: weights are compiled
  into the FPGA kernel rather than streamed from DDR.

## Routed Kernel Utilization Excerpt

The routed kernel utilization excerpt showed:

| Vitis report column | Used | Utilization |
|---|---:|---:|
| Column 1 | 63,973 | 6.16% |
| Column 2 | 69,722 | 13.46% |
| Column 3 | 5,205 | 2.00% |
| BRAM-class column | 250 | 41.67% |
| URAM | 115 | 43.56% |
| DSP | 335 | 25.53% |

The original pasted terminal excerpt did not include the full report header for
the first three columns. The BRAM/URAM/DSP interpretation is consistent with
the VEK280 resource totals and the HLS estimate, but the routed report should
remain the source of truth when writing final tables.

## Latency Note

The HLS report excerpt included a very large conservative maximum latency:

```text
187,693,025 cycles ~= 0.601 s
```

This should not be treated as measured board runtime. It appears to come from
conservative HLS analysis around variable loop bounds, especially the
`AdaptiveAvgPool3D` path. The same excerpt also showed a much smaller fixed-side
latency near:

```text
161,825 cycles ~= 0.518 ms
```

Action item for performance reporting:

- Use XRT profiling on the physical board for measured end-to-end runtime.
- Use the HLS latency report only as a synthesis estimate.
- Before optimizing latency, specialize fixed-size loops so Vitis does not infer
  overly broad variable bounds.

## Architecture Meaning of These Numbers

The current build is a correctness-first implementation:

- DDR is used only for `branch_input`, `trunk_input`, and `output`.
- Weights are generated as `static const fp16_t` arrays and compiled into the
  kernel.
- Intermediate activations are local/static arrays inside `cronet_kernel`.
- No external memory ports exist for weights or intermediate tensors.

That satisfies the major AIE-paper-style data movement requirement for a fair
accelerator comparison: inference should avoid DRAM traffic for weights and
intermediate activations.

## Report Files to Preserve

Keep these files with the board handoff package:

```text
build/cronet_kernel.hw.xo.compile_summary
build/cronet_kernel.hw.xsa.link_summary
build/v++.package_summary
_x/reports/cronet_kernel.hw/hls_reports/cronet_kernel_csynth.rpt
_x/reports/link/imp/impl_1_route_report_timing_summary_0.rpt
_x/reports/link/imp/impl_1_kernel_util_routed.rpt
_x/reports/link/imp/impl_1_full_util_routed.rpt
```

## Suggested Final Status Statement

```text
The CRONet 60x20 FP16 PL/HLS implementation completed Vitis hardware packaging
for VEK280 using Vitis 2025.2. The generated SD image is self-contained and
includes the xclbin, host executable, runner script, and vector/golden package.
Routed timing constraints were met. HLS estimated utilization was 32% BRAM_18K,
29% DSP, 7% FF, 32% LUT, and 51% URAM. Board execution remains to be measured on
physical VEK280 hardware.
```
