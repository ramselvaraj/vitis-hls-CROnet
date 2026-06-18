# CRONet HLS (FP16) — Vitis PL Kernel

Clean-room CRONet implementations as Vitis HLS programmable-logic kernels for the
**AMD Versal VEK280** (Vitis 2024.1). This repository bundles **three mesh-size
variants** of the same network so they can be built and validated from one place.

The network is a two-branch model (a recurrent **branch** path + a convolutional
**trunk** path) that regresses a per-node displacement field over a structural
mesh. Weights are stored as FP16 ROMs compiled into the kernel; computation runs
in FP16 storage with FP32 accumulation.

## Variants

Each variant differs only in the mesh element grid (`NELX × NELY`), which changes
every downstream tensor shape and the final output size:

| Folder            | `NELX` | `NELY` | Node grid (`H×W`) | Output neurons (`H·W·2`) |
|-------------------|:------:|:------:|:-----------------:|:------------------------:|
| [`60x20/`](60x20) |   60   |   20   |      21 × 61      |          2562            |
| [`30x20/`](30x20) |   30   |   20   |      21 × 31      |          1302            |
| [`30x10/`](30x10) |   30   |   10   |      11 × 31      |           682            |

Common to all: `HISTORY = 10` timesteps, RNN hidden = 64, FC width = 40, 2 DOF/node.

Because the tensor shapes are baked into the code, the variant-specific files are
**not** shared — each folder is fully self-contained:

- `src/cronet_kernel.cpp` — the HLS kernel (variant-specific dataflow/shapes)
- `include/cronet_config.hpp` — the dimension constants
- `host/host.cpp` — the XRT host (variant-specific I/O sizes)
- `scripts/gen_fp16_*.py` — weight-header generators (variant-specific tensor shapes)

Everything else (testbenches, `vitis/`, `Makefile`, `env_source.sh`, docs) is
identical across the three but duplicated so each folder builds on its own.

## Repository layout

```text
.
├── README.md            ← you are here
├── .gitignore
├── reference/           ← shared spec & reference material (network spec, PDFs,
│                          vector exporter, manifest) — not needed to build
├── 60x20/               ┐
├── 30x20/               ├ one self-contained HLS project per variant
└── 30x10/               ┘
```

Each variant folder contains:

```text
src/        cronet_kernel.cpp, cronet_fp32.cpp, cronet_half_debug.cpp
include/    config + AXI + half-type + bin-io headers   (generated/ is regenerated)
host/       host.cpp (XRT host application)
scripts/    gen_fp16_*.py, run_hw_board.sh, prepare_board_handoff.sh, xrt_profile.ini
vitis/      system.cfg, half_layerwise_csim.tcl
tests/      native C++ layerwise testbenches
docs/       utilization / timing summary + AIE-alignment notes
data/       README.md only — the vector package is regenerated (see below)
Makefile, env_source.sh, README.md, HW_RUN.md, CODEX_PORTING_SUMMARY.md
```

## What is **not** in this repo (and how to get it)

To keep the upload small, two large, regenerable inputs are intentionally
excluded (see [`.gitignore`](.gitignore)):

1. **The FP16 vector package** — per-variant inputs + golden tensors used by the
   testbenches and the host (`<variant>/data/cronet_hls_fp16_vectors/`, ~30–100 MB
   each). Regenerate it with the exporter in
   [`reference/layer_wise_weights/`](reference/layer_wise_weights), or copy a
   prebuilt package into place. See each variant's `data/README.md` for the
   expected layout (it must contain `manifest.json`).

2. **The generated FP16 weight ROM headers** — `include/generated/*.hpp`
   (~10–15 MB each). Regenerate them from the vector package:

   ```bash
   cd 60x20                          # or 30x20 / 30x10
   python3 scripts/gen_fp16_weight_header.py data/cronet_hls_fp16_vectors
   python3 scripts/gen_fp16_half_header.py   data/cronet_hls_fp16_vectors
   # → writes include/generated/cronet_weights_fp16_words.hpp
   #          include/generated/cronet_weights_fp16_half.hpp
   ```

Board deliverables (`sd_card.img`, `*.xclbin`, `BOOT.BIN`, …) and all Vitis build
artifacts are likewise excluded — they are produced by the build.

## Quick start (per variant)

All commands are run from inside a variant folder (e.g. `cd 60x20`). See that
folder's `README.md` for the full staged flow and expected output.

```bash
# 0. Put the vector package at data/cronet_hls_fp16_vectors/ (see data/README.md),
#    then generate the weight headers (commands above).

# 1. Native correctness checks (no Vitis needed):
make fp32-test            # FP32 layerwise golden check
make fp16-storage-test    # FP16-storage / FP32-accum golden check

# 2. Vitis flow on the VEK280 machine (Vitis 2024.1):
source env_source.sh 24
make vitis-check
make vitis-sw-emu CASE=sanity     # software emulation
make vitis-hw-emu-run             # hardware emulation
make vitis-hw-build               # build the hw SD-card image
```

The 60x20 hardware build has been board-validated on the VEK280
(~410k kernel cycles @ 230 MHz; see commit history and `60x20/docs/`).

## Requirements

- **Native checks:** a C++11 compiler (g++/clang).
- **Vitis flow:** AMD Vitis 2024.1 + XRT, VEK280 base platform
  (`xilinx_vek280_base_202410_1`), and the Versal `Image`/`rootfs.ext4`.
  Source the environment with `source env_source.sh 24` before building.
