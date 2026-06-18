#!/usr/bin/env python3
"""
CRONet HLS vector exporter.

Generates deterministic dummy weights, deterministic test inputs, and layer-by-layer
reference outputs for the CRONet model in the current project.

Default model size matches the current cronet.py settings:
  nelx = 60, nely = 20, history = 10, neurons = 40
  output_neurons = (nely + 1) * (nelx + 1) * 2 = 2562

Main outputs:
  - raw little-endian float32 .bin files for HLS C/RTL simulation
  - raw little-endian uint16 BF16-word .bin files for BF16 HLS designs
  - manifest.json with shapes, order, dtype, file paths, and SHA256 checksums
  - cronet_hls_shapes.hpp and tb_bin_io.hpp helper headers
  - EXPORT_REFERENCE.md describing every exported tensor

Run:
  python3 cronet_hls_vector_exporter.py --outdir cronet_hls_vectors_export --cases sanity,ramp,sparse,random

Quick check:
  python3 cronet_hls_vector_exporter.py --outdir cronet_hls_vectors_export --check

Optional text exports for debugging:
  python3 cronet_hls_vector_exporter.py --outdir cronet_hls_vectors_export --emit-txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_NELX = 60
DEFAULT_NELY = 20
DEFAULT_HISTORY = 10
DEFAULT_NEURONS = 40


# ----------------------------- BF16 utilities -----------------------------

def float32_to_bf16_words(x: np.ndarray) -> np.ndarray:
    """Convert float32 values to raw BF16 uint16 words with round-to-nearest-even."""
    x32 = np.asarray(x, dtype=np.float32)
    u32 = x32.view(np.uint32)
    lsb = (u32 >> 16) & 1
    rounding_bias = np.uint32(0x7FFF) + lsb.astype(np.uint32)
    rounded = u32 + rounding_bias
    return (rounded >> 16).astype(np.uint16)


def bf16_words_to_float32(words: np.ndarray) -> np.ndarray:
    """Convert raw BF16 uint16 words to float32 values."""
    w = np.asarray(words, dtype=np.uint16).astype(np.uint32)
    u32 = w << 16
    return u32.view(np.float32)


def bf16_round_to_float32(x: np.ndarray) -> np.ndarray:
    """Round float32 values to BF16 precision, returned as float32."""
    return bf16_words_to_float32(float32_to_bf16_words(x))


# ----------------------------- CRONet model -----------------------------

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class CRONetHLSReference(nn.Module):
    """Side-effect-free CRONet inference graph matching the current PyTorch model.

    Tensor conventions:
      branch input: [B, history, nely, nelx]
      trunk input:  [B, nely + 1, nelx + 1, 2]

    PyTorch Conv3d consumes [N, C, D, H, W]. Therefore trunk_input.unsqueeze(1)
    is interpreted as [B, 1, D=nely+1, H=nelx+1, W=2].
    """

    def __init__(
        self,
        nelx: int = DEFAULT_NELX,
        nely: int = DEFAULT_NELY,
        history: int = DEFAULT_HISTORY,
        neurons: int = DEFAULT_NEURONS,
    ) -> None:
        super().__init__()
        self.nelx = int(nelx)
        self.nely = int(nely)
        self.history = int(history)
        self.neurons = int(neurons)
        self.output_neurons = (self.nely + 1) * (self.nelx + 1) * 2

        # BranchNet
        self.branch_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.branch_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.branch_rnn = nn.RNN(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.branch_fc1 = nn.Linear(64, self.neurons)
        self.branch_fc2 = nn.Linear(self.neurons, self.output_neurons)

        # TrunkNet
        self.trunk_conv1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 2), stride=1, padding=1)
        self.trunk_conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 2), stride=1, padding=1)
        self.trunk_fc1 = nn.Linear(32 * 15 * 10 * 1, self.neurons)
        self.trunk_fc2 = nn.Linear(self.neurons, self.output_neurons)

        self.out_bias = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
        return_intermediates: bool = False,
    ):
        z: Dict[str, torch.Tensor] = {}
        bsz, history, height, width = branch_input.shape
        assert height == self.nely and width == self.nelx
        assert history == self.history
        assert tuple(trunk_input.shape) == (bsz, self.nely + 1, self.nelx + 1, 2)

        # BranchNet
        x = branch_input.reshape(bsz * history, 1, height, width)
        z["branch_input_reshaped"] = x

        x = self.branch_conv1(x)
        z["branch_conv1_pre_silu"] = x
        x = silu(x)
        z["branch_conv1_silu"] = x

        x = self.branch_conv2(x)
        z["branch_conv2_pre_silu"] = x
        x = silu(x)
        z["branch_conv2_silu"] = x

        x = F.max_pool2d(x, kernel_size=2, stride=2)
        z["branch_maxpool2d"] = x

        x = F.adaptive_avg_pool2d(x, output_size=1)
        z["branch_aap2d"] = x

        x = x.reshape(bsz, history, 32)
        z["branch_rnn_input"] = x
        x_all, _ = self.branch_rnn(x)
        z["branch_rnn_all"] = x_all
        for step in range(self.history):
            z[f"branch_rnn_step_{step:02d}"] = x_all[:, step, :]
        x = x_all[:, -1, :]
        z["branch_rnn_last"] = x

        x = self.branch_fc1(x)
        z["branch_fc1_pre_silu"] = x
        x = silu(x)
        z["branch_fc1_silu"] = x

        x = self.branch_fc2(x)
        z["branch_fc2"] = x
        branch_out = x.reshape(bsz, self.nely + 1, self.nelx + 1, 2)
        z["branch_reshaped"] = branch_out

        # TrunkNet
        t = trunk_input.unsqueeze(1)
        z["trunk_unsqueezed"] = t

        t = self.trunk_conv1(t)
        z["trunk_conv1_pre_silu"] = t
        t = silu(t)
        z["trunk_conv1_silu"] = t

        t = self.trunk_conv2(t)
        z["trunk_conv2_pre_silu"] = t
        t = silu(t)
        z["trunk_conv2_silu"] = t

        t = F.adaptive_avg_pool3d(t, output_size=(15, 10, 1))
        z["trunk_aap3d"] = t

        t = torch.flatten(t, start_dim=1)
        z["trunk_flatten"] = t

        t = self.trunk_fc1(t)
        z["trunk_fc1_pre_silu"] = t
        t = silu(t)
        z["trunk_fc1_silu"] = t

        t = self.trunk_fc2(t)
        z["trunk_fc2"] = t
        trunk_out = t.reshape(bsz, self.nely + 1, self.nelx + 1, 2)
        z["trunk_reshaped"] = trunk_out

        y = branch_out * trunk_out
        z["final_mul_pre_bias"] = y
        y = y + self.out_bias.reshape(1, 1, 1, 1)
        z["final_output"] = y

        if return_intermediates:
            return y, z
        return y


# ----------------------------- deterministic data -----------------------------

def deterministic_values(shape: Tuple[int, ...], scale: float, offset: int = 0) -> np.ndarray:
    """Small repeatable signed pattern; no RNG required."""
    n = int(np.prod(shape))
    idx = np.arange(offset, offset + n, dtype=np.float32)
    vals = ((idx % 37.0) - 18.0) * scale
    return vals.reshape(shape).astype(np.float32)


def initialize_dummy_weights(model: CRONetHLSReference) -> None:
    """Fill all model parameters with deterministic dummy values."""
    with torch.no_grad():
        for i, (name, param) in enumerate(model.named_parameters()):
            if "weight" in name:
                scale = 1.0 / 1024.0
            else:
                scale = 1.0 / 512.0
            arr = deterministic_values(tuple(param.shape), scale=scale, offset=1009 * (i + 1))
            param.copy_(torch.from_numpy(arr))
        model.out_bias.copy_(torch.tensor([1.0 / 32.0], dtype=torch.float32))


def make_case(case_name: str, nelx: int, nely: int, history: int) -> Tuple[np.ndarray, np.ndarray]:
    branch_shape = (1, history, nely, nelx)
    trunk_shape = (1, nely + 1, nelx + 1, 2)

    if case_name == "sanity":
        branch = np.ones(branch_shape, dtype=np.float32) * 0.25
        trunk = np.zeros(trunk_shape, dtype=np.float32)
    elif case_name == "ramp":
        branch = deterministic_values(branch_shape, scale=1.0 / 128.0, offset=7)
        trunk = deterministic_values(trunk_shape, scale=1.0 / 128.0, offset=7003)
    elif case_name == "sparse":
        branch = np.zeros(branch_shape, dtype=np.float32)
        for h in range(history):
            y0 = (2 * h) % max(1, nely - 4)
            x0 = (5 * h) % max(1, nelx - 8)
            branch[0, h, y0 : y0 + 4, x0 : x0 + 8] = 0.125 * (h + 1)
        trunk = np.zeros(trunk_shape, dtype=np.float32)
    elif case_name == "random":
        rng = np.random.default_rng(12345)
        branch = rng.normal(loc=0.0, scale=0.1, size=branch_shape).astype(np.float32)
        trunk = rng.normal(loc=0.0, scale=0.1, size=trunk_shape).astype(np.float32)
    else:
        raise ValueError(f"Unknown case '{case_name}'. Use sanity,ramp,sparse,random.")

    # Add load-like entries in every case so TrunkNet sees sparse nonzero structure.
    trunk[0, 0, 0, 1] = -1.0
    trunk[0, nely // 2, nelx // 2, 0] = 0.5
    trunk[0, nely, nelx, 1] = 0.25
    return np.ascontiguousarray(branch, dtype=np.float32), np.ascontiguousarray(trunk, dtype=np.float32)


def clone_with_bf16_rounded_params(model: CRONetHLSReference) -> CRONetHLSReference:
    new_model = CRONetHLSReference(model.nelx, model.nely, model.history, model.neurons)
    new_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        for param in new_model.parameters():
            arr = param.detach().cpu().numpy().astype(np.float32)
            param.copy_(torch.from_numpy(bf16_round_to_float32(arr).reshape(tuple(param.shape))))
    return new_model


# ----------------------------- export helpers -----------------------------

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def safe_name(name: str) -> str:
    return name.replace(".", "__").replace("/", "__")


def save_raw_array(
    path: Path,
    arr: np.ndarray,
    dtype: str,
    emit_txt: bool = False,
    emit_npy: bool = False,
) -> Dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)

    if dtype == "float32":
        a = np.ascontiguousarray(arr, dtype="<f4")
    elif dtype == "uint16":
        a = np.ascontiguousarray(arr, dtype="<u2")
    else:
        raise ValueError(dtype)

    raw = a.tobytes(order="C")
    path.write_bytes(raw)

    info: Dict[str, object] = {
        "path": str(path),
        "dtype": dtype,
        "shape": list(arr.shape),
        "count": int(np.prod(arr.shape)),
        "bytes": len(raw),
        "flatten_order": "C row-major / numpy C-contiguous",
        "sha256": sha256_bytes(raw),
    }

    if dtype == "float32":
        flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        info.update({
            "min": float(np.min(flat)) if flat.size else 0.0,
            "max": float(np.max(flat)) if flat.size else 0.0,
            "mean": float(np.mean(flat)) if flat.size else 0.0,
            "first_8": [float(v) for v in flat[:8]],
        })
    else:
        flat_u = np.asarray(arr, dtype=np.uint16).reshape(-1)
        info.update({
            "first_8_hex": [f"0x{int(v):04x}" for v in flat_u[:8]],
        })

    if emit_txt:
        txt_path = path.with_suffix(".txt")
        if dtype == "float32":
            np.savetxt(txt_path, np.asarray(arr, dtype=np.float32).reshape(-1), fmt="%.9e")
        else:
            np.savetxt(txt_path, np.asarray(arr, dtype=np.uint16).reshape(-1), fmt="%u")
        info["txt_path"] = str(txt_path)

    if emit_npy:
        npy_path = path.with_suffix(".npy")
        np.save(npy_path, arr)
        info["npy_path"] = str(npy_path)

    return info


def relpaths(manifest: Dict[str, object], root: Path) -> Dict[str, object]:
    """Make path strings relative to root before JSON writing."""
    def conv(obj):
        if isinstance(obj, dict):
            return {k: conv(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [conv(v) for v in obj]
        if isinstance(obj, str):
            try:
                p = Path(obj)
                if p.is_absolute():
                    return str(p.relative_to(root))
            except Exception:
                pass
        return obj
    return conv(manifest)


EXPORT_TENSOR_ORDER = [
    "branch_input_reshaped",
    "branch_conv1_pre_silu",
    "branch_conv1_silu",
    "branch_conv2_pre_silu",
    "branch_conv2_silu",
    "branch_maxpool2d",
    "branch_aap2d",
    "branch_rnn_input",
    "branch_rnn_all",
    "branch_rnn_last",
    "branch_fc1_pre_silu",
    "branch_fc1_silu",
    "branch_fc2",
    "branch_reshaped",
    "trunk_unsqueezed",
    "trunk_conv1_pre_silu",
    "trunk_conv1_silu",
    "trunk_conv2_pre_silu",
    "trunk_conv2_silu",
    "trunk_aap3d",
    "trunk_flatten",
    "trunk_fc1_pre_silu",
    "trunk_fc1_silu",
    "trunk_fc2",
    "trunk_reshaped",
    "final_mul_pre_bias",
    "final_output",
]


PARAMETER_LAYOUTS = {
    "branch_conv1.weight": "[out_channel=16, in_channel=1, kernel_h=3, kernel_w=3]",
    "branch_conv1.bias": "[out_channel=16]",
    "branch_conv2.weight": "[out_channel=32, in_channel=16, kernel_h=3, kernel_w=3]",
    "branch_conv2.bias": "[out_channel=32]",
    "branch_rnn.weight_ih_l0": "[hidden_size=64, input_size=32]",
    "branch_rnn.weight_hh_l0": "[hidden_size=64, hidden_size=64]",
    "branch_rnn.bias_ih_l0": "[hidden_size=64]",
    "branch_rnn.bias_hh_l0": "[hidden_size=64]",
    "branch_fc1.weight": "[out_features=40, in_features=64]",
    "branch_fc1.bias": "[out_features=40]",
    "branch_fc2.weight": "[out_features=2562, in_features=40]",
    "branch_fc2.bias": "[out_features=2562]",
    "trunk_conv1.weight": "[out_channel=16, in_channel=1, kernel_d=3, kernel_h=3, kernel_w=2]",
    "trunk_conv1.bias": "[out_channel=16]",
    "trunk_conv2.weight": "[out_channel=32, in_channel=16, kernel_d=3, kernel_h=3, kernel_w=2]",
    "trunk_conv2.bias": "[out_channel=32]",
    "trunk_fc1.weight": "[out_features=40, in_features=4800]",
    "trunk_fc1.bias": "[out_features=40]",
    "trunk_fc2.weight": "[out_features=2562, in_features=40]",
    "trunk_fc2.bias": "[out_features=2562]",
    "out_bias": "[1] scalar broadcast over final output",
}


# ----------------------------- generated support files -----------------------------

def write_shapes_header(outdir: Path, nelx: int, nely: int, history: int, neurons: int) -> None:
    output_neurons = (nely + 1) * (nelx + 1) * 2
    text = f"""#pragma once
#include <cstddef>

// Generated CRONet HLS vector constants.
static constexpr int CRONET_NELX = {nelx};
static constexpr int CRONET_NELY = {nely};
static constexpr int CRONET_HISTORY = {history};
static constexpr int CRONET_NEURONS = {neurons};
static constexpr int CRONET_OUT_NEURONS = {output_neurons};

static constexpr int CRONET_BRANCH_IN_COUNT = CRONET_HISTORY * CRONET_NELY * CRONET_NELX;
static constexpr int CRONET_TRUNK_IN_COUNT = (CRONET_NELY + 1) * (CRONET_NELX + 1) * 2;
static constexpr int CRONET_FINAL_OUT_COUNT = CRONET_TRUNK_IN_COUNT;

static constexpr int BRANCH_CONV1_OUT_COUNT = CRONET_HISTORY * 16 * CRONET_NELY * CRONET_NELX;
static constexpr int BRANCH_CONV2_OUT_COUNT = CRONET_HISTORY * 32 * CRONET_NELY * CRONET_NELX;
static constexpr int BRANCH_MAXPOOL_OUT_COUNT = CRONET_HISTORY * 32 * (CRONET_NELY / 2) * (CRONET_NELX / 2);
static constexpr int BRANCH_AAP2D_OUT_COUNT = CRONET_HISTORY * 32;
static constexpr int BRANCH_RNN_ALL_COUNT = CRONET_HISTORY * 64;
static constexpr int BRANCH_RNN_LAST_COUNT = 64;
static constexpr int BRANCH_FC1_OUT_COUNT = CRONET_NEURONS;
static constexpr int BRANCH_FC2_OUT_COUNT = CRONET_OUT_NEURONS;

static constexpr int TRUNK_CONV1_OUT_COUNT = 16 * (CRONET_NELY + 1) * (CRONET_NELX + 1) * 3;
static constexpr int TRUNK_CONV2_OUT_COUNT = 32 * (CRONET_NELY + 1) * (CRONET_NELX + 1) * 4;
static constexpr int TRUNK_AAP3D_OUT_COUNT = 32 * 15 * 10 * 1;
static constexpr int TRUNK_FLATTEN_COUNT = TRUNK_AAP3D_OUT_COUNT;
static constexpr int TRUNK_FC1_OUT_COUNT = CRONET_NEURONS;
static constexpr int TRUNK_FC2_OUT_COUNT = CRONET_OUT_NEURONS;
"""
    (outdir / "include").mkdir(parents=True, exist_ok=True)
    (outdir / "include" / "cronet_hls_shapes.hpp").write_text(text, encoding="utf-8")


def write_tb_io_header(outdir: Path) -> None:
    text = r'''#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

template <typename T>
bool load_bin(const char* path, T* dst, std::size_t count) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "Failed to open %s\n", path);
        return false;
    }
    const std::size_t got = std::fread(dst, sizeof(T), count, f);
    std::fclose(f);
    if (got != count) {
        std::fprintf(stderr, "Read %zu elements from %s, expected %zu\n", got, path, count);
        return false;
    }
    return true;
}

template <typename T>
bool save_bin(const char* path, const T* src, std::size_t count) {
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "Failed to open %s\n", path);
        return false;
    }
    const std::size_t wrote = std::fwrite(src, sizeof(T), count, f);
    std::fclose(f);
    return wrote == count;
}

inline float bf16_bits_to_float(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

inline uint16_t float_to_bf16_bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(uint32_t));
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

inline bool compare_float_arrays(
    const float* got,
    const float* ref,
    std::size_t count,
    float abs_tol,
    float rel_tol,
    std::size_t max_print = 16
) {
    bool ok = true;
    std::size_t printed = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        const float denom = std::fmax(std::fabs(ref[i]), 1e-12f);
        const float rel = diff / denom;
        if (diff > max_abs) max_abs = diff;
        if (rel > max_rel) max_rel = rel;
        if (diff > abs_tol && rel > rel_tol) {
            ok = false;
            if (printed < max_print) {
                std::printf("mismatch[%zu]: got %.9e ref %.9e abs %.9e rel %.9e\n", i, got[i], ref[i], diff, rel);
                ++printed;
            }
        }
    }
    std::printf("compare: %s max_abs %.9e max_rel %.9e\n", ok ? "PASS" : "FAIL", max_abs, max_rel);
    return ok;
}
'''
    (outdir / "include").mkdir(parents=True, exist_ok=True)
    (outdir / "include" / "tb_bin_io.hpp").write_text(text, encoding="utf-8")


def write_reference_md(outdir: Path, manifest: Mapping[str, object]) -> None:
    nelx = manifest["nelx"]
    nely = manifest["nely"]
    history = manifest["history"]
    neurons = manifest["neurons"]
    outn = manifest["output_neurons"]
    lines: List[str] = []
    lines.append("# CRONet HLS Test Vector Export Reference\n")
    lines.append("This file documents the generated dummy weights, inputs, and golden outputs used to validate the HLS C++ layers.\n")
    lines.append("## Model configuration\n")
    lines.append(f"- `nelx = {nelx}`\n")
    lines.append(f"- `nely = {nely}`\n")
    lines.append(f"- `history = {history}`\n")
    lines.append(f"- `neurons = {neurons}`\n")
    lines.append(f"- `output_neurons = (nely + 1) * (nelx + 1) * 2 = {outn}`\n")
    lines.append("- Batch size for all exported cases is `1`.\n")
    lines.append("\n## File formats\n")
    lines.append("- `fp32/*.bin`: raw little-endian IEEE float32 values. Load as `float`.\n")
    lines.append("- `bf16_words/*.bin`: raw little-endian unsigned 16-bit BF16 words. Load as `uint16_t` or `ap_uint<16>`.\n")
    lines.append("- `bf16_io_f32/*.bin`: float32 reference outputs computed after inputs and weights were rounded to BF16, with PyTorch/FP32 arithmetic. This is useful if your HLS uses BF16 inputs/weights but accumulates to float.\n")
    lines.append("- Optional `.txt` files are flat debug dumps in the same order as the `.bin` files.\n")
    lines.append("- `manifest.json` contains paths, shapes, dtypes, counts, byte sizes, first values, and SHA256 hashes.\n")
    lines.append("\n## Flattening and layout rules\n")
    lines.append("All tensors are flattened in C row-major order, equivalent to `numpy.reshape(-1)` on a C-contiguous array.\n")
    lines.append("\nFor nested C arrays, the rightmost index changes fastest. Example: `branch_input[1][10][20][60]` is flattened as `b, history, y, x`.\n")
    lines.append("\n## Input tensors\n")
    lines.append("| Tensor | Shape | Meaning |\n")
    lines.append("|---|---:|---|\n")
    lines.append(f"| `branch_input` | `[1,{history},{nely},{nelx}]` | 10 history frames of material density X. |\n")
    lines.append(f"| `trunk_input` | `[1,{nely + 1},{nelx + 1},2]` | Load/displacement-coordinate input F. In Conv3D this becomes `[1,1,{nely + 1},{nelx + 1},2]`. |\n")
    lines.append("\n## Weight tensor layouts\n")
    lines.append("| Parameter | Layout |\n")
    lines.append("|---|---|\n")
    for name, layout in PARAMETER_LAYOUTS.items():
        lines.append(f"| `{name}` | `{layout}` |\n")
    lines.append("\n## Golden output tensors\n")
    lines.append("| Tensor | Shape for default 60x20 | Use in HLS bring-up |\n")
    lines.append("|---|---:|---|\n")
    default_shapes = {
        "branch_input_reshaped": f"[{history},1,{nely},{nelx}]",
        "branch_conv1_pre_silu": f"[{history},16,{nely},{nelx}]",
        "branch_conv1_silu": f"[{history},16,{nely},{nelx}]",
        "branch_conv2_pre_silu": f"[{history},32,{nely},{nelx}]",
        "branch_conv2_silu": f"[{history},32,{nely},{nelx}]",
        "branch_maxpool2d": f"[{history},32,{nely // 2},{nelx // 2}]",
        "branch_aap2d": f"[{history},32,1,1]",
        "branch_rnn_input": f"[1,{history},32]",
        "branch_rnn_all": f"[1,{history},64]",
        "branch_rnn_last": "[1,64]",
        "branch_fc1_pre_silu": f"[1,{neurons}]",
        "branch_fc1_silu": f"[1,{neurons}]",
        "branch_fc2": f"[1,{outn}]",
        "branch_reshaped": f"[1,{nely + 1},{nelx + 1},2]",
        "trunk_unsqueezed": f"[1,1,{nely + 1},{nelx + 1},2]",
        "trunk_conv1_pre_silu": f"[1,16,{nely + 1},{nelx + 1},3]",
        "trunk_conv1_silu": f"[1,16,{nely + 1},{nelx + 1},3]",
        "trunk_conv2_pre_silu": f"[1,32,{nely + 1},{nelx + 1},4]",
        "trunk_conv2_silu": f"[1,32,{nely + 1},{nelx + 1},4]",
        "trunk_aap3d": "[1,32,15,10,1]",
        "trunk_flatten": "[1,4800]",
        "trunk_fc1_pre_silu": f"[1,{neurons}]",
        "trunk_fc1_silu": f"[1,{neurons}]",
        "trunk_fc2": f"[1,{outn}]",
        "trunk_reshaped": f"[1,{nely + 1},{nelx + 1},2]",
        "final_mul_pre_bias": f"[1,{nely + 1},{nelx + 1},2]",
        "final_output": f"[1,{nely + 1},{nelx + 1},2]",
    }
    bringup_notes = {
        "branch_conv1_pre_silu": "Compare raw Conv2D1 before activation.",
        "branch_conv1_silu": "Compare Conv2D1 with fused SiLU.",
        "branch_conv2_pre_silu": "Compare raw Conv2D2 before activation.",
        "branch_conv2_silu": "Compare Conv2D2 with fused SiLU.",
        "branch_maxpool2d": "Validate 2x2 stride-2 max pooling.",
        "branch_aap2d": "Validate average over each 10x30 pooled feature map.",
        "branch_rnn_all": "Validate all ten RNN steps.",
        "branch_rnn_last": "Validate only the hidden state used by FC1.",
        "branch_fc1_silu": "Validate Linear 64->40 with SiLU.",
        "branch_fc2": "Validate Linear 40->2562.",
        "trunk_conv1_silu": "Validate Conv3D1 with fused SiLU.",
        "trunk_conv2_silu": "Validate Conv3D2 with fused SiLU.",
        "trunk_aap3d": "Validate adaptive average pool to 15x10x1.",
        "trunk_flatten": "Validate flatten address order before trunk FC1.",
        "trunk_fc1_silu": "Validate Linear 4800->40 with SiLU.",
        "trunk_fc2": "Validate Linear 40->2562.",
        "final_output": "Validate full CRONet output after branch*trunk+bias.",
    }
    for name in EXPORT_TENSOR_ORDER:
        lines.append(f"| `{name}` | `{default_shapes.get(name, 'see manifest')}` | {bringup_notes.get(name, '')} |\n")
    lines.append("\nThe script also exports `branch_rnn_step_00` through `branch_rnn_step_09`, each with shape `[1,64]`, for debugging a fully unrolled RNN implementation.\n")
    lines.append("\n## Operation formulas used by the reference\n")
    lines.append("- SiLU: `y = x * sigmoid(x)`.\n")
    lines.append("- RNN step: `h_t = tanh(x_t @ W_ih.T + b_ih + h_{t-1} @ W_hh.T + b_hh)`, with zero initial hidden state.\n")
    lines.append("- Final combine: reshape BranchNet and TrunkNet FC2 outputs to `[1,nely+1,nelx+1,2]`, then `out = branch * trunk + out_bias`.\n")
    lines.append("- PyTorch adaptive average pooling window for output index `i`: `start = floor(i * input_size / output_size)`, `end = ceil((i + 1) * input_size / output_size)`. Average over `[start,end)`.\n")
    lines.append("\n## Suggested HLS validation order\n")
    lines.append("1. Load FP32 inputs and weights, validate one layer at a time.\n")
    lines.append("2. Validate Conv2D1/Conv2D2 and Conv3D1/Conv3D2 before and after SiLU.\n")
    lines.append("3. Validate MaxPool2D and AAP2D/AAP3D separately.\n")
    lines.append("4. Validate the 10 RNN steps using the `branch_rnn_step_XX` dumps.\n")
    lines.append("5. Validate GEMV layers.\n")
    lines.append("6. Validate final multiply and bias.\n")
    lines.append("7. Switch to BF16-word files once FP32 indexing and arithmetic are correct.\n")
    lines.append("\n## HLS loader helpers\n")
    lines.append("Use `include/tb_bin_io.hpp` for simple `load_bin` and compare helpers, and `include/cronet_hls_shapes.hpp` for element counts.\n")
    (outdir / "EXPORT_REFERENCE.md").write_text("".join(lines), encoding="utf-8")


# ----------------------------- export core -----------------------------

def export_weights(outdir: Path, model: CRONetHLSReference, emit_txt: bool, emit_npy: bool) -> Dict[str, object]:
    weights_info: Dict[str, object] = {}
    for name, param in model.state_dict().items():
        arr = np.ascontiguousarray(param.detach().cpu().numpy().astype(np.float32))
        fname = safe_name(name) + ".bin"
        weights_info[name] = {
            "layout": PARAMETER_LAYOUTS.get(name, "see shape"),
            "fp32": save_raw_array(outdir / "weights" / "fp32" / fname, arr, "float32", emit_txt, emit_npy),
            "bf16_words": save_raw_array(outdir / "weights" / "bf16_words" / fname, float32_to_bf16_words(arr), "uint16", emit_txt, emit_npy),
        }
    return weights_info


def run_model_to_numpy(model: CRONetHLSReference, branch_np: np.ndarray, trunk_np: np.ndarray):
    model.eval()
    with torch.no_grad():
        branch = torch.from_numpy(np.ascontiguousarray(branch_np, dtype=np.float32))
        trunk = torch.from_numpy(np.ascontiguousarray(trunk_np, dtype=np.float32))
        y, z = model(branch, trunk, return_intermediates=True)
    outputs = {name: tensor.detach().cpu().numpy().astype(np.float32) for name, tensor in z.items()}
    return y.detach().cpu().numpy().astype(np.float32), outputs


def export_case(
    outdir: Path,
    case_name: str,
    model_fp32: CRONetHLSReference,
    model_bf16_params: CRONetHLSReference,
    nelx: int,
    nely: int,
    history: int,
    emit_txt: bool,
    emit_npy: bool,
) -> Dict[str, object]:
    case_root = outdir / "cases" / case_name
    branch_np, trunk_np = make_case(case_name, nelx, nely, history)

    # FP32 reference.
    _, fp32_outputs = run_model_to_numpy(model_fp32, branch_np, trunk_np)

    # BF16-IO reference: inputs and weights rounded to BF16, ops executed in FP32.
    branch_bf16_f32 = bf16_round_to_float32(branch_np)
    trunk_bf16_f32 = bf16_round_to_float32(trunk_np)
    _, bf16_io_outputs = run_model_to_numpy(model_bf16_params, branch_bf16_f32, trunk_bf16_f32)

    info: Dict[str, object] = {"inputs": {}, "golden": {"fp32": {}, "bf16_io_f32": {}, "bf16_words": {}}}

    info["inputs"]["branch_input"] = {
        "fp32": save_raw_array(case_root / "inputs" / "fp32" / "branch_input.bin", branch_np, "float32", emit_txt, emit_npy),
        "bf16_words": save_raw_array(case_root / "inputs" / "bf16_words" / "branch_input.bin", float32_to_bf16_words(branch_np), "uint16", emit_txt, emit_npy),
    }
    info["inputs"]["trunk_input"] = {
        "fp32": save_raw_array(case_root / "inputs" / "fp32" / "trunk_input.bin", trunk_np, "float32", emit_txt, emit_npy),
        "bf16_words": save_raw_array(case_root / "inputs" / "bf16_words" / "trunk_input.bin", float32_to_bf16_words(trunk_np), "uint16", emit_txt, emit_npy),
    }

    names = list(EXPORT_TENSOR_ORDER)
    for step in range(history):
        names.append(f"branch_rnn_step_{step:02d}")

    for name in names:
        if name not in fp32_outputs:
            continue
        arr_fp32 = np.ascontiguousarray(fp32_outputs[name], dtype=np.float32)
        arr_bf16io = np.ascontiguousarray(bf16_io_outputs[name], dtype=np.float32)
        info["golden"]["fp32"][name] = save_raw_array(case_root / "golden" / "fp32" / f"{name}.bin", arr_fp32, "float32", emit_txt, emit_npy)
        info["golden"]["bf16_io_f32"][name] = save_raw_array(case_root / "golden" / "bf16_io_f32" / f"{name}.bin", arr_bf16io, "float32", emit_txt, emit_npy)
        info["golden"]["bf16_words"][name] = save_raw_array(case_root / "golden" / "bf16_words" / f"{name}.bin", float32_to_bf16_words(arr_bf16io), "uint16", emit_txt, emit_npy)

    return info


def export_all(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    model_fp32 = CRONetHLSReference(args.nelx, args.nely, args.history, args.neurons)
    initialize_dummy_weights(model_fp32)
    model_fp32.eval()

    model_bf16_params = clone_with_bf16_rounded_params(model_fp32)
    model_bf16_params.eval()

    case_names = [c.strip() for c in args.cases.split(",") if c.strip()]

    manifest: Dict[str, object] = {
        "description": "CRONet deterministic dummy-weight HLS vectors",
        "nelx": args.nelx,
        "nely": args.nely,
        "history": args.history,
        "neurons": args.neurons,
        "output_neurons": (args.nely + 1) * (args.nelx + 1) * 2,
        "batch_size": 1,
        "flatten_order": "C row-major / numpy C-contiguous",
        "endianness": "little-endian",
        "precision_sets": {
            "fp32": "raw IEEE float32 .bin; use for first HLS correctness bring-up",
            "bf16_words": "raw uint16 BF16 words .bin; use for BF16 HLS inputs/weights/outputs",
            "bf16_io_f32": "float32 golden outputs after BF16-rounded inputs and weights; operations are PyTorch FP32",
        },
        "conv3d_layout_note": "trunk_input [B,nely+1,nelx+1,2] is unsqueezed to [B,1,D=nely+1,H=nelx+1,W=2] for PyTorch Conv3D",
        "parameter_layouts": PARAMETER_LAYOUTS,
        "weights": export_weights(outdir, model_fp32, args.emit_txt, args.emit_npy),
        "cases": {},
    }

    for case_name in case_names:
        print(f"Exporting case: {case_name}")
        manifest["cases"][case_name] = export_case(
            outdir=outdir,
            case_name=case_name,
            model_fp32=model_fp32,
            model_bf16_params=model_bf16_params,
            nelx=args.nelx,
            nely=args.nely,
            history=args.history,
            emit_txt=args.emit_txt,
            emit_npy=args.emit_npy,
        )

    write_shapes_header(outdir, args.nelx, args.nely, args.history, args.neurons)
    write_tb_io_header(outdir)
    write_reference_md(outdir, manifest)

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(relpaths(manifest, outdir), indent=2), encoding="utf-8")
    print(f"Wrote: {outdir}")
    print(f"Manifest: {manifest_path}")
    print(f"Reference: {outdir / 'EXPORT_REFERENCE.md'}")


def check_all(args: argparse.Namespace) -> None:
    # Basic check: re-run all cases and compare final_output fp32 and bf16_io_f32.
    outdir: Path = args.outdir
    manifest_path = outdir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    model_fp32 = CRONetHLSReference(manifest["nelx"], manifest["nely"], manifest["history"], manifest["neurons"])
    initialize_dummy_weights(model_fp32)
    model_fp32.eval()
    model_bf16 = clone_with_bf16_rounded_params(model_fp32)
    model_bf16.eval()

    ok_all = True
    for case_name in manifest["cases"].keys():
        branch_np, trunk_np = make_case(case_name, manifest["nelx"], manifest["nely"], manifest["history"])
        _, out_fp32 = run_model_to_numpy(model_fp32, branch_np, trunk_np)
        _, out_bf16 = run_model_to_numpy(model_bf16, bf16_round_to_float32(branch_np), bf16_round_to_float32(trunk_np))

        ref_fp32 = np.fromfile(outdir / "cases" / case_name / "golden" / "fp32" / "final_output.bin", dtype="<f4")
        ref_bf16 = np.fromfile(outdir / "cases" / case_name / "golden" / "bf16_io_f32" / "final_output.bin", dtype="<f4")
        got_fp32 = out_fp32["final_output"].reshape(-1)
        got_bf16 = out_bf16["final_output"].reshape(-1)
        ok_fp32 = np.allclose(got_fp32, ref_fp32, atol=args.atol, rtol=args.rtol)
        ok_bf16 = np.allclose(got_bf16, ref_bf16, atol=args.atol, rtol=args.rtol)
        print(f"{case_name}: fp32={ok_fp32} bf16_io_f32={ok_bf16}")
        ok_all = ok_all and ok_fp32 and ok_bf16

    if not ok_all:
        raise SystemExit(1)
    print("All checks passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CRONet HLS dummy weights and test vectors.")
    parser.add_argument("--outdir", type=Path, default=Path("cronet_hls_vectors_export"))
    parser.add_argument("--nelx", type=int, default=DEFAULT_NELX)
    parser.add_argument("--nely", type=int, default=DEFAULT_NELY)
    parser.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    parser.add_argument("--neurons", type=int, default=DEFAULT_NEURONS)
    parser.add_argument("--cases", type=str, default="sanity,ramp,sparse,random", help="Comma list: sanity,ramp,sparse,random")
    parser.add_argument("--emit-txt", action="store_true", help="Also write flat .txt debug dumps. Large for full model.")
    parser.add_argument("--emit-npy", action="store_true", help="Also write .npy files for Python debugging.")
    parser.add_argument("--check", action="store_true", help="Recompute final outputs and compare against existing export.")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        check_all(args)
    else:
        export_all(args)


if __name__ == "__main__":
    main()
