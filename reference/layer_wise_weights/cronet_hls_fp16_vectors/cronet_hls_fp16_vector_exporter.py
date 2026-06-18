#!/usr/bin/env python3
"""
CRONet HLS FP16 vector exporter.

Generates deterministic dummy weights, deterministic test inputs, and layer-by-layer
reference outputs for FP16-oriented HLS C++ testbenches.

This script is intentionally aligned with cronet_hls_vector_exporter.py, but its
main low-precision export is IEEE FP16 / binary16 rather than BF16.

Outputs:
  weights/fp32/*.bin              raw little-endian float32 weights
  weights/fp16/*.bin              raw little-endian IEEE half weights
  weights/fp16_words/*.bin        same FP16 payload as uint16_t/ap_uint<16> words

  cases/<case>/inputs/fp32/*.bin
  cases/<case>/inputs/fp16/*.bin
  cases/<case>/inputs/fp16_words/*.bin

  cases/<case>/golden/fp32/*.bin
      FP32 reference.

  cases/<case>/golden/fp16_io_f32/*.bin
      Float32 golden outputs computed after inputs and weights are rounded to FP16,
      with the layer math executed in FP32. Use this if your HLS design stores
      weights/inputs in half but accumulates MACs in float.

  cases/<case>/golden/fp16_torch/*.bin
      Raw IEEE half golden outputs from a PyTorch half-precision forward pass.
      Use this if your HLS design computes/stores outputs as half.

  cases/<case>/golden/fp16_torch_f32/*.bin
      The same PyTorch half outputs converted to float32 for easier debugging.

Run:
  python3 cronet_hls_fp16_vector_exporter.py --outdir cronet_hls_fp16_vectors

Check:
  python3 cronet_hls_fp16_vector_exporter.py --outdir cronet_hls_fp16_vectors --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch

# Import the already-generated reference model and deterministic case/weight setup.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from cronet_hls_vector_exporter import (  # noqa: E402
    DEFAULT_HISTORY,
    DEFAULT_NELX,
    DEFAULT_NELY,
    DEFAULT_NEURONS,
    CRONetHLSReference,
    EXPORT_TENSOR_ORDER,
    PARAMETER_LAYOUTS,
    initialize_dummy_weights,
    make_case,
    relpaths,
    safe_name,
    write_shapes_header,
)


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


# ----------------------------- FP16 utilities -----------------------------

def float32_to_fp16_words(x: np.ndarray) -> np.ndarray:
    """Convert float32 values to IEEE FP16/binary16 uint16 words."""
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32).astype(np.float16)).view(np.uint16)


def fp16_words_to_float32(words: np.ndarray) -> np.ndarray:
    """Convert raw IEEE FP16/binary16 uint16 words to float32 values."""
    return np.ascontiguousarray(np.asarray(words, dtype=np.uint16)).view(np.float16).astype(np.float32)


def fp16_round_to_float32(x: np.ndarray) -> np.ndarray:
    """Round float32 values to IEEE FP16 precision, returned as float32."""
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def clone_with_fp16_rounded_params(model: CRONetHLSReference) -> CRONetHLSReference:
    new_model = CRONetHLSReference(model.nelx, model.nely, model.history, model.neurons)
    new_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        for param in new_model.parameters():
            arr = param.detach().cpu().numpy().astype(np.float32)
            param.copy_(torch.from_numpy(fp16_round_to_float32(arr).reshape(tuple(param.shape))))
    return new_model


def clone_as_torch_fp16(model: CRONetHLSReference) -> CRONetHLSReference:
    new_model = CRONetHLSReference(model.nelx, model.nely, model.history, model.neurons)
    new_model.load_state_dict(model.state_dict())
    new_model.half()
    return new_model


# ----------------------------- file helpers -----------------------------

def save_raw_array(path: Path, arr: np.ndarray, dtype: str, emit_txt: bool = False, emit_npy: bool = False) -> Dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)

    if dtype == "float32":
        a = np.ascontiguousarray(arr, dtype="<f4")
    elif dtype == "float16":
        a = np.ascontiguousarray(arr, dtype="<f2")
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

    if dtype in ("float32", "float16"):
        flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        info.update({
            "min": float(np.min(flat)) if flat.size else 0.0,
            "max": float(np.max(flat)) if flat.size else 0.0,
            "mean": float(np.mean(flat)) if flat.size else 0.0,
            "first_8": [float(v) for v in flat[:8]],
        })
    else:
        flat_u = np.asarray(arr, dtype=np.uint16).reshape(-1)
        info.update({"first_8_hex": [f"0x{int(v):04x}" for v in flat_u[:8]]})

    if emit_txt:
        txt_path = path.with_suffix(".txt")
        if dtype == "uint16":
            np.savetxt(txt_path, np.asarray(arr, dtype=np.uint16).reshape(-1), fmt="%u")
        else:
            np.savetxt(txt_path, np.asarray(arr, dtype=np.float32).reshape(-1), fmt="%.9e")
        info["txt_path"] = str(txt_path)

    if emit_npy:
        npy_path = path.with_suffix(".npy")
        np.save(npy_path, arr)
        info["npy_path"] = str(npy_path)

    return info


def run_model_to_numpy(model: CRONetHLSReference, branch_np: np.ndarray, trunk_np: np.ndarray, torch_dtype: torch.dtype):
    model.eval()
    with torch.no_grad():
        branch = torch.from_numpy(np.ascontiguousarray(branch_np, dtype=np.float32)).to(dtype=torch_dtype)
        trunk = torch.from_numpy(np.ascontiguousarray(trunk_np, dtype=np.float32)).to(dtype=torch_dtype)
        y, z = model(branch, trunk, return_intermediates=True)
    outputs = {name: tensor.detach().cpu().numpy() for name, tensor in z.items()}
    return y.detach().cpu().numpy(), outputs


# ----------------------------- generated HLS helper -----------------------------

def write_tb_io_header_fp16(outdir: Path) -> None:
    text = r'''#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifdef __SYNTHESIS__
#include <hls_half.h>
#else
// Vitis HLS C-sim normally provides hls_half.h. Keep this include outside
// synthesis too when compiling with Vitis. If your host compiler cannot find it,
// load FP16 files as uint16_t and convert/check in Python instead.
#include <hls_half.h>
#endif

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

inline float fp16_bits_to_float(uint16_t bits) {
    half h;
    std::memcpy(&h, &bits, sizeof(uint16_t));
    return static_cast<float>(h);
}

inline uint16_t float_to_fp16_bits(float x) {
    half h = static_cast<half>(x);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(uint16_t));
    return bits;
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

inline bool compare_half_arrays(
    const half* got,
    const half* ref,
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
        const float g = static_cast<float>(got[i]);
        const float r = static_cast<float>(ref[i]);
        const float diff = std::fabs(g - r);
        const float denom = std::fmax(std::fabs(r), 1e-12f);
        const float rel = diff / denom;
        if (diff > max_abs) max_abs = diff;
        if (rel > max_rel) max_rel = rel;
        if (diff > abs_tol && rel > rel_tol) {
            ok = false;
            if (printed < max_print) {
                std::printf("mismatch[%zu]: got %.9e ref %.9e abs %.9e rel %.9e\n", i, g, r, diff, rel);
                ++printed;
            }
        }
    }
    std::printf("compare_half: %s max_abs %.9e max_rel %.9e\n", ok ? "PASS" : "FAIL", max_abs, max_rel);
    return ok;
}
'''
    (outdir / "include").mkdir(parents=True, exist_ok=True)
    (outdir / "include" / "tb_bin_io_fp16.hpp").write_text(text, encoding="utf-8")


# ----------------------------- reference MD -----------------------------

def write_reference_md_fp16(outdir: Path, manifest: Mapping[str, object]) -> None:
    nelx = manifest["nelx"]
    nely = manifest["nely"]
    history = manifest["history"]
    neurons = manifest["neurons"]
    outn = manifest["output_neurons"]

    lines: List[str] = []
    lines.append("# CRONet HLS FP16 Test Vector Export Reference\n\n")
    lines.append("This package contains deterministic dummy weights, inputs, and golden outputs for validating the CRONet HLS C++ implementation using IEEE FP16 / binary16.\n\n")
    lines.append("## Model configuration\n\n")
    lines.append(f"- `nelx = {nelx}`\n")
    lines.append(f"- `nely = {nely}`\n")
    lines.append(f"- `history = {history}`\n")
    lines.append(f"- `neurons = {neurons}`\n")
    lines.append(f"- `output_neurons = (nely + 1) * (nelx + 1) * 2 = {outn}`\n")
    lines.append("- Batch size for all exported cases is `1`.\n\n")

    lines.append("## Precision sets\n\n")
    lines.append("| Directory | File dtype | Use |\n")
    lines.append("|---|---|---|\n")
    lines.append("| `fp32` | little-endian IEEE float32 | First bring-up and debugging. |\n")
    lines.append("| `fp16` | little-endian IEEE binary16 | Main HLS `half` weights, inputs, and half golden outputs. |\n")
    lines.append("| `fp16_words` | little-endian `uint16_t` words containing IEEE binary16 payloads | Use if your HLS testbench loads into `ap_uint<16>` or reinterprets manually. |\n")
    lines.append("| `fp16_io_f32` | little-endian IEEE float32 | Golden outputs computed with FP16-rounded inputs/weights but FP32 math. Use if your HLS design uses half storage but float accumulation. |\n")
    lines.append("| `fp16_torch` | little-endian IEEE binary16 | Golden outputs from PyTorch half forward pass. Use if your HLS design computes/stores layer outputs as `half`. |\n")
    lines.append("| `fp16_torch_f32` | little-endian IEEE float32 | Same as `fp16_torch`, converted to float32 for easier inspection. |\n\n")

    lines.append("## Flattening and layout\n\n")
    lines.append("All files are flattened in C row-major order. The rightmost tensor index changes fastest.\n\n")
    lines.append("Input shapes:\n\n")
    lines.append(f"- `branch_input`: `[1,{history},{nely},{nelx}]`, representing `[batch, history, y, x]`.\n")
    lines.append(f"- `trunk_input`: `[1,{nely + 1},{nelx + 1},2]`. For Conv3D, this is unsqueezed to `[1,1,{nely + 1},{nelx + 1},2]`, interpreted by PyTorch as `[N,C,D,H,W]`.\n\n")

    lines.append("## Weight tensor layouts\n\n")
    lines.append("| Parameter | Layout |\n")
    lines.append("|---|---|\n")
    for name, layout in PARAMETER_LAYOUTS.items():
        lines.append(f"| `{name}` | `{layout}` |\n")

    lines.append("\n## Golden tensors exported per case\n\n")
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
    lines.append("| Tensor | Shape | HLS debug purpose |\n")
    lines.append("|---|---:|---|\n")
    notes = {
        "branch_conv1_pre_silu": "Raw Branch Conv2D1.",
        "branch_conv1_silu": "Branch Conv2D1 with fused SiLU.",
        "branch_conv2_pre_silu": "Raw Branch Conv2D2.",
        "branch_conv2_silu": "Branch Conv2D2 with fused SiLU.",
        "branch_maxpool2d": "2x2 stride-2 MaxPool.",
        "branch_aap2d": "Adaptive average to 1x1 per channel/history frame.",
        "branch_rnn_all": "All ten RNN hidden states.",
        "branch_rnn_last": "Final RNN hidden state used by Branch FC1.",
        "branch_fc1_silu": "Branch Linear 64->40 with SiLU.",
        "branch_fc2": "Branch Linear 40->2562.",
        "trunk_conv1_silu": "Trunk Conv3D1 with fused SiLU.",
        "trunk_conv2_silu": "Trunk Conv3D2 with fused SiLU.",
        "trunk_aap3d": "Adaptive average to 15x10x1.",
        "trunk_flatten": "Flatten order before Trunk FC1.",
        "trunk_fc1_silu": "Trunk Linear 4800->40 with SiLU.",
        "trunk_fc2": "Trunk Linear 40->2562.",
        "final_output": "Full CRONet output after branch * trunk + scalar bias.",
    }
    for name in EXPORT_TENSOR_ORDER:
        lines.append(f"| `{name}` | `{default_shapes.get(name, 'see manifest')}` | {notes.get(name, '')} |\n")
    lines.append("\nThe package also exports `branch_rnn_step_00` through `branch_rnn_step_09`, each with shape `[1,64]`.\n\n")

    lines.append("## Recommended HLS comparison flow\n\n")
    lines.append("1. Bring up each layer using `fp32` weights/inputs/goldens.\n")
    lines.append("2. Switch weights/inputs to `fp16`. If your HLS kernel accumulates in `float`, compare to `golden/fp16_io_f32`.\n")
    lines.append("3. If your HLS kernel computes and stores in `half`, compare to `golden/fp16_torch` using a relaxed tolerance such as `abs_tol=2e-2`, `rel_tol=2e-2` to start. Tighten after indexing is correct.\n")
    lines.append("4. Validate block-by-block before checking `final_output`. Conv2D/Conv3D indexing and flatten order are the most common failure points.\n\n")

    lines.append("## HLS loader helper\n\n")
    lines.append("Use `include/tb_bin_io_fp16.hpp`. For raw half files, load directly into `half[]` under Vitis HLS, or load `fp16_words` into `uint16_t[]`/`ap_uint<16>[]` and reinterpret as needed.\n")
    (outdir / "FP16_EXPORT_REFERENCE.md").write_text("".join(lines), encoding="utf-8")


# ----------------------------- export core -----------------------------

def export_weights(outdir: Path, model: CRONetHLSReference, emit_txt: bool, emit_npy: bool) -> Dict[str, object]:
    weights_info: Dict[str, object] = {}
    for name, param in model.state_dict().items():
        arr = np.ascontiguousarray(param.detach().cpu().numpy().astype(np.float32))
        fname = safe_name(name) + ".bin"
        arr_fp16 = np.ascontiguousarray(arr.astype(np.float16))
        weights_info[name] = {
            "layout": PARAMETER_LAYOUTS.get(name, "see shape"),
            "fp32": save_raw_array(outdir / "weights" / "fp32" / fname, arr, "float32", emit_txt, emit_npy),
            "fp16": save_raw_array(outdir / "weights" / "fp16" / fname, arr_fp16, "float16", emit_txt, emit_npy),
            "fp16_words": save_raw_array(outdir / "weights" / "fp16_words" / fname, arr_fp16.view(np.uint16), "uint16", emit_txt, emit_npy),
        }
    return weights_info


def export_case(
    outdir: Path,
    case_name: str,
    model_fp32: CRONetHLSReference,
    model_fp16_io_f32: CRONetHLSReference,
    model_fp16_torch: CRONetHLSReference,
    nelx: int,
    nely: int,
    history: int,
    emit_txt: bool,
    emit_npy: bool,
) -> Dict[str, object]:
    case_root = outdir / "cases" / case_name
    branch_np, trunk_np = make_case(case_name, nelx, nely, history)

    _, fp32_outputs = run_model_to_numpy(model_fp32, branch_np, trunk_np, torch.float32)

    branch_fp16_f32 = fp16_round_to_float32(branch_np)
    trunk_fp16_f32 = fp16_round_to_float32(trunk_np)
    _, fp16_io_outputs = run_model_to_numpy(model_fp16_io_f32, branch_fp16_f32, trunk_fp16_f32, torch.float32)

    # Native PyTorch half pass. Inputs are rounded to half by dtype conversion.
    _, fp16_torch_outputs = run_model_to_numpy(model_fp16_torch, branch_np, trunk_np, torch.float16)

    info: Dict[str, object] = {
        "inputs": {},
        "golden": {"fp32": {}, "fp16_io_f32": {}, "fp16_torch": {}, "fp16_torch_f32": {}},
    }

    branch_half = np.ascontiguousarray(branch_np.astype(np.float16))
    trunk_half = np.ascontiguousarray(trunk_np.astype(np.float16))
    info["inputs"]["branch_input"] = {
        "fp32": save_raw_array(case_root / "inputs" / "fp32" / "branch_input.bin", branch_np, "float32", emit_txt, emit_npy),
        "fp16": save_raw_array(case_root / "inputs" / "fp16" / "branch_input.bin", branch_half, "float16", emit_txt, emit_npy),
        "fp16_words": save_raw_array(case_root / "inputs" / "fp16_words" / "branch_input.bin", branch_half.view(np.uint16), "uint16", emit_txt, emit_npy),
    }
    info["inputs"]["trunk_input"] = {
        "fp32": save_raw_array(case_root / "inputs" / "fp32" / "trunk_input.bin", trunk_np, "float32", emit_txt, emit_npy),
        "fp16": save_raw_array(case_root / "inputs" / "fp16" / "trunk_input.bin", trunk_half, "float16", emit_txt, emit_npy),
        "fp16_words": save_raw_array(case_root / "inputs" / "fp16_words" / "trunk_input.bin", trunk_half.view(np.uint16), "uint16", emit_txt, emit_npy),
    }

    names = list(EXPORT_TENSOR_ORDER)
    for step in range(history):
        names.append(f"branch_rnn_step_{step:02d}")

    for name in names:
        if name not in fp32_outputs:
            continue
        arr_fp32 = np.ascontiguousarray(fp32_outputs[name].astype(np.float32))
        arr_fp16io = np.ascontiguousarray(fp16_io_outputs[name].astype(np.float32))
        arr_torch_half = np.ascontiguousarray(fp16_torch_outputs[name].astype(np.float16))
        arr_torch_f32 = np.ascontiguousarray(arr_torch_half.astype(np.float32))

        info["golden"]["fp32"][name] = save_raw_array(case_root / "golden" / "fp32" / f"{name}.bin", arr_fp32, "float32", emit_txt, emit_npy)
        info["golden"]["fp16_io_f32"][name] = save_raw_array(case_root / "golden" / "fp16_io_f32" / f"{name}.bin", arr_fp16io, "float32", emit_txt, emit_npy)
        info["golden"]["fp16_torch"][name] = save_raw_array(case_root / "golden" / "fp16_torch" / f"{name}.bin", arr_torch_half, "float16", emit_txt, emit_npy)
        info["golden"]["fp16_torch_f32"][name] = save_raw_array(case_root / "golden" / "fp16_torch_f32" / f"{name}.bin", arr_torch_f32, "float32", emit_txt, emit_npy)

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

    model_fp16_io_f32 = clone_with_fp16_rounded_params(model_fp32)
    model_fp16_io_f32.eval()

    model_fp16_torch = clone_as_torch_fp16(model_fp32)
    model_fp16_torch.eval()

    case_names = [c.strip() for c in args.cases.split(",") if c.strip()]

    manifest: Dict[str, object] = {
        "description": "CRONet deterministic dummy-weight HLS vectors for IEEE FP16",
        "nelx": args.nelx,
        "nely": args.nely,
        "history": args.history,
        "neurons": args.neurons,
        "output_neurons": (args.nely + 1) * (args.nelx + 1) * 2,
        "batch_size": 1,
        "flatten_order": "C row-major / numpy C-contiguous",
        "endianness": "little-endian",
        "precision_sets": {
            "fp32": "raw IEEE float32 .bin",
            "fp16": "raw IEEE binary16 .bin; load as HLS half",
            "fp16_words": "raw uint16 words containing IEEE binary16 payloads",
            "fp16_io_f32": "float32 goldens after FP16-rounded inputs and weights, FP32 arithmetic",
            "fp16_torch": "raw IEEE binary16 goldens from PyTorch half forward pass",
            "fp16_torch_f32": "fp16_torch goldens converted to float32 for debugging",
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
            model_fp16_io_f32=model_fp16_io_f32,
            model_fp16_torch=model_fp16_torch,
            nelx=args.nelx,
            nely=args.nely,
            history=args.history,
            emit_txt=args.emit_txt,
            emit_npy=args.emit_npy,
        )

    write_shapes_header(outdir, args.nelx, args.nely, args.history, args.neurons)
    write_tb_io_header_fp16(outdir)
    write_reference_md_fp16(outdir, manifest)

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(relpaths(manifest, outdir), indent=2), encoding="utf-8")

    # Include the base reference module in the package so this script is reproducible standalone from the zip.
    src_base = THIS_DIR / "cronet_hls_vector_exporter.py"
    if src_base.exists() and src_base.resolve() != (outdir / "cronet_hls_vector_exporter.py").resolve():
        shutil.copy2(src_base, outdir / "cronet_hls_vector_exporter.py")

    print(f"Wrote: {outdir}")
    print(f"Manifest: {manifest_path}")
    print(f"Reference: {outdir / 'FP16_EXPORT_REFERENCE.md'}")


def check_all(args: argparse.Namespace) -> None:
    outdir: Path = args.outdir
    manifest_path = outdir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    model_fp32 = CRONetHLSReference(manifest["nelx"], manifest["nely"], manifest["history"], manifest["neurons"])
    initialize_dummy_weights(model_fp32)
    model_fp32.eval()
    model_fp16_io_f32 = clone_with_fp16_rounded_params(model_fp32)
    model_fp16_io_f32.eval()
    model_fp16_torch = clone_as_torch_fp16(model_fp32)
    model_fp16_torch.eval()

    ok_all = True
    for case_name in manifest["cases"].keys():
        branch_np, trunk_np = make_case(case_name, manifest["nelx"], manifest["nely"], manifest["history"])
        _, out_fp32 = run_model_to_numpy(model_fp32, branch_np, trunk_np, torch.float32)
        _, out_fp16io = run_model_to_numpy(model_fp16_io_f32, fp16_round_to_float32(branch_np), fp16_round_to_float32(trunk_np), torch.float32)
        _, out_half = run_model_to_numpy(model_fp16_torch, branch_np, trunk_np, torch.float16)

        ref_fp32 = np.fromfile(outdir / "cases" / case_name / "golden" / "fp32" / "final_output.bin", dtype="<f4")
        ref_fp16io = np.fromfile(outdir / "cases" / case_name / "golden" / "fp16_io_f32" / "final_output.bin", dtype="<f4")
        ref_half = np.fromfile(outdir / "cases" / case_name / "golden" / "fp16_torch" / "final_output.bin", dtype="<f2")

        got_fp32 = out_fp32["final_output"].reshape(-1).astype(np.float32)
        got_fp16io = out_fp16io["final_output"].reshape(-1).astype(np.float32)
        got_half = out_half["final_output"].reshape(-1).astype(np.float16)

        ok_fp32 = np.allclose(got_fp32, ref_fp32, atol=args.atol, rtol=args.rtol)
        ok_fp16io = np.allclose(got_fp16io, ref_fp16io, atol=args.atol, rtol=args.rtol)
        ok_half = np.array_equal(got_half.view(np.uint16), ref_half.view(np.uint16))
        print(f"{case_name}: fp32={ok_fp32} fp16_io_f32={ok_fp16io} fp16_torch_bits={ok_half}")
        ok_all = ok_all and ok_fp32 and ok_fp16io and ok_half

    if not ok_all:
        raise SystemExit(1)
    print("All checks passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CRONet HLS FP16 dummy weights and test vectors.")
    parser.add_argument("--outdir", type=Path, default=Path("cronet_hls_fp16_vectors"))
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
