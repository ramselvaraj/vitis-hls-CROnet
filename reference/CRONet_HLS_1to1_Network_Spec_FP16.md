# CRONet HLS 1:1 Network Architecture Specification - FP16 Revision

This document is the implementation contract for recreating the CRONet inference network in Vitis HLS / PL logic on the AMD Versal VEK280.

This revision targets the **IEEE FP16 / binary16 HLS export package** described by `CRONET_HLS_FP16_VECTOR_EXPORT_REFERENCE.md`. The goal is a layer-accurate HLS implementation with the same tensor shapes, operator semantics, weight layouts, activation layouts, precision policy, and validation vectors used by that export package.

Important comparison note: the AIE paper reports a BF16 Versal implementation. This HLS revision uses FP16 because the current HLS vectors and reference package are FP16. Therefore, report the precision explicitly in benchmark tables. If the AIE implementation remains BF16, label the comparison as **AIE BF16 vs PL/HLS FP16**. For a strict same-datatype comparison, both sides must be rebuilt or re-exported using the same precision.

---

## 1. Scope and comparison requirements

The HLS design shall implement only CRONet inference, not training and not the finite-element topology-optimization loop.

For a fair PL/HLS implementation and a clean comparison against the AIE version, the HLS implementation should follow these rules:

| Requirement | HLS requirement |
|---|---|
| Target board | AMD Versal VEK280 |
| Batch size | `1` |
| Model size | `nelx = 60`, `nely = 20` |
| Branch input | 10 previous material-density frames |
| Trunk input | Load field over the nodal grid |
| HLS precision for this revision | IEEE FP16 / binary16 for weights, inputs, named FP16 layer outputs, and final output |
| Accumulator behavior | Must be documented as an HLS implementation choice |
| Weights | Stored on chip as ROM/BRAM/URAM initialized at bitstream build time |
| Intermediate activations | Kept on chip in BRAM/URAM/registers/streams |
| DDR usage | No DDR for weights or intermediate activations |
| Top-level interface | Prefer AXI4-Stream or local BRAM test harness; avoid `m_axi` ports for internal tensors |
| Validation | Layer-by-layer comparison against exported golden tensors before full-system comparison |

Allowed external data movement for the final benchmark:

1. Load the input tensors into the FPGA-side design.
2. Run the full CRONet inference pipeline.
3. Read the final displacement output.

Weights and intermediate feature maps should not be fetched from or spilled to DDR during inference.

---

## 2. Global model constants

```text
nelx            = 60
nely            = 20
history         = 10
neurons         = 40
node_h          = nely + 1 = 21
node_w          = nelx + 1 = 61
output_neurons  = node_h * node_w * 2 = 2562
batch_size      = 1
```

CRONet predicts a displacement vector at each nodal location. For the 2D problem there are two displacement components per node.

```text
final output shape = [1, 21, 61, 2]
                   = [batch, node_y, node_x, dof]
                   = 2562 scalar values
```

---

## 3. Top-level network view

CRONet has two independent subnetworks:

1. **BranchNet** processes the recent material-density history.
2. **TrunkNet** processes the load/displacement-coordinate tensor.

The two subnetworks produce equal-sized vectors. These are reshaped to `[1,21,61,2]`, multiplied elementwise, and shifted by a learned scalar bias.

```text
branch = BranchNet(branch_input)       -> [1,2562] -> reshape -> [1,21,61,2]
trunk  = TrunkNet(trunk_input)         -> [1,2562] -> reshape -> [1,21,61,2]
out    = branch * trunk + out_bias     -> [1,21,61,2]
```

Mathematically:

```text
U_hat[b,y,x,d] = B[b,y,x,d] * T[b,y,x,d] + out_bias
```

where `out_bias` is a learned scalar broadcast across all 2562 output values.

---

## 4. Tensor layout and flattening rules

All exported tensors and recommended HLS arrays use C row-major order. The rightmost index changes fastest.

Examples:

```text
branch_input[1][10][20][60]
flatten order: b, history, y, x

trunk_input[1][21][61][2]
flatten order: b, y, x, dof

conv2d_weight[OC][IC][KH][KW]
flatten order: oc, ic, kh, kw

conv3d_weight[OC][IC][KD][KH][KW]
flatten order: oc, ic, kd, kh, kw

linear_weight[OUT][IN]
flatten order: out_feature, in_feature
```

For HLS, define indexing macros/functions so that the hardware address order is visibly identical to the exported reference files.

---

## 5. Precision policy - FP16 revision

### 5.1 Required storage precision

Use IEEE FP16 / binary16 for the main HLS implementation in this revision:

- input tensors,
- weights,
- biases,
- named FP16 intermediate layer outputs,
- final output tensor.

Recommended HLS type for direct arithmetic:

```cpp
#include <hls_half.h>
using fp16_t = half;
```

Recommended packed word type for testbench loading or raw storage:

```cpp
#include <ap_int.h>
using fp16_word_t = ap_uint<16>;
```

Use `fp16_word_t` only when the testbench or storage path needs raw binary16 payloads. Convert or reinterpret to `half` at compute boundaries in a controlled way.

### 5.2 Arithmetic modes

The FP16 export package supports multiple comparison paths. Do not mix them in one result. Report each arithmetic mode separately.

#### Mode FP16-A: FP16 storage with float accumulation

This is recommended for first low-precision bring-up after FP32 indexing is correct.

```text
FP16 input/weight storage -> convert to float -> FP32 MAC accumulation -> store named outputs according to the selected comparison path
```

Use this mode when the HLS convolution, RNN, and linear kernels store weights/inputs in `half`, but perform MAC reductions in `float`.

Compare against:

```text
golden/fp16_io_f32
```

Those references are float32 golden outputs computed after weights and inputs are rounded to FP16, but the arithmetic is performed in float32.

#### Mode FP16-B: PyTorch-style half forward path

Use this mode if the HLS kernels compute and store named layer outputs as `half` and you want to compare against the PyTorch half forward path.

Compare against:

```text
golden/fp16_torch
```

or, for easier inspection while keeping the same values:

```text
golden/fp16_torch_f32
```

Start with a relaxed tolerance such as:

```text
abs_tol = 2e-2
rel_tol = 2e-2
```

Then tighten after indexing and operation semantics are confirmed.

#### Mode FP32-debug: full float bring-up

Use this only for indexing and operation debugging.

```text
FP32 input/weight/golden tensors -> float arithmetic -> compare every named checkpoint
```

Compare against:

```text
fp32
```

Do not report this as the final precision comparison unless it is clearly labeled as FP32 debug or FP32 baseline.

### 5.3 Rounding and debug points

For reproducible HLS validation, define where the debug build exposes named tensors. These are comparison checkpoints, not necessarily physical memory boundaries in the optimized design.

Recommended debug checkpoints:

```text
Conv pre-SiLU, SiLU, MaxPool, AAP, RNN step/output, Linear pre-SiLU, Linear/FC output, final output
```

A fused implementation may not physically store all of these tensors, but its C-simulation or debug build should be able to expose equivalent values for comparison against the exported references.

---

## 6. Operation definitions

### 6.1 Conv2D

Used in BranchNet.

PyTorch-style parameters:

```text
stride   = 1
padding  = 1
kernel   = 3 x 3
dilation = 1
groups   = 1
```

Input/output layout:

```text
input  = [N,C,H,W]
weight = [OC,IC,KH,KW]
bias   = [OC]
output = [N,OC,H,W]
```

Formula:

```text
out[n,oc,y,x] = bias[oc]
              + sum_ic sum_ky sum_kx input_padded[n,ic,y+ky-1,x+kx-1]
                                      * weight[oc,ic,ky,kx]
```

Values outside the image are zero due to padding.

### 6.2 Conv3D

Used in TrunkNet.

PyTorch-style parameters:

```text
stride   = 1
padding  = 1 in D, H, and W
kernel   = 3 x 3 x 2
dilation = 1
groups   = 1
```

Input/output layout:

```text
input  = [N,C,D,H,W]
weight = [OC,IC,KD,KH,KW]
bias   = [OC]
output = [N,OC,D_out,H_out,W_out]
```

Formula:

```text
out[n,oc,d,h,w] = bias[oc]
                + sum_ic sum_kd sum_kh sum_kw input_padded[n,ic,d+kd-1,h+kh-1,w+kw-1]
                                                 * weight[oc,ic,kd,kh,kw]
```

For this model, PyTorch Conv3D treats the trunk tensor as:

```text
trunk_input       = [1,21,61,2]
trunk_unsqueezed  = [1,1,21,61,2]
                   = [N,C,D,H,W]
```

The final dimension grows from 2 to 3 after Conv3D layer 1 and from 3 to 4 after Conv3D layer 2 because padding is applied on the W dimension and the kernel width is 2.

### 6.3 SiLU activation

Used after all convolution layers and after the first fully connected layer in both subnetworks.

```text
SiLU(x) = x * sigmoid(x)
sigmoid(x) = 1 / (1 + exp(-x))
```

For HLS, the approximation method must be fixed and reported. Options:

- exact `hls::expf` for correctness bring-up,
- LUT or piecewise approximation for optimized design,
- polynomial approximation.

The optimized approximation should be compared against the reference with a documented tolerance.

### 6.4 MaxPool2D

Used in BranchNet.

```text
kernel = 2 x 2
stride = 2
padding = 0
```

Input:

```text
[10,32,20,60]
```

Output:

```text
[10,32,10,30]
```

Formula:

```text
out[n,c,y,x] = max(
    in[n,c,2*y,  2*x],
    in[n,c,2*y,  2*x+1],
    in[n,c,2*y+1,2*x],
    in[n,c,2*y+1,2*x+1]
)
```

### 6.5 AdaptiveAvgPool2D

Used in BranchNet.

Input:

```text
[10,32,10,30]
```

Output:

```text
[10,32,1,1]
```

Since the output is `1 x 1`, this is simply the average over the full `10 x 30` spatial map for each history frame and channel:

```text
out[n,c,0,0] = sum_y sum_x in[n,c,y,x] / 300
```

### 6.6 AdaptiveAvgPool3D

Used in TrunkNet.

Input:

```text
[1,32,21,61,4]
```

Output:

```text
[1,32,15,10,1]
```

Use the PyTorch adaptive average pooling window rule for each output index:

```text
start = floor(out_index * input_size / output_size)
end   = ceil((out_index + 1) * input_size / output_size)
```

Average over the half-open interval `[start,end)` in each dimension.

For this model:

```text
D: 21 -> 15
H: 61 -> 10
W: 4  -> 1
```

### 6.7 RNN

Used in BranchNet after the CNN feature extractor.

Configuration:

```text
input_size    = 32
hidden_size   = 64
num_layers    = 1
batch_first   = true
nonlinearity  = tanh
sequence_len  = 10
initial h_0   = zeros
```

Input:

```text
[1,10,32]
```

All hidden outputs:

```text
[1,10,64]
```

Only the final hidden state is passed to the next layer:

```text
branch_rnn_last = rnn_all[0,9,:] -> [1,64]
```

Formula for timestep `t`:

```text
h_t = tanh(x_t @ W_ih.T + b_ih + h_{t-1} @ W_hh.T + b_hh)
```

where:

```text
W_ih = branch_rnn.weight_ih_l0 = [64,32]
W_hh = branch_rnn.weight_hh_l0 = [64,64]
b_ih = branch_rnn.bias_ih_l0   = [64]
b_hh = branch_rnn.bias_hh_l0   = [64]
```

### 6.8 Linear / GEMV

All fully connected layers use PyTorch `nn.Linear` semantics.

Input:

```text
x = [IN]
```

Weights:

```text
weight = [OUT,IN]
bias   = [OUT]
```

Formula:

```text
y[out] = bias[out] + sum_i x[i] * weight[out,i]
```

---

## 7. BranchNet exact architecture

Input:

```text
branch_input = [1,10,20,60]
```

The batch dimension is 1. The PyTorch model reshapes this tensor so each history frame is processed by the same CNN:

```text
[1,10,20,60] -> [10,1,20,60]
```

Layer-by-layer specification:

| Step | Name | Operation | Parameters | Output shape |
|---:|---|---|---|---|
| B0 | `branch_input` | Input | material-density history | `[1,10,20,60]` |
| B1 | `branch_input_reshaped` | Reshape | merge batch and history | `[10,1,20,60]` |
| B2 | `branch_conv1_pre_silu` | Conv2D | `IC=1`, `OC=16`, `K=3x3`, `stride=1`, `padding=1` | `[10,16,20,60]` |
| B3 | `branch_conv1_silu` | SiLU | elementwise | `[10,16,20,60]` |
| B4 | `branch_conv2_pre_silu` | Conv2D | `IC=16`, `OC=32`, `K=3x3`, `stride=1`, `padding=1` | `[10,32,20,60]` |
| B5 | `branch_conv2_silu` | SiLU | elementwise | `[10,32,20,60]` |
| B6 | `branch_maxpool2d` | MaxPool2D | `K=2x2`, `stride=2` | `[10,32,10,30]` |
| B7 | `branch_aap2d` | AdaptiveAvgPool2D | output `1x1` | `[10,32,1,1]` |
| B8 | `branch_rnn_input` | Reshape | restore batch/history | `[1,10,32]` |
| B9 | `branch_rnn_all` | RNN | `input=32`, `hidden=64`, `seq=10`, `tanh` | `[1,10,64]` |
| B10 | `branch_rnn_last` | Slice | last timestep only | `[1,64]` |
| B11 | `branch_fc1_pre_silu` | Linear | `64 -> 40` | `[1,40]` |
| B12 | `branch_fc1_silu` | SiLU | elementwise | `[1,40]` |
| B13 | `branch_fc2` | Linear | `40 -> 2562` | `[1,2562]` |
| B14 | `branch_reshaped` | Reshape | nodal displacement layout | `[1,21,61,2]` |

BranchNet weights:

| Parameter | Shape |
|---|---:|
| `branch_conv1.weight` | `[16,1,3,3]` |
| `branch_conv1.bias` | `[16]` |
| `branch_conv2.weight` | `[32,16,3,3]` |
| `branch_conv2.bias` | `[32]` |
| `branch_rnn.weight_ih_l0` | `[64,32]` |
| `branch_rnn.weight_hh_l0` | `[64,64]` |
| `branch_rnn.bias_ih_l0` | `[64]` |
| `branch_rnn.bias_hh_l0` | `[64]` |
| `branch_fc1.weight` | `[40,64]` |
| `branch_fc1.bias` | `[40]` |
| `branch_fc2.weight` | `[2562,40]` |
| `branch_fc2.bias` | `[2562]` |

---

## 8. TrunkNet exact architecture

Input:

```text
trunk_input = [1,21,61,2]
```

The PyTorch model adds a channel dimension before Conv3D:

```text
[1,21,61,2] -> [1,1,21,61,2]
```

Layer-by-layer specification:

| Step | Name | Operation | Parameters | Output shape |
|---:|---|---|---|---|
| T0 | `trunk_input` | Input | load tensor | `[1,21,61,2]` |
| T1 | `trunk_unsqueezed` | Unsqueeze | add channel dimension | `[1,1,21,61,2]` |
| T2 | `trunk_conv1_pre_silu` | Conv3D | `IC=1`, `OC=16`, `K=3x3x2`, `stride=1`, `padding=1` | `[1,16,21,61,3]` |
| T3 | `trunk_conv1_silu` | SiLU | elementwise | `[1,16,21,61,3]` |
| T4 | `trunk_conv2_pre_silu` | Conv3D | `IC=16`, `OC=32`, `K=3x3x2`, `stride=1`, `padding=1` | `[1,32,21,61,4]` |
| T5 | `trunk_conv2_silu` | SiLU | elementwise | `[1,32,21,61,4]` |
| T6 | `trunk_aap3d` | AdaptiveAvgPool3D | output `(15,10,1)` | `[1,32,15,10,1]` |
| T7 | `trunk_flatten` | Flatten | C row-major | `[1,4800]` |
| T8 | `trunk_fc1_pre_silu` | Linear | `4800 -> 40` | `[1,40]` |
| T9 | `trunk_fc1_silu` | SiLU | elementwise | `[1,40]` |
| T10 | `trunk_fc2` | Linear | `40 -> 2562` | `[1,2562]` |
| T11 | `trunk_reshaped` | Reshape | nodal displacement layout | `[1,21,61,2]` |

TrunkNet weights:

| Parameter | Shape |
|---|---:|
| `trunk_conv1.weight` | `[16,1,3,3,2]` |
| `trunk_conv1.bias` | `[16]` |
| `trunk_conv2.weight` | `[32,16,3,3,2]` |
| `trunk_conv2.bias` | `[32]` |
| `trunk_fc1.weight` | `[40,4800]` |
| `trunk_fc1.bias` | `[40]` |
| `trunk_fc2.weight` | `[2562,40]` |
| `trunk_fc2.bias` | `[2562]` |

---

## 9. Final combine layer

Inputs:

```text
branch_reshaped = [1,21,61,2]
trunk_reshaped  = [1,21,61,2]
out_bias        = [1]
```

Output:

```text
final_output = [1,21,61,2]
```

Formula:

```text
final_mul_pre_bias[b,y,x,d] = branch_reshaped[b,y,x,d] * trunk_reshaped[b,y,x,d]
final_output[b,y,x,d]       = final_mul_pre_bias[b,y,x,d] + out_bias[0]
```

Final flatten order for output stream or output buffer:

```text
b, y, x, d
```

---

## 10. Recommended HLS top-level organization

A fair PL/HLS implementation should mirror the AIE dataflow rather than using a generic accelerator runtime.

Recommended top function shape:

```cpp
void cronet_hls(
    hls::stream<fp16_t>& branch_input_stream,
    hls::stream<fp16_t>& trunk_input_stream,
    hls::stream<fp16_t>& output_stream
) {
#pragma HLS DATAFLOW
    branch_net(...);
    trunk_net(...);
    final_mul_bias(...);
}
```

Alternative for raw packed input/output:

```cpp
void cronet_hls(
    hls::stream<fp16_word_t>& branch_input_stream,
    hls::stream<fp16_word_t>& trunk_input_stream,
    hls::stream<fp16_word_t>& output_stream
);
```

Alternative for early bring-up:

```cpp
void cronet_hls(
    const fp16_t branch_input[1][10][20][60],
    const fp16_t trunk_input[1][21][61][2],
    fp16_t output[1][21][61][2]
);
```

For final comparison, avoid large external-memory `m_axi` interfaces for weights or intermediates. Use local static arrays, BRAM/URAM arrays, or streams.

---

## 11. On-chip storage requirements

### 11.1 Weights

All weights should be stored as static constant arrays in the HLS design or initialized into BRAM/URAM ROMs.

Recommended storage mapping:

| Weight group | Recommended storage |
|---|---|
| Small conv/RNN/FC1 weights | BRAM or LUTROM, depending on synthesis results |
| Large FC2 weights | BRAM or URAM |
| Biases | LUTRAM, registers, or BRAM |

### 11.2 Intermediate activations

The implementation may either:

1. materialize named tensors in BRAM/URAM for bring-up and debug, or
2. fuse/stream layers once the debug version is validated.

The optimized design should still be able to build a debug variant that emits named golden checkpoints.

Suggested fusion after correctness:

```text
Conv2D + SiLU
Conv3D + SiLU
Linear + SiLU
MaxPool2D + AAP2D where practical
Final reshape + multiply + bias
```

Do not fuse layers before the standalone layer outputs match the reference.

---

## 12. Exported weights and test-vector package - FP16

The companion export package is defined by `CRONET_HLS_FP16_VECTOR_EXPORT_REFERENCE.md`. It contains deterministic dummy weights, inputs, intermediate golden outputs, and final golden outputs for validating the CRONet HLS C++ implementation using IEEE FP16 / binary16.

The package is the source of truth for:

- file names,
- tensor shapes,
- flattening order,
- weight layouts,
- binary file formats,
- comparison directories,
- HLS loader helper names.

Expected export directory structure:

```text
export_root/
  manifest.json
  fp32/
    *.bin
    optional *.txt
  fp16/
    *.bin
    optional *.txt
  fp16_words/
    *.bin
    optional *.txt
  fp16_io_f32/
    *.bin
    optional *.txt
  fp16_torch/
    *.bin
    optional *.txt
  fp16_torch_f32/
    *.bin
    optional *.txt
  include/
    tb_bin_io_fp16.hpp
```

### 12.1 File formats

| Directory | File dtype | HLS use |
|---|---|---|
| `fp32` | Little-endian IEEE float32 | First bring-up and debugging |
| `fp16` | Little-endian IEEE binary16 | Main HLS `half` weights, inputs, and half golden outputs |
| `fp16_words` | Little-endian `uint16_t` containing IEEE binary16 payloads | Load into `uint16_t` or `ap_uint<16>` and reinterpret manually |
| `fp16_io_f32` | Little-endian IEEE float32 | Golden outputs for FP16-rounded inputs/weights with FP32 math |
| `fp16_torch` | Little-endian IEEE binary16 | Golden outputs from a PyTorch half forward pass |
| `fp16_torch_f32` | Little-endian IEEE float32 | Same as `fp16_torch`, converted to float32 for easier inspection |
| `manifest.json` | Metadata | Tensor paths, shapes, dtypes, counts, byte sizes, first values, and hashes |
| `include/tb_bin_io_fp16.hpp` | HLS testbench helper | Binary loader and comparison helper |

### 12.2 Available named golden tensors

The export package provides layer-by-layer golden outputs. These names should be used directly in the HLS testbench.

BranchNet tensors:

```text
branch_input
branch_input_reshaped
branch_conv1_pre_silu
branch_conv1_silu
branch_conv2_pre_silu
branch_conv2_silu
branch_maxpool2d
branch_aap2d
branch_rnn_input
branch_rnn_all
branch_rnn_last
branch_rnn_step_00 ... branch_rnn_step_09
branch_fc1_pre_silu
branch_fc1_silu
branch_fc2
branch_reshaped
```

TrunkNet tensors:

```text
trunk_input
trunk_unsqueezed
trunk_conv1_pre_silu
trunk_conv1_silu
trunk_conv2_pre_silu
trunk_conv2_silu
trunk_aap3d
trunk_flatten
trunk_fc1_pre_silu
trunk_fc1_silu
trunk_fc2
trunk_reshaped
```

Final tensors:

```text
final_mul_pre_bias
final_output
```

### 12.3 Available weights

The export package provides weights by the parameter names listed in Sections 7 and 8, plus:

```text
out_bias
```

Use the weight layout tables exactly. Do not transpose weights when loading unless your HLS kernel explicitly expects a transposed local layout. If a transposed or tiled internal layout is used for performance, create a deterministic conversion script and verify it against `manifest.json` hashes before synthesis.

---

## 13. Suggested validation sequence

Use the exported golden vectors in this order:

1. **File I/O sanity check**
   - Load `manifest.json`.
   - Verify element counts and byte sizes.
   - Print the first few values of `branch_input`, `trunk_input`, and one weight tensor.

2. **FP32 debug path**
   - Use `fp32` files and `float` arithmetic.
   - Validate every named tensor to prove indexing and operation formulas.

3. **BranchNet CNN**
   - Validate `branch_input_reshaped`.
   - Validate `branch_conv1_pre_silu`.
   - Validate `branch_conv1_silu`.
   - Validate `branch_conv2_pre_silu`.
   - Validate `branch_conv2_silu`.

4. **BranchNet pooling**
   - Validate `branch_maxpool2d`.
   - Validate `branch_aap2d`.
   - Validate `branch_rnn_input`.

5. **BranchNet RNN**
   - Validate `branch_rnn_step_00` through `branch_rnn_step_09`.
   - Validate `branch_rnn_all`.
   - Validate `branch_rnn_last`.

6. **BranchNet FC layers**
   - Validate `branch_fc1_pre_silu`.
   - Validate `branch_fc1_silu`.
   - Validate `branch_fc2`.
   - Validate `branch_reshaped`.

7. **TrunkNet Conv3D**
   - Validate `trunk_unsqueezed`.
   - Validate `trunk_conv1_pre_silu`.
   - Validate `trunk_conv1_silu`.
   - Validate `trunk_conv2_pre_silu`.
   - Validate `trunk_conv2_silu`.

8. **TrunkNet pooling and FC**
   - Validate `trunk_aap3d`.
   - Validate `trunk_flatten`.
   - Validate `trunk_fc1_pre_silu`.
   - Validate `trunk_fc1_silu`.
   - Validate `trunk_fc2`.
   - Validate `trunk_reshaped`.

9. **Final combine**
   - Validate `final_mul_pre_bias`.
   - Validate `final_output`.

10. **Switch precision path**
   - For FP16 storage with float accumulation, compare to `fp16_io_f32`.
   - For half compute/storage, compare to `fp16_torch` or `fp16_torch_f32`.
   - Keep tolerances documented in the testbench report.

---

## 14. HLS implementation notes

### 14.1 Conv2D and Conv3D

For a first implementation, direct nested loops are acceptable. Then optimize with:

- line buffers for Conv2D,
- small plane/window buffers for Conv3D,
- output-channel tiling,
- input-channel reduction tiling,
- partial unrolling of kernel loops,
- explicit array partitioning on small kernel windows.

### 14.2 RNN

The RNN sequence length is fixed at 10. A fully unrolled or statically scheduled implementation is recommended.

At each timestep:

```text
x_t = branch_rnn_input[0,t,:]  // 32 values
h_prev = zeros for t=0, otherwise previous hidden vector
h_t = tanh(W_ih*x_t + b_ih + W_hh*h_prev + b_hh)
```

Use the exported `branch_rnn_step_XX` tensors to debug each timestep independently.

### 14.3 Flatten before trunk FC1

`trunk_flatten` must preserve C row-major order from `[1,32,15,10,1]` to `[1,4800]`.

Equivalent loop order:

```cpp
int idx = 0;
for (int n = 0; n < 1; n++)
  for (int c = 0; c < 32; c++)
    for (int d = 0; d < 15; d++)
      for (int h = 0; h < 10; h++)
        for (int w = 0; w < 1; w++)
          flat[idx++] = aap3d[n][c][d][h][w];
```

### 14.4 Reshape after FC2

Both BranchNet and TrunkNet FC2 outputs reshape from `[1,2562]` to `[1,21,61,2]` in C row-major order.

Equivalent loop order:

```cpp
int idx = 0;
for (int b = 0; b < 1; b++)
  for (int y = 0; y < 21; y++)
    for (int x = 0; x < 61; x++)
      for (int d = 0; d < 2; d++)
        reshaped[b][y][x][d] = fc2[idx++];
```

---

## 15. Acceptance criteria

An HLS implementation is considered layer-accurate when:

1. All tensor shapes match this specification.
2. All flattening and weight layouts match the FP16 export package.
3. The FP32 debug build matches `fp32` golden tensors within a small floating-point tolerance.
4. The FP16-storage-with-float-accumulation build matches `fp16_io_f32` within the declared tolerance.
5. The half-compute build matches `fp16_torch` or `fp16_torch_f32` within the declared tolerance.
6. The final full-network output matches `final_output` for the chosen arithmetic mode.
7. The final benchmark design stores all weights and intermediate activations on chip.
8. Any use of DDR is limited to benchmark input injection and output collection, unless a stricter PL-only test harness is used.
9. The benchmark report states the exact precision mode used: FP32-debug, FP16+float accumulation, or FP16 half compute.
10. If compared to the AIE paper's BF16 result, the benchmark report clearly labels the datatype difference.

---

## 16. Minimum tensor count summary

Input scalar counts:

```text
branch_input = 1*10*20*60  = 12000
trunk_input  = 1*21*61*2   = 2562
output       = 1*21*61*2   = 2562
```

Major intermediate scalar counts:

```text
branch_conv1_silu  = 10*16*20*60    = 192000
branch_conv2_silu  = 10*32*20*60    = 384000
branch_maxpool2d   = 10*32*10*30    = 96000
branch_aap2d       = 10*32*1*1      = 320
branch_rnn_all     = 1*10*64        = 640
branch_fc2         = 1*2562         = 2562

trunk_conv1_silu   = 1*16*21*61*3   = 61488
trunk_conv2_silu   = 1*32*21*61*4   = 164064
trunk_aap3d        = 1*32*15*10*1   = 4800
trunk_fc2          = 1*2562         = 2562
```

These tensors do not all need to be materialized in the optimized design. They are listed because they exist as named validation points in the reference export.

---

## 17. Practical build plan

Recommended development stages:

1. Implement all layers with simple loops and `float` using `fp32` exports.
2. Validate every named tensor against the `fp32` golden outputs.
3. Replace input/weight storage with `half` or raw `fp16_words` and explicit conversion/reinterpretation.
4. Validate against `fp16_io_f32` for FP16 storage plus float accumulation.
5. Validate against `fp16_torch` or `fp16_torch_f32` for half compute/storage.
6. Convert weights to HLS static ROMs.
7. Move intermediate tensors from external testbench arrays to local BRAM/URAM/streams.
8. Add `DATAFLOW` between BranchNet, TrunkNet, and final combine.
9. Fuse Conv/Linear with SiLU after standalone validation.
10. Tune unroll factors and storage partitioning.
11. Run C simulation, C/RTL cosimulation, synthesis, implementation, then on-board latency/power measurement.

---

## 18. Non-negotiable 1:1 compatibility rules

- Do not change tensor shapes.
- Do not change flatten order.
- Do not change Conv2D or Conv3D padding semantics.
- Do not transpose linear weights unless the testbench conversion is explicit and verified.
- Do not approximate adaptive pooling with uniform bins unless it matches the PyTorch start/end rule.
- Do not use a different RNN equation or omit either RNN bias vector.
- Do not compare FP32 HLS against AIE or FP16 HLS unless explicitly labeled as a separate precision experiment.
- Do not compare FP16 HLS against BF16 AIE without labeling the datatype difference.
- Do not use DDR for weights or intermediates in the final comparison design.
