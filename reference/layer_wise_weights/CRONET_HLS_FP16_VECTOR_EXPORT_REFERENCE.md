# CRONet HLS FP16 Test Vector Export Reference

This package contains deterministic dummy weights, inputs, and golden outputs for validating the CRONet HLS C++ implementation using IEEE FP16 / binary16.

## Model configuration

- `nelx = 60`
- `nely = 20`
- `history = 10`
- `neurons = 40`
- `output_neurons = (nely + 1) * (nelx + 1) * 2 = 2562`
- Batch size for all exported cases is `1`.

## Precision sets

| Directory | File dtype | Use |
|---|---|---|
| `fp32` | little-endian IEEE float32 | First bring-up and debugging. |
| `fp16` | little-endian IEEE binary16 | Main HLS `half` weights, inputs, and half golden outputs. |
| `fp16_words` | little-endian `uint16_t` words containing IEEE binary16 payloads | Use if your HLS testbench loads into `ap_uint<16>` or reinterprets manually. |
| `fp16_io_f32` | little-endian IEEE float32 | Golden outputs computed with FP16-rounded inputs/weights but FP32 math. Use if your HLS design uses half storage but float accumulation. |
| `fp16_torch` | little-endian IEEE binary16 | Golden outputs from PyTorch half forward pass. Use if your HLS design computes/stores layer outputs as `half`. |
| `fp16_torch_f32` | little-endian IEEE float32 | Same as `fp16_torch`, converted to float32 for easier inspection. |

## Flattening and layout

All files are flattened in C row-major order. The rightmost tensor index changes fastest.

Input shapes:

- `branch_input`: `[1,10,20,60]`, representing `[batch, history, y, x]`.
- `trunk_input`: `[1,21,61,2]`. For Conv3D, this is unsqueezed to `[1,1,21,61,2]`, interpreted by PyTorch as `[N,C,D,H,W]`.

## Weight tensor layouts

| Parameter | Layout |
|---|---|
| `branch_conv1.weight` | `[out_channel=16, in_channel=1, kernel_h=3, kernel_w=3]` |
| `branch_conv1.bias` | `[out_channel=16]` |
| `branch_conv2.weight` | `[out_channel=32, in_channel=16, kernel_h=3, kernel_w=3]` |
| `branch_conv2.bias` | `[out_channel=32]` |
| `branch_rnn.weight_ih_l0` | `[hidden_size=64, input_size=32]` |
| `branch_rnn.weight_hh_l0` | `[hidden_size=64, hidden_size=64]` |
| `branch_rnn.bias_ih_l0` | `[hidden_size=64]` |
| `branch_rnn.bias_hh_l0` | `[hidden_size=64]` |
| `branch_fc1.weight` | `[out_features=40, in_features=64]` |
| `branch_fc1.bias` | `[out_features=40]` |
| `branch_fc2.weight` | `[out_features=2562, in_features=40]` |
| `branch_fc2.bias` | `[out_features=2562]` |
| `trunk_conv1.weight` | `[out_channel=16, in_channel=1, kernel_d=3, kernel_h=3, kernel_w=2]` |
| `trunk_conv1.bias` | `[out_channel=16]` |
| `trunk_conv2.weight` | `[out_channel=32, in_channel=16, kernel_d=3, kernel_h=3, kernel_w=2]` |
| `trunk_conv2.bias` | `[out_channel=32]` |
| `trunk_fc1.weight` | `[out_features=40, in_features=4800]` |
| `trunk_fc1.bias` | `[out_features=40]` |
| `trunk_fc2.weight` | `[out_features=2562, in_features=40]` |
| `trunk_fc2.bias` | `[out_features=2562]` |
| `out_bias` | `[1] scalar broadcast over final output` |

## Golden tensors exported per case

| Tensor | Shape | HLS debug purpose |
|---|---:|---|
| `branch_input_reshaped` | `[10,1,20,60]` |  |
| `branch_conv1_pre_silu` | `[10,16,20,60]` | Raw Branch Conv2D1. |
| `branch_conv1_silu` | `[10,16,20,60]` | Branch Conv2D1 with fused SiLU. |
| `branch_conv2_pre_silu` | `[10,32,20,60]` | Raw Branch Conv2D2. |
| `branch_conv2_silu` | `[10,32,20,60]` | Branch Conv2D2 with fused SiLU. |
| `branch_maxpool2d` | `[10,32,10,30]` | 2x2 stride-2 MaxPool. |
| `branch_aap2d` | `[10,32,1,1]` | Adaptive average to 1x1 per channel/history frame. |
| `branch_rnn_input` | `[1,10,32]` |  |
| `branch_rnn_all` | `[1,10,64]` | All ten RNN hidden states. |
| `branch_rnn_last` | `[1,64]` | Final RNN hidden state used by Branch FC1. |
| `branch_fc1_pre_silu` | `[1,40]` |  |
| `branch_fc1_silu` | `[1,40]` | Branch Linear 64->40 with SiLU. |
| `branch_fc2` | `[1,2562]` | Branch Linear 40->2562. |
| `branch_reshaped` | `[1,21,61,2]` |  |
| `trunk_unsqueezed` | `[1,1,21,61,2]` |  |
| `trunk_conv1_pre_silu` | `[1,16,21,61,3]` |  |
| `trunk_conv1_silu` | `[1,16,21,61,3]` | Trunk Conv3D1 with fused SiLU. |
| `trunk_conv2_pre_silu` | `[1,32,21,61,4]` |  |
| `trunk_conv2_silu` | `[1,32,21,61,4]` | Trunk Conv3D2 with fused SiLU. |
| `trunk_aap3d` | `[1,32,15,10,1]` | Adaptive average to 15x10x1. |
| `trunk_flatten` | `[1,4800]` | Flatten order before Trunk FC1. |
| `trunk_fc1_pre_silu` | `[1,40]` |  |
| `trunk_fc1_silu` | `[1,40]` | Trunk Linear 4800->40 with SiLU. |
| `trunk_fc2` | `[1,2562]` | Trunk Linear 40->2562. |
| `trunk_reshaped` | `[1,21,61,2]` |  |
| `final_mul_pre_bias` | `[1,21,61,2]` |  |
| `final_output` | `[1,21,61,2]` | Full CRONet output after branch * trunk + scalar bias. |

The package also exports `branch_rnn_step_00` through `branch_rnn_step_09`, each with shape `[1,64]`.

## Recommended HLS comparison flow

1. Bring up each layer using `fp32` weights/inputs/goldens.
2. Switch weights/inputs to `fp16`. If your HLS kernel accumulates in `float`, compare to `golden/fp16_io_f32`.
3. If your HLS kernel computes and stores in `half`, compare to `golden/fp16_torch` using a relaxed tolerance such as `abs_tol=2e-2`, `rel_tol=2e-2` to start. Tighten after indexing is correct.
4. Validate block-by-block before checking `final_output`. Conv2D/Conv3D indexing and flatten order are the most common failure points.

## HLS loader helper

Use `include/tb_bin_io_fp16.hpp`. For raw half files, load directly into `half[]` under Vitis HLS, or load `fp16_words` into `uint16_t[]`/`ap_uint<16>[]` and reinterpret as needed.
