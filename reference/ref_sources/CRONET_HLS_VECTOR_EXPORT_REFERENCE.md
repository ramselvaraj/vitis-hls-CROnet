# CRONet HLS Test Vector Export Reference
This file documents the generated dummy weights, inputs, and golden outputs used to validate the HLS C++ layers.
## Model configuration
- `nelx = 60`
- `nely = 20`
- `history = 10`
- `neurons = 40`
- `output_neurons = (nely + 1) * (nelx + 1) * 2 = 2562`
- Batch size for all exported cases is `1`.

## File formats
- `fp32/*.bin`: raw little-endian IEEE float32 values. Load as `float`.
- `bf16_words/*.bin`: raw little-endian unsigned 16-bit BF16 words. Load as `uint16_t` or `ap_uint<16>`.
- `bf16_io_f32/*.bin`: float32 reference outputs computed after inputs and weights were rounded to BF16, with PyTorch/FP32 arithmetic. This is useful if your HLS uses BF16 inputs/weights but accumulates to float.
- Optional `.txt` files are flat debug dumps in the same order as the `.bin` files.
- `manifest.json` contains paths, shapes, dtypes, counts, byte sizes, first values, and SHA256 hashes.

## Flattening and layout rules
All tensors are flattened in C row-major order, equivalent to `numpy.reshape(-1)` on a C-contiguous array.

For nested C arrays, the rightmost index changes fastest. Example: `branch_input[1][10][20][60]` is flattened as `b, history, y, x`.

## Input tensors
| Tensor | Shape | Meaning |
|---|---:|---|
| `branch_input` | `[1,10,20,60]` | 10 history frames of material density X. |
| `trunk_input` | `[1,21,61,2]` | Load/displacement-coordinate input F. In Conv3D this becomes `[1,1,21,61,2]`. |

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

## Golden output tensors
| Tensor | Shape for default 60x20 | Use in HLS bring-up |
|---|---:|---|
| `branch_input_reshaped` | `[10,1,20,60]` |  |
| `branch_conv1_pre_silu` | `[10,16,20,60]` | Compare raw Conv2D1 before activation. |
| `branch_conv1_silu` | `[10,16,20,60]` | Compare Conv2D1 with fused SiLU. |
| `branch_conv2_pre_silu` | `[10,32,20,60]` | Compare raw Conv2D2 before activation. |
| `branch_conv2_silu` | `[10,32,20,60]` | Compare Conv2D2 with fused SiLU. |
| `branch_maxpool2d` | `[10,32,10,30]` | Validate 2x2 stride-2 max pooling. |
| `branch_aap2d` | `[10,32,1,1]` | Validate average over each 10x30 pooled feature map. |
| `branch_rnn_input` | `[1,10,32]` |  |
| `branch_rnn_all` | `[1,10,64]` | Validate all ten RNN steps. |
| `branch_rnn_last` | `[1,64]` | Validate only the hidden state used by FC1. |
| `branch_fc1_pre_silu` | `[1,40]` |  |
| `branch_fc1_silu` | `[1,40]` | Validate Linear 64->40 with SiLU. |
| `branch_fc2` | `[1,2562]` | Validate Linear 40->2562. |
| `branch_reshaped` | `[1,21,61,2]` |  |
| `trunk_unsqueezed` | `[1,1,21,61,2]` |  |
| `trunk_conv1_pre_silu` | `[1,16,21,61,3]` |  |
| `trunk_conv1_silu` | `[1,16,21,61,3]` | Validate Conv3D1 with fused SiLU. |
| `trunk_conv2_pre_silu` | `[1,32,21,61,4]` |  |
| `trunk_conv2_silu` | `[1,32,21,61,4]` | Validate Conv3D2 with fused SiLU. |
| `trunk_aap3d` | `[1,32,15,10,1]` | Validate adaptive average pool to 15x10x1. |
| `trunk_flatten` | `[1,4800]` | Validate flatten address order before trunk FC1. |
| `trunk_fc1_pre_silu` | `[1,40]` |  |
| `trunk_fc1_silu` | `[1,40]` | Validate Linear 4800->40 with SiLU. |
| `trunk_fc2` | `[1,2562]` | Validate Linear 40->2562. |
| `trunk_reshaped` | `[1,21,61,2]` |  |
| `final_mul_pre_bias` | `[1,21,61,2]` |  |
| `final_output` | `[1,21,61,2]` | Validate full CRONet output after branch*trunk+bias. |

The script also exports `branch_rnn_step_00` through `branch_rnn_step_09`, each with shape `[1,64]`, for debugging a fully unrolled RNN implementation.

## Operation formulas used by the reference
- SiLU: `y = x * sigmoid(x)`.
- RNN step: `h_t = tanh(x_t @ W_ih.T + b_ih + h_{t-1} @ W_hh.T + b_hh)`, with zero initial hidden state.
- Final combine: reshape BranchNet and TrunkNet FC2 outputs to `[1,nely+1,nelx+1,2]`, then `out = branch * trunk + out_bias`.
- PyTorch adaptive average pooling window for output index `i`: `start = floor(i * input_size / output_size)`, `end = ceil((i + 1) * input_size / output_size)`. Average over `[start,end)`.

## Suggested HLS validation order
1. Load FP32 inputs and weights, validate one layer at a time.
2. Validate Conv2D1/Conv2D2 and Conv3D1/Conv3D2 before and after SiLU.
3. Validate MaxPool2D and AAP2D/AAP3D separately.
4. Validate the 10 RNN steps using the `branch_rnn_step_XX` dumps.
5. Validate GEMV layers.
6. Validate final multiply and bias.
7. Switch to BF16-word files once FP32 indexing and arithmetic are correct.

## HLS loader helpers
Use `include/tb_bin_io.hpp` for simple `load_bin` and compare helpers, and `include/cronet_hls_shapes.hpp` for element counts.
