# Local Vector Data

Place the CRONet FP16 vector package here:

```text
data/cronet_hls_fp16_vectors/
```

This directory is intentionally ignored by git because it contains generated binary weights, inputs, and golden tensors. The code and scripts expect this in-repo path by default so the same commands work on Windows and on the Vitis Linux machine after the package is copied.

Minimum expected files:

```text
data/cronet_hls_fp16_vectors/manifest.json
data/cronet_hls_fp16_vectors/weights/fp32/*.bin
data/cronet_hls_fp16_vectors/weights/fp16_words/*.bin
data/cronet_hls_fp16_vectors/cases/sanity/inputs/fp32/*.bin
data/cronet_hls_fp16_vectors/cases/sanity/golden/fp32/*.bin
data/cronet_hls_fp16_vectors/cases/*/inputs/fp16_words/*.bin
data/cronet_hls_fp16_vectors/cases/*/golden/fp16_io_f32/*.bin
data/cronet_hls_fp16_vectors/cases/*/golden/fp16_torch_f32/*.bin
```
