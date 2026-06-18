#pragma once

#if defined(CRONET_USE_NATIVE_FLOAT_HALF)
namespace cronet {
using fp16_t = float;
} // namespace cronet
#else
#include <hls_half.h>
namespace cronet {
using fp16_t = half;
} // namespace cronet
#endif
