// This file gives Vitis HLS a simple design top for the debug-only half C-sim
// project. The layerwise testbench includes the real debug implementation in
// one translation unit so Vitis cannot omit it from csim linking.

extern "C" void cronet_half_layerwise_csim_dummy() {}
