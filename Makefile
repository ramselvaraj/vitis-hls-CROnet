CXX ?= g++
CXXFLAGS ?= -std=c++11 -O2 -Wall -Wextra -Iinclude
VITIS_HLS ?= vitis_hls
BUILD_DIR ?= build
VECTORS ?= ../layer_wise_weights/cronet_hls_fp16_vectors

FP32_TB := $(BUILD_DIR)/tb_fp32_layerwise
FP16_STORAGE_TB := $(BUILD_DIR)/tb_fp16_storage_f32acc

.PHONY: all fp32-test fp16-storage-test half-native-smoke vitis-half-csim clean

all: $(FP32_TB) $(FP16_STORAGE_TB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(FP32_TB): src/cronet_fp32.cpp tests/tb_fp32_layerwise.cpp include/cronet_config.hpp include/bin_io.hpp include/cronet_fp32.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) src/cronet_fp32.cpp tests/tb_fp32_layerwise.cpp -o $@

$(FP16_STORAGE_TB): src/cronet_fp32.cpp tests/tb_fp16_storage_f32acc.cpp include/cronet_config.hpp include/bin_io.hpp include/cronet_fp32.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) src/cronet_fp32.cpp tests/tb_fp16_storage_f32acc.cpp -o $@

fp32-test: $(FP32_TB)
	$(FP32_TB) $(VECTORS) all

fp16-storage-test: $(FP16_STORAGE_TB)
	$(FP16_STORAGE_TB) $(VECTORS) all

half-native-smoke: src/cronet_half_debug.cpp tests/tb_half_layerwise.cpp include/cronet_half_debug.hpp include/cronet_half_type.hpp include/generated/cronet_weights_fp16_half.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DCRONET_USE_NATIVE_FLOAT_HALF -DCRONET_HALF_DEBUG_SINGLE_TRANSLATION_UNIT tests/tb_half_layerwise.cpp -o $(BUILD_DIR)/tb_half_layerwise_native_stub
	$(BUILD_DIR)/tb_half_layerwise_native_stub $(VECTORS) all

vitis-half-csim:
	CRONET_VECTOR_ROOT=$(VECTORS) $(VITIS_HLS) -f vitis/half_layerwise_csim.tcl

clean:
	rm -rf $(BUILD_DIR)
