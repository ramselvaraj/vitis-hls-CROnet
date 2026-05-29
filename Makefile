CXX ?= g++
CXXFLAGS ?= -std=c++11 -O2 -Wall -Wextra -Iinclude
BUILD_DIR ?= build
VECTORS ?= ../layer_wise_weights/cronet_hls_fp16_vectors

FP32_TB := $(BUILD_DIR)/tb_fp32_layerwise
FP16_STORAGE_TB := $(BUILD_DIR)/tb_fp16_storage_f32acc

.PHONY: all fp32-test clean

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

clean:
	rm -rf $(BUILD_DIR)
