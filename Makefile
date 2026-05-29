CXX ?= g++
CXXFLAGS ?= -std=c++11 -O2 -Wall -Wextra -Iinclude
VITIS_HLS ?= vitis_hls
VPP ?= v++
BUILD_DIR ?= build
VECTORS ?= data/cronet_hls_fp16_vectors
TARGET ?= sw_emu
CASE ?= sanity
ABS_TOL ?= 2e-2
REL_TOL ?= 2e-2

VITIS_HOME ?= $(shell dirname "$$(dirname "$$(which vitis 2>/dev/null)")" 2>/dev/null)
PLATFORM_USE ?= xilinx_vek280_base_202410_1
BASE_PLATFORM ?= $(VITIS_HOME)/base_platforms/$(PLATFORM_USE)/$(PLATFORM_USE).xpfm

KERNEL_NAME := cronet_kernel
KERNEL_XO := $(BUILD_DIR)/$(KERNEL_NAME).$(TARGET).xo
XCLBIN := $(BUILD_DIR)/$(KERNEL_NAME).$(TARGET).xclbin
HOST_EXE_SW := $(BUILD_DIR)/host_ps_on_x86
HOST_EXE_HW := $(BUILD_DIR)/host.exe
HOST_EXE := $(if $(filter $(TARGET),sw_emu),$(HOST_EXE_SW),$(HOST_EXE_HW))
VITIS_CFG := vitis/system.cfg

FP32_TB := $(BUILD_DIR)/tb_fp32_layerwise
FP16_STORAGE_TB := $(BUILD_DIR)/tb_fp16_storage_f32acc

.PHONY: all fp32-test fp16-storage-test half-native-smoke vitis-half-csim \
        vitis-check vitis-kernel vitis-xclbin vitis-host vitis-sw-emu vitis-clean clean

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

vitis-check:
	@test -n "$(VITIS_HOME)" || (echo "ERROR: vitis not found. Run: source env_source.sh 24"; exit 1)
	@test -f "$(BASE_PLATFORM)" || (echo "ERROR: platform not found: $(BASE_PLATFORM)"; exit 1)
	@test -f "$(VECTORS)/manifest.json" || (echo "ERROR: missing vectors at $(VECTORS)"; exit 1)
	@if [ "$(TARGET)" = "sw_emu" ]; then test -n "$(XILINX_XRT)" || (echo "ERROR: XILINX_XRT is not set. Run: source env_source.sh 24"; exit 1); fi
	@echo "VITIS_HOME=$(VITIS_HOME)"
	@echo "BASE_PLATFORM=$(BASE_PLATFORM)"
	@echo "TARGET=$(TARGET)"
	@echo "VECTORS=$(VECTORS)"

vitis-kernel: $(KERNEL_XO)

$(KERNEL_XO): src/cronet_kernel.cpp include/cronet_config.hpp include/cronet_half_type.hpp include/generated/cronet_weights_fp16_half.hpp | $(BUILD_DIR)
	$(MAKE) vitis-check TARGET=$(TARGET)
	$(VPP) -c -t $(TARGET) --platform $(BASE_PLATFORM) --save-temps -g \
		-Iinclude -k $(KERNEL_NAME) src/cronet_kernel.cpp -o $(KERNEL_XO)

vitis-xclbin: $(XCLBIN)

$(XCLBIN): $(KERNEL_XO) $(VITIS_CFG)
	$(VPP) -l -t $(TARGET) --platform $(BASE_PLATFORM) --save-temps -g \
		--config $(VITIS_CFG) $(KERNEL_XO) -o $(XCLBIN)

ifeq ($(TARGET),sw_emu)
vitis-host: $(HOST_EXE_SW)

$(HOST_EXE_SW): host/host.cpp | $(BUILD_DIR)
	$(MAKE) vitis-check TARGET=$(TARGET)
	g++ -std=c++17 -Wall -Wextra -I$(XILINX_XRT)/include -Iinclude \
		host/host.cpp -L$(XILINX_XRT)/lib -lxrt_coreutil -o $(HOST_EXE_SW)
else
vitis-host: $(HOST_EXE_HW)

$(HOST_EXE_HW): host/host.cpp | $(BUILD_DIR)
	$(MAKE) vitis-check TARGET=$(TARGET)
	$(CXX) -std=c++17 -Wall -Wextra -I$(SDKTARGETSYSROOT)/usr/include/xrt -Iinclude \
		host/host.cpp -lxrt_coreutil -o $(HOST_EXE_HW)
endif

vitis-sw-emu: TARGET=sw_emu
vitis-sw-emu: vitis-xclbin vitis-host
	cd $(BUILD_DIR) && emconfigutil --platform $(BASE_PLATFORM) --nd 1
	cd $(BUILD_DIR) && XCL_EMULATION_MODE=sw_emu ./host_ps_on_x86 $(KERNEL_NAME).sw_emu.xclbin ../$(VECTORS) $(CASE) $(ABS_TOL) $(REL_TOL)

vitis-clean:
	rm -rf $(BUILD_DIR)/$(KERNEL_NAME).* $(BUILD_DIR)/host_ps_on_x86 $(BUILD_DIR)/host.exe
	rm -rf _x .Xil *.log *.jou *.str vitis_hls.log

clean:
	rm -rf $(BUILD_DIR)
