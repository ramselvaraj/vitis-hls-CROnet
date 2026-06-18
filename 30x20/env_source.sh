#!/usr/bin/env bash

# Source this file from the repository root:
#   source env_source.sh advent_rigel_2024_1
#
# This mirrors the VEK280/Vitis environment style used by the archived
# implementation, trimmed to the toolchains relevant to this clean HLS repo.

usage() {
    echo "Usage: source env_source.sh <env>"
    echo ""
    echo "Environments:"
    echo "  advent_rigel_2024_1  Vitis 2024.1 + VEK280 common image paths"
    echo "  advent_rigel_2025_2  Vitis 2025.2 + VEK280 common image paths"
    echo "  help                 Show this message"
}

source_if_exists() {
    local path="$1"
    if [ -f "$path" ]; then
        # shellcheck disable=SC1090
        source "$path"
    else
        echo "WARNING: missing source file: $path"
    fi
}

advent_rigel_2024_1_env() {
    echo "Loading Vitis 2024.1 | VEK280"

    source_if_exists /mnt/vault0/tools/Vitis_2024.1/Vitis/2024.1/settings64.sh

    export XILINX_VERSAL=/mnt/vault0/tools/Vitis_2024.1/xilinx-versal-common-v2024.1
    source_if_exists "$XILINX_VERSAL/environment-setup-cortexa72-cortexa53-xilinx-linux"

    export ROOTFS=$XILINX_VERSAL/rootfs.ext4
    export IMAGE=$XILINX_VERSAL/Image
    export PLATFORM_REPO_PATHS=/mnt/vault0/tools/Vitis_2024.1/Vitis/2024.1/base_platforms

    source_if_exists /opt/xilinx/xrt/setup.sh
}

advent_rigel_2025_2_env() {
    echo "Loading Vitis 2025.2 | VEK280"

    source_if_exists /mnt/vault0/tools/Vitis_2025.2/2025.2/Vitis/settings64.sh

    export XILINX_VERSAL=/mnt/vault0/tools/Vitis_2025.2/xilinx-versal-common-v2025.2
    source_if_exists "$XILINX_VERSAL/environment-setup-cortexa72-cortexa53-amd-linux"

    export ROOTFS=$XILINX_VERSAL/rootfs.ext4
    export IMAGE=$XILINX_VERSAL/Image
    export PLATFORM_REPO_PATHS=/mnt/vault0/tools/Vitis_2025.2/2025.2/Vitis/base_platforms

    source_if_exists /opt/xilinx/xrt/setup.sh
}

print_env() {
    echo ""
    echo "Tool paths:"
    echo -n "  vitis_hls: "
    which vitis_hls || true
    echo -n "  vitis:     "
    which vitis || true
    echo -n "  v++:       "
    which v++ || true
    echo -n "  vivado:    "
    which vivado || true
    echo ""
    echo "Key variables:"
    echo "  XILINX_VERSAL=$XILINX_VERSAL"
    echo "  ROOTFS=$ROOTFS"
    echo "  IMAGE=$IMAGE"
    echo "  PLATFORM_REPO_PATHS=$PLATFORM_REPO_PATHS"
    echo ""
}

if [ -z "$1" ]; then
    usage
    return 1 2>/dev/null || exit 1
fi

case "$1" in
    advent_rigel_2024_1|vitis_2024_1|2024.1|24)
        advent_rigel_2024_1_env
        print_env
        ;;
    advent_rigel_2025_2|vitis_2025_2|2025.2|25)
        advent_rigel_2025_2_env
        print_env
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Unknown environment: $1"
        usage
        return 1 2>/dev/null || exit 1
        ;;
esac
