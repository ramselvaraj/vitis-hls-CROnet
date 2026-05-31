#!/usr/bin/env sh

set -eu

case_name="${1:-sanity}"
abs_tol="${2:-2e-2}"
rel_tol="${3:-2e-2}"
n_iter="${4:-1}"

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir"

vectors="data/cronet_hls_fp16_vectors"

if [ ! -f "$vectors/manifest.json" ]; then
    echo "ERROR: missing $vectors/manifest.json"
    echo "Run this from the packaged SD-card directory or ensure data/ was packaged."
    exit 1
fi

xclbin=""
for candidate in a.xclbin binary_container_1.xclbin cronet_kernel.hw.xclbin cronet_kernel.hw_emu.xclbin cronet_kernel.sw_emu.xclbin; do
    if [ -f "$candidate" ]; then
        xclbin="$candidate"
        break
    fi
done

if [ -z "$xclbin" ]; then
    for candidate in *.xclbin; do
        if [ -f "$candidate" ]; then
            xclbin="$candidate"
            break
        fi
    done
fi

if [ -z "$xclbin" ]; then
    echo "ERROR: no .xclbin found in $script_dir"
    exit 1
fi

if [ ! -x ./host.exe ]; then
    chmod +x ./host.exe 2>/dev/null || true
fi

echo "Running CRONet hardware check"
echo "  xclbin: $xclbin"
echo "  case:   $case_name"
echo "  abs:    $abs_tol"
echo "  rel:    $rel_tol"
echo "  iter:   $n_iter"

./host.exe "$xclbin" "$vectors" "$case_name" "$abs_tol" "$rel_tol" "$n_iter"
