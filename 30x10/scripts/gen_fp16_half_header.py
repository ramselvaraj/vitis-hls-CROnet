#!/usr/bin/env python3
"""Generate a C++ header containing CRONet FP16 weights as HLS half constants."""

import argparse
import os
import struct


WEIGHTS = [
    ("out_bias", "out_bias.bin", 1),
    ("branch_conv1_weight", "branch_conv1__weight.bin", 16 * 1 * 3 * 3),
    ("branch_conv1_bias", "branch_conv1__bias.bin", 16),
    ("branch_conv2_weight", "branch_conv2__weight.bin", 32 * 16 * 3 * 3),
    ("branch_conv2_bias", "branch_conv2__bias.bin", 32),
    ("branch_rnn_weight_ih_l0", "branch_rnn__weight_ih_l0.bin", 64 * 32),
    ("branch_rnn_weight_hh_l0", "branch_rnn__weight_hh_l0.bin", 64 * 64),
    ("branch_rnn_bias_ih_l0", "branch_rnn__bias_ih_l0.bin", 64),
    ("branch_rnn_bias_hh_l0", "branch_rnn__bias_hh_l0.bin", 64),
    ("branch_fc1_weight", "branch_fc1__weight.bin", 40 * 64),
    ("branch_fc1_bias", "branch_fc1__bias.bin", 40),
    ("branch_fc2_weight", "branch_fc2__weight.bin", 682 * 40),
    ("branch_fc2_bias", "branch_fc2__bias.bin", 682),
    ("trunk_conv1_weight", "trunk_conv1__weight.bin", 16 * 1 * 3 * 3 * 2),
    ("trunk_conv1_bias", "trunk_conv1__bias.bin", 16),
    ("trunk_conv2_weight", "trunk_conv2__weight.bin", 32 * 16 * 3 * 3 * 2),
    ("trunk_conv2_bias", "trunk_conv2__bias.bin", 32),
    ("trunk_fc1_weight", "trunk_fc1__weight.bin", 40 * 4800),
    ("trunk_fc1_bias", "trunk_fc1__bias.bin", 40),
    ("trunk_fc2_weight", "trunk_fc2__weight.bin", 682 * 40),
    ("trunk_fc2_bias", "trunk_fc2__bias.bin", 682),
]


def fp16_to_float(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def read_values(path, count):
    with open(path, "rb") as f:
        data = f.read()
    expected = count * 2
    if len(data) != expected:
        raise RuntimeError(f"{path}: got {len(data)} bytes, expected {expected}")
    words = struct.unpack("<" + "H" * count, data)
    return [fp16_to_float(x) for x in words]


def emit_array(out, name, values):
    out.write(f"static const fp16_t {name}[{len(values)}] = {{\n")
    for i in range(0, len(values), 8):
        chunk = values[i:i + 8]
        body = ", ".join(f"fp16_t({x:.9e}f)" for x in chunk)
        suffix = "," if i + 8 < len(values) else ""
        out.write(f"    {body}{suffix}\n")
    out.write("};\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vector_root", help="Path to cronet_hls_fp16_vectors")
    parser.add_argument(
        "-o",
        "--output",
        default=os.path.join("include", "generated", "cronet_weights_fp16_half.hpp"),
        help="Output header path",
    )
    args = parser.parse_args()

    weights_root = os.path.join(args.vector_root, "weights", "fp16_words")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", newline="\n") as out:
        out.write("#pragma once\n")
        out.write("#include \"cronet_half_type.hpp\"\n\n")
        out.write("// Generated from weights/fp16_words as HLS half constants.\n")
        out.write("// Do not edit by hand; rerun scripts/gen_fp16_half_header.py instead.\n\n")
        out.write("namespace cronet {\n")
        out.write("namespace weights_fp16 {\n\n")

        for symbol, filename, count in WEIGHTS:
            path = os.path.join(weights_root, filename)
            emit_array(out, symbol, read_values(path, count))

        out.write("} // namespace weights_fp16\n")
        out.write("} // namespace cronet\n")

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
