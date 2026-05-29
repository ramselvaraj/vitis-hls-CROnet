# Vitis HLS C-simulation harness for the CRONet half debug checkpoint.
#
# Optional environment overrides:
#   CRONET_VECTOR_ROOT  path to cronet_hls_fp16_vectors
#   CRONET_CASE         all|sanity|ramp|sparse|random
#   CRONET_ABS_TOL      default 2e-2
#   CRONET_REL_TOL      default 2e-2
#   CRONET_PART         default VEK280 engineering sample part

set script_dir [file dirname [file normalize [info script]]]
set repo_dir [file normalize [file join $script_dir ".."]]

if {[info exists ::env(CRONET_VECTOR_ROOT)]} {
    set vector_root $::env(CRONET_VECTOR_ROOT)
} else {
    set vector_root [file normalize [file join $repo_dir ".." "layer_wise_weights" "cronet_hls_fp16_vectors"]]
}

if {[info exists ::env(CRONET_CASE)]} {
    set case_name $::env(CRONET_CASE)
} else {
    set case_name "all"
}

if {[info exists ::env(CRONET_ABS_TOL)]} {
    set abs_tol $::env(CRONET_ABS_TOL)
} else {
    set abs_tol "2e-2"
}

if {[info exists ::env(CRONET_REL_TOL)]} {
    set rel_tol $::env(CRONET_REL_TOL)
} else {
    set rel_tol "2e-2"
}

if {[info exists ::env(CRONET_PART)]} {
    set part_name $::env(CRONET_PART)
} else {
    set part_name "xcve2802-vsvh1760-2MP-e-S"
}

open_project -reset [file join $repo_dir "build" "vitis_half_layerwise_csim"]

set src_file [file join $repo_dir "src" "cronet_half_debug.cpp"]
set tb_file [file join $repo_dir "tests" "tb_half_layerwise.cpp"]
set include_dir [file join $repo_dir "include"]
set cflags "-std=c++11 -I$include_dir"

if {![file exists $src_file]} {
    error "Missing design source: $src_file"
}
if {![file exists $tb_file]} {
    error "Missing testbench source: $tb_file"
}

set_top "cronet::run_cronet_half_debug"

# Keep options before file paths. Some Vitis HLS versions do not reliably
# parse file-first add_files commands and can then omit the design object
# from csim linking.
add_files -cflags $cflags $src_file
add_files -tb -cflags $cflags $tb_file

open_solution -reset "solution1" -flow_target vitis
set_part $part_name
create_clock -period 5 -name default

puts "CRONET_VECTOR_ROOT=$vector_root"
puts "CRONET_CASE=$case_name"
puts "CRONET_ABS_TOL=$abs_tol"
puts "CRONET_REL_TOL=$rel_tol"
puts "CRONET_PART=$part_name"

csim_design -argv "$vector_root $case_name $abs_tol $rel_tol"
exit
