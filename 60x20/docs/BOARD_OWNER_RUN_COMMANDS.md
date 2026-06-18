# CRONet Board Run Commands

After booting the VEK280 from `sd_card.img`, find the SD-card mount:

```bash
find /run/media /mnt -name run_hw_board.sh 2>/dev/null
```

Then enter that directory. A typical path is:

```bash
cd /run/media/mmcblk0p1
chmod +x host.exe run_hw_board.sh 2>/dev/null || true
```

Run a single sanity inference:

```bash
sh ./run_hw_board.sh sanity
```

Run the sanity case with 100 timed iterations:

```bash
sh ./run_hw_board.sh sanity 2e-2 2e-2 100
```

Run all exported cases with 100 timed iterations each:

```bash
for case in sanity ramp sparse random; do
  echo "=== $case ==="
  sh ./run_hw_board.sh "$case" 2e-2 2e-2 100
done
```

Argument order:

```text
run_hw_board.sh <case> <abs_tol> <rel_tol> <n_iter>
```

For example, in:

```bash
sh ./run_hw_board.sh sanity 2e-2 2e-2 100
```

`100` is `n_iter`, so the host runs the FPGA kernel path 100 times and reports total and average timing.

Expected successful output includes:

```text
compare final_output: PASS ...
timing_iterations 100
kernel_core_cycles ...
kernel_core_ms_at_250mhz ...
timing_ms kernel_run    total ... avg ...
timing_ms inference_io  total ... avg ...
CRONET XRT TEST PASS
```

Timing meanings:

- `kernel_core_cycles`: cycle counter measured inside the HLS kernel after packed
  DDR inputs are copied into local memory and before packed output is written
  back to DDR.
- `kernel_core_ms_at_250mhz`: `kernel_core_cycles` converted using the 250 MHz
  handoff clock.
- `kernel_run avg`: average XRT kernel execution time.
- `inference_io avg`: average host-to-device sync + kernel run + device-to-host sync time.
