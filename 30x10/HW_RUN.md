# Running the CRONet 60x20 FP16 kernel on the VEK280 board

This build is a **pure PL/HLS** design (no AIE graph) — `cronet_kernel`, FP16,
built with Vitis 2025.2.

**Current design:** the optimized **~410k-cycle** configuration — trunk conv2 with
4× output-channel parallelism and the large feature maps placed in URAM. Kernel
clock is **230 MHz** (the 250 MHz target missed timing by 0.15 ns at place-&-route,
so the bitstream is built at 230 MHz). The board host compares the kernel's final
output against the packaged golden vectors and reports timing.

> **Clock label:** the host computes ms as `kernel_core_cycles ÷ clock`. The clock
> defaults to 250 MHz unless you set `CRONET_KERNEL_CLOCK_HZ`. For this 230 MHz
> bitstream you **must** export `CRONET_KERNEL_CLOCK_HZ=230000000` so the ms is
> correct (the line then reads `kernel_core_ms_at_230mhz`). The clock-independent
> number to trust is always `kernel_core_cycles` (expected ~410,000).

## SSH access

- Board: `root@10.218.101.208` (key-based, no password prompt)
- **Use a tmux session** for long board interactions — keeps shell state across
  commands and lets long runs survive between tool invocations.

```bash
tmux new-session -d -s board
tmux send-keys -t board "ssh root@10.218.101.208" Enter
tmux capture-pane -t board -p
```

> **IP may change after a reboot.** If `ssh root@10.218.101.208` fails after a
> reboot, re-check the board's IP (or ask the board owner) before retrying.
> Quick reachability check from the workstation:
> ```bash
> timeout 5 bash -c 'echo > /dev/tcp/10.218.101.208/22' && echo "SSH up" || echo "unreachable"
> ```

## Files on the board

The SD-card boot partition is mounted at `/run/media/mmcblk0p1/`. Contents:

| File               | Purpose                                                       |
|--------------------|---------------------------------------------------------------|
| `BOOT.BIN`         | First-stage boot image (PMC, PLM, FSBL, PL bitstream)         |
| `Image`            | Linux kernel                                                  |
| `boot.scr`         | U-Boot boot script                                            |
| `a.xclbin`         | Linked PL/HLS design (`cronet_kernel`)                        |
| `host.exe`         | aarch64 XRT host application (clock-aware, see env var below) |
| `run_hw_board.sh`  | Board run wrapper (selects case + tolerances + iteration count)|
| `data/`            | FP16 input vectors + goldens used for the on-board comparison |

Source of these files on the workstation, after the hardware build:

```text
build/package.hw/sd_card/
```

## Deploy + run directly from the workstation

This is a kernel change (new `a.xclbin` / `BOOT.BIN`), so the boot files must be
copied and the board rebooted. Run these from the repo root on the workstation.

```bash
SDCARD=build/package.hw/sd_card
BOARD=root@10.218.101.208

# 1. Copy boot + kernel + host (+ runner). data/ only needs copying the first
#    time or when vectors/goldens change.
scp "$SDCARD/BOOT.BIN" "$SDCARD/Image" "$SDCARD/boot.scr" \
    "$SDCARD/a.xclbin" "$SDCARD/host.exe" "$SDCARD/run_hw_board.sh" \
    "$BOARD":/run/media/mmcblk0p1/
scp -r "$SDCARD/data" "$BOARD":/run/media/mmcblk0p1/      # first time / vectors changed only

# 2. Reboot so the new bitstream (BOOT.BIN) loads.
ssh "$BOARD" 'sync; (sleep 1; reboot) >/dev/null 2>&1 &'

# 3. Wait ~30-60 s, re-establish SSH (IP may change on reboot), then run.
ssh "$BOARD" 'cd /run/media/mmcblk0p1 && chmod +x host.exe run_hw_board.sh; \
  CRONET_KERNEL_CLOCK_HZ=230000000 sh ./run_hw_board.sh sanity 2e-2 2e-2 100'
```

## Running an experiment (from a shell on the board)

```bash
cd /run/media/mmcblk0p1/
export CRONET_KERNEL_CLOCK_HZ=230000000        # match the 230 MHz bitstream

sh ./run_hw_board.sh sanity                     # single correctness run
sh ./run_hw_board.sh sanity 2e-2 2e-2 100       # 100 timed iterations
```

Argument order: `run_hw_board.sh <case> <abs_tol> <rel_tol> <n_iter>`.
Available cases: `sanity`, `ramp`, `sparse`, `random`.

Run all four cases, 100 timed iterations each:

```bash
export CRONET_KERNEL_CLOCK_HZ=230000000
for case in sanity ramp sparse random; do
  echo "=== $case ==="
  sh ./run_hw_board.sh "$case" 2e-2 2e-2 100
done
```

The wrapper auto-detects `a.xclbin` and calls the host. To call the host directly:

```bash
CRONET_KERNEL_CLOCK_HZ=230000000 \
  ./host.exe a.xclbin data/cronet_hls_fp16_vectors sanity 2e-2 2e-2 100
```

Expected ending of a passing run:

```text
compare final_output: PASS  max_abs ...  max_rel ...  mismatches 0
timing_iterations 100
kernel_core_cycles 410xxx
kernel_core_ms_at_230mhz <ms>
timing_ms kernel_run   total ... avg ...
CRONET XRT TEST PASS
```

> `kernel_core_cycles` is the exact in-fabric cycle count (the real metric, ~410k).
> `kernel_core_ms_at_230mhz` = `cycles ÷ 230 MHz`, correct only when
> `CRONET_KERNEL_CLOCK_HZ` matches the built bitstream clock. If you rebuild at a
> different `KERNEL_FREQ`, set the env var to the same value.

## When only the host application changes

Host-only changes don't affect the FPGA fabric — no reboot required.

```bash
scp build/package.hw/sd_card/host.exe root@10.218.101.208:/run/media/mmcblk0p1/
# then on the board:
cd /run/media/mmcblk0p1/ && CRONET_KERNEL_CLOCK_HZ=230000000 sh ./run_hw_board.sh sanity
```

## Quick reference

| Change scope                  | Copy                                                              | Reboot? |
|-------------------------------|------------------------------------------------------------------|---------|
| HLS PL kernel (`a.xclbin`)    | `BOOT.BIN`, `Image`, `boot.scr`, `a.xclbin`, `host.exe`, `run_hw_board.sh` | **yes** |
| Host application only         | `host.exe`                                                       | no      |
| Input vectors / goldens only  | `data/`                                                          | no      |

> Set `CRONET_KERNEL_CLOCK_HZ=230000000` for every run of this bitstream so the
> reported ms matches the real 230 MHz clock.
