# Running the CRONet 60x20 FP16 kernel on the VEK280 board

This build is a **pure PL/HLS** design (no AIE graph) — `cronet_kernel`, FP16,
built with Vitis 2025.2. The current bitstream is an `-O quick`, **100 MHz**
functional build. The board host compares the kernel's final output against the
packaged golden vectors and reports timing.

## SSH access

- Board: `root@10.218.101.208` (key-based, no password prompt)
- **Always use a tmux session** for board interactions — keeps shell state across commands and lets long runs survive between tool invocations.

```bash
tmux new-session -d -s board
tmux send-keys -t board "ssh root@10.218.101.111" Enter
tmux capture-pane -t board -p
```

> **IP may change after a reboot.** If `ssh root@10.218.101.111` fails after a reboot, **ask the user for the new IP** before retrying.

## Files on the board

The SD-card boot partition is mounted at `/run/media/mmcblk0p1/`. Contents:

| File               | Purpose                                                       |
|--------------------|---------------------------------------------------------------|
| `BOOT.BIN`         | First-stage boot image (PMC, PLM, FSBL, PL bitstream)         |
| `Image`            | Linux kernel                                                  |
| `boot.scr`         | U-Boot boot script                                            |
| `a.xclbin`         | Linked PL/HLS design (`cronet_kernel`)                        |
| `host.exe`         | aarch64 XRT host application                                  |
| `run_hw_board.sh`  | Board run wrapper (selects case + tolerances + iteration count)|
| `data/`            | FP16 input vectors + goldens used for the on-board comparison |

## Source of these files on the workstation

Everything lives in the packaged SD directory after the hardware build:

```text
build/package.hw/sd_card/
```

## Running an experiment

```bash
cd /run/media/mmcblk0p1/
sh ./run_hw_board.sh sanity                 # single correctness run
sh ./run_hw_board.sh sanity 2e-2 2e-2 100   # 100 timed iterations
```

Argument order: `run_hw_board.sh <case> <abs_tol> <rel_tol> <n_iter>`.
Available cases: `sanity`, `ramp`, `sparse`, `random`.

Run all four cases, 100 timed iterations each:

```bash
for case in sanity ramp sparse random; do
  echo "=== $case ==="
  sh ./run_hw_board.sh "$case" 2e-2 2e-2 100
done
```

The wrapper auto-detects `a.xclbin` and calls the host. To call the host
directly instead:

```bash
./host.exe a.xclbin data/cronet_hls_fp16_vectors sanity 2e-2 2e-2 100
```

Expected ending of a passing run:

```text
compare final_output: PASS  max_abs ...  max_rel ...  mismatches 0
timing_iterations 100
kernel_core_cycles <non-zero>
kernel_core_ms_at_100mhz <ms>
timing_ms kernel_run   total ... avg ...
CRONET XRT TEST PASS
```

> This bitstream runs at **100 MHz**, so the host line reads
> `kernel_core_ms_at_100mhz`. The clock-independent number to trust for
> comparison is `kernel_core_cycles`; the ms value scales with the 100 MHz clock.

## When the HLS PL kernel changes (new `a.xclbin` / `BOOT.BIN`)

The new design is baked into `a.xclbin` and `BOOT.BIN`. You must:

1. From the workstation, copy the rebuilt boot + kernel + host files to the board:
   ```bash
   SDCARD=build/package.hw/sd_card
   scp "$SDCARD/BOOT.BIN" "$SDCARD/Image" "$SDCARD/boot.scr" \
       "$SDCARD/a.xclbin" "$SDCARD/host.exe" "$SDCARD/run_hw_board.sh" \
       root@10.218.101.111:/run/media/mmcblk0p1/
   ```
2. **Reboot the board** so the new bitstream loads:
   ```bash
   tmux send-keys -t board "sync; reboot" Enter
   ```
3. **Re-establish SSH** — `ssh root@10.218.101.111`. If this fails, the IP probably changed on reboot; ask the user for the new IP.
4. `cd /run/media/mmcblk0p1/ && sh ./run_hw_board.sh sanity`

> `data/` only needs to be copied the **first time** (or when the vectors/goldens
> change) — it is larger and does not change between kernel rebuilds:
> ```bash
> scp -r build/package.hw/sd_card/data root@10.218.101.111:/run/media/mmcblk0p1/
> ```

## When only the host application changes

Host-only changes don't affect the FPGA fabric — no reboot required.

1. Copy just `host.exe` (and `run_hw_board.sh` if it changed):
   ```bash
   scp build/package.hw/sd_card/host.exe root@10.218.101.111:/run/media/mmcblk0p1/
   ```
2. `cd /run/media/mmcblk0p1/ && sh ./run_hw_board.sh sanity` from the existing SSH session.

## Quick reference

| Change scope                  | Copy                                                              | Reboot? |
|-------------------------------|------------------------------------------------------------------|---------|
| HLS PL kernel (`a.xclbin`)    | `BOOT.BIN`, `Image`, `boot.scr`, `a.xclbin`, `host.exe`, `run_hw_board.sh` | **yes** |
| Host application only         | `host.exe`                                                       | no      |
| Input vectors / goldens only  | `data/`                                                          | no      |
