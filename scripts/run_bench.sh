#!/bin/sh
# Full benchmark with power rails. Needs root or a NOPASSWD sudoers rule for
# /usr/bin/powermetrics, e.g.:
#   echo "$USER ALL=(root) NOPASSWD: /usr/bin/powermetrics" | sudo tee /etc/sudoers.d/powermetrics
#   sudo chmod 440 /etc/sudoers.d/powermetrics
# Close other heavy workloads first; ane-bench aborts if the idle baseline is noisy.
set -e
cd "$(dirname "$0")/.."
cargo build --release --bins
./target/release/ane-bench --secs 5 --out bench_results "$@" 2>&1 | tee bench_results/run.log
