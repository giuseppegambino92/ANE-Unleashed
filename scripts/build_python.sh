#!/bin/sh
# Build the Rust cdylib and place it inside the Python package.
set -e
cd "$(dirname "$0")/.."
cargo build --release
cp target/release/libane_unleashed.dylib python/ane_unleashed/
echo "Python package ready: pip install ./python"
