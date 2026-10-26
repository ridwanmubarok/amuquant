#!/usr/bin/env bash
set -e

# Setup script to clone and build external/llama.cpp for AMU-QUANT full-weight benchmarking
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
EXTERNAL_DIR="$ROOT_DIR/external"

echo "=== AMU-QUANT: Setting up external/llama.cpp ==="

mkdir -p "$EXTERNAL_DIR"
cd "$EXTERNAL_DIR"

if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp repository..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp
else
    echo "llama.cpp directory already exists. Pulling latest changes..."
    cd llama.cpp && git pull && cd ..
fi

echo "Building llama.cpp libraries..."
cmake -B llama.cpp/build -S llama.cpp -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_METAL=ON
cmake --build llama.cpp/build -j

echo "=== Setup complete! You can now reconfigure AMU-QUANT: ==="
echo "cmake -B build -S . && cmake --build build -j"
echo "This will enable 'amuquant_chat' and 'live_benchmark'."
