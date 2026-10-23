#!/bin/bash
# get_orfs_libs.sh —— 只拉 OpenROAD-flow-scripts 的 3 个工艺库 lib 目录（稀疏检出）
# 用法: bash get_orfs_libs.sh [DEST]   (默认 DEST=./libs/OpenROAD-flow-scripts)
# 验证: 检出 commit 应为 3451ca1f56902a49baf50bb8eba345eb709cf9c8 或按 lib_manifest.sha256 核对
set -u
DEST="${1:-./libs/OpenROAD-flow-scripts}"
REPO="https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git"

for i in 1 2 3 4 5 6; do
  echo "=== attempt $i ==="
  rm -rf "$DEST"
  if git clone --filter=blob:none --sparse --depth 1 "$REPO" "$DEST" 2>&1 | tail -2; then
    cd "$DEST" || exit 1
    git sparse-checkout set flow/platforms/nangate45/lib flow/platforms/sky130hd/lib flow/platforms/asap7/lib 2>&1 | tail -1
    if [ -d flow/platforms/nangate45/lib ] && [ -d flow/platforms/sky130hd/lib ] && [ -d flow/platforms/asap7/lib ]; then
      echo "=== OK ==="
      git rev-parse HEAD
      find flow/platforms -name "*.lib*" | head -20
      exit 0
    fi
  fi
  echo "attempt failed, retry in 60s..."
  sleep 60
done
echo "ORFS_LIBS_FAILED"
exit 1
