#!/bin/bash
# ビジュアライザを実行するスクリプト
# 使い方: ./vis.sh <input> <output>
# 例: ./vis.sh testcases/a/in/0000.txt testcases/a/out/0000.txt

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input> <output>"
    exit 1
fi

cd "$(dirname "$0")"
cargo run -r --manifest-path tools/Cargo.toml --bin vis -- "$1" "$2"
