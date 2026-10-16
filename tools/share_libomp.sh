#!/bin/bash
# Renames the libomp a delocated wheel bundles to @rpath/libomp.dylib. delocate
# gives each wheel's copy its own id, so two wheels start two openmp runtimes
# in one process, which crash or hang each other's parallel loops. With one
# shared id dyld loads whichever copy comes first and reuses it.
# usage: share_libomp.sh <wheel>   (rewrites the wheel in place)
set -euo pipefail
whl=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
tmp=$(mktemp -d)
python -m wheel unpack -d "$tmp" "$whl"
root=$(echo "$tmp"/*)
find "$root" -path '*dylibs/libomp.dylib' | while read -r lib; do
  install_name_tool -id @rpath/libomp.dylib "$lib"
  codesign -f -s - "$lib"
done
find "$root" -name '*.so' | while read -r so; do
  old=$(otool -L "$so" | awk '/dylibs\/libomp\.dylib/ {print $1}')
  [ -n "$old" ] || continue
  dir=$(dirname "${old#@loader_path/}")
  install_name_tool -change "$old" @rpath/libomp.dylib -add_rpath "@loader_path/$dir" "$so"
  codesign -f -s - "$so"
done
rm "$whl"
python -m wheel pack -d "$(dirname "$whl")" "$root"
rm -rf "$tmp"
