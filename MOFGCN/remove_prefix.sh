#!/bin/bash

for file in mofgcn_*.csv; do
  # 新しいファイル名 = "mofgcn_" を除いたもの
  newname="${file#mofgcn_}"
  mv "$file" "$newname"
  echo "Renamed: $file → $newname"
done

