#!/bin/bash

for file in *.csv; do
  # パターン: mofgcn_mofgcn_{dataset}_{datatype}_{status}.csv
  if [[ "$file" =~ ^mofgcn_mofgcn_([a-z0-9]+)_([a-z]+)_(true|pred)\.csv$ ]]; then
    dataset="${BASH_REMATCH[1]}"
    datatype="${BASH_REMATCH[2]}"
    status="${BASH_REMATCH[3]}"
    corrected_name="mofgcn_${status}_${dataset}_${datatype}.csv"
    mv "$file" "$corrected_name"
    echo "Fixed: $file → $corrected_name"
    continue
  fi

  # 正しい形式ならスキップ
  if [[ "$file" =~ ^mofgcn_(true|pred)_(ctrp|gdsc1|gdsc2|nci)_(cell|drug)\.csv$ ]]; then
    echo "Already correct: $file"
    continue
  fi

  # 例: new_cell_true_gdsc1.csv → mofgcn_true_gdsc1_cell.csv
  if [[ "$file" =~ ^new_(cell|drug)_(true|pred)_(ctrp|gdsc1|gdsc2|nci)\.csv$ ]]; then
    datatype="${BASH_REMATCH[1]}"
    status="${BASH_REMATCH[2]}"
    dataset="${BASH_REMATCH[3]}"
    newname="mofgcn_${status}_${dataset}_${datatype}.csv"
    mv "$file" "$newname"
    echo "Renamed: $file → $newname"
    continue
  fi

  echo "Unrecognized format: $file (skipped)"
done

