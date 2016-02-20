#!/bin/sh

./XGB_offset_reg.py \
  -o sub83.csv \
  -n 800 \
  -m Powell \
  --clf-params="{'learning_rate': 0.05, 'min_child_weight': 240, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'initial_params': [0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1, 2, 3, 5, 6, 7]}" \
  -b 7 \
  -E PrudentialRegressorFO \
#  -f 5
