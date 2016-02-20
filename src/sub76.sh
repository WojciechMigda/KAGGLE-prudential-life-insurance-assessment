#!/bin/sh

./XGB_offset_reg.py \
  -o sub76.csv \
  -n 700 \
  -m Powell \
  --clf-params="{'learning_rate': 0.045, 'min_child_weight': 50, 'subsample': 0.8, 'colsample_bytree': 0.7, 'max_depth': 7, 'initial_params': [0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1]}" \
#  -f 10
