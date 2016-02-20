#!/bin/sh

./XGB_offset_reg.py \
  -o sub85.csv \
  -n 800 \
  -m Powell \
  --clf-params="{'learning_rate': 0.05, 'min_child_weight': 240, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'initial_params': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4, 7.2]}" \
  -b 10 \
  -E PrudentialRegressorFO \
#  -f 5
