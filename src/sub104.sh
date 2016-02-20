#!/bin/sh

## b = 9
CLF_PARAMS="{'int_fold': 7, 'learning_rate': 0.03, 'learning_rates': [0.03] * 200 + [0.02] * 500, 'min_child_weight': 240, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 10, 'initial_params': [-0.01, 0.1, 0.1, 0.1, -0.1, -0.08, 0.02, -0.08, -0.01]}"
## b = 8
CLF_PARAMS="{'int_fold': 7, 'learning_rate': 0.03, 'learning_rates': [0.03] * 200 + [0.02] * 500, 'min_child_weight': 240, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 10, 'initial_params': [-0.01, 0.1, 0.1, -0.1, -0.08, 0.02, -0.08, -0.01]}"

./XGB_offset_reg.py \
  -o sub104.csv \
  -n 700 \
  -j -1 \
  -m Powell \
  --clf-params="${CLF_PARAMS}" \
  --cv-grid="{'n_estimators': [700], 'gamma': [0.0], 'max_depth': [10], 'colsample_bytree': [0.67], 'subsample': [.9], 'min_child_weight': [240], 'int_fold': [7], 'learning_rate': [0.03]}" \
  -b 8 \
  -E PrudentialRegressorCVO2 \
  --int-fold=7 \
#  -f 5
