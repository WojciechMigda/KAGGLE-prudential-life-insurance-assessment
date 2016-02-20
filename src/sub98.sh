#!/bin/sh

#CLF_PARAMS="{'learning_rate': 0.05, 'min_child_weight': 240, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'initial_params': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4, 7.2]}"
#CLF_PARAMS="{'learning_rate': 0.05, 'min_child_weight': 240, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'initial_params': [0.1, 0.1, 0.1, -1, -0.8, 0.02, 0.8, 1, 2, 3, 5, 6, 7]}" \

#CLF_PARAMS="{'learning_rate': 0.05, 'learning_rates': [0.03] * 200 + [0.02] * 500, 'min_child_weight': 240, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'initial_params': [-0.01, -0.01, -0.01, 0.1, 0.1, 0.1, -0.1, -0.08, 0.02, -0.08, -0.01, -0.01]}" \
CLF_PARAMS="{'learning_rate': 0.03, 'learning_rates': [0.03] * 200 + [0.02] * 500, 'min_child_weight': 240, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 10, 'initial_params': [0.1, 0.1, 0.1, -0.1, -0.08, 0.02, -0.08, -0.01]}" \
#CLF_PARAMS="{'learning_rate': 0.05, 'learning_rates': [0.03] * 200 + [0.02] * 500, 'min_child_weight': 240, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'initial_params': [-0.1, 0.1, -0.1, -0.08, 0.02, -0.08, -0.01]}" \
#CLF_PARAMS="{'learning_rate': 0.05, 'min_child_weight': 240, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'initial_params': [0.1, 0.1, 0.1, -0.1, -0.08, 0.02, -0.08, -0.01]}" \
#CLF_PARAMS="{'learning_rate': 0.05, 'min_child_weight': 240, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'initial_params': [0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1]}" \

./XGB_offset_reg.py \
  -o sub98.csv \
  -n 700 \
  -j -1 \
  -m Powell \
  --clf-params="${CLF_PARAMS}" \
  --cv-grid="{'n_estimators': [700], 'max_depth': [10], 'colsample_bytree': [0.67], 'subsample': [0.9], 'min_child_weight': [240], 'int_fold': [7], 'learning_rate': [0.03]}" \
  -b 8 \
  -E PrudentialRegressorCVO3 \
  --int-fold=7 \
#  -f 5
