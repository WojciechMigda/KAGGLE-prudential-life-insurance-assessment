#!/opt/anaconda2/bin/python
# -*- coding: utf-8 -*-

"""
################################################################################
#
#  Copyright (c) 2016 Wojciech Migda
#  All rights reserved
#  Distributed under the terms of the MIT license
#
################################################################################
#
#  Filename: XGB_offset_reg.py
#
#  Decription:
#      XGBoost with offset fitting (based on Kaggle scripts)
#
#  Authors:
#       Wojciech Migda
#
################################################################################
#
#  History:
#  --------
#  Date         Who  Ticket     Description
#  ----------   ---  ---------  ------------------------------------------------
#  2016-01-22   wm              Initial version
#
################################################################################
"""

from __future__ import print_function


DEBUG = False

try:
    import ml_metrics
except ImportError:
    KAGGLE = False
    pass
else:
    DEBUG = True
    KAGGLE = True
    pass

__all__ = []
__version__ = "0.0.1"
__date__ = '2016-01-22'
__updated__ = '2016-01-22'


NOMINALS = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3',
            'Product_Info_5', 'Product_Info_6', 'Product_Info_7',
            'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5',
            'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4',
            'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7',
            'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3',
            'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8',
            'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2',
            'Medical_History_3', 'Medical_History_4', 'Medical_History_5',
            'Medical_History_6', 'Medical_History_7', 'Medical_History_8',
            'Medical_History_9', 'Medical_History_11', 'Medical_History_12',
            'Medical_History_13', 'Medical_History_14', 'Medical_History_16',
            'Medical_History_17', 'Medical_History_18', 'Medical_History_19',
            'Medical_History_20', 'Medical_History_21', 'Medical_History_22',
            'Medical_History_23', 'Medical_History_25', 'Medical_History_26',
            'Medical_History_27', 'Medical_History_28', 'Medical_History_29',
            'Medical_History_30', 'Medical_History_31', 'Medical_History_33',
            'Medical_History_34', 'Medical_History_35', 'Medical_History_36',
            'Medical_History_37', 'Medical_History_38', 'Medical_History_39',
            'Medical_History_40', 'Medical_History_41']

CONTINUOUS = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI',
              'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6',
              'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3',
              'Family_Hist_4', 'Family_Hist_5']

DISCRETE = ['Medical_History_1', 'Medical_History_10', 'Medical_History_15',
            'Medical_History_24', 'Medical_History_32']

BOOLEANS = ['Medical_Keyword_' + str(i + 1) for i in range(48)]

def OneHot(df, colnames):
    from pandas import get_dummies, concat
    for col in colnames:
        dummies = get_dummies(df[col])
        #ndumcols = dummies.shape[1]
        dummies.rename(columns={p: col + '_' + str(i + 1)  for i, p in enumerate(dummies.columns.values)}, inplace=True)
        df = concat([df, dummies], axis=1)
        pass
    df = df.drop(colnames, axis=1)
    return df


def Kappa(y_true, y_pred, **kwargs):
    if not KAGGLE:
        from skll import kappa
        #from kappa import kappa
    return kappa(y_true, y_pred, **kwargs)


def NegQWKappaScorer(y_hat, y):
    from numpy import clip
    #MIN, MAX = (-3, 12)
    MIN, MAX = (1, 8)
    return -Kappa(clip(y, MIN, MAX), clip(y_hat, MIN, MAX),
                  weights='quadratic', min_rating=MIN, max_rating=MAX)


from sklearn.base import BaseEstimator, RegressorMixin
class PrudentialRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                objective='reg:linear',
                learning_rate=0.045,
                min_child_weight=50,
                subsample=0.8,
                colsample_bytree=0.7,
                max_depth=7,
                n_estimators=700,
                nthread=-1,
                seed=0,
                n_buckets=8,
                initial_params=[-1.5, -2.6, -3.6, -1.2, -0.8, 0.04, 0.7, 3.6,
                                #1., 2., 3., 4., 5., 6., 7.
                                ],
                minimizer='BFGS',
                scoring=NegQWKappaScorer):

        self.objective = objective
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.nthread = nthread
        self.seed = seed
        self.n_buckets = n_buckets
        self.initial_params = initial_params
        self.minimizer = minimizer
        self.scoring = scoring

        return


    def fit(self, X, y):
        from xgboost import XGBRegressor
        if not KAGGLE:
            from OptimizedOffsetRegressor import DigitizedOptimizedOffsetRegressor

        self.xgb = XGBRegressor(
                       objective=self.objective,
                       learning_rate=self.learning_rate,
                       min_child_weight=self.min_child_weight,
                       subsample=self.subsample,
                       colsample_bytree=self.colsample_bytree,
                       max_depth=self.max_depth,
                       n_estimators=self.n_estimators,
                       nthread=self.nthread,
                       missing=0.0,
                       seed=self.seed)
        #from OptimizedOffsetRegressor import FullDigitizedOptimizedOffsetRegressor
        #self.off = FullDigitizedOptimizedOffsetRegressor(n_buckets=self.n_buckets,
        #               basinhopping=True,
        self.off = DigitizedOptimizedOffsetRegressor(n_buckets=self.n_buckets,
                       initial_params=self.initial_params,
                       minimizer=self.minimizer,
                       scoring=self.scoring)

        self.xgb.fit(X, y)

        tr_y_hat = self.xgb.predict(X,
                                    ntree_limit=self.xgb.booster().best_iteration)
        print('Train score is:', -self.scoring(tr_y_hat, y))
        self.off.fit(tr_y_hat, y)
        print("Offsets:", self.off.params)

        return self


    def predict(self, X):
        from numpy import clip
        te_y_hat = self.xgb.predict(X, ntree_limit=self.xgb.booster().best_iteration)
        return clip(self.off.predict(te_y_hat), 1, 8)

    pass


class PrudentialRegressorFO(BaseEstimator, RegressorMixin):
    def __init__(self,
                objective='reg:linear',
                learning_rate=0.045,
                min_child_weight=50,
                subsample=0.8,
                colsample_bytree=0.7,
                max_depth=7,
                n_estimators=700,
                nthread=-1,
                seed=0,
                n_buckets=8,
                initial_params=[-1.5, -2.6, -3.6, -1.2, -0.8, 0.04, 0.7, 3.6,
                                #1., 2., 3., 4., 5., 6., 7.
                                ],
                minimizer='BFGS',
                scoring=NegQWKappaScorer):

        self.objective = objective
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.nthread = nthread
        self.seed = seed
        self.n_buckets = n_buckets
        self.initial_params = initial_params
        self.minimizer = minimizer
        self.scoring = scoring

        return


    def fit(self, X, y):
        from xgboost import XGBRegressor
        if not KAGGLE:
            from OptimizedOffsetRegressor import DigitizedOptimizedOffsetRegressor

        self.xgb = XGBRegressor(
                       objective=self.objective,
                       learning_rate=self.learning_rate,
                       min_child_weight=self.min_child_weight,
                       subsample=self.subsample,
                       colsample_bytree=self.colsample_bytree,
                       max_depth=self.max_depth,
                       n_estimators=self.n_estimators,
                       nthread=self.nthread,
                       missing=0.0,
                       seed=self.seed)
        from OptimizedOffsetRegressor import FullDigitizedOptimizedOffsetRegressor
        self.off = FullDigitizedOptimizedOffsetRegressor(n_buckets=self.n_buckets,
#                       basinhopping=True,
                       initial_params=self.initial_params,
                       minimizer=self.minimizer,
                       scoring=self.scoring)

        self.xgb.fit(X, y)

        tr_y_hat = self.xgb.predict(X,
                                    ntree_limit=self.xgb.booster().best_iteration)
        print('Train score is:', -self.scoring(tr_y_hat, y))
        self.off.fit(tr_y_hat, y)
        print("Offsets:", self.off.params)

        return self


    def predict(self, X):
        from numpy import clip
        te_y_hat = self.xgb.predict(X, ntree_limit=self.xgb.booster().best_iteration)
        return clip(self.off.predict(te_y_hat), 1, 8)

    pass


class PrudentialRegressorCVO(BaseEstimator, RegressorMixin):
    def __init__(self,
                objective='reg:linear',
                learning_rate=0.045,
                min_child_weight=50,
                subsample=0.8,
                colsample_bytree=0.7,
                max_depth=7,
                n_estimators=700,
                nthread=-1,
                seed=0,
                n_buckets=8,
                initial_params=[-1.5, -2.6, -3.6, -1.2, -0.8, 0.04, 0.7, 3.6,
                                #1., 2., 3., 4., 5., 6., 7.
                                ],
                minimizer='BFGS',
                scoring=NegQWKappaScorer):

        self.objective = objective
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.nthread = nthread
        self.seed = seed
        self.n_buckets = n_buckets
        self.initial_params = initial_params
        self.minimizer = minimizer
        self.scoring = scoring

        return


    def fit(self, X, y):
        from xgboost import XGBRegressor
        if not KAGGLE:
            from OptimizedOffsetRegressor import DigitizedOptimizedOffsetRegressor

        #from OptimizedOffsetRegressor import FullDigitizedOptimizedOffsetRegressor
        #self.off = FullDigitizedOptimizedOffsetRegressor(n_buckets=self.n_buckets,
        #               basinhopping=True,

        """
2 / 5
grid scores:
  mean: 0.65531, std: 0.00333, params: {'n_estimators': 700, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 240}
best score: 0.65531

3 / 5
grid scores:
  mean: 0.65474, std: 0.00308, params: {'n_estimators': 700, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 240}
best score: 0.65474

4 / 5
grid scores:
  mean: 0.65490, std: 0.00302, params: {'n_estimators': 700, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 240}
best score: 0.65490


2 / 10
grid scores:
  mean: 0.65688, std: 0.00725, params: {'n_estimators': 700, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 240}
best score: 0.65688

3 / 10
grid scores:
  mean: 0.65705, std: 0.00714, params: {'n_estimators': 700, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 240}
best score: 0.65705

4 / 10
grid scores:
  mean: 0.65643, std: 0.00715, params: {'n_estimators': 700, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 240}
best score: 0.65643

5 / 10
grid scores:
  mean: 0.65630, std: 0.00699, params: {'n_estimators': 700, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 240}
best score: 0.65630

        """
        from sklearn.cross_validation import StratifiedKFold
        kf = StratifiedKFold(y, n_folds=2)
        print(kf)
        params = []
        for itrain, itest in kf:
            ytrain = y[itrain]
            Xtrain = X.iloc[list(itrain)]
            ytest = y[itest]
            Xtest = X.iloc[list(itest)]

            self.xgb = XGBRegressor(
                           objective=self.objective,
                           learning_rate=self.learning_rate,
                           min_child_weight=self.min_child_weight,
                           subsample=self.subsample,
                           colsample_bytree=self.colsample_bytree,
                           max_depth=self.max_depth,
                           n_estimators=self.n_estimators,
                           nthread=self.nthread,
                           missing=0.0,
                           seed=self.seed)
            self.xgb.fit(Xtrain, ytrain)
            te_y_hat = self.xgb.predict(Xtest,
                                        ntree_limit=self.xgb.booster().best_iteration)
            print('XGB Test score is:', -self.scoring(te_y_hat, ytest))

            self.off = DigitizedOptimizedOffsetRegressor(n_buckets=self.n_buckets,
                           initial_params=self.initial_params,
                           minimizer=self.minimizer,
                           scoring=self.scoring)
            self.off.fit(te_y_hat, ytest)
            print("Offsets:", self.off.params)
            params += [list(self.off.params)]

            pass

        from numpy import array
        self.off.params = array(params).mean(axis=0)
        print("Mean Offsets:", self.off.params)
        self.xgb.fit(X, y)

        return self


    def predict(self, X):
        from numpy import clip
        te_y_hat = self.xgb.predict(X, ntree_limit=self.xgb.booster().best_iteration)
        return clip(self.off.predict(te_y_hat), 1, 8)

    pass


class PrudentialRegressorCVO2(BaseEstimator, RegressorMixin):
    def __init__(self,
                objective='reg:linear',
                learning_rate=0.045,
                min_child_weight=50,
                subsample=0.8,
                colsample_bytree=0.7,
                max_depth=7,
                n_estimators=700,
                nthread=-1,
                seed=0,
                n_buckets=8,
                int_fold=6,
                initial_params=[-1.5, -2.6, -3.6, -1.2, -0.8, 0.04, 0.7, 3.6,
                                #1., 2., 3., 4., 5., 6., 7.
                                ],
                minimizer='BFGS',
                scoring=NegQWKappaScorer):

        self.objective = objective
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.nthread = nthread
        self.seed = seed
        self.n_buckets = n_buckets
        self.int_fold = int_fold
        self.initial_params = initial_params
        self.minimizer = minimizer
        self.scoring = scoring

        return


    def fit(self, X, y):
        if not KAGGLE:
            from OptimizedOffsetRegressor import DigitizedOptimizedOffsetRegressor

        #from OptimizedOffsetRegressor import FullDigitizedOptimizedOffsetRegressor
        #self.off = FullDigitizedOptimizedOffsetRegressor(n_buckets=self.n_buckets,
        #               basinhopping=True,

        """
5-fold Stratified CV
grid scores:
  mean: 0.64475, std: 0.00483, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 2, 'max_depth': 6}
  mean: 0.64926, std: 0.00401, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 3, 'max_depth': 6}
  mean: 0.65281, std: 0.00384, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 4, 'max_depth': 6}
  mean: 0.65471, std: 0.00422, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 5, 'max_depth': 6}
  mean: 0.65563, std: 0.00440, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 6, 'max_depth': 6}
  mean: 0.65635, std: 0.00433, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 6}
  mean: 0.65600, std: 0.00471, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 8, 'max_depth': 6}
best score: 0.65635
best params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 6}


reversed params [8 bins]:
  mean: 0.65588, std: 0.00417, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 6, 'max_depth': 6}
  mean: 0.65640, std: 0.00438, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 6}


with Scirpus obj
grid scores:
  mean: 0.65775, std: 0.00429, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 6}
best score: 0.65775

+1 na trzech Product_info_2*
  mean: 0.65555, std: 0.00462, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 6, 'max_depth': 6}
  mean: 0.65613, std: 0.00438, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 6}

DISCRETE: NaN=most_common, +Medical_History_10,24, (24 jest znaczacy)
  mean: 0.65589, std: 0.00490, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 6}


PROPER DATA + Scirpus + reversed params + no-drops
  mean: 0.65783, std: 0.00444, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 6}

PROPER DATA + Scirpus + reversed params + no-drops, EVAL_SET@30.RMSE
  mean: 0.65790, std: 0.00421, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 6}

jak wyzej, max_depth=7
  mean: 0.65802, std: 0.00420, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 7}

jak wyzej, max_depth=10
  mean: 0.65833, std: 0.00387, params: {'colsample_bytree': 0.67, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}

jak wyzej, max_depth=10, eta=0.03
  mean: 0.65888, std: 0.00391, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}

jak wyzej, max_depth=30, eta=0.02
  mean: 0.65798, std: 0.00340, params: {'colsample_bytree': 0.67, 'learning_rate': 0.02, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 30}

jak wyzej, max_depth=10, eta=0.03, eval_metric=Scirpus
  mean: 0.65891, std: 0.00395, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}

jak wyzej, max_depth=10, eta=0.03, eval_metric=QWKappa
  mean: 0.65827, std: 0.00368, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}


jak wyzej, max_depth=10, eta=0.03, eval_metric=Scirpus, GMM6,GMM17
  mean: 0.65862, std: 0.00423, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}
jak wyzej, max_depth=10, eta=0.03, eval_metric=Scirpus, Gvector
  mean: 0.65864, std: 0.00384, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}

jak wyzej, max_depth=10, eta=0.03, eval_metric=Scirpus, learning_rates=[0.03] * 200 + [0.02] * 500,
  mean: 0.65910, std: 0.00384, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}

        """

        from sklearn.cross_validation import StratifiedKFold
        kf = StratifiedKFold(y, n_folds=self.int_fold)
        print(kf)
        self.xgb = []
        self.off = []
        for i, (itrain, itest) in enumerate(kf):
            ytrain = y[itrain]
            Xtrain = X.iloc[list(itrain)]
            ytest = y[itest]
            Xtest = X.iloc[list(itest)]

            self.xgb += [None]

            from xgb_sklearn import XGBRegressor
            #from xgboost import XGBRegressor
            self.xgb[i] = XGBRegressor(
                           objective=self.objective,
                           learning_rate=self.learning_rate,
                           min_child_weight=self.min_child_weight,
                           subsample=self.subsample,
                           colsample_bytree=self.colsample_bytree,
                           max_depth=self.max_depth,
                           n_estimators=self.n_estimators,
                           nthread=self.nthread,
                           missing=0.0,
                           seed=self.seed)
            self.xgb[i].fit(Xtrain, ytrain,
                            eval_set=[(Xtest, ytest)],
                            #eval_metric=self.scoring,
                            #eval_metric='rmse',
                            eval_metric=scirpus_error,
                            #eval_metric=qwkappa_error,
                            verbose=False,
                            early_stopping_rounds=30,
                            #learning_rates=[self.learning_rate] * 200 + [0.02] * 500,
                            obj=scirpus_regobj
                            #obj=qwkappa_regobj
                            )
            print("best iteration:", self.xgb[i].booster().best_iteration)
            te_y_hat = self.xgb[i].predict(Xtest,
                                        ntree_limit=self.xgb[i].booster().best_iteration)
            print('XGB Test score is:', -self.scoring(te_y_hat, ytest))

            self.off += [None]
            self.off[i] = DigitizedOptimizedOffsetRegressor(n_buckets=self.n_buckets,
                           initial_params=self.initial_params,
                           minimizer=self.minimizer,
                           scoring=self.scoring)
            self.off[i].fit(te_y_hat, ytest)
            print("Offsets:", self.off[i].params)
            pass

        return self


    def predict(self, X):
        from numpy import clip, array
        result = []
        for xgb, off in zip(self.xgb, self.off):
            te_y_hat = xgb.predict(X, ntree_limit=xgb.booster().best_iteration)
            result.append(off.predict(te_y_hat))
        result = clip(array(result).mean(axis=0), 1, 8)
        return result

    pass


class PrudentialRegressorCVO2FO(BaseEstimator, RegressorMixin):
    def __init__(self,
                objective='reg:linear',
                learning_rate=0.045,
                min_child_weight=50,
                subsample=0.8,
                colsample_bytree=0.7,
                max_depth=7,
                n_estimators=700,
                nthread=-1,
                seed=0,
                n_buckets=8,
                int_fold=6,
                initial_params=[-1.5, -2.6, -3.6, -1.2, -0.8, 0.04, 0.7, 3.6,
                                #1., 2., 3., 4., 5., 6., 7.
                                ],
                minimizer='BFGS',
                scoring=NegQWKappaScorer):

        self.objective = objective
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.nthread = nthread
        self.seed = seed
        self.n_buckets = n_buckets
        self.int_fold = int_fold
        self.initial_params = initial_params
        self.minimizer = minimizer
        self.scoring = scoring

#        from numpy.random import seed as random_seed
#        random_seed(seed)

        return


    def __call__(self, i, te_y_hat, ytest):
        print('XGB[{}] Test score is:'.format(i + 1), -self.scoring(te_y_hat, ytest))

        from OptimizedOffsetRegressor import FullDigitizedOptimizedOffsetRegressor
        off = FullDigitizedOptimizedOffsetRegressor(n_buckets=self.n_buckets,
                       basinhopping=True,
                       initial_params=self.initial_params,
                       minimizer=self.minimizer,
                       scoring=self.scoring)
        off.fit(te_y_hat, ytest)
        print("Offsets[{}]:".format(i + 1), off.params)

        return off


    def fit(self, X, y):
        from sklearn.cross_validation import StratifiedKFold
        kf = StratifiedKFold(y, n_folds=self.int_fold)
        print(kf)
        self.xgb = []
        self.off = []

        datamap = {i: (itrain, itest) for i, (itrain, itest) in enumerate(kf)}

        for i, (itrain, _) in datamap.items():
            ytrain = y[itrain]
            Xtrain = X.iloc[list(itrain)]
            self.xgb += [None]

            #from xgboost import XGBRegressor
            from xgb_sklearn import XGBRegressor
            self.xgb[i] = XGBRegressor(
                           objective=self.objective,
                           learning_rate=self.learning_rate,
                           min_child_weight=self.min_child_weight,
                           subsample=self.subsample,
                           colsample_bytree=self.colsample_bytree,
                           max_depth=self.max_depth,
                           n_estimators=self.n_estimators,
                           nthread=self.nthread,
                           missing=0.0 + 1e6,
                           seed=self.seed)
            self.xgb[i].fit(Xtrain, ytrain, obj=scirpus_regobj)
            pass

        from joblib import Parallel, delayed
        from sklearn.base import clone
        off = Parallel(
            n_jobs=self.nthread, verbose=2,
            #pre_dispatch='n_jobs',
        )(
            delayed(clone(self))(i,
                         self.xgb[i].predict(X.iloc[list(itest)],
                               ntree_limit=self.xgb[i].booster().best_iteration),
                         y[itest])
                         for i, (_, itest) in datamap.items())
        self.off = off
        return self


    def predict(self, X):
        from numpy import clip, array
        result = []
        for xgb, off in zip(self.xgb, self.off):
            te_y_hat = xgb.predict(X, ntree_limit=xgb.booster().best_iteration)
            result.append(off.predict(te_y_hat))
        result = clip(array(result).mean(axis=0), 1, 8)
        return result

    pass


def scirpus_regobj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    from numpy import exp as npexp
    grad = 2 * x * npexp(-(x ** 2)) * (npexp(x ** 2) + x ** 2 + 1)
    hess = 2 * npexp(-(x ** 2)) * (npexp(x ** 2) - 2 * (x ** 4) + 5 * (x ** 2) - 1)
    return grad, hess


def scirpus_error(preds, dtrain):
    labels = dtrain.get_label()
    x = (labels - preds)
    from numpy import exp as npexp
    error = (x ** 2) * (1 - npexp(-(x ** 2)))
    from numpy import mean
    return 'error', mean(error)


def qwkappa_error(preds, dtrain):
    labels = dtrain.get_label()
    kappa = NegQWKappaScorer(labels, preds)
    return 'kappa', kappa


def qwkappa_regobj(preds, dtrain):
    labels = dtrain.get_label()

    work = preds.copy()
    from numpy import empty_like
    grad = empty_like(preds)
    for i in range(len(preds)):
        work[i] += 1
        score = NegQWKappaScorer(labels, work)
        work[i] -= 2
        grad[i] = (score - NegQWKappaScorer(labels, work)) / 2.
        work[i] += 1
        pass

    from numpy import ones
    hess = ones(len(preds)) / len(preds)
    return grad, hess


def work(out_csv_file,
         estimator,
         nest,
         njobs,
         nfolds,
         cv_grid,
         minimizer,
         nbuckets,
         mvector,
         imputer,
         clf_kwargs,
         int_fold):

    from numpy.random import seed as random_seed
    random_seed(1)


    from zipfile import ZipFile
    from pandas import read_csv,factorize
    from numpy import rint,clip,savetxt,stack

    if KAGGLE:
        train = read_csv("../input/train.csv")
        test = read_csv("../input/test.csv")
    else:
        train = read_csv(ZipFile("../../data/train.csv.zip", 'r').open('train.csv'))
        test = read_csv(ZipFile("../../data/test.csv.zip", 'r').open('test.csv'))

#    gmm17_train = read_csv('GMM_17_full_train.csv')
#    gmm17_test = read_csv('GMM_17_full_test.csv')
#    gmm6_train = read_csv('GMM_6_full_train.csv')
#    gmm6_test = read_csv('GMM_6_full_test.csv')
#
#    train['GMM17'] = gmm17_train['Response']
#    test['GMM17'] = gmm17_test['Response']
#    train['GMM6'] = gmm6_train['Response']
#    test['GMM6'] = gmm6_test['Response']

    # combine train and test
    all_data = train.append(test)

#    G_vectors = read_csv('../../data/G_vectors.csv')
#    #all_data = all_data.join(G_vectors.drop(['G3'], axis=1))
#    all_data = all_data.join(
#        G_vectors[['G8', 'G11', 'G12', 'G13', 'G17', 'G18', 'G19', 'G20']])

    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    all_data[DISCRETE] = imp.fit_transform(all_data[DISCRETE])
#    from numpy import bincount
#    for col in all_data[DISCRETE]:
#        top = bincount(all_data[col].astype(int)).argmax()
#        all_data[col] -= top
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    all_data[CONTINUOUS] = imp.fit_transform(all_data[CONTINUOUS])
#    all_data[BOOLEANS] = all_data[BOOLEANS] + 1e6


#    from sklearn.preprocessing import StandardScaler
#    from sklearn.decomposition import PCA
#    std = StandardScaler(copy=True)
#    all_data[CONTINUOUS] = std.fit_transform(all_data[CONTINUOUS])
#    pca = PCA(whiten=False, copy=True)
#    all_data[CONTINUOUS] = pca.fit_transform(all_data[CONTINUOUS])


    # create any new variables
    all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
    all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]


    # factorize categorical variables
    all_data['Product_Info_2'] = factorize(all_data['Product_Info_2'])[0]# + 1
    all_data['Product_Info_2_char'] = factorize(all_data['Product_Info_2_char'])[0]# + 1
    all_data['Product_Info_2_num'] = factorize(all_data['Product_Info_2_num'])[0]# + 1

    """
    Both: 0.65576
    BmiAge: 0.65578
    MedCount: 0.65638
    None: 0.65529
    """
    all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']
    med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
    all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)


    """
    print('BOOLEANS:')
    for col in all_data[BOOLEANS]:
        print(col, all_data[col].dtype, min(all_data[col]), max(all_data[col]), float(sum(all_data[col] == 0)) / len(all_data[col]))
    print('DISCRETE:')
    for col in all_data[DISCRETE]:
        print(col, all_data[col].dtype, min(all_data[col]), max(all_data[col]), float(sum(all_data[col] == 0)) / len(all_data[col]))
    print('CONTINUOUS:')
    for col in all_data[CONTINUOUS]:
        print(col, all_data[col].dtype, min(all_data[col]), max(all_data[col]), float(sum(all_data[col] == 0)) / len(all_data[col]))
    print('NOMINALS:')
    for col in all_data[NOMINALS]:
        print(col, all_data[col].dtype, min(all_data[col]), max(all_data[col]), float(sum(all_data[col] == 0)) / len(all_data[col]))
    return
    """

    # Use -1 for any others
    if imputer is None:
        all_data.fillna(-1, inplace=True)
    else:
        all_data['Response'].fillna(-1, inplace=True)

    # fix the dtype on the label column
    all_data['Response'] = all_data['Response'].astype(int)

    # split train and test
    train = all_data[all_data['Response'] > 0].copy()
    test = all_data[all_data['Response'] < 1].copy()

    #dropped_cols = ['Id', 'Response', 'Medical_History_10', 'Medical_History_24']#, 'Medical_History_32']
    dropped_cols = ['Id', 'Response']

    train_y = train['Response'].values
    train_X = train.drop(dropped_cols, axis=1)
    test_X = test.drop(dropped_cols, axis=1)

    if imputer is not None:
        from sklearn.preprocessing import Imputer
        imp = Imputer(missing_values='NaN', strategy=imputer, axis=0)
        train_X = imp.fit_transform(train_X)
        test_X = imp.transform(test_X)

    prudential_kwargs = \
    {
        'objective': 'reg:linear',
        'learning_rate': 0.045,
        'min_child_weight': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'max_depth': 7,
        'n_estimators': nest,
        'nthread': njobs,
        'seed': 0,
        'n_buckets': nbuckets,
        'initial_params': mvector,
        'minimizer': minimizer,
        'scoring': NegQWKappaScorer
    }
    if estimator == 'PrudentialRegressorCVO2FO' or estimator == 'PrudentialRegressorCVO2':
        prudential_kwargs['int_fold'] = int_fold
        pass

    # override kwargs with any changes
    for k, v in clf_kwargs.items():
        prudential_kwargs[k] = v
    clf = globals()[estimator](**prudential_kwargs)
    print(estimator, clf.get_params())

    if nfolds > 1:
        param_grid = {
                    'n_estimators': [700],
                    'max_depth': [6],
                    'colsample_bytree': [0.67],
                    'subsample': [0.9],
                    'min_child_weight': [240],
                    #'initial_params': [[-0.71238755, -1.4970176, -1.73800531, -1.13361266, -0.82986203, -0.06473039, 0.69008725, 0.94815881]]
                    }
        for k, v in cv_grid.items():
            param_grid[k] = v

        from sklearn.metrics import make_scorer
        MIN, MAX = (1, 8)
        qwkappa = make_scorer(Kappa, weights='quadratic',
                              min_rating=MIN, max_rating=MAX)

        from sklearn.cross_validation import StratifiedKFold
        from sklearn.grid_search import GridSearchCV
        grid = GridSearchCV(estimator=clf,
                            param_grid=param_grid,
                            cv=StratifiedKFold(train_y, n_folds=nfolds),
                            scoring=qwkappa, n_jobs=1,
                            verbose=1,
                            refit=False)
        grid.fit(train_X, train_y)
        print('grid scores:')
        for item in grid.grid_scores_:
            print('  {:s}'.format(item))
        print('best score: {:.5f}'.format(grid.best_score_))
        print('best params:', grid.best_params_)

        pass

    else:
        clf.fit(train_X, train_y)


        final_test_preds = clf.predict(test_X)
        final_test_preds = rint(clip(final_test_preds, 1, 8))

        savetxt(out_csv_file,
                stack(zip(test['Id'].values, final_test_preds), axis=1).T,
                delimiter=',',
                fmt=['%d', '%d'],
                header='"Id","Response"', comments='')

        importance = clf.xgb.booster().get_fscore()
        import operator
        print(sorted(importance.items()), "\n")
        importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
        print(importance, "\n")
        features = [k for k, _ in importance]
        print(len(features), features)

    return



def main(argv=None): # IGNORE:C0111
    '''Command line options.'''
    from sys import argv as Argv

    if argv is None:
        argv = Argv
        pass
    else:
        Argv.extend(argv)
        pass

    from os.path import basename
    program_name = basename(Argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    try:
        program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    except:
        program_shortdesc = __import__('__main__').__doc__
    program_license = '''%s

  Created by Wojciech Migda on %s.
  Copyright 2016 Wojciech Migda. All rights reserved.

  Licensed under the MIT License

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        from argparse import ArgumentParser
        from argparse import RawDescriptionHelpFormatter
        from argparse import FileType
        from sys import stdout

        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)

        parser.add_argument("-n", "--num-est",
            type=int, default=700, action='store', dest="nest",
            help="number of Random Forest estimators")

        parser.add_argument("-j", "--jobs",
            type=int, default=-1, action='store', dest="njobs",
            help="number of jobs")

        parser.add_argument("-f", "--cv-fold",
            type=int, default=0, action='store', dest="nfolds",
            help="number of cross-validation folds")

        parser.add_argument("--int-fold",
            type=int, default=6, action='store', dest="int_fold",
            help="internal fold for PrudentialRegressorCVO2FO")

        parser.add_argument("-b", "--n-buckets",
            type=int, default=8, action='store', dest="nbuckets",
            help="number of buckets for digitizer")

        parser.add_argument("-o", "--out-csv",
            action='store', dest="out_csv_file", default=stdout,
            type=FileType('w'),
            help="output CSV file name")

        parser.add_argument("-m", "--minimizer",
            action='store', dest="minimizer", default='BFGS',
            type=str, choices=['Powell', 'CG', 'BFGS'],
            help="minimizer method for scipy.optimize.minimize")

        parser.add_argument("-M", "--mvector",
            action='store', dest="mvector", default=[-1.5, -2.6, -3.6, -1.2, -0.8, 0.04, 0.7, 3.6],
            type=float, nargs='*',
            help="minimizer's initial params vector")

        parser.add_argument("-I", "--imputer",
            action='store', dest="imputer", default=None,
            type=str, choices=['mean', 'median', 'most_frequent'],
            help="Imputer strategy, None is -1")

        parser.add_argument("--clf-params",
            type=str, default="{}", action='store', dest="clf_params",
            help="classifier parameters subset to override defaults")

        parser.add_argument("-G", "--cv-grid",
            type=str, default="{}", action='store', dest="cv_grid",
            help="cross-validation grid params (used if NFOLDS > 0)")

        parser.add_argument("-E", "--estimator",
            action='store', dest="estimator", default='PrudentialRegressor',
            type=str,# choices=['mean', 'median', 'most_frequent'],
            help="Estimator class to use")

        # Process arguments
        args = parser.parse_args()

        for k, v in args.__dict__.items():
            print(str(k) + ' => ' + str(v))
            pass

        work(args.out_csv_file,
             args.estimator,
             args.nest,
             args.njobs,
             args.nfolds,
             eval(args.cv_grid),
             args.minimizer,
             args.nbuckets,
             args.mvector,
             args.imputer,
             eval(args.clf_params),
             args.int_fold)


        return 0
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception as e:
        if DEBUG:
            raise(e)
            pass
        indent = len(program_name) * " "
        from sys import stderr
        stderr.write(program_name + ": " + repr(e) + "\n")
        stderr.write(indent + "  for help use --help")
        return 2

    pass


if __name__ == "__main__":
    if DEBUG:
        from sys import argv
        argv.append("-n 700")
        argv.append("--minimizer=Powell")
        argv.append("--clf-params={'learning_rate': 0.05, 'min_child_weight': 240, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'initial_params': [0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1]}")
        argv.append("-f 10")
        pass
    from sys import exit as Exit
    Exit(main())
    pass
