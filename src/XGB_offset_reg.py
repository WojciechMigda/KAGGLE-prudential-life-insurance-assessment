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
  mean: 0.64539, std: 0.00389, params: {'n_estimators': 700, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 240}
best score: 0.64539

3 / 5
grid scores:
  mean: 0.65007, std: 0.00436, params: {'n_estimators': 700, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 240}
best score: 0.65007

4 / 5
grid scores:
  mean: 0.65336, std: 0.00361, params: {'n_estimators': 700, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 240}
best score: 0.65336

5 / 5
grid scores:
  mean: 0.65588, std: 0.00378, params: {'n_estimators': 700, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 240}
best score: 0.65588

6 / 5
grid scores:
  mean: 0.65657, std: 0.00316, params: {'n_estimators': 700, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 240}
best score: 0.65657

7 / 5
grid scores:
  mean: 0.65622, std: 0.00296, params: {'n_estimators': 700, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 240}
best score: 0.65622


8 / 5
grid scores:
  mean: 0.65601, std: 0.00372, params: {'n_estimators': 700, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 240}
best score: 0.65601

        """

        from sklearn.cross_validation import StratifiedKFold
        kf = StratifiedKFold(y, n_folds=6)
        print(kf)
        self.xgb = []
        self.off = []
        for i, (itrain, itest) in enumerate(kf):
            ytrain = y[itrain]
            Xtrain = X.iloc[list(itrain)]
            ytest = y[itest]
            Xtest = X.iloc[list(itest)]

            self.xgb += [None]
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
            self.xgb[i].fit(Xtrain, ytrain)
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


def work(out_csv_file,
         estimator,
         nest,
         njobs,
         nfolds,
         minimizer,
         mvector,
         imputer,
         clf_kwargs):


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

#    from sklearn.preprocessing import Imputer
#    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
#    all_data[DISCRETE] = imp.fit_transform(all_data[DISCRETE])
#    from numpy import bincount
#    for col in all_data[DISCRETE]:
#        top = bincount(all_data[col].astype(int)).argmax()
#        all_data[col] -= top
#    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
#    all_data[CONTINUOUS] = imp.fit_transform(all_data[CONTINUOUS])
#    all_data[BOOLEANS] = all_data[BOOLEANS] + 1e6

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

    #all_data = OneHot(all_data, ['Employment_Info_2', 'Employment_Info_3'])
    #all_data = all_data.drop(ranked_features[100:], axis=1)


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

    dropped_cols = ['Id', 'Response', 'Medical_History_10', 'Medical_History_24']#, 'Medical_History_32']
#    dropped_cols = ['Id', 'Response']

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
        'n_buckets': 8,
        'initial_params': mvector,
        'minimizer': minimizer,
        'scoring': NegQWKappaScorer
    }
    # override kwargs with any changes
    for k, v in clf_kwargs.items():
        prudential_kwargs[k] = v
    clf = globals()[estimator](**prudential_kwargs)
    print(estimator, clf.get_params())

    if nfolds > 1:
        param_grid={
                    'n_estimators': [700],
                    'max_depth': [6],
                    'colsample_bytree': [0.67],
                    'subsample': [0.9],
                    'min_child_weight': [240],
                    #'initial_params': [[-0.71238755, -1.4970176, -1.73800531, -1.13361266, -0.82986203, -0.06473039, 0.69008725, 0.94815881]]
                    }
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
             args.minimizer,
             args.mvector,
             args.imputer,
             eval(args.clf_params))


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
