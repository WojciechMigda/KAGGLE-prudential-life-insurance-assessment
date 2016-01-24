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
__all__ = []
__version__ = "0.0.1"
__date__ = '2016-01-22'
__updated__ = '2016-01-22'


def Kappa(y_true, y_pred, **kwargs):
    from numpy import clip
    from skll import kappa
    return kappa(clip(y_true, 1, 8), clip(y_pred, 1, 8), **kwargs)


def NegQWKappaScorer(y_hat, y):
    return -Kappa(y, y_hat, weights='quadratic')


def work(out_csv_file,
         nest,
         njobs):


    from zipfile import ZipFile
    from pandas import read_csv,factorize
    from numpy import rint,clip,savetxt,stack


    train = read_csv(ZipFile("../../data/train.csv.zip", 'r').open('train.csv'))
    test = read_csv(ZipFile("../../data/test.csv.zip", 'r').open('test.csv'))

    gmm17_train = read_csv('GMM_17_full_train.csv')
    gmm17_test = read_csv('GMM_17_full_test.csv')
    gmm6_train = read_csv('GMM_6_full_train.csv')
    gmm6_test = read_csv('GMM_6_full_test.csv')


    train['GMM17'] = gmm17_train['Response']
    test['GMM17'] = gmm17_test['Response']
    train['GMM6'] = gmm6_train['Response']
    test['GMM6'] = gmm6_test['Response']

    # combine train and test
    all_data = train.append(test)

    # create any new variables
    all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
    all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]

    # factorize categorical variables
    all_data['Product_Info_2'] = factorize(all_data['Product_Info_2'])[0]
    all_data['Product_Info_2_char'] = factorize(all_data['Product_Info_2_char'])[0]
    all_data['Product_Info_2_num'] = factorize(all_data['Product_Info_2_num'])[0]


    print('Eliminate missing values')
    # Use -1 for any others
    all_data.fillna(-1, inplace=True)

    # fix the dtype on the label column
    all_data['Response'] = all_data['Response'].astype(int)

    # Provide split column
    #from numpy.random import randint
    #all_data['Split'] = randint(5, size=all_data.shape[0])

    # split train and test
    train = all_data[all_data['Response'] > 0].copy()
    test = all_data[all_data['Response'] < 1].copy()


    train_y = train['Response'].as_matrix()
#    train_X = train.drop(['Id', 'Response', 'Medical_History_1'], axis=1).as_matrix()
#    test_X = test.drop(['Id', 'Response', 'Medical_History_1'], axis=1).as_matrix()
    train_X = train.drop(['Id', 'Response'], axis=1).as_matrix()
    test_X = test.drop(['Id', 'Response'], axis=1).as_matrix()


    from sklearn.base import BaseEstimator, RegressorMixin
    class PrudentialRegressor(BaseEstimator, RegressorMixin):
        def __init__(self,
                    max_depth=7,
                    n_estimators=50,
                    n_jobs=-1,
                    random_state=1,
                    max_features=1.0,
                    min_samples_leaf=1.0,
                    verbose=1,
                    n_buckets=8,
                    initial_offsets=[-1.5, -2.6, -3.6, -1.2, -0.8, 0.04, 0.7, 3.6],
                    scoring=NegQWKappaScorer):

            self.max_depth = max_depth
            self.n_estimators = n_estimators
            self.n_jobs = n_jobs
            self.random_state = random_state
            self.max_features = max_features
            self.min_samples_leaf = min_samples_leaf
            self.verbose=verbose

            self.n_buckets = n_buckets
            self.initial_offsets = initial_offsets
            self.scoring = scoring

            return


        def epsilon(self, Tp):
            from numpy import finfo
            return finfo(Tp).eps


        def clip(self, arr):
            from numpy import clip
            return clip(arr, 0., self.n_buckets * (1. - self.epsilon(arr.dtype)))


        def fit(self, X, y):
            from sklearn.ensemble import RandomForestRegressor
            from OptimizedOffsetRegressor import OptimizedOffsetRegressor

            self.rfr = RandomForestRegressor(
                           max_depth=self.max_depth,
                           n_estimators=self.n_estimators,
                           n_jobs=self.n_jobs,
                           random_state=self.random_state,
                           max_features=self.max_features,
                           min_samples_leaf=self.min_samples_leaf,
                           verbose=self.verbose)
            self.off = OptimizedOffsetRegressor(n_buckets=self.n_buckets,
                           initial_offsets=self.initial_offsets,
                           scoring=self.scoring)

            self.rfr.fit(X, y)
            tr_y_hat = self.clip(self.rfr.predict(X))
            print('Train score is:', -self.scoring(tr_y_hat, y))
            self.off.fit(tr_y_hat, y)
            print("Offsets:", self.off.offsets_)
            return self


        def predict(self, X):
            te_y_hat = self.clip(self.rfr.predict(X))
            return self.off.predict(te_y_hat)

        pass

    clf = PrudentialRegressor(
        max_depth=7,
        n_estimators=nest,
        n_jobs=njobs,
        random_state=1,
        max_features=1.0,
        min_samples_leaf=1.0,
        verbose=1,
        n_buckets=8,
        initial_offsets=[-1.5, -2.6, -3.6, -1.2, -0.8, 0.04, 0.7, 3.6],
        scoring=NegQWKappaScorer)


    CrossVal=False
    CrossVal=True
    if CrossVal:
        param_grid={
                    'n_estimators': [100],
                    'max_depth': [15, 20, 30, 40],
                    }
        from sklearn.metrics import make_scorer
        qwkappa = make_scorer(Kappa, weights='quadratic')
        from sklearn.grid_search import GridSearchCV
        grid = GridSearchCV(estimator=clf,
                            param_grid=param_grid,
                            cv=3, scoring=qwkappa, n_jobs=1,
                            verbose=1,
                            refit=False)
        grid.fit(train_X, train_y)
        print('grid scores:')
        for item in grid.grid_scores_:
            print('  {:s}'.format(item))
        print('best score: {:.5f}'.format(grid.best_score_))
        print('best params:', grid.best_params_)

        """
CV=3
grid scores:
  mean: 0.59014, std: 0.00513, params: {'n_estimators': 60, 'max_depth': 7}
  mean: 0.58991, std: 0.00550, params: {'n_estimators': 100, 'max_depth': 7}
  mean: 0.60015, std: 0.00507, params: {'n_estimators': 60, 'max_depth': 8}
  mean: 0.60093, std: 0.00523, params: {'n_estimators': 100, 'max_depth': 8}
  mean: 0.61423, std: 0.00421, params: {'n_estimators': 60, 'max_depth': 10}
  mean: 0.61451, std: 0.00435, params: {'n_estimators': 100, 'max_depth': 10}
best score: 0.61451
best params: {'n_estimators': 100, 'max_depth': 10}

grid scores:
  mean: 0.62720, std: 0.00369, params: {'n_estimators': 100, 'max_depth': 15}
  mean: 0.62954, std: 0.00357, params: {'n_estimators': 100, 'max_depth': 20}
  mean: 0.62450, std: 0.00335, params: {'n_estimators': 100, 'max_depth': 30}
  mean: 0.61914, std: 0.00270, params: {'n_estimators': 100, 'max_depth': 40}
best score: 0.62954
best params: {'n_estimators': 100, 'max_depth': 20}

        """
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
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
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
        from sys import stdout,stdin

        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)

        parser.add_argument("-n", "--num-est",
            type=int, default=50, action='store', dest="nest",
            help="number of Random Forest estimators")

        parser.add_argument("-j", "--jobs",
            type=int, default=-1, action='store', dest="njobs",
            help="number of jobs")

        parser.add_argument("-o", "--out-csv",
            action='store', dest="out_csv_file", default=stdout,
            type=FileType('w'),
            help="output CSV file name")

        """
        parser.add_argument("--in-test-csv",
            action='store', dest="in_test_csv", default='test.csv',
            type=str,
            help="input CSV with test data zipped inside IN_TEST_ARCH")

        parser.add_argument("-o", "--out-h5",
            action='store', dest="out_h5", default='raw-data.h5',
            type=str,
            help="output HDF5 filename for data")

        """

        # Process arguments
        args = parser.parse_args()

        for k, v in args.__dict__.items():
            print(str(k) + ' => ' + str(v))
            pass


        work(args.out_csv_file,
             args.nest,
             args.njobs)


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
        argv.append("-h")
        pass
    from sys import exit as Exit
    Exit(main())
    pass
