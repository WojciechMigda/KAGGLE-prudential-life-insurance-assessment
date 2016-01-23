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


#def OffsetMinimizer(args):
#    def apply_offset(data, bin_offset, sv, scorer=lambda y_hat, y: -Kappa(y, y_hat, weights='quadratic') ):
#        # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
#        data[1, data[0].astype(int) == sv] = data[0, data[0].astype(int) == sv] + bin_offset
#        return scorer(data[1], data[2])
#
#    from scipy.optimize import fmin_powell
#    j, data, offset0 = args
#    return fmin_powell(lambda x: apply_offset(data, x, j), offset0)


from sklearn.base import BaseEstimator, RegressorMixin
class OptimizedOffsetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_jobs=-1, offset_scale=1.0, n_buckets=2, initial_offsets=None):
        self.n_jobs = int(n_jobs)
        self.offset_scale = float(offset_scale)
        self.n_buckets = int(n_buckets)
        if initial_offsets is None:
            self.initial_offsets_ = [-0.5] * self.n_buckets
        else:
            self.initial_offsets_ = list(initial_offsets)
            assert(len(self.initial_offsets_) == self.n_buckets)
        pass


    def __call__(self, args):
        return self.OffsetMinimizer(args)


    def Kappa(self, y_true, y_pred, **kwargs):
        from numpy import clip
        from skll import kappa
        return kappa(clip(y_true, 1, 8), clip(y_pred, 1, 8), **kwargs)


    def OffsetMinimizer(self, args):
        def apply_offset(data, bin_offset, sv, scorer=lambda y_hat, y: -self.Kappa(y, y_hat, weights='quadratic') ):
            data[1, data[0].astype(int) == sv] = data[0, data[0].astype(int) == sv] + bin_offset
            return scorer(data[1], data[2])

        from scipy.optimize import fmin_powell
        j, data, offset0 = args
        return fmin_powell(lambda x: apply_offset(data, x, j), offset0)


    def fit(self, X, y):
        from multiprocessing import Pool
        pool = Pool(processes=None if self.n_jobs is -1 else self.n_jobs)

        from numpy import vstack
        self.data_  = vstack((X, X, y))
        for j in range(self.n_buckets):
            self.data_[1, self.data_[0].astype(int) == j] = self.data_[0, self.data_[0].astype(int) == j] + self.initial_offsets_[j]

        from numpy import array
        self.offsets_ = array(pool.map(self,
                                       zip(range(self.n_buckets),
                                           [self.data_] * self.n_buckets,
                                           self.initial_offsets_)))
        return self


    def predict(self, X):
        from numpy import vstack
        data = vstack((X, X))
        for j in range(self.n_buckets):
            data[1, data[0].astype(int) == j] = data[0, data[0].astype(int) == j] + self.offsets_[j]
        return data[1]

    pass


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


    from xgboost import XGBRegressor
    f_hat = XGBRegressor(
        objective='reg:linear',
        learning_rate=0.045,
        min_child_weight=50,
        subsample=0.8,
        colsample_bytree=0.7,
        max_depth=7,
        n_estimators=nest,
        nthread=njobs,
        seed=1)
    f_hat.fit(train_X, train_y)

    tr_y_hat = f_hat.predict(train_X)
    te_y_hat = f_hat.predict(test_X)
    print('Train score is:', Kappa(train_y, tr_y_hat, weights='quadratic'))

    off = OptimizedOffsetRegressor(n_buckets=8, initial_offsets=[-1.5, -2.6, -3.6, -1.2, -0.8, 0.04, 0.7, 3.6])
    off.fit(tr_y_hat, train_y)
    print("Offsets:", off.offsets_)

    final_test_preds = off.predict(te_y_hat)
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
            type=int, default=700, action='store', dest="nest",
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
