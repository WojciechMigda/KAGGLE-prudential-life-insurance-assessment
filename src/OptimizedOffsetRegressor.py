#!/usr/bin/python
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
#  Filename: OptimizedOffsetRegressor.py
#
#  Decription:
#      Regressor implementing optimized ofsets in a scikit-learn fashion.
#      Based on scripts on Kaggle
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
#  2016-01-23   wm              Initial version
#
################################################################################
"""

from __future__ import print_function

__author__ = 'Wojciech Migda'
__date__ = '2016-01-23'
__version__ = '0.0.1'
__all__ = [
    'OptimizedOffsetRegressor'
]


from sklearn.base import BaseEstimator, RegressorMixin


class OptimizedOffsetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_jobs=-1, offset_scale=1.0, n_buckets=2, initial_offsets=None, scoring='accuracy'):
        self.n_jobs = int(n_jobs)
        self.offset_scale = float(offset_scale)
        self.n_buckets = int(n_buckets)
        if initial_offsets is None:
            self.initial_offsets_ = [-0.5] * self.n_buckets
            pass
        else:
            self.initial_offsets_ = list(initial_offsets)
            assert(len(self.initial_offsets_) == self.n_buckets)
            pass
        from sklearn.metrics import get_scorer
        self.scoring = get_scorer(scoring)
        pass


    def __call__(self, args):
        return self.OffsetMinimizer_(args)


    def apply_offset(self, data, bin_offset, sv):
        mask = data[0].astype(int) == sv
        data[1, mask] = data[0, mask] + bin_offset
        return data

    def OffsetMinimizer_(self, args):
        def apply_offset_and_score(data, bin_offset, sv):
            data = self.apply_offset(data, bin_offset, sv)
            return self.scoring(data[1], data[2])

        j, data, offset0 = args
        from scipy.optimize import fmin_powell
        return fmin_powell(lambda x: apply_offset_and_score(data, x, j), offset0, disp=True)


    def fit(self, X, y):
        from multiprocessing import Pool
        pool = Pool(processes=None if self.n_jobs is -1 else self.n_jobs)

        from numpy import vstack
        self.data_ = vstack((X, X, y))
        for j in range(self.n_buckets):
            self.data_ = self.apply_offset(self.data_, self.initial_offsets_[j], j)

        from numpy import array
        self.offsets_ = array(pool.map(self,
                                       zip(range(self.n_buckets),
                                           [self.data_] * self.n_buckets,
                                           self.initial_offsets_)))
#        self.offsets_ = array(map(self,
#                                       zip(range(self.n_buckets),
#                                           [self.data_] * self.n_buckets,
#                                           self.initial_offsets_)))
        return self


    def predict(self, X):
        from numpy import vstack
        data = vstack((X, X))
        for j in range(self.n_buckets):
            data = self.apply_offset(data, self.offsets_[j], j)
        return data[1]

    pass


class DigitizedOptimizedOffsetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 n_jobs=-1,
                 offset_scale=1.0,
                 n_buckets=2,
                 initial_params=None,
                 scoring='accuracy'):

        from numpy import array

        self.n_jobs = int(n_jobs)
        self.offset_scale = float(offset_scale)
        self.n_buckets = int(n_buckets)
        if initial_params is None:
            #self.initial_offsets_ = [-0.5] * self.n_buckets
            pass
        else:
            self.params = array(initial_params)
            #assert(len(self.initial_offsets_) == self.n_buckets)
            pass
        from sklearn.metrics import get_scorer
        self.scoring = get_scorer(scoring)
        pass


    def apply_params(self, params, data):
        from numpy import digitize
        offsets = params[:self.n_buckets]

        # both give #40: 0.67261
        splits = [1., 2., 3., 4., 5., 6., 7.]
        response = digitize(data[0], splits)
        #splits = [2., 3., 4., 5., 6., 7., 8.]
        #response = digitize(data[0], splits) + 1

        for i, off in enumerate(offsets):
            mask = response == i
            data[1, mask] = data[0, mask] + offsets[i]

        return data


    def apply_params_and_score(self, params, data):
        data = self.apply_params(params, data)
        return self.scoring(data[1], data[2])

    def fit(self, X, y):
        from numpy import vstack
        data = vstack((X, X, y))

        from scipy.optimize import fmin_powell
        self.params = fmin_powell(
            self.apply_params_and_score,
            self.params,
            args=(data,)
            )
        return self


    def predict(self, X):
        from numpy import vstack
        data = vstack((X, X))
        params = self.params.copy()
        params[:self.n_buckets] = self.offset_scale * params[:self.n_buckets]
        data = self.apply_params(params, data)
        return data[1]

    pass


class FullDigitizedOptimizedOffsetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 n_jobs=-1,
                 offset_scale=1.0,
                 n_buckets=2,
                 initial_params=None,
                 scoring='accuracy'):

        from numpy import array

        self.n_jobs = int(n_jobs)
        self.offset_scale = float(offset_scale)
        self.n_buckets = int(n_buckets)
        if initial_params is None:
            #self.initial_offsets_ = [-0.5] * self.n_buckets
            pass
        else:
            self.params = array(initial_params)
            #assert(len(self.initial_offsets_) == self.n_buckets)
            pass
        from sklearn.metrics import get_scorer
        self.scoring = get_scorer(scoring)
        pass


    def apply_params(self, params, data):
        from numpy import digitize
        offsets = params[:self.n_buckets]

        splits = sorted(list(params[self.n_buckets:2 * self.n_buckets - 1]))
        response = digitize(data[0], splits)

        for i, off in enumerate(offsets):
            mask = response == i
            data[1, mask] = data[0, mask] + offsets[i]

        return data


    def apply_params_and_score(self, params, data):
        data = self.apply_params(params, data)
        return self.scoring(data[1], data[2])

    def fit(self, X, y):
        from numpy import vstack
        data = vstack((X, X, y))

        from scipy.optimize import fmin_powell
        self.params = fmin_powell(
            self.apply_params_and_score,
            self.params,
            args=(data,)
            )
        return self


    def predict(self, X):
        from numpy import vstack
        data = vstack((X, X))
        params = self.params.copy()
        params[:self.n_buckets] = self.offset_scale * params[:self.n_buckets]
        data = self.apply_params(params, data)
        return data[1]

    pass


if __name__ == "__main__":
    pass
