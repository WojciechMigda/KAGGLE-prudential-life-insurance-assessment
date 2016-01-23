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


    def OffsetMinimizer_(self, args):
        def apply_offset(data, bin_offset, sv):
            data[1, data[0].astype(int) == sv] = data[0, data[0].astype(int) == sv] + bin_offset
            return self.scoring(data[1], data[2])

        j, data, offset0 = args
        from scipy.optimize import fmin_powell
        return fmin_powell(lambda x: apply_offset(data, x, j), offset0)


    def fit(self, X, y):
        from multiprocessing import Pool
        pool = Pool(processes=None if self.n_jobs is -1 else self.n_jobs)

        from numpy import vstack
        self.data_ = vstack((X, X, y))
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


if __name__ == "__main__":
    pass
