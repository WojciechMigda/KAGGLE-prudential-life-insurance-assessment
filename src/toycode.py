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
#  Filename: toycode.py
#
#  Decription:
#      Toycode
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
#  2016-01-02   wm              Initial version
#  2016-01-09   wm              Playing with data
#
################################################################################
"""

from __future__ import print_function


DEBUG = False
__all__ = []
__version__ = "0.0.2"
__date__ = '2016-01-02'
__updated__ = '2016-01-09'

from sys import path as sys_path
sys_path.insert(0, './mlpipes')
sys_path.insert(0, './Pipe')
import pipe as P


def work():

    from h5pipes import h5open
    from pypipes import getitem,as_key
    from nppipes import as_array
    from skll import kappa

    data = (
        ('raw-data.h5',)
        | h5open
        | as_key('file')
        | as_key('train_X', lambda d:
            (d['file'],)
            | getitem('train_X')
            | as_array
            | P.first
            )
        | as_key('train_y', lambda d:
            (d['file'],)
            | getitem('train_y')
            | as_array
            | P.first
            )
        | as_key('test_X', lambda d:
            (d['file'],)
            | getitem('test_X')
            | as_array
            | P.first
            )
        | as_key('train_labels', lambda d:
            (d['file'],)
            | getitem('train_labels')
            | as_array
            | P.first
            )
        | as_key('test_labels', lambda d:
            (d['file'],)
            | getitem('test_labels')
            | as_array
            | P.first
            )

        | P.first
    )


    nominal_cidx = [0, 1, 2, 4, 5, 6, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23,
                 24, 25, 26, 27, 29, 30, 31, 32, 38, 39, 40, 41, 42, 43, 44, 45,
                 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59,
                 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77]


    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(categorical_features=nominal_cidx, sparse=False)
    data['train_X'] = enc.fit_transform(data['train_X'])
    data['test_X'] = enc.transform(data['test_X'])


    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    data['train_X'] = ss.fit_transform(data['train_X'])
    data['test_X'] = ss.transform(data['test_X'])



    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=1, n_estimators=10, n_jobs=1)
    rfc = RandomForestClassifier(random_state=1, n_jobs=3)

    #from sklearn.ensemble import GradientBoostingClassifier
    #clf = GradientBoostingClassifier(n_estimators=10)
    #from sklearn.ensemble import AdaBoostClassifier
    #clf = AdaBoostClassifier(rfc, n_estimators=30, random_state=1)
    #from sklearn.ensemble import ExtraTreesClassifier
    #clf = ExtraTreesClassifier(n_jobs=3, n_estimators=50, random_state=1)

    from sklearn.metrics import make_scorer
    qwkappa = make_scorer(kappa, weights='quadratic')
#    from sklearn.cross_validation import cross_val_score
#    scores = cross_val_score(clf, data['train_X'], data['train_y'], cv=10,
#                            scoring=qwkappa, n_jobs=2)
#    print("Kappa: {:.5f} (+/- {:.5f})".format(scores.mean(), scores.std()))

    from sklearn.grid_search import GridSearchCV
    grid = GridSearchCV(estimator=clf,
                        param_grid={'n_estimators': [10, 20, 50, 100, 200, 500]},
                        cv=10, scoring=qwkappa, n_jobs=2,
                        verbose=1)
    grid.fit(data['train_X'], data['train_y'])
    print('grid scores:', grid.grid_scores_)
    print('best score:', grid.best_score_)
    print('best params:', grid.best_params_)

    pass


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

        """
        parser.add_argument("--in-train-archive",
            action='store', dest="in_train_arch",
            type=str,
            help="input ZIP archive with training data")

        parser.add_argument("--in-test-archive",
            action='store', dest="in_test_arch",
            type=str,
            help="input ZIP archive with test data")

        parser.add_argument("--in-train-csv",
            action='store', dest="in_train_csv", default='train.csv',
            type=str,
            help="input CSV with training data zipped inside IN_TRAIN_ARCH")

        parser.add_argument("--in-test-csv",
            action='store', dest="in_test_csv", default='test.csv',
            type=str,
            help="input CSV with test data zipped inside IN_TEST_ARCH")

        parser.add_argument("-o", "--out-h5",
            action='store', dest="out_h5", default='raw-data.h5',
            type=str,
            help="output HDF5 filename for data")

        parser.add_argument("-o", "--out-csv",
            action='store', dest="out_csv_file", default=stdout,
            type=FileType('w'),
            help="output CSV file name")
        parser.add_argument("-e", "--epsilon",
            type=float, default=2.1, action='store', dest="epsilon",
            help="epsilon for DBSCAN")
        """

        # Process arguments
        args = parser.parse_args()

        for k, v in args.__dict__.items():
            print(str(k) + ' => ' + str(v))
            pass


        work()


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
