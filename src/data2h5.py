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
#  Filename: data2h5.py
#
#  Decription:
#      Reads input ZIP archives, processes data filling in missing entries,
#      replacing nominal variable values in test data unseen in the training
#      set, and writes out to HDF5 storage file.
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
#  2016-01-08   wm              Initial version
#
################################################################################
"""

from __future__ import print_function


DEBUG = False
__all__ = []
__version__ = 0.1
__date__ = '2016-01-08'
__updated__ = '2016-01-08'

from sys import path as sys_path
sys_path.insert(0, './mlpipes')
sys_path.insert(0, './Pipe')
import pipe as P


def work(in_train_arch,
         in_test_arch,
         in_train_csv,
         in_test_csv,
         out_h5):

    from pypipes import unzip,as_key,del_key,getitem,setitem
    from nppipes import (genfromtxt,
                         place,astype,as_columns,label_encoder,fit_transform,
                         transform,stack
                         )
    from nppipes import take as np_take
    from numpy.core.defchararray import strip
    from numpy import s_,mean,in1d,putmask
    from collections import Counter
    from h5pipes import h5new


    @P.Pipe
    def replace_missing_with(iterable, ftor):
        from numpy import isnan
        for item in iterable:
            for i in range(item.shape[1]):
                mask = isnan(item[:, i])
                value = ftor(item[~mask, i])
                item[mask, i] = value
                pass
            yield item


    missing_cidx = [11, 14, 16, 28, 33, 34, 35, 36, 37, 46, 51, 60, 68]
    unseen_nominal_cidx = [2, 12, 38, 69, 74]
    seen_nominal_cidx = [0, 1, 4, 5, 6, 13, 15, 17, 18, 19, 20, 21, 22, 23,
                 24, 25, 26, 27, 29, 30, 31, 32, 39, 40, 41, 42, 43, 44, 45,
                 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59,
                 61, 62, 63, 64, 65, 66, 67, 70, 71, 72, 73, 75, 76, 77]
    nominal_cidx = seen_nominal_cidx + unseen_nominal_cidx


    data = (
        in_train_arch
        | unzip(in_train_csv)
        | genfromtxt(delimiter=',', dtype=str)
        | place(lambda d: d == '', 'nan')
        | as_key('train')
        | as_key('train_col_names', lambda d: strip(d['train'][0], '"'))
        | as_key('train_labels',    lambda d: d['train'][1:, 0].astype(int))
        | as_key('train_X',         lambda d: d['train'][1:, 1:-1])
        | as_key('train_y',         lambda d: d['train'][1:, -1].astype(int))
        | del_key('train')


        | as_key('test', lambda d:
                in_test_arch
                | unzip(in_test_csv)
                | genfromtxt(delimiter=',', dtype=str)
                | place(lambda d: d == '', 'nan')
                | P.first
                )
        | as_key('test_col_names', lambda d: strip(d['test'][0], '"'))
        | as_key('test_labels',    lambda d: d['test'][1:, 0].astype(int))
        | as_key('test_X',         lambda d: d['test'][1:, 1:])
        | del_key('test')

        | as_key('train_X', lambda d:
                (d['train_X'],)
                | np_take(missing_cidx, axis=1)
                | astype(float)

                | replace_missing_with(mean)

                | astype(str)
                | setitem(d['train_X'].copy(), s_[:, missing_cidx])
                | P.first
                )

        | as_key('label_encoders', lambda d:
                len(nominal_cidx)
                | label_encoder
                | P.as_tuple
                )

        | as_key('train_X', lambda d:
                (d['train_X'],)
                | np_take(nominal_cidx, axis=1)
                | as_columns
                | fit_transform(d['label_encoders'])
                | stack(axis=1)
                | setitem(d['train_X'].copy(), s_[:, nominal_cidx])
                | P.first
                )

        | as_key('test_X', lambda d:
                (d['test_X'],)
                | np_take(seen_nominal_cidx, axis=1)
                | as_columns
                | transform(d['label_encoders'][:-len(unseen_nominal_cidx)])
                | stack(axis=1)
                | setitem(d['test_X'].copy(), s_[:, seen_nominal_cidx])
                | P.first
                )

        | as_key('test_X', lambda d:
                (d['test_X'],)
                | np_take(unseen_nominal_cidx, axis=1)
                | as_key('test_unseen_nominals_features')

                | as_key('test_unseen_nominals', lambda d2:
                        zip(d2['test_unseen_nominals_features'].T,
                            d['label_encoders'][-len(unseen_nominal_cidx):])
                        | P.select(lambda t: list(set(t[0]) - set(t[1].classes_)))
                        | P.as_list
                        )

                | as_key('train_most_common_nominals', lambda d2:
                        zip(d['train_X'][:, unseen_nominal_cidx].T.astype(int),
                            d['label_encoders'][-len(unseen_nominal_cidx):])
                        | P.select(lambda t: t[1].inverse_transform(t[0]))
                        | P.select(lambda s: Counter(s).most_common(1)[0][0])
                        | P.as_list
                        )

                | as_key('test_corrected_features', lambda d2:
                        zip(d2['test_unseen_nominals_features'].copy().T,
                            d2['test_unseen_nominals'],
                            d2['train_most_common_nominals'])
                        | P.select(lambda t: putmask(t[0], in1d(t[0], t[1]), t[2]) or t[0].T)
                        | stack(axis=1)
                        | P.first
                        )

                | getitem('test_corrected_features')
                | as_columns
                | transform(d['label_encoders'][-len(unseen_nominal_cidx):])
                | stack(axis=1)
                | setitem(d['test_X'].copy(), s_[:, unseen_nominal_cidx])
                | P.first
                )

        | del_key('label_encoders')

        | as_key('test_X', lambda d:
                (d['test_X'],)
                | np_take(missing_cidx, axis=1)
                | astype(float)

                | replace_missing_with(mean)

                | astype(str)
                | setitem(d['test_X'].copy(), s_[:, missing_cidx])
                | P.first
                )

        | P.first
        )

    #print(data.keys())

    (
        (out_h5,)
        | h5new
        | as_key('train_X',         lambda _: data['train_X'].astype(float))
        | as_key('train_y',         lambda _: data['train_y'].astype(float))
        | as_key('test_X',          lambda _: data['test_X'].astype(float))
        | as_key('train_labels',    lambda _: data['train_labels'])
        | as_key('test_labels',     lambda _: data['test_labels'])
        | P.first
    )

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

        """
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


        work(args.in_train_arch,
             args.in_test_arch,
             args.in_train_csv,
             args.in_test_csv,
             args.out_h5)


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
