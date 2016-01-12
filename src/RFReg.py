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
#  Filename: RFClf.py
#
#  Decription:
#      Simple RandomForectRegressor model
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
#  2016-01-11   wm              Initial version
#
################################################################################
"""

from __future__ import print_function


DEBUG = False
__all__ = []
__version__ = "0.0.1"
__date__ = '2016-01-11'
__updated__ = '2016-01-11'

from sys import path as sys_path
sys_path.insert(0, './mlpipes')
sys_path.insert(0, './Pipe')
import pipe as P


def work(in_h5,
         out_csv_file,
         nest,
         njobs):

    from h5pipes import h5open
    from pypipes import getitem,as_key,del_key
    from nppipes import as_array,fit_transform,transform,fit,predict,savetxt,stack
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor


    nominal_cidx = [0, 1, 2, 4, 5, 6, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23,
                 24, 25, 26, 27, 29, 30, 31, 32, 38, 39, 40, 41, 42, 43, 44, 45,
                 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59,
                 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77]

    data = (
        (in_h5,)
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

        | as_key('one_hot', lambda _:
            (OneHotEncoder(categorical_features=nominal_cidx, sparse=False),))
        | as_key('train_X', lambda d:
            (d['train_X'].copy(),)
            | fit_transform(d['one_hot'])
            | P.first
            )
        | as_key('test_X', lambda d:
            (d['test_X'].copy(),)
            | transform(d['one_hot'])
            | P.first
            )
        | del_key('one_hot')

        | as_key('std_scaler', lambda _: (StandardScaler(),))
        | as_key('train_X', lambda d:
            (d['train_X'].copy(),)
            | fit_transform(d['std_scaler'])
            | P.first
            )
        | as_key('test_X', lambda d:
            (d['test_X'].copy(),)
            | transform(d['std_scaler'])
            | P.first
            )
        | del_key('std_scaler')

        | as_key('RFReg', lambda d:
            (RandomForestRegressor(random_state=1,
                                   n_estimators=nest, n_jobs=njobs,
                                   verbose=1,
                                   max_features=1.0, min_samples_leaf=1.0,
                                   max_depth=50),)
            | fit((d['train_X'],), (d['train_y'],))
            | P.first
            )
        | as_key('y_hat', lambda d:
            (d['test_X'],)
            | predict((d['RFReg'],))
            | P.first
            )
        | del_key('RFReg')

        | P.first
    )

    (
        (data['test_labels'], data['y_hat'])
        | stack(axis=1)
        | savetxt(out_csv_file,
                  delimiter=',',
                  fmt=['%d', '%d'],
                  header='"Id","Response"', comments='')
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

        parser.add_argument("-i", "--in-hdf5-file",
            action='store', dest="in_h5",
            type=str,
            help="input HDF5 file with contest data")

        parser.add_argument("-n", "--num-est",
            type=int, default=50, action='store', dest="nest",
            help="number of Random Forest estimators")

        parser.add_argument("-j", "--jobs",
            type=int, default=2, action='store', dest="njobs",
            help="number of Random Forest jobs")

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


        work(args.in_h5,
             args.out_csv_file,
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
