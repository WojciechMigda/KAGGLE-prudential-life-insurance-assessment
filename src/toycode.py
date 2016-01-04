#!/opt/anaconda2/bin/python
# -*- coding: utf-8 -*-

"""
################################################################################
#
#  Copyright (c) 2015 Wojciech Migda
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
#
################################################################################
"""

from __future__ import print_function


DEBUG = False
__all__ = []
__version__ = 0.1
__date__ = '2016-01-02'
__updated__ = '2016-01-02'

from sys import path as sys_path
sys_path.insert(0, './mlpipes')
sys_path.insert(0, './Pipe')
import pipe as P


def work():

    from pypipes import unzip,as_key,del_key
    from nppipes import genfromtxt
    from nppipes import place
    from numpy.core.defchararray import strip


    """
Id
cat     Product_Info_1-3, {1: 1-2, 3:1-38}
cont    Product_Info_4, {0.-1.}
cat     Product_Info_5-7, {5: 2-3, 6: 1-3, 7: 1-3}
cont    Ins_Age {0.-1.}
cont    Ht {0.-1.}
cont    Wt {0.-1.}
cont    BMI {0.-1.}
cont    Employment_Info_1 {0.-1.}
cat     Employment_Info_2-3 {2: 1-38, 3: 1-3}   13,14
cont    Employment_Info_4
cat     Employment_Info_5   16
cont    Employment_Info_6
cat     InsuredInfo_1-7     18,19,20,21,22,23,24
cat     Insurance_History_1-4   25,26,27,28
cont    Insurance_History_5
cat     Insurance_History_7-9   30,31,32
cat     Family_Hist_1   33
cont    Family_Hist_2-5
disc    Medical_History_1   38*
cat     Medical_History_2-9     39,40,41,42,43,44,45,46
disc    Medical_History_10
cat     Medical_History_11-14   48,49,50,51
disc    Medical_History_15
cat     Medical_History_16-23   53,54,55,56,57,58,59,60
disc    Medical_History_24
cat     Medical_History_25-31   62,63,64,65,66,67,68
disc    Medical_History_32
cat     Medical_History_33-41   70,71,72,73,74,75,76,77,78
*?      Medical_Keyword_1-48
int     Response
    """


    data = (
        '../../data/train.csv.zip'
        | unzip('train.csv')
        | genfromtxt(delimiter=',', dtype=str)
        | place(lambda d: d == '', 'nan')
        | as_key('train')
        | as_key('train_col_names', lambda d: strip(d['train'][0], '"'))
        | as_key('train_labels',    lambda d: d['train'][1:, 0].astype(int))
        | as_key('train_X',         lambda d: d['train'][1:, 1:-1])
        | as_key('train_y',         lambda d: d['train'][1:, -1].astype(int))
        | del_key('train')


        | as_key('test', lambda d:
                '../../data/test.csv.zip'
                | unzip('test.csv')
                | genfromtxt(delimiter=',', dtype=str)
                | place(lambda d: d == '', 'nan')
                | P.first
                )
        | as_key('test_col_names', lambda d: strip(d['test'][0], '"'))
        | as_key('test_labels',    lambda d: d['test'][1:, 0].astype(int))
        | as_key('test_X',         lambda d: d['test'][1:, 1:])
        | del_key('test')


        | P.first
        )

    print(data.keys())
    print(data['train_col_names'])
    print(data['train_X'][:, 16]) # 'nan'

    return

    """
    Missing:
    12, 15, 17, 29, 34, 35, 36, 37, 38, 47, 52, 61, 69,
    """

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    for fidx in [1, 2, 5, 6, 7, 14, 16, 18, 19, 20, 21, 22, 23, 24,
                 25, 26, 27, 28, 30, 31, 32, 33, 40, 41, 42, 43, 44, 45, 46,
                 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60,
                 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 76, 77, 78]:
        fidx -= 1
        X_train[:, fidx] = le.fit_transform(X_train[:, fidx])
        X_test[:, fidx] = le.transform(X_test[:, fidx])
        print(fidx + 1, le.classes_)
        pass

    for fidx, unseen in {3: [], 13: [], 39: [], 70: [], 75: ['3']}.items():
        print(fidx, unseen)
        fidx -= 1
        X_train[:, fidx] = le.fit_transform(X_train[:, fidx])
        from numpy import place,in1d
        place(X_test[:, fidx], in1d(X_test[:, fidx], unseen), 'nan')
        X_test[:, fidx] = le.transform(X_test[:, fidx])
        print(fidx + 1, le.classes_)
        pass


    """
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(foo)
    print(enc.n_values_)
    """

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
        parser.add_argument("-i", "--in-csv",
            action='store', dest="in_csv_file", default=stdin,
            type=FileType('r'),
            help="input CSV file name")
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
