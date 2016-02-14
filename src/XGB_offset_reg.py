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

NOMINALS_2 = ['Product_Info_1', 'Product_Info_5', 'Product_Info_6',
              'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_2',
              'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7',
              'Insurance_History_1', 'Medical_History_4', 'Medical_History_22']

NOMINALS_3 = ['Product_Info_7', 'InsuredInfo_1', 'Insurance_History_2',
              'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7',
              'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1',
              'Medical_History_3', 'Medical_History_5',
              'Medical_History_6', 'Medical_History_7', 'Medical_History_8',
              'Medical_History_9', 'Medical_History_11', 'Medical_History_12',
              'Medical_History_13', 'Medical_History_14', 'Medical_History_16',
              'Medical_History_17', 'Medical_History_18', 'Medical_History_19',
              'Medical_History_20', 'Medical_History_21',
             'Medical_History_23', 'Medical_History_25', 'Medical_History_26',
             'Medical_History_27', 'Medical_History_28', 'Medical_History_29',
             'Medical_History_30', 'Medical_History_31', 'Medical_History_33',
             'Medical_History_34', 'Medical_History_35', 'Medical_History_36',
             'Medical_History_37', 'Medical_History_38', 'Medical_History_39',
             'Medical_History_40', 'Medical_History_41']

NOMINALS_GE4 = [
    'InsuredInfo_3', # 11
    'Product_Info_2', # 19
    'Product_Info_3', # 38
    'Employment_Info_2', # 38
    'Medical_History_2' # 628
    ]

CONTINUOUS = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI',
              'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6',
              'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3',
              'Family_Hist_4', 'Family_Hist_5']

DISCRETE = ['Medical_History_1', 'Medical_History_10', 'Medical_History_15',
            'Medical_History_24', 'Medical_History_32']

BOOLEANS = ['Medical_Keyword_' + str(i + 1) for i in range(48)]


TO_DROP = ['Medical_Keyword_13', 'Product_Info_7_2', 'InsuredInfo_1_3',
 'Insurance_History_2_2', 'Insurance_History_3_2', 'Medical_History_3_1',
 'Medical_History_5_3', 'Medical_History_6_2', 'Medical_History_8_1',
 'Medical_History_9_3', 'Medical_History_11_1', 'Medical_History_12_1',
 'Medical_History_13_2', 'Medical_History_16_2', 'Medical_History_17_1',
 'Medical_History_18_3', 'Medical_History_19_3', 'Medical_History_20_3',
 'Medical_History_21_3', 'Medical_History_23_2', 'Medical_History_25_3',
 'Medical_History_26_1', 'Medical_History_27_2', 'Medical_History_28_3',
 'Medical_History_29_2', 'Medical_History_30_1', 'Medical_History_31_2',
 'Medical_History_33_2', 'Medical_History_34_2', 'Medical_History_35_1',
 'Medical_History_35_2', 'Medical_History_35_3', 'Medical_History_37_3',
 'Medical_History_38_3', 'Medical_History_39_2', 'Medical_History_40_2',
 'Medical_History_41_2', 'InsuredInfo_3_9', 'Product_Info_2_7',
 'Product_Info_2_9', 'Product_Info_2_13', 'Product_Info_2_14',
 'Product_Info_3_1', 'Product_Info_3_2', 'Product_Info_3_3',
 'Product_Info_3_4', 'Product_Info_3_5', 'Product_Info_3_6',
 'Product_Info_3_7', 'Product_Info_3_8', 'Product_Info_3_9',
 'Product_Info_3_11', 'Product_Info_3_12', 'Product_Info_3_13',
 'Product_Info_3_14', 'Product_Info_3_15', 'Product_Info_3_16',
 'Product_Info_3_17', 'Product_Info_3_18', 'Product_Info_3_19',
 'Product_Info_3_20', 'Product_Info_3_21', 'Product_Info_3_22',
 'Product_Info_3_23', 'Product_Info_3_24', 'Product_Info_3_25',
 'Product_Info_3_27', 'Product_Info_3_28', 'Product_Info_3_30',
 'Product_Info_3_32', 'Product_Info_3_33', 'Product_Info_3_34',
 'Product_Info_3_35', 'Product_Info_3_36', 'Product_Info_3_37',
 'Product_Info_3_38', 'Employment_Info_2_2', 'Employment_Info_2_4',
 'Employment_Info_2_5', 'Employment_Info_2_6', 'Employment_Info_2_7',
 'Employment_Info_2_8', 'Employment_Info_2_10', 'Employment_Info_2_13',
 'Employment_Info_2_15', 'Employment_Info_2_16', 'Employment_Info_2_17',
 'Employment_Info_2_18', 'Employment_Info_2_19', 'Employment_Info_2_20',
 'Employment_Info_2_21', 'Employment_Info_2_22', 'Employment_Info_2_23',
 'Employment_Info_2_24', 'Employment_Info_2_25', 'Employment_Info_2_26',
 'Employment_Info_2_27', 'Employment_Info_2_28', 'Employment_Info_2_29',
 'Employment_Info_2_30', 'Employment_Info_2_31', 'Employment_Info_2_33',
 'Employment_Info_2_34', 'Employment_Info_2_35', 'Employment_Info_2_36',
 'Employment_Info_2_37', 'Employment_Info_2_38', 'Medical_History_2_1',
 'Medical_History_2_2', 'Medical_History_2_4', 'Medical_History_2_5',
 'Medical_History_2_6', 'Medical_History_2_7', 'Medical_History_2_8',
 'Medical_History_2_9', 'Medical_History_2_10', 'Medical_History_2_11',
 'Medical_History_2_12', 'Medical_History_2_13', 'Medical_History_2_15',
 'Medical_History_2_17', 'Medical_History_2_18', 'Medical_History_2_19',
 'Medical_History_2_20', 'Medical_History_2_21', 'Medical_History_2_22',
 'Medical_History_2_23', 'Medical_History_2_24', 'Medical_History_2_25',
 'Medical_History_2_26', 'Medical_History_2_27', 'Medical_History_2_28',
 'Medical_History_2_29', 'Medical_History_2_30', 'Medical_History_2_31',
 'Medical_History_2_32', 'Medical_History_2_33', 'Medical_History_2_34',
 'Medical_History_2_35', 'Medical_History_2_36', 'Medical_History_2_37',
 'Medical_History_2_38', 'Medical_History_2_39', 'Medical_History_2_40',
 'Medical_History_2_41', 'Medical_History_2_42', 'Medical_History_2_43',
 'Medical_History_2_44', 'Medical_History_2_45', 'Medical_History_2_46',
 'Medical_History_2_47', 'Medical_History_2_48', 'Medical_History_2_49',
 'Medical_History_2_50', 'Medical_History_2_51', 'Medical_History_2_52',
 'Medical_History_2_53', 'Medical_History_2_54', 'Medical_History_2_55',
 'Medical_History_2_56', 'Medical_History_2_58', 'Medical_History_2_59',
 'Medical_History_2_60', 'Medical_History_2_61', 'Medical_History_2_62',
 'Medical_History_2_63', 'Medical_History_2_64', 'Medical_History_2_65',
 'Medical_History_2_66', 'Medical_History_2_67', 'Medical_History_2_68',
 'Medical_History_2_69', 'Medical_History_2_70', 'Medical_History_2_71',
 'Medical_History_2_72', 'Medical_History_2_73', 'Medical_History_2_74',
 'Medical_History_2_75', 'Medical_History_2_76', 'Medical_History_2_77',
 'Medical_History_2_78', 'Medical_History_2_79', 'Medical_History_2_80',
 'Medical_History_2_81', 'Medical_History_2_82', 'Medical_History_2_83',
 'Medical_History_2_84', 'Medical_History_2_85', 'Medical_History_2_86',
 'Medical_History_2_87', 'Medical_History_2_88', 'Medical_History_2_89',
 'Medical_History_2_90', 'Medical_History_2_91', 'Medical_History_2_92',
 'Medical_History_2_93', 'Medical_History_2_94', 'Medical_History_2_95',
 'Medical_History_2_96', 'Medical_History_2_97', 'Medical_History_2_98',
 'Medical_History_2_99', 'Medical_History_2_100', 'Medical_History_2_101',
 'Medical_History_2_102', 'Medical_History_2_103', 'Medical_History_2_104',
 'Medical_History_2_105', 'Medical_History_2_106', 'Medical_History_2_107',
 'Medical_History_2_108', 'Medical_History_2_109', 'Medical_History_2_110',
 'Medical_History_2_112', 'Medical_History_2_113', 'Medical_History_2_114',
 'Medical_History_2_115', 'Medical_History_2_116', 'Medical_History_2_117',
 'Medical_History_2_118', 'Medical_History_2_119', 'Medical_History_2_120',
 'Medical_History_2_121', 'Medical_History_2_122', 'Medical_History_2_123',
 'Medical_History_2_124', 'Medical_History_2_125', 'Medical_History_2_126',
 'Medical_History_2_127', 'Medical_History_2_128', 'Medical_History_2_130',
 'Medical_History_2_131', 'Medical_History_2_132', 'Medical_History_2_133',
 'Medical_History_2_134', 'Medical_History_2_135', 'Medical_History_2_136',
 'Medical_History_2_137', 'Medical_History_2_138', 'Medical_History_2_139',
 'Medical_History_2_140', 'Medical_History_2_141', 'Medical_History_2_143',
 'Medical_History_2_144', 'Medical_History_2_145', 'Medical_History_2_146',
 'Medical_History_2_147', 'Medical_History_2_148', 'Medical_History_2_149',
 'Medical_History_2_150', 'Medical_History_2_151', 'Medical_History_2_154',
 'Medical_History_2_155', 'Medical_History_2_156', 'Medical_History_2_157',
 'Medical_History_2_160', 'Medical_History_2_161', 'Medical_History_2_162',
 'Medical_History_2_163', 'Medical_History_2_164', 'Medical_History_2_165',
 'Medical_History_2_166', 'Medical_History_2_167', 'Medical_History_2_168',
 'Medical_History_2_169', 'Medical_History_2_170', 'Medical_History_2_171',
 'Medical_History_2_172', 'Medical_History_2_173', 'Medical_History_2_174',
 'Medical_History_2_175', 'Medical_History_2_176', 'Medical_History_2_178',
 'Medical_History_2_179', 'Medical_History_2_180', 'Medical_History_2_181',
 'Medical_History_2_182', 'Medical_History_2_183', 'Medical_History_2_184',
 'Medical_History_2_185', 'Medical_History_2_186', 'Medical_History_2_187',
 'Medical_History_2_189', 'Medical_History_2_190', 'Medical_History_2_191',
 'Medical_History_2_192', 'Medical_History_2_193', 'Medical_History_2_194',
 'Medical_History_2_195', 'Medical_History_2_196', 'Medical_History_2_197',
 'Medical_History_2_198', 'Medical_History_2_199', 'Medical_History_2_200',
 'Medical_History_2_201', 'Medical_History_2_202', 'Medical_History_2_203',
 'Medical_History_2_204', 'Medical_History_2_205', 'Medical_History_2_206',
 'Medical_History_2_207', 'Medical_History_2_208', 'Medical_History_2_209',
 'Medical_History_2_210', 'Medical_History_2_211', 'Medical_History_2_212',
 'Medical_History_2_213', 'Medical_History_2_214', 'Medical_History_2_215',
 'Medical_History_2_216', 'Medical_History_2_217', 'Medical_History_2_218',
 'Medical_History_2_219', 'Medical_History_2_220', 'Medical_History_2_221',
 'Medical_History_2_222', 'Medical_History_2_223', 'Medical_History_2_224',
 'Medical_History_2_225', 'Medical_History_2_226', 'Medical_History_2_227',
 'Medical_History_2_228', 'Medical_History_2_229', 'Medical_History_2_230',
 'Medical_History_2_231', 'Medical_History_2_232', 'Medical_History_2_233',
 'Medical_History_2_234', 'Medical_History_2_235', 'Medical_History_2_236',
 'Medical_History_2_237', 'Medical_History_2_238', 'Medical_History_2_239',
 'Medical_History_2_240', 'Medical_History_2_241', 'Medical_History_2_242',
 'Medical_History_2_243', 'Medical_History_2_244', 'Medical_History_2_245',
 'Medical_History_2_246', 'Medical_History_2_247', 'Medical_History_2_248',
 'Medical_History_2_249', 'Medical_History_2_250', 'Medical_History_2_251',
 'Medical_History_2_252', 'Medical_History_2_253', 'Medical_History_2_254',
 'Medical_History_2_256', 'Medical_History_2_257', 'Medical_History_2_258',
 'Medical_History_2_259', 'Medical_History_2_260', 'Medical_History_2_261',
 'Medical_History_2_262', 'Medical_History_2_263', 'Medical_History_2_264',
 'Medical_History_2_265', 'Medical_History_2_266', 'Medical_History_2_267',
 'Medical_History_2_268', 'Medical_History_2_269', 'Medical_History_2_270',
 'Medical_History_2_271', 'Medical_History_2_272', 'Medical_History_2_273',
 'Medical_History_2_274', 'Medical_History_2_275', 'Medical_History_2_276',
 'Medical_History_2_277', 'Medical_History_2_278', 'Medical_History_2_279',
 'Medical_History_2_280', 'Medical_History_2_281', 'Medical_History_2_282',
 'Medical_History_2_283', 'Medical_History_2_284', 'Medical_History_2_285',
 'Medical_History_2_286', 'Medical_History_2_287', 'Medical_History_2_288',
 'Medical_History_2_289', 'Medical_History_2_290', 'Medical_History_2_291',
 'Medical_History_2_292', 'Medical_History_2_293', 'Medical_History_2_294',
 'Medical_History_2_295', 'Medical_History_2_296', 'Medical_History_2_297',
 'Medical_History_2_298', 'Medical_History_2_299', 'Medical_History_2_300',
 'Medical_History_2_301', 'Medical_History_2_302', 'Medical_History_2_303',
 'Medical_History_2_304', 'Medical_History_2_305', 'Medical_History_2_306',
 'Medical_History_2_307', 'Medical_History_2_308', 'Medical_History_2_309',
 'Medical_History_2_310', 'Medical_History_2_311', 'Medical_History_2_312',
 'Medical_History_2_313', 'Medical_History_2_314', 'Medical_History_2_315',
 'Medical_History_2_316', 'Medical_History_2_317', 'Medical_History_2_318',
 'Medical_History_2_319', 'Medical_History_2_320', 'Medical_History_2_321',
 'Medical_History_2_322', 'Medical_History_2_324', 'Medical_History_2_325',
 'Medical_History_2_326', 'Medical_History_2_327', 'Medical_History_2_328',
 'Medical_History_2_329', 'Medical_History_2_330', 'Medical_History_2_331',
 'Medical_History_2_332', 'Medical_History_2_333', 'Medical_History_2_334',
 'Medical_History_2_335', 'Medical_History_2_336', 'Medical_History_2_337',
 'Medical_History_2_338', 'Medical_History_2_339', 'Medical_History_2_340',
 'Medical_History_2_341', 'Medical_History_2_342', 'Medical_History_2_343',
 'Medical_History_2_344', 'Medical_History_2_345', 'Medical_History_2_346',
 'Medical_History_2_347', 'Medical_History_2_348', 'Medical_History_2_349',
 'Medical_History_2_350', 'Medical_History_2_351', 'Medical_History_2_353',
 'Medical_History_2_354', 'Medical_History_2_355', 'Medical_History_2_356',
 'Medical_History_2_357', 'Medical_History_2_358', 'Medical_History_2_360',
 'Medical_History_2_361', 'Medical_History_2_362', 'Medical_History_2_363',
 'Medical_History_2_364', 'Medical_History_2_365', 'Medical_History_2_366',
 'Medical_History_2_367', 'Medical_History_2_368', 'Medical_History_2_369',
 'Medical_History_2_370', 'Medical_History_2_371', 'Medical_History_2_372',
 'Medical_History_2_374', 'Medical_History_2_375', 'Medical_History_2_376',
 'Medical_History_2_377', 'Medical_History_2_378', 'Medical_History_2_379',
 'Medical_History_2_380', 'Medical_History_2_381', 'Medical_History_2_382',
 'Medical_History_2_383', 'Medical_History_2_384', 'Medical_History_2_385',
 'Medical_History_2_386', 'Medical_History_2_387', 'Medical_History_2_388',
 'Medical_History_2_389', 'Medical_History_2_390', 'Medical_History_2_391',
 'Medical_History_2_392', 'Medical_History_2_394', 'Medical_History_2_395',
 'Medical_History_2_396', 'Medical_History_2_397', 'Medical_History_2_398',
 'Medical_History_2_399', 'Medical_History_2_400', 'Medical_History_2_401',
 'Medical_History_2_402', 'Medical_History_2_403', 'Medical_History_2_404',
 'Medical_History_2_405', 'Medical_History_2_406', 'Medical_History_2_407',
 'Medical_History_2_408', 'Medical_History_2_409', 'Medical_History_2_410',
 'Medical_History_2_411', 'Medical_History_2_412', 'Medical_History_2_413',
 'Medical_History_2_414', 'Medical_History_2_415', 'Medical_History_2_416',
 'Medical_History_2_417', 'Medical_History_2_418', 'Medical_History_2_419',
 'Medical_History_2_421', 'Medical_History_2_422', 'Medical_History_2_423',
 'Medical_History_2_424', 'Medical_History_2_425', 'Medical_History_2_426',
 'Medical_History_2_427', 'Medical_History_2_428', 'Medical_History_2_429',
 'Medical_History_2_430', 'Medical_History_2_431', 'Medical_History_2_432',
 'Medical_History_2_433', 'Medical_History_2_434', 'Medical_History_2_435',
 'Medical_History_2_436', 'Medical_History_2_437', 'Medical_History_2_438',
 'Medical_History_2_440', 'Medical_History_2_441', 'Medical_History_2_442',
 'Medical_History_2_443', 'Medical_History_2_444', 'Medical_History_2_445',
 'Medical_History_2_446', 'Medical_History_2_447', 'Medical_History_2_448',
 'Medical_History_2_449', 'Medical_History_2_450', 'Medical_History_2_451',
 'Medical_History_2_452', 'Medical_History_2_453', 'Medical_History_2_454',
 'Medical_History_2_455', 'Medical_History_2_456', 'Medical_History_2_457',
 'Medical_History_2_458', 'Medical_History_2_459', 'Medical_History_2_460',
 'Medical_History_2_461', 'Medical_History_2_462', 'Medical_History_2_464',
 'Medical_History_2_465', 'Medical_History_2_466', 'Medical_History_2_467',
 'Medical_History_2_468', 'Medical_History_2_469', 'Medical_History_2_470',
 'Medical_History_2_471', 'Medical_History_2_472', 'Medical_History_2_473',
 'Medical_History_2_474', 'Medical_History_2_475', 'Medical_History_2_477',
 'Medical_History_2_478', 'Medical_History_2_479', 'Medical_History_2_480',
 'Medical_History_2_481', 'Medical_History_2_482', 'Medical_History_2_483',
 'Medical_History_2_484', 'Medical_History_2_485', 'Medical_History_2_486',
 'Medical_History_2_487', 'Medical_History_2_488', 'Medical_History_2_489',
 'Medical_History_2_490', 'Medical_History_2_491', 'Medical_History_2_492',
 'Medical_History_2_493', 'Medical_History_2_494', 'Medical_History_2_495',
 'Medical_History_2_496', 'Medical_History_2_497', 'Medical_History_2_498',
 'Medical_History_2_499', 'Medical_History_2_500', 'Medical_History_2_501',
 'Medical_History_2_502', 'Medical_History_2_503', 'Medical_History_2_504',
 'Medical_History_2_505', 'Medical_History_2_506', 'Medical_History_2_507',
 'Medical_History_2_508', 'Medical_History_2_509', 'Medical_History_2_510',
 'Medical_History_2_511', 'Medical_History_2_512', 'Medical_History_2_513',
 'Medical_History_2_514', 'Medical_History_2_515', 'Medical_History_2_516',
 'Medical_History_2_517', 'Medical_History_2_518', 'Medical_History_2_519',
 'Medical_History_2_520', 'Medical_History_2_521', 'Medical_History_2_522',
 'Medical_History_2_523', 'Medical_History_2_524', 'Medical_History_2_525',
 'Medical_History_2_526', 'Medical_History_2_527', 'Medical_History_2_528',
 'Medical_History_2_529', 'Medical_History_2_530', 'Medical_History_2_531',
 'Medical_History_2_532', 'Medical_History_2_533', 'Medical_History_2_534',
 'Medical_History_2_535', 'Medical_History_2_536', 'Medical_History_2_537',
 'Medical_History_2_538', 'Medical_History_2_539', 'Medical_History_2_540',
 'Medical_History_2_541', 'Medical_History_2_542', 'Medical_History_2_543',
 'Medical_History_2_544', 'Medical_History_2_545', 'Medical_History_2_548',
 'Medical_History_2_549', 'Medical_History_2_550', 'Medical_History_2_551',
 'Medical_History_2_552', 'Medical_History_2_553', 'Medical_History_2_554',
 'Medical_History_2_555', 'Medical_History_2_556', 'Medical_History_2_557',
 'Medical_History_2_558', 'Medical_History_2_559', 'Medical_History_2_560',
 'Medical_History_2_561', 'Medical_History_2_562', 'Medical_History_2_563',
 'Medical_History_2_564', 'Medical_History_2_565', 'Medical_History_2_566',
 'Medical_History_2_567', 'Medical_History_2_568', 'Medical_History_2_569',
 'Medical_History_2_570', 'Medical_History_2_571', 'Medical_History_2_572',
 'Medical_History_2_573', 'Medical_History_2_574', 'Medical_History_2_575',
 'Medical_History_2_576', 'Medical_History_2_577', 'Medical_History_2_578',
 'Medical_History_2_579', 'Medical_History_2_580', 'Medical_History_2_582',
 'Medical_History_2_583', 'Medical_History_2_584', 'Medical_History_2_585',
 'Medical_History_2_586', 'Medical_History_2_587', 'Medical_History_2_588',
 'Medical_History_2_589', 'Medical_History_2_590', 'Medical_History_2_591',
 'Medical_History_2_592', 'Medical_History_2_593', 'Medical_History_2_594',
 'Medical_History_2_595', 'Medical_History_2_596', 'Medical_History_2_597',
 'Medical_History_2_598', 'Medical_History_2_599', 'Medical_History_2_600',
 'Medical_History_2_601', 'Medical_History_2_602', 'Medical_History_2_603',
 'Medical_History_2_604', 'Medical_History_2_605', 'Medical_History_2_606',
 'Medical_History_2_607', 'Medical_History_2_609', 'Medical_History_2_610',
 'Medical_History_2_611', 'Medical_History_2_612', 'Medical_History_2_613',
 'Medical_History_2_614', 'Medical_History_2_615', 'Medical_History_2_616',
 'Medical_History_2_617', 'Medical_History_2_618', 'Medical_History_2_619',
 'Medical_History_2_620', 'Medical_History_2_621', 'Medical_History_2_622',
 'Medical_History_2_623', 'Medical_History_2_624', 'Medical_History_2_625',
 'Medical_History_2_626', 'Medical_History_2_627', 'Medical_History_2_628']



def CSV_w_coro():
    with open('xgbgen.csv', 'w') as f:
        while True:
            arr = yield
            f.write(','.join([str(x) for x in arr]) + '\n')
    pass
#it = CSV_w_coro()
#next(it)

def CSV_r_coro():
    with open('xgbgen.csv', 'r') as f:
        lines = f.readlines()
        for l in lines:
            from numpy import asarray
            yield asarray(l.strip().split(','))
it = CSV_r_coro()


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
                learning_rates=None,
                min_child_weight=50,
                subsample=0.8,
                colsample_bytree=0.7,
                max_depth=7,
                gamma=0.0,
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
        self.learning_rates = learning_rates
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.nthread = nthread
        self.seed = seed
        self.n_buckets = n_buckets
        self.int_fold = int_fold
        self.initial_params = initial_params
        self.minimizer = minimizer
        self.scoring = scoring
        self.feature_importances_ = None

        return


    def _update_feature_iportances(self, feature_names):
        from numpy import zeros
        feature_importances = zeros(len(feature_names))

        for xgb in self.xgb:
            importances = xgb.booster().get_fscore()
            for i, feat in enumerate(feature_names):
                if feat in importances:
                    feature_importances[i] += importances[feat]
                    pass
                pass
            pass

        self.feature_importances_ = feature_importances / sum(feature_importances)
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

jak wyzej, +nowy bucketing (max_depth=10, eta=0.03, eval_metric=Scirpus, learning_rates=[0.03] * 200 + [0.02] * 500,)
  n_buckets=7:
  mean: 0.65876, std: 0.00405, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}
  n_buckets=8:
  mean: 0.65966, std: 0.00412, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}
  n_buckets=9:
  mean: 0.65965, std: 0.00390, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}
  n_buckets=10:
  mean: 0.65359, std: 0.00405, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}
  n_buckets=12:
  mean: 0.65705, std: 0.00442, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}
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
                           gamma=self.gamma,
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
                            learning_rates=self.learning_rates,
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

        self._update_feature_iportances(X.columns.values.tolist())

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


class PrudentialRegressorCVO3(BaseEstimator, RegressorMixin):
    def __init__(self,
                objective='reg:linear',
                learning_rate=0.045,
                learning_rates=None,
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
        self.learning_rates = learning_rates
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

        """
nah

grid scores:
  mean: 0.65882, std: 0.00382, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}
best score: 0.65882
best params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}
        """

    def fit(self, X, y):
        if not KAGGLE:
            from OptimizedOffsetRegressor import DigitizedOptimizedOffsetRegressor

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
                            learning_rates=self.learning_rates,
                            obj=scirpus_regobj
                            #obj=qwkappa_regobj
                            )
            print("best iteration:", self.xgb[i].booster().best_iteration)
            te_y_hat = self.xgb[i].predict(Xtest,
                                        ntree_limit=self.xgb[i].booster().best_iteration)
            print('XGB Test score is:', -self.scoring(te_y_hat, ytest))

            pass

        xgb_result = []
        for xgb in self.xgb:
            tr_y_hat = xgb.predict(Xtrain, ntree_limit=xgb.booster().best_iteration)
            xgb_result.append(tr_y_hat)

        from numpy import array
        xgb_result = array(xgb_result).mean(axis=0)

        self.off = DigitizedOptimizedOffsetRegressor(n_buckets=self.n_buckets,
                       initial_params=self.initial_params,
                       minimizer=self.minimizer,
                       scoring=self.scoring)
        self.off.fit(xgb_result, ytrain)
        print("Offsets:", self.off.params)

        return self


    def predict(self, X):
        from numpy import clip, array
        xgb_result = []
        for xgb in self.xgb:
            te_y_hat = xgb.predict(X, ntree_limit=xgb.booster().best_iteration)
            xgb_result.append(te_y_hat)
        xgb_result = array(xgb_result).mean(axis=0)

        result = clip(self.off.predict(xgb_result), 1, 8)
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

#    for col in all_data[CONTINUOUS]:
#        from numpy import median, mean as npmean
#        _min = min(all_data[col])
#        _max = max(all_data[col])
#        _median = median(all_data[col])
#        _mean = npmean(all_data[col])
#        if _median != _min and _median != _max:
#            all_data[col + '_median'] = all_data[col] > _median
#            pass
#        if _mean != _min and _mean != _max:
#            all_data[col + '_mean'] = all_data[col] > _mean
#            pass
#        pass

#    from sklearn.preprocessing import StandardScaler
#    from sklearn.decomposition import PCA
#    std = StandardScaler(copy=True)
#    all_data[CONTINUOUS] = std.fit_transform(all_data[CONTINUOUS])
#    pca = PCA(whiten=False, copy=True)
#    all_data[CONTINUOUS] = pca.fit_transform(all_data[CONTINUOUS])


    # create any new variables
#    all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
#    all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]


    # factorize categorical variables
#    all_data['Product_Info_2'] = factorize(all_data['Product_Info_2'])[0]# + 1
#    all_data['Product_Info_2_char'] = factorize(all_data['Product_Info_2_char'])[0]# + 1
#    all_data['Product_Info_2_num'] = factorize(all_data['Product_Info_2_num'])[0]# + 1

    #all_data = all_data.drop(NOMINALS_3, axis=1)
    all_data = OneHot(all_data, NOMINALS_3)
    all_data = OneHot(all_data, NOMINALS_GE4)
    #all_data = OneHot(all_data, NOMINALS_GE4[:-1])
    #all_data = all_data.drop(NOMINALS_GE4[-1:], axis=1)
    """
all_data = all_data.drop(NOMINALS_GE4, axis=1)
all_data = OneHot(all_data, NOMINALS_3)
  mean: 0.65158, std: 0.00558, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}

    all_data = OneHot(all_data, NOMINALS_3)
    all_data = OneHot(all_data, NOMINALS_GE4[:2])
    all_data = all_data.drop(NOMINALS_GE4[2:], axis=1)
  mean: 0.65712, std: 0.00514, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}

    all_data = OneHot(all_data, NOMINALS_3)
    all_data = OneHot(all_data, NOMINALS_GE4[:-1])
    all_data = all_data.drop(NOMINALS_GE4[-1:], axis=1)
  mean: 0.65903, std: 0.00418, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}

    all_data = OneHot(all_data, NOMINALS_3)
    all_data = OneHot(all_data, NOMINALS_GE4)
  mean: 0.66028, std: 0.00443, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10}

inf_fold=5, 28 minutes
  mean: 0.66003, std: 0.00465, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 5, 'max_depth': 10}
inf_fold=5, bez learning_rates, 22 minutes
  mean: 0.66010, std: 0.00416, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 5, 'max_depth': 10}
inf_fold=4, eta=0.05, bez learning_rates, 11 minutes
  mean: 0.65861, std: 0.00391, params: {'colsample_bytree': 0.67, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 4, 'max_depth': 10}

    grid scores:
  mean: 0.65888, std: 0.00443, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.8, 'int_fold': 4, 'max_depth': 10}
  mean: 0.65914, std: 0.00422, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 4, 'max_depth': 10}
  mean: 0.65978, std: 0.00360, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10}
  mean: 0.65875, std: 0.00458, params: {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.8, 'int_fold': 4, 'max_depth': 10}
  mean: 0.65913, std: 0.00416, params: {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 4, 'max_depth': 10}
  mean: 0.65860, std: 0.00387, params: {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10}
  mean: 0.65941, std: 0.00377, params: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.8, 'int_fold': 4, 'max_depth': 10}
  mean: 0.65938, std: 0.00420, params: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 4, 'max_depth': 10}
  mean: 0.65874, std: 0.00378, params: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10}
best score: 0.65978
best params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10}
  mean: 0.65933, std: 0.00412, params: {'colsample_bytree': 0.55, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10}
  mean: 0.65961, std: 0.00411, params: {'colsample_bytree': 0.45, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10}
  mean: 0.65953, std: 0.00370, params: {'colsample_bytree': 0.35, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10}

onehot + poly@CONT, po obcięciu nieważnych
grid scores:
  mean: 0.65729, std: 0.00337, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10}
best score: 0.65729

onehot + poly@CONT, po obcięciu nieważnych
  mean: 0.65660, std: 0.00293, params: {'colsample_bytree': 0.67, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 4, 'max_depth': 10, 'gamma': 0.0}
grid scores:
  mean: 0.65712, std: 0.00384, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 200, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 0.0}
  mean: 0.65695, std: 0.00361, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 0.0}
  mean: 0.65705, std: 0.00351, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 200, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10, 'gamma': 0.0}
  mean: 0.65729, std: 0.00337, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10, 'gamma': 0.0}
  mean: 0.65690, std: 0.00402, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 200, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 1.0}
  mean: 0.65678, std: 0.00347, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 1.0}
  mean: 0.65748, std: 0.00388, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 200, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10, 'gamma': 1.0}
  mean: 0.65727, std: 0.00351, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10, 'gamma': 1.0}
best score: 0.65748
best params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 200, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10, 'gamma': 1.0}

  mean: 0.65645, std: 0.00403, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 160, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 6, 'gamma': 1.0}
  mean: 0.65699, std: 0.00394, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 200, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 6, 'gamma': 1.0}
  mean: 0.65746, std: 0.00367, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 160, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 1.0}
  mean: 0.65690, std: 0.00402, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 200, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 1.0}
best score: 0.65746
best params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 160, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 1.0}

efekt +/- mean, +/- median, vs. 0.65748
  mean: 0.65728, std: 0.00347, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 200, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10, 'gamma': 1.0}

bez poly (psuje)
  mean: 0.65946, std: 0.00392, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 200, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10, 'gamma': 1.0}

grid scores:
> mean: 0.66006, std: 0.00437, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 160, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 0.0}
  mean: 0.65847, std: 0.00445, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 0.0}
  mean: 0.65917, std: 0.00514, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 160, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 12, 'gamma': 0.0}
  mean: 0.65956, std: 0.00452, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 12, 'gamma': 0.0}
  mean: 0.65950, std: 0.00425, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 160, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 1.0}
  mean: 0.65888, std: 0.00380, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 1.0}
  mean: 0.65953, std: 0.00419, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 160, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 12, 'gamma': 1.0}
  mean: 0.65934, std: 0.00400, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 12, 'gamma': 1.0}
best score: 0.66006
best params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 160, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 0.0}

grid scores:
  mean: 0.65942, std: 0.00459, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 80, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 6, 'gamma': 0.0}
  mean: 0.65927, std: 0.00386, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 120, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 6, 'gamma': 0.0}
  mean: 0.65969, std: 0.00381, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 80, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 0.0}
  mean: 0.65998, std: 0.00414, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 120, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 0.0}
  mean: 0.65946, std: 0.00363, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 80, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10, 'gamma': 0.0}
  mean: 0.65967, std: 0.00400, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 120, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 10, 'gamma': 0.0}
best score: 0.65998
best params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 120, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 0.0}
  mean: 0.65945, std: 0.00351, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 140, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 0.0}
  mean: 0.65982, std: 0.00414, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 150, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 0.0}
  mean: 0.65935, std: 0.00393, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 170, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 0.0}
  mean: 0.65903, std: 0.00280, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 180, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 0.0}
best score: 0.65982
best params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 150, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 0.0}

  mean: 0.65947, std: 0.00332, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 160, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 7, 'gamma': 0.0}
  mean: 0.65956, std: 0.00395, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 160, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 9, 'gamma': 0.0}
best score: 0.65956
best params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 160, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 9, 'gamma': 0.0}
  mean: 0.65951, std: 0.00416, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 160, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 2.0}
  mean: 0.65975, std: 0.00367, params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 160, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 4.0}
best score: 0.65975
best params: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'min_child_weight': 160, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 4, 'max_depth': 8, 'gamma': 4.0}

=====
full one hot bez poly
full CV, cf. 0.66028
grid scores:
  mean: 0.65966, std: 0.00394, params: {'colsample_bytree': 0.6, 'learning_rate': 0.03, 'min_child_weight': 160, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 7, 'max_depth': 8, 'gamma': 0.0}
best score: 0.65966
best params: {'colsample_bytree': 0.6, 'learning_rate': 0.03, 'min_child_weight': 160, 'n_estimators': 700, 'subsample': 1.0, 'int_fold': 7, 'max_depth': 8, 'gamma': 0.0}
grid scores:
  mean: 0.66047, std: 0.00471, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10, 'gamma': 0.0}
best score: 0.66047
best params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 240, 'n_estimators': 700, 'subsample': 0.9, 'int_fold': 7, 'max_depth': 10, 'gamma': 0.0}


    """

    """
    Both: 0.65576
    BmiAge: 0.65578
    MedCount: 0.65638
    None: 0.65529
    """
    med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
    all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)

    # poly_15
    all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(all_data[CONTINUOUS])
    poly = poly[:, len(CONTINUOUS):]
#    for i in range(poly.shape[1]):
#        all_data['poly_' + str(i + 1)] = poly[:, i]
    best_poly_120 = ['poly_64', 'poly_55', 'poly_54', 'poly_57', 'poly_56', 'poly_50', 'poly_52', 'poly_68', 'poly_11', 'poly_10', 'poly_13', 'poly_34', 'poly_15', 'poly_14', 'poly_31', 'poly_16', 'poly_73', 'poly_18', 'poly_75', 'poly_77', 'poly_76', 'poly_39', 'poly_74', 'poly_5', 'poly_4', 'poly_7', 'poly_1', 'poly_3', 'poly_2', 'poly_9', 'poly_12', 'poly_37', 'poly_78', 'poly_35', 'poly_42', 'poly_43', 'poly_40', 'poly_41', 'poly_47', 'poly_45', 'poly_33', 'poly_48', 'poly_49', 'poly_32', 'poly_24', 'poly_25', 'poly_26', 'poly_20', 'poly_21', 'poly_22', 'poly_23', 'poly_30', 'poly_28', 'poly_65', 'poly_66', 'poly_67']
    # T4
    best_poly = ['poly_5', 'poly_13', 'poly_14', 'poly_15']
    #for n in best_poly:
    #    all_data[n] = poly[:, int(n[5:]) - 1]
    # 3x3
    #all_data['MH1_BMI'] = all_data['Medical_History_1'] * all_data['BMI']
    #all_data['MH1_MKC'] = all_data['Medical_History_1'] * all_data['Med_Keywords_Count']
    #all_data['BMI_MKC'] = all_data['BMI'] * all_data['Med_Keywords_Count']

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

    all_data = all_data.drop(TO_DROP, axis=1)

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
                            verbose=2,
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

#        if not isinstance(clf.xgb, list):
#            xgb_ensemble = [clf.xgb]
#        else:
#            xgb_ensemble = clf.xgb
#        for xgb in xgb_ensemble:
#            importance = xgb.booster().get_fscore()
#            import operator
#            print(sorted(importance.items()), "\n")
#            importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
#            print(importance, "\n")
#            features = [k for k, _ in importance]
#            print(len(features), features)
        feat_imp = clf.feature_importances_
        nonzero_features = train_X.columns.values[feat_imp > 0.]
        print("Features with importance != 0",
              len(nonzero_features),
              nonzero_features,
              sorted(zip(feat_imp[feat_imp > 0.], nonzero_features)))
        zero_features = train_X.columns.values[feat_imp == 0.]
        print("Features with importance == 0", zero_features)

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
