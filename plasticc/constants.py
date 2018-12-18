from pathlib import Path

import numpy as np

TEST_SET_SIZE = 453653104  # 453M roughly
SUB_SIZE = 3492890  # 3.5m

TRAIN_SET_SHAPE = 1421705

FLUX_RATIO_PREFIX = 'flux_by_flux_ratio_sq'

GALACTIC_CLASSES = (6, 16, 53, 65, 92)
UNDERWEIGHTED_CLASSES = (53,)
# we say shit is class 90 that is not: eg (42, 52, 67) are the 3 largest cultrups
# need to work on separating 52 and 90!

# docs say 3492890
DTYPES = {
    'object_id': np.int32,
    'mjd': np.float32,
    'passband': np.int8,
    'flux': np.float32,
    'flux_err': np.float32,
    'detected': np.int8
}
COLUMN_TO_TYPE = DTYPES

DATA_DIR = Path('/Users/shleifer/plasticc')
DROPBOX_DATA_DIR = Path('/Users/shleifer/Dropbox/plasticc_data_shleifer/')
if not DATA_DIR.exists():
    DATA_DIR = Path('/home/paperspace/data')
    DROPBOX_DATA_DIR = Path('/home/paperspace/Dropbox/plasticc_data_shleifer/')

assert DATA_DIR.exists()
assert DROPBOX_DATA_DIR.exists()


aggs = {
    'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'detected': ['mean'],
    'flux_ratio_sq': ['sum', 'skew'],  # no sort
    'flux_by_flux_ratio_sq': ['sum', 'skew'],
}

fcp_improved = {
    '4': {'fft_aggregated': [{'aggtype': 'kurtosis'}],
          'cid_ce': [{'normalize': True}],
          'binned_entropy': [{'max_bins': 10}]},
    '2': {'cid_ce': [{'normalize': True}],
          'fft_aggregated': [{'aggtype': 'skew'}, {'aggtype': 'kurtosis'}],
          'median': None,
          'binned_entropy': [{'max_bins': 10}],
          'partial_autocorrelation': [{'lag': 1}]},
    '1': {'quantile': [{'q': 0.7}], 'fft_aggregated': [{'aggtype': 'skew'}]},
    '0': {'quantile': [{'q': 0.9}], 'cid_ce': [{'normalize': True}]},
    '5': {'autocorrelation': [{'lag': 1}],
          'fft_aggregated': [{'aggtype': 'kurtosis'}]}
}

fcp_improved = {
    '4': {'fft_aggregated': [{'aggtype': 'kurtosis'}],
          'cid_ce': [{'normalize': True}],
          'binned_entropy': [{'max_bins': 10}]},
    '2': {'cid_ce': [{'normalize': True}],
          'fft_aggregated': [{'aggtype': 'skew'}, {'aggtype': 'kurtosis'}],
          'median': None,
          'binned_entropy': [{'max_bins': 10}],
          'partial_autocorrelation': [{'lag': 1}]},
    '1': {'quantile': [{'q': 0.7}], 'fft_aggregated': [{'aggtype': 'skew'}]},
    '0': {'quantile': [{'q': 0.9}], 'cid_ce': [{'normalize': True}]},
    '5': {'autocorrelation': [{'lag': 1}],
          'fft_aggregated': [{'aggtype': 'kurtosis'}]}
}

FC_PASSBAND_V2 = {
    'fft_aggregated': [{'aggtype': 'kurtosis'}, {'aggtype': 'skew'}],
    'cid_ce': [{'normalize': True}],
    'binned_entropy': [{'max_bins': 10}],
    'median': None,
    'partial_autocorrelation': [{'lag': 1}],
    'quantile': [{'q': 0.7}, {'q': 0.9}],
    'autocorrelation': [{'lag': 1}],
}

fcp = {  #
    'flux': {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
        'mean_change': None,
        'mean_abs_change': None,
        'length': None,
    },

    'flux_by_flux_ratio_sq': {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
    },

    'flux_passband': {
        'fft_coefficient': [
            {'coeff': 0, 'attr': 'abs'},
            {'coeff': 1, 'attr': 'abs'}
        ],
        'kurtosis': None,
        'skewness': None,
    },

    'mjd': {
        'maximum': None,
        'minimum': None,
        'mean_change': None,
        'mean_abs_change': None,
    },
}
import tsfresh

SMALLER_RATIO_FEATS = {
    'linear_trend': [{'attr': 'stderr'}],
    'quantile': [
        {'q': 0.8},
        {'q': 0.7},
        {'q': 0.1},
        {'q': 0.6},
    ],
    'change_quantiles': [
        {'f_agg': 'mean', 'isabs': True, 'qh': 0.4, 'ql': 0.0},
        {'f_agg': 'mean', 'isabs': True, 'qh': 0.6, 'ql': 0.0},
        {'f_agg': 'mean', 'isabs': True, 'qh': 0.6, 'ql': 0.2},
        {'f_agg': 'var', 'isabs': True, 'qh': 0.2, 'ql': 0.0}],
    'minimum': None,
    'skewness': None,
    'median': None
}

HANDFIXED_RATIO_FEATS = {
    'c3': [{'lag': 1}],
    'change_quantiles': [
        {'f_agg': 'mean', 'isabs': True, 'qh': 0.8, 'ql': 0.2},
        {'f_agg': 'mean', 'isabs': True, 'qh': 0.4, 'ql': 0.0},
        {'f_agg': 'mean', 'isabs': True, 'qh': 0.6, 'ql': 0.0},
        {'f_agg': 'mean', 'isabs': True, 'qh': 0.2, 'ql': 0.0},
        {'f_agg': 'mean', 'isabs': True, 'qh': 0.6, 'ql': 0.2},
        {'f_agg': 'var', 'isabs': False, 'qh': 0.2, 'ql': 0.0},
        {'f_agg': 'var', 'isabs': True, 'qh': 0.2, 'ql': 0.0}
    ],
    'linear_trend': [{'attr': 'stderr'}],
    'quantile': [
        {'q': 0.8},
        {'q': 0.9},
        {'q': 0.6},
        {'q': 0.7},
        {'q': 0.1},
        {'q': 0.4}, ],
    'minimum': None,
    'skewness': None,
    'agg_linear_trend': [{'f_agg': 'min', 'chunk_len': 5, 'attr': 'stderr'}],
    'median': None,
    'fft_aggregated': [{'aggtype': 'kurtosis'}]
}
_useful_flux_features = [
    'flux__ar_coefficient__k_10__coeff_4',
    'flux__augmented_dickey_fuller__attr_"pvalue"',
    'flux__autocorrelation__lag_6',
    'flux__binned_entropy__max_bins_10',
    'flux__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8',
    'flux__cid_ce__normalize_True',
    'flux__friedrich_coefficients__m_3__r_30__coeff_0',
    'flux__friedrich_coefficients__m_3__r_30__coeff_2',
    'flux__kurtosis',
    'flux__median',
    'flux__number_crossing_m__m_-1',
    'flux__number_crossing_m__m_0',
    'flux__number_crossing_m__m_1',
    'flux__partial_autocorrelation__lag_2',
    'flux__partial_autocorrelation__lag_4',
    'flux__quantile__q_0.2',
    'flux__quantile__q_0.3',
    'flux__quantile__q_0.4',
    'flux__quantile__q_0.6',
    'flux__quantile__q_0.8',
    'flux__quantile__q_0.9',
    'flux__ratio_beyond_r_sigma__r_0.5',
    'flux__ratio_beyond_r_sigma__r_1.5',
    'flux__ratio_beyond_r_sigma__r_2',
    'flux__sample_entropy',
    'flux__skewness',
    'flux__time_reversal_asymmetry_statistic__lag_1',
    'flux__time_reversal_asymmetry_statistic__lag_2',
    'flux__time_reversal_asymmetry_statistic__lag_3'
]
EXTRA_FLUX_PARS = tsfresh.feature_extraction.settings.from_columns(
_useful_flux_features)['flux']  # very expensive
LGB_PARAMS = {
    'device': 'cpu',
    'objective': 'multiclass',
    'num_class': 14,
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'max_depth': 7,
    'n_estimators': 5000,  # avoid updating
    'subsample_freq': 2,
    'subsample_for_bin': 5000,
    'min_data_per_group': 100,
    'max_cat_to_onehot': 4,
    'cat_l2': 1.0,
    'cat_smooth': 59.5,
    'max_cat_threshold': 32,
    'metric_freq': 10,
    'verbosity': -1,
    'metric': 'multi_logloss',
    'xgboost_dart_mode': False,
    'uniform_drop': False,
    'colsample_bytree': 0.5,
    'drop_rate': 0.173,
    'learning_rate': 0.0267,
    'max_drop': 5,
    'min_child_samples': 10,
    'min_child_weight': 100.0,
    'min_split_gain': 0.1,
    'num_leaves': 7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.00023,
    'skip_drop': 0.44,
    'subsample': 0.6,
}
RAY_CHG = [
    #'boosting type',
    'max_depth', 'num_leaves', 'min_child_weight',
    'reg_alpha', 'subsample',
]

pars_to_test = [
    {'max_depth': 6},
    {'max_depth': 8},
    {'num_leaves': 6},
    {'num_leaves': 8},
    {'min_child_weight': 80},
    {'min_child_weight': 120},
    {'reg_alpha': .05},
    {'reg_alpha': .15},
    {'subsample': .9},
    {'subsample': .6},
]
# 'skip_drop', 'drop_rate',
# changeable lgbm params
# n_splits
# max_depth
# subsample freq
# subsample for bin
# max_cat_to_onehot:
# min data per group
# cat l2

FAST_PARAMS = LGB_PARAMS.copy()
FAST_PARAMS['n_estimators'] = 3
DART_PARAMS = LGB_PARAMS.copy()
DART_PARAMS['boosting_type'] = 'dart'
BASE_AGGS = ['min', 'max', 'mean', 'median', 'std', 'skew']
OBJECT_ID = 'object_id'
OID = OBJECT_ID
PASSBAND = 'passband'

PREVIOUSLY_UNUSED_FLUX_PASSBAND_FEATS = [
    '0__fft_aggregated__aggtype_"kurtosis"',
    '2__autocorrelation__lag_1',
    '2__quantile__q_0.9',
    '3__fft_aggregated__aggtype_"kurtosis"',
    '3__skewness',
    '4__autocorrelation__lag_1',
    '4__fft_aggregated__aggtype_"skew"',
    'flux__friedrich_coefficients__m_3__r_30__coeff_3',
]

FNAMES_126_ADDITIONS = PREVIOUSLY_UNUSED_FLUX_PASSBAND_FEATS + [
    # should be cached across machines
    'flux_by_flux_ratio_sq0__linear_trend__attr_"stderr"',
    'flux_by_flux_ratio_sq0__quantile__q_0.8',
    'flux_by_flux_ratio_sq1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0',
    'flux_by_flux_ratio_sq1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0',
    'flux_by_flux_ratio_sq1__minimum',
    'flux_by_flux_ratio_sq1__quantile__q_0.7',
    'flux_by_flux_ratio_sq1__skewness',
    'flux_by_flux_ratio_sq2__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0',
    'flux_by_flux_ratio_sq2__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.2',
    'flux_by_flux_ratio_sq2__median',
    'flux_by_flux_ratio_sq2__minimum',
    'flux_by_flux_ratio_sq2__quantile__q_0.1',
    'flux_by_flux_ratio_sq2__quantile__q_0.6',
    'flux_by_flux_ratio_sq2__quantile__q_0.7',
    'flux_by_flux_ratio_sq2__quantile__q_0.8',
    'flux_by_flux_ratio_sq2__skewness',
    'flux_by_flux_ratio_sq5__change_quantiles__f_agg_"var"__isabs_True__qh_0.2__ql_0.0'
]

MJD_SETTINGS = {'cid_ce': [{'normalize': False}],
  'standard_deviation': None,
  'variance': None,
  'fft_coefficient': [{'coeff': 1, 'attr': 'abs'}],
  'linear_trend': [{'attr': 'slope'}]}

TO_DROP4 = [
    'flux__ratio_beyond_r_sigma__r_0.5',
    'flux__ratio_beyond_r_sigma__r_1.5', 'flux__ratio_beyond_r_sigma__r_2',
    'flux_by_flux_ratio_sq_0__linear_trend__attr_"stderr"',
    'flux_by_flux_ratio_sq_0__quantile__q_0.8',
    'flux_by_flux_ratio_sq_1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0',
    'flux_by_flux_ratio_sq_1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0',
    'flux_by_flux_ratio_sq_1__minimum',
    'flux_by_flux_ratio_sq_1__quantile__q_0.7',
    'flux_by_flux_ratio_sq_1__skewness',
    'flux_by_flux_ratio_sq_2__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0',
    'flux_by_flux_ratio_sq_2__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.2',
    'flux_by_flux_ratio_sq_2__median', 'flux_by_flux_ratio_sq_2__minimum',
    'flux_by_flux_ratio_sq_2__quantile__q_0.1',
    'flux_by_flux_ratio_sq_2__quantile__q_0.6',
    'flux_by_flux_ratio_sq_2__quantile__q_0.7',
    'flux_by_flux_ratio_sq_2__quantile__q_0.8',
    'flux_by_flux_ratio_sq_2__skewness',
    'flux_by_flux_ratio_sq_5__change_quantiles__f_agg_"var"__isabs_True__qh_0.2__ql_0.0',
]



MASSIVE_RENAMER = {
    'flux_by_flux_ratio_sq0__linear_trend__attr_"stderr"': 'flux_by_flux_ratio_sq_0__linear_trend__attr_"stderr"',
 'flux_by_flux_ratio_sq0__quantile__q_0.8': 'flux_by_flux_ratio_sq_0__quantile__q_0.8',
 'flux_by_flux_ratio_sq1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0': 'flux_by_flux_ratio_sq_1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0',
 'flux_by_flux_ratio_sq1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0': 'flux_by_flux_ratio_sq_1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0',
 'flux_by_flux_ratio_sq1__minimum': 'flux_by_flux_ratio_sq_1__minimum',
 'flux_by_flux_ratio_sq1__quantile__q_0.7': 'flux_by_flux_ratio_sq_1__quantile__q_0.7',
 'flux_by_flux_ratio_sq1__skewness': 'flux_by_flux_ratio_sq_1__skewness',
 'flux_by_flux_ratio_sq2__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0': 'flux_by_flux_ratio_sq_2__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0',
 'flux_by_flux_ratio_sq2__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.2': 'flux_by_flux_ratio_sq_2__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.2',
 'flux_by_flux_ratio_sq2__median': 'flux_by_flux_ratio_sq_2__median',
 'flux_by_flux_ratio_sq2__minimum': 'flux_by_flux_ratio_sq_2__minimum',
 'flux_by_flux_ratio_sq2__quantile__q_0.1': 'flux_by_flux_ratio_sq_2__quantile__q_0.1',
 'flux_by_flux_ratio_sq2__quantile__q_0.6': 'flux_by_flux_ratio_sq_2__quantile__q_0.6',
 'flux_by_flux_ratio_sq2__quantile__q_0.7': 'flux_by_flux_ratio_sq_2__quantile__q_0.7',
 'flux_by_flux_ratio_sq2__quantile__q_0.8': 'flux_by_flux_ratio_sq_2__quantile__q_0.8',
 'flux_by_flux_ratio_sq2__skewness': 'flux_by_flux_ratio_sq_2__skewness',
 'flux_by_flux_ratio_sq5__change_quantiles__f_agg_"var"__isabs_True__qh_0.2__ql_0.0': 'flux_by_flux_ratio_sq_5__change_quantiles__f_agg_"var"__isabs_True__qh_0.2__ql_0.0',
 'flux__partial_autocorrelation__lag_1': 'efficient_flux__partial_autocorrelation__lag_1'
}
CLASSES = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
CLASS_WEIGHTS = { 64: 2, 15: 2, 6: 1, 16: 1, 42: 1, 52: 1, 53: 1,
                 62: 1,  65: 1, 67: 1, 88: 1,
                 90: 1, 92: 1, 95: 1}
DEFAULT_SAMPLE_WEIGHTS = {  # 7848 / num appearances
    90: 10.18,
    42: 19.74,
    65: 24.0,
    16: 25.48,
    15: 47.56,
    62: 48.64,
    88: 63.63,
    92: 98.51,
    67: 113.19,
    52: 128.66,
    95: 134.54,
    6: 155.92,
    64: 230.82,
    53: 784.8
}

BEST_SWEIGHTS = {
    90: 3.3594,
    42: 19.74,
    65: 24.0,
    16: 25.48,
    15: 95.12,
    62: 48.64,
    88: 63.63,
    92: 98.51,
    67: 113.19,
    52: 385.98,
    95: 403.62,
    6: 467.76,
    64: 461.64,
    53: 2354.4
}

JIM_FNAMES = [
    'outlierScore', 'hipd', 'lipd', 'highEnergy_transitory_1.0_TF',
    'highEnergy_transitory_1.5_TF', 'lowEnergy_transitory_1.0_TF',
    'lowEnergy_transitory_1.5_TF'
]


KNOWN_BAD_FEATS =  [
    'flux__kurtosis_times_sq_dist',
    'undet_over_det_flux_mean',
    'undet_over_det_flux_median',
    'flux_by_flux_ratio_sq0__linear_trend__attr_"stderr"_over_det_min_over_mjd_det__cid_ce__normalize_False',
    '4__skewness_over_mjd_det__cid_ce__normalize_False'
]
