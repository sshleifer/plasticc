from pathlib import Path

import numpy as np

TEST_SET_SIZE = 453653104  # 453M roughly
SUB_SIZE = 3492890  # 3.5m

GALACTIC_CLASSES = (6, 16, 53, 65, 92)
# docs say 3492890
DTYPES = {
    'object_id': np.int32,
    'mjd': np.float32,
    'passband': np.int8,
    'flux': np.float32,
    'flux_err': np.float32,
    'detected': np.int8
}

DATA_DIR = Path('/Users/shleifer/plasticc')

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
best_params = {
    'device': 'cpu',
    'objective': 'multiclass',
    'num_class': 14,
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'max_depth': 7,
    'n_estimators': 2000,  # avoid updating
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
    'subsample': 0.75
}

# DART_KERNEL_PARAMS = {
#     'device': 'cpu',
#     'objective': 'multiclass',
#     'num_class': 14,
#     'boosting_type': 'gbdt',
#     'n_jobs': -1,
#     'max_depth': 7,
#     'n_estimators': 500,
#     'subsample_freq': 2,
#     'subsample_for_bin': 5000,
#     'min_data_per_group': 100,
#     'max_cat_to_onehot': 4,
#     'cat_l2': 1.0,
#     'cat_smooth': 59.5,
#     'max_cat_threshold': 32,
#     'metric_freq': 10,
#     'verbosity': -1,
#     'metric': 'multi_logloss',
#     'xgboost_dart_mode': False,
#     'uniform_drop': False,
#     'colsample_bytree': 0.5,
#     'drop_rate': 0.173,
#     'learning_rate': 0.0267,
#     'max_drop': 5,
#     'min_child_samples': 10,
#     'min_child_weight': 100.0,
#     'min_split_gain': 0.1,
#     'num_leaves': 7,
#     'reg_alpha': 0.1,
#     'reg_lambda': 0.00023,
#     'skip_drop': 0.44,
#     'subsample': 0.75
# }
OBJECT_ID = 'object_id'
