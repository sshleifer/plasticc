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
    'flux_ratio_sq': ['sum', 'skew'], # no sort
    'flux_by_flux_ratio_sq': ['sum', 'skew'],
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
best_params = {
    'device': 'cpu',
    'objective': 'multiclass',
    'num_class': 14,
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'max_depth': 7,
    'n_estimators': 500,
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
