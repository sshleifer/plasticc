from tsfresh import extract_features

COL_ORDER = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53',
       'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 'class_90',
       'class_92', 'class_95', 'class_99', 'object_id']
from sklearn.model_selection import StratifiedKFold
from tsfresh.feature_extraction import extract_features
from lightgbm import LGBMClassifier
from tqdm import *
from numba import jit

import gc
import os
import gc
import glob
import sys
import time
from datetime import datetime as dt

import os

from plasticc.constants import OBJECT_ID, fcp_improved

PRED_99_AVG = 0.14

gc.enable()
from functools import partial

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra

np.warnings.filterwarnings('ignore')
from .forked_lgbm_script import make_pred_df

def make_sub(chunk_paths, save_path, fnames_final, clfs, feature_add_fn):
    for pth in tqdm_notebook(chunk_paths):
        i_c = os.path.basename(pth)[:-3]
        feat_cache_path = f'{feat_dir}/{i_c}.mp'
        df = pd.read_msgpack(pth)
        feats = pd.read_msgpack(feat_cache_path).replace(0, np.nan).set_index(OBJECT_ID)
        if feature_add_fn is not None:
            test_feat_df = feature_add_fn(df, feats)
        else:
            test_feat_df = feats
        preds_df = make_pred_df(clfs, fnames_final, test_feat_df.reset_index())
        if not os.path.exists(save_path):
            preds_df[COL_ORDER].to_csv(save_path, index=False)
        else:
            preds_df[COL_ORDER].to_csv(save_path, header=False, mode='a', index=False)
        test_feat_df.to_msgpack(f'feature_cache_nov_28/{i_c}.mp')
        del preds_df, test_feat_df
        gc.collect()


def tsfresh_adder(mock_tr, feat_df, settings, disable_bar=True, **kwargs):
    X = extract_features(
        mock_tr, default_fc_parameters=settings, column_id=OBJECT_ID, profile=True,
        column_sort='mjd', column_value='flux', disable_progressbar=disable_bar, **kwargs
    ).rename_axis(OBJECT_ID)
    return feat_df.join(X)


def add_acor_feat(mock_tr, feat_df):
    acor_params = {'partial_autocorrelation': [{'lag': 1}, ]}
    X = extract_features(mock_tr, default_fc_parameters=acor_params,
                         column_id=OBJECT_ID, profile=True,
                         column_sort='mjd',
                         column_value='flux', disable_progressbar=True).rename_axis(OBJECT_ID)
    X.columns  = ['efficient_flux__partial_autocorrelation__lag_1']  # HACK
    return feat_df.join(X)


def add_improved_feats(mock_tr, feat_df):
    X = extract_features(mock_tr, default_fc_parameters=fcp_improved,
                         column_id=OBJECT_ID, profile=True,
                         column_kind='passband',
                         column_sort='mjd',
                         column_value='flux', disable_progressbar=True).rename_axis(OBJECT_ID)
    return feat_df.join(X)

COL_ORDER = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53',
       'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 'class_90',
       'class_92', 'class_95', 'class_99', 'object_id']

def subdringus(clfs, fnames_final, save_path):
    raw_chunks_pat = 'feature_cache_dec_2b/*.mp'
    chunk_paths = sorted(list(glob.glob(raw_chunks_pat)))
    for pth in tqdm_notebook(chunk_paths):
        test_feat_df = pd.read_msgpack(pth)
        preds_df = make_pred_df(clfs, fnames_final, test_feat_df.reset_index())
        if not os.path.exists(save_path):
            preds_df[COL_ORDER].to_csv(save_path, index=False)
        else:
            preds_df[COL_ORDER].to_csv(save_path, header=False, mode='a', index=False)
        #test_feat_df.to_msgpack(f'feature_cache_nov_28/{i_c}.mp')
        del preds_df, test_feat_df
        gc.collect()
    feat_dir = 'feature_cache_nov_28'
    feature_add_fn = add_improved_feats
    save_path = 'no_drop_77_features_oof_6231.csv'
    raw_chunks_pat = 'chunked_test_df/*.mp'
    chunk_paths = sorted(list(glob.glob(raw_chunks_pat)))
    col_order = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53',
                 'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 'class_90',
                 'class_92', 'class_95', 'class_99', 'object_id']


def patcher(test_feat_df, clfs, fnames_final, save_path):

    pth = 'feature_cache_dec_2b/0.mp'
    test_feat_df = pd.read_msgpack(pth)
    preds_df = make_pred_df(clfs, fnames_final, test_feat_df.reset_index())
    preds_df[COL_ORDER].to_csv(save_path, header=False, mode='a', index=False)

