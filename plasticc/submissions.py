col_order = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53',
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

from plasticc.constants import OBJECT_ID

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
            preds_df[col_order].to_csv(save_path, index=False)
        else:
            preds_df[col_order].to_csv(save_path, header=False, mode='a', index=False)
        test_feat_df.to_msgpack(f'feature_cache_nov_28/{i_c}.mp')
        del preds_df, test_feat_df
        gc.collect()


