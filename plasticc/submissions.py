from plasticc.inspection import classes

COL_ORDER = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53',
       'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 'class_90',
       'class_92', 'class_95', 'class_99', 'object_id']
import gc
import glob

import os
from tqdm import *
from tsfresh.feature_extraction import extract_features

from plasticc.constants import OBJECT_ID, fcp_improved, SUB_SIZE

PRED_99_AVG = 0.14  # old

gc.enable()

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra

np.warnings.filterwarnings('ignore')
from .forked_lgbm_script import make_pred_df
from .constants import MASSIVE_RENAMER


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


import funcy
def sub_from_dir(feat_dir, clfs, fnames, save_path=None):
    preds = []
    chunk_paths = glob.glob(f'{feat_dir}/*.mp')
    for pth in tqdm_notebook(chunk_paths):
        test_feat_df = pd.read_msgpack(pth).rename(columns=funcy.flip(MASSIVE_RENAMER))
        preds_df = make_pred_df(clfs, fnames, test_feat_df.reset_index()).set_index(OBJECT_ID)
        preds.append(preds_df)
    df = pd.concat(preds)
    class99 = GenUnknown(df)
    df['class_99'] = class99
    if save_path is not None:
        if OBJECT_ID in df.columns:
            df = df.set_index(OBJECT_ID)
        df.to_csv(save_path)
        print(f'saved to {save_path}')
    return df





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


def validate_sub(df) -> None:
    desired_cols = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53',
     'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 'class_90',
     'class_92', 'class_95', 'class_99', 'object_id']
    desired_shape = (3492890, 16)
    extra_cols = df.columns.difference(desired_cols)
    missing_cols = set(desired_cols).difference(df.columns)
    assert len(missing_cols) == 0, f'missing column {missing_cols}'
    assert len(extra_cols) == 0, f'extra column {missing_cols}'
    assert df.shape == desired_shape, f'expected shape {desired_shape} got {df.shape}'


def patcher(test_feat_df, clfs, fnames_final, save_path):

    pth = 'feature_cache_dec_2b/0.mp'
    test_feat_df = pd.read_msgpack(pth)
    preds_df = make_pred_df(clfs, fnames_final, test_feat_df.reset_index())
    preds_df[COL_ORDER].to_csv(save_path, header=False, mode='a', index=False)


def dedup_and_gen_unknown(df, new_path):
    if df.shape[0] != SUB_SIZE:
        df = dedup_sub(df, ['object_id'])
        if df.shape[0] != SUB_SIZE:
            print(f'df is only {df.shape[0]} rows. Expecting {SUB_SIZE:,}.')
    class_99 = GenUnknown(df)
    df['class_99'] = class_99
    assert 'object_id' in df.columns, df.columns
    df.to_csv(new_path, index=False)
    return new_path


def compare_subs(a, b):
    """Total row distance, averaged over rows."""
    intersection =   a.index.intersection(b.index)
    return (a.loc[intersection] - b.loc[intersection]).abs().sum(1).mean()



def dedup_sub(df, subset):
    dups = df.duplicated(subset=subset)
    if dups.sum() > 0:
        print(f'Deleting {dups.sum()} duplicate {subset}')
        df = df.loc[~dups]
    return df


def GenUnknown(preds_df):
    feats = classes
    data = pd.DataFrame()
    data['mymean'] = preds_df[feats].mean(axis=1)
    data['mymedian'] = preds_df[feats].median(axis=1)
    data['mymax'] = preds_df[feats].max(axis=1)
    return (0.5 + 0.5 * data["mymedian"] + 0.25 * data["mymean"] - 0.5 * data["mymax"] ** 3) / 2


def msg_pack_to_csv(path):
    base, _ = os.path.splitext(path)
    new_path = f'{base}.csv'
    assert not os.path.exists(new_path), new_path
    pd.read_msgpack(path).to_csv(new_path)
    return new_path


def compare_subs(sub1, sub2, n=300):
    deltas = sub1.sort_values('object_id').tail(n) - sub2.sort_values('object_id').tail(n)
    by_class = deltas.abs().describe()
    overall = deltas.stack().abs().describe()
    by_class.loc['OVERALL'] = overall
    return by_class


def mn_compare_df(uneven, best):
    return pd.DataFrame(dict(delta=uneven.mean() - best.mean(), orig=best.mean())).round(3).sort_values(
        'orig', ascending=False)
