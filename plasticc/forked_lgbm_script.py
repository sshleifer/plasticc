"""
This script is forked from iprapas's notebook 
https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135

#    https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data
#    https://www.kaggle.com/c/PLAsTiCC-2018/discussion/70908
#    https://www.kaggle.com/meaninglesslives/simple-neural-net-for-time-series-classification
#
"""

import gc
import glob
import sys
import time
from datetime import datetime as dt

import os

from plasticc.constants import OBJECT_ID, CLASSES, CLASS_WEIGHTS, BEST_SWEIGHTS

PRED_99_AVG = 0.14

gc.enable()
from functools import partial

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra

np.warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from tsfresh.feature_extraction import extract_features
from lightgbm import LGBMClassifier
from tqdm import *
from numba import jit

from .constants import DATA_DIR, DTYPES, aggs, fcp, LGB_PARAMS, TEST_SET_SIZE


@jit
def haversine_plus(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) from 
    #https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance
    -between-two-gps-points
    """
    # Convert decimal degrees to Radians:
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    # Implementing Haversine Formula:
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),
               np.multiply(np.cos(lat1),
                           np.multiply(np.cos(lat2),
                                       np.power(np.sin(np.divide(dlon, 2)), 2))))

    haversine = np.multiply(2, np.arcsin(np.sqrt(a)))
    return {
        'haversine': haversine,
        'latlon1': np.subtract(np.multiply(lon1, lat1), np.multiply(lon2, lat2)),
    }


@jit
def process_flux(df):
    if 'flux_by_flux_ratio_sq' in df.columns:
        return df
    flux_ratio_sq = np.power(df['flux'].values / df['flux_err'].values, 2.0)

    df_flux = pd.DataFrame({
        'flux_ratio_sq': flux_ratio_sq,
        'flux_by_flux_ratio_sq': df['flux'].values * flux_ratio_sq, },
        index=df.index)

    return pd.concat([df, df_flux], axis=1)


@jit
def process_flux_agg(df):
    flux_w_mean = df['flux_by_flux_ratio_sq_sum'].values / df['flux_ratio_sq_sum'].values
    flux_diff = df['flux_max'].values - df['flux_min'].values

    df_flux_agg = pd.DataFrame({
        'flux_w_mean': flux_w_mean,
        'flux_diff1': flux_diff,
        'flux_diff2': flux_diff / df['flux_mean'].values,
        'flux_diff3': flux_diff / flux_w_mean,
    }, index=df.index)

    return pd.concat([df, df_flux_agg], axis=1)


def featurize(df, df_meta, aggs, fcp):
    """
    Extracting Features from train set
    Features from olivier's kernel
    very smart and powerful feature that is generously given here
    https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    per passband features with tsfresh library. fft features added to capture periodicity
    https://www.kaggle.com/c/PLAsTiCC-2018/discussion/70346#415506
    """

    df = process_flux(df)

    agg_df = df.groupby(OBJECT_ID).agg(aggs)
    agg_df.columns = ['{}_{}'.format(k, agg) for k in aggs.keys() for agg in aggs[k]]
    agg_df = process_flux_agg(agg_df)
    df.sort_values('mjd', inplace=True)
    default_params = dict(column_id=OBJECT_ID, disable_progressbar=True, column_sort='mjd')
    # Add more features with tsfresh
    agg_df_ts_flux_passband = extract_features(
        df, column_kind='passband', column_value='flux',
        default_fc_parameters=fcp['flux_passband'], **default_params
    )

    agg_df_ts_flux = extract_features(
        df, column_value='flux', default_fc_parameters=fcp['flux'], **default_params)

    agg_df_ts_flux_by_flux_ratio_sq = extract_features(
        df, column_value='flux_by_flux_ratio_sq',
        default_fc_parameters=fcp['flux_by_flux_ratio_sq'], **default_params
    )

    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    # dt[detected==1, mjd_diff:=max(mjd)-min(mjd), by=object_id]
    df_det = df[df['detected'] == 1].copy()
    agg_df_mjd = extract_features(
        df_det, column_id=OBJECT_ID,
        column_value='mjd', default_fc_parameters=fcp['mjd'], disable_progressbar=True)
    agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'].values - agg_df_mjd[
        'mjd__minimum'].values
    del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']
    agg_df_ts = pd.concat([agg_df,
                           agg_df_ts_flux_passband,
                           agg_df_ts_flux,
                           agg_df_ts_flux_by_flux_ratio_sq,
                           agg_df_mjd], axis=1).rename_axis(OBJECT_ID).reset_index()

    result = agg_df_ts.merge(right=df_meta, how='left', on=OBJECT_ID)
    return result


def process_meta(filename):
    meta_df = pd.read_csv(filename)

    meta_dict = dict()
    # distance
    meta_dict.update(haversine_plus(meta_df['ra'].values, meta_df['decl'].values,
                                    meta_df['gal_l'].values, meta_df['gal_b'].values))
    #
    meta_dict['hostgal_photoz_certain'] = np.multiply(
        meta_df['hostgal_photoz'].values,
        np.exp(meta_df['hostgal_photoz_err'].values))

    meta_df = pd.concat([meta_df, pd.DataFrame(meta_dict, index=meta_df.index)], axis=1)
    return meta_df


OOF_PRED_COLS = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]


def make_oof_pred_df(oof_preds, columns=OOF_PRED_COLS):
    return pd.DataFrame(oof_preds, columns=columns).add_prefix('class_')


def multi_weighted_logloss(y_true, y_preds, classes=CLASSES, class_weights=CLASS_WEIGHTS):
    """Refactor from @author olivier https://www.kaggle.com/ogrellier."""
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weights[k] for k in sorted(class_weights.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


def lgbm_multi_weighted_logloss(y_true, y_preds, classes=CLASSES, class_weights=CLASS_WEIGHTS):
    """Refactor from olivier.multi logloss for PLAsTiCC challenge."""
    loss = multi_weighted_logloss(y_true, y_preds, classes=classes, class_weights=class_weights)
    return 'wloss', loss, False


def agg_importances(imp_df):
    return (imp_df.groupby('feature').gain.agg([np.mean, np.std])
            .add_suffix('_gain').sort_values(by='mean_gain', ascending=False)).round()


ILLEGAL_FNAMES = ['target', OBJECT_ID, 'hostgal_specz',
                  'gal_b', 'gal_l', 'ra', 'ddf', 'decl',  # 'distmod'
                  ]


def my_rfe(x, y, sorted_fnames, classes=CLASSES, class_weights=CLASS_WEIGHTS, max_n_to_delete=None,
           order=-1):
    '''if order is -1 try deleting least important features first.'''
    if max_n_to_delete is None:
        max_n_to_delete = len(sorted_fnames) - 1
    scores = {}
    for i in range(max_n_to_delete):
        fnames = sorted_fnames[:i * order]
        _, score, _, _ = lgbm_modeling_cross_validation(
            LGB_PARAMS, x[fnames], y, classes, class_weights, nr_fold=3,
        )
        scores[i] = score
    return scores

def add_jims_features():
    pass


def smoteAdataset(Xig_train, yig_train, Xig_test, yig_test):
    sm = SMOTE(random_state=2)
    Xig_train_res, yig_train_res = sm.fit_sample(Xig_train, yig_train.ravel())
    return Xig_train_res, pd.Series(yig_train_res), Xig_test, pd.Series(yig_test)


def lgbm_modeling_cross_validation(params, full_train, y, classes=CLASSES,
                                   class_weights=CLASS_WEIGHTS,
                                   nr_fold=5, random_state=1, sweights=BEST_SWEIGHTS, smote=False):
    full_train = full_train.drop(ILLEGAL_FNAMES, axis=1, errors='ignore')
    # assert 'distmod' in full_train.columns
    if sweights is None:
        # Compute weights
        w = y.value_counts()
        sweights = {i: np.sum(w) / w[i] for i in w.index}

    clfs = []
    importances = pd.DataFrame()
    folds = StratifiedKFold(n_splits=nr_fold,
                            shuffle=True,
                            random_state=random_state)

    oof_preds = np.zeros((len(full_train), np.unique(y).shape[0]))
    for fold_, (trn_, val_) in tqdm_notebook(enumerate(folds.split(y, y)), total=nr_fold):
        trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_train.iloc[val_], y.iloc[val_]
        if smote:
            trn_xa, trn_y, val_xa, val_y = smoteAdataset(
                trn_x.values, trn_y.values, val_x.values, val_y.values)
            trn_x = pd.DataFrame(data=trn_xa, columns=trn_x.columns)
            val_x = pd.DataFrame(data=val_xa, columns=val_x.columns)

        clf = LGBMClassifier(**params)
        loss_fn = lambda y, ypred: lgbm_multi_weighted_logloss(
            y, ypred, classes=classes, class_weights=class_weights
        )
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=loss_fn,
            verbose=-1,
            early_stopping_rounds=50,
            sample_weight=trn_y.map(sweights)
        )
        clfs.append(clf)

        oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        imp_df = pd.DataFrame({
            'feature': full_train.columns,
            'gain': clf.feature_importances_,
            'fold': [fold_ + 1] * len(full_train.columns),
        })
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

    score = multi_weighted_logloss(y_true=y, y_preds=oof_preds,
                                   classes=classes, class_weights=class_weights)
    print(f'OOF:{score:.4f} n_folds={nr_fold}, nfeat={full_train.shape[1]}')
    normal_weight_score = multi_weighted_logloss(y, oof_preds)
    if class_weights != CLASS_WEIGHTS:
        print(
            f'OOF Default weights:{normal_weight_score:.4f} n_folds={nr_fold}, '
            f'nfeat={full_train.shape[1]}')
    df_importances = agg_importances(importances)
    oof_df = make_oof_pred_df(oof_preds, columns=clf.classes_)
    return clfs, score, df_importances, oof_df, normal_weight_score


from sklearn.metrics import log_loss


def binary_lgbm_oof(params, xdf, y, nr_fold=3, random_state=1):
    xdf = xdf.drop(ILLEGAL_FNAMES, axis=1, errors='ignore')
    # assert 'distmod' in xdf.columns
    # Compute weights
    w = y.value_counts()
    weights = {i: np.sum(w) / w[i] for i in w.index}

    clfs = []
    folds = StratifiedKFold(n_splits=nr_fold,
                            shuffle=True,
                            random_state=random_state)
    oof_preds = np.zeros((len(xdf), np.unique(y).shape[0]))
    for fold_, (trn_, val_) in tqdm_notebook(enumerate(folds.split(y, y)), total=nr_fold):
        trn_x, trn_y = xdf.iloc[trn_], y.iloc[trn_]
        val_x, val_y = xdf.iloc[val_], y.iloc[val_]

        clf = LGBMClassifier(**params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            verbose=-1,
            early_stopping_rounds=50,
            sample_weight=trn_y.map(weights)
        )
        oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        clfs.append(clf)
    score = log_loss(y, oof_preds[:, 1])
    return clfs, score, oof_preds


def binary_lgbm(params, xdf, y, nr_fold=3, random_state=1):
    ret = []
    for class_num in tqdm_notebook(set(y)):
        targ = (y == class_num)
        ret.append(binary_lgbm(LGB_PARAMS, xdf, targ, nr_fold=nr_fold, random_state=random_state))
    return ret


def predict_chunk(df, clfs, meta_, fnames, featurize_configs, train_mean,
                  feat_cache_path=None):
    test_feat_df = featurize(df, meta_,
                             featurize_configs['aggs'],
                             featurize_configs['fcp'])
    test_feat_df.fillna(0, inplace=True)
    if feat_cache_path is not None:
        test_feat_df.to_msgpack(feat_cache_path)

    return make_pred_df(clfs, fnames, test_feat_df)


def make_pred_df(clfs, features, test_feat_df):
    preds = avg_predict_proba(clfs, test_feat_df[features])
    preds_df_ = pd.DataFrame(preds, columns=['class_{}'.format(s) for s in clfs[0].classes_])
    preds_df_['object_id'] = test_feat_df['object_id']
    return preds_df_


def avg_predict_proba(clfs, X_test):
    # Make predictions
    preds_ = None
    for clf in clfs:
        if preds_ is None:
            preds_ = clf.predict_proba(X_test)
        else:
            preds_ += clf.predict_proba(X_test)
    preds_ = preds_ / len(clfs)
    return preds_


CHUNKSIZE = 5000000


class AbstractFeatureAdder:
    def __init___(self, clfs, feat_dir, fnames):

        pass

    @staticmethod
    def add_features(raw_df, feat_df):
        """Make feat_df look like whatever clfs were trained on"""
        acor_params = {'partial_autocorrelation': [{'lag': 1}, ]}
        # raise NotImplementedError()
        return feat_df  # placeholder

    @staticmethod
    def predict(feature_add_fn, clfs, fnames, save_path, raw_chunks_pat='chunked_test_df/*.mp'):
        paths = list(glob.glob(raw_chunks_pat))
        for pth in tqdm_notebook(sorted(paths)):
            i_c = os.path.basename(pth)[:-3]
            feat_cache_path = f'{feat_dir}/{i_c}.mp'
            df = pd.read_msgpack(pth)
            feats = pd.read_msgpack(feat_cache_path).replace(0, np.nan)
            test_feat_df = feature_add_fn(df, feats)
            preds_df = make_pred_df(clfs, fnames, test_feat_df)
            if not os.path.exists(save_path):
                preds_df.to_csv(save_path, index=False)
            else:
                preds_df.to_csv(save_path, header=False, mode='a', index=False)
            del preds_df
            gc.collect()


def add_acor_feat(mock_tr, feat_df):
    acor_params = {'partial_autocorrelation': [{'lag': 1}, ]}
    X = extract_features(mock_tr, default_fc_parameters=acor_params,
                         column_id=OBJECT_ID, profile=True,
                         column_sort='mjd',
                         column_value='flux', disable_progressbar=True).rename_axis(OBJECT_ID)
    return feat_df.join(X)


def process_test(clfs, features, featurize_configs, train_mean,
                 feat_dir=None, filename='predictions.csv', chunksize=CHUNKSIZE):
    print(feat_dir)
    print(DTYPES)
    start = time.time()

    meta_test = process_meta(DATA_DIR / 'test_set_metadata.csv')
    # meta_test.set_index('object_id',inplace=True)

    remain_df = None
    chunked_df = pd.read_csv(DATA_DIR / 'test_set.csv', dtype=DTYPES,
                             chunksize=chunksize, iterator=True)
    n_chunks = int(np.ceil(TEST_SET_SIZE / chunksize))
    for i_c, df in tqdm_notebook(enumerate(chunked_df), total=n_chunks):
        # Check object_ids
        # I believe np.unique keeps the order of group_ids as they appear in the file
        unique_ids = np.unique(df['object_id'])

        new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()
        if remain_df is None:
            df = df.loc[df['object_id'].isin(unique_ids[:-1])]
        else:
            df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)
        # Create remaining samples df
        remain_df = new_remain_df
        if feat_dir is not None:
            feat_cache_path = f'{feat_dir}/{i_c}.mp'
            df.to_msgpack(f'chunked_test_df/{i_c}.mp')
            remain_df.to_msgpack('cur_remain_df.mp')  # for easier restarts

        preds_df = predict_chunk(df=df,
                                 clfs=clfs,
                                 meta_=meta_test,
                                 fnames=features,
                                 featurize_configs=featurize_configs,
                                 train_mean=train_mean,
                                 feat_cache_path=feat_cache_path)

        if i_c == 0:
            preds_df.to_csv(filename, header=True, mode='a', index=False)
        else:
            preds_df.to_csv(filename, header=False, mode='a', index=False)

        del preds_df
        gc.collect()
        print('{:15d} done in {:5.1f} minutes'.format(
            chunksize * (i_c + 1), (time.time() - start) / 60), flush=True)

    i_c = i_c + 1
    if feat_dir is not None:
        feat_cache_path = f'{feat_dir}/{i_c}.mp'
        remain_df.to_msgpack(f'chunked_test_df/{i_c}.mp')
    # Compute last object in remain_df
    preds_df = predict_chunk(df=remain_df,
                             clfs=clfs,
                             meta_=meta_test,
                             fnames=features,
                             featurize_configs=featurize_configs,
                             train_mean=train_mean,
                             feat_cache_path=feat_cache_path)

    preds_df.to_csv(filename, header=False, mode='a', index=False)
    return
