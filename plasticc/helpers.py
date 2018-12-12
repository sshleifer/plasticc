import os
from tsfresh import extract_features
import funcy
import pandas as pd
from tqdm import *

from .constants import *
import funcy
import os
import pandas as pd
from tqdm import *
from tsfresh import extract_features

from .constants import *


def settings_from_cols(cols):
    return tsfresh.feature_extraction.settings.from_columns(cols)

def flatten(names, sep='_'):
    """turn iterable of strings into _ separated string, or return itself if string is passed."""
    return sep.join(map(str, names)) if not isinstance(names, str) else names


def flatten_cols(arg_df, sep='_'):
    """Turn multiindex into single index. Does not mutate."""
    df = arg_df.copy()
    df.columns = df.columns.map(lambda x: flatten(x, sep=sep))
    return df

import gc
def ray_target_encode(train, col_groups, kf):
    col_no = 0
    for col in col_groups:
        print('deal_probability: {}'.format(col))
        if col != 'deal_probability':
            vals = []
            for train_inx, val_inx in kf.split(list(range(len(train)))):
                gby = train.loc[train_inx, col + ['deal_probability']].groupby(col).agg(
                    {'deal_probability': 'mean'}).reset_index()
                colname = 'mean_attributed_{}'.format(str(col_no))
                gby.columns = col + [colname]
                gby[colname] = gby[colname].astype(np.float32)
                val = train.loc[val_inx, col].reset_index(drop=False)
                val = val.merge(gby, on=col, how='left')
                del gby
                gc.collect()
                vals.append(val[['index', colname]])
                del val
                gc.collect()

            cur_val = pd.concat(vals, axis=0).set_index('index')
            cur_val.reindex(train.index)
            train[colname] = cur_val[colname]
            del cur_val, vals
            gc.collect()
        col_no += 1
    col_no = 0
    for col in col_groups:
        colname = 'mean_attributed_{}'.format(str(col_no))
        gby = train.loc[:, col + ['deal_probability']].groupby(col).agg(
            {'deal_probability': 'mean'}).reset_index()
        gby.columns = col + [colname]
        gby[colname] = gby[colname].astype(np.float32)
        test = test.merge(gby, how='left', on=col)
        col_no += 1





def add_dope_features(xdf10):
    xdf10['sq_dist'] = xdf10['hostgal_photoz'] ** 2
    xdf10['min_over_max_fband'] = xdf10['max_fluxband'] / xdf10['min_fluxband']
    xdf10['det_flux_max_over_min'] = xdf10['det_flux_max'] / xdf10['det_flux_min']
    xdf10['max_fluxband_times_flux_mean'] = xdf10['flux_mean'] * xdf10['max_fluxband']
    xdf10['flux__longest_strike_above_mean_times_sq_dist'] = xdf10['sq_dist'] * xdf10['flux__longest_strike_above_mean']



def _add_useless_median_features_(xdf):
    xdf['max_median_passband'] = xdf[['0__median', '1__median', '2__median',
                                      '3__median', '4__median', '5__median', ]].idxmax(1).str.slice(
        0, 1).astype(int)
    xdf['max_median_val'] = xdf[['0__median', '1__median', '2__median',
                                 '3__median', '4__median', '5__median', ]].max(1)
    more_feats = xdf[['0__median', '1__median', '2__median',
                      '3__median', '4__median', '5__median', ]].apply(
        lambda x: x / xdf['max_median_val']).add_suffix('_over_max_median_val')
    xdf['median_2_over_5'] = xdf['2__median'] / xdf['5__median']
    xdf['median_2_over_4'] = xdf['2__median'] / xdf['4__median']
    xdf = xdf.join(more_feats)

    xdf['flux_err_std_ratio'] = xdf['flux_std'] / xdf['flux_err_std']
    xdf['flux_std_over_err_max'] = xdf['flux_std'] / xdf['flux_err_max']
    xdf['flux_std_over_err_max'] = xdf['flux_std'] / xdf['flux_err_max']


def add_more_dope_features(xdf):
    for stat in ['flux_min', 'flux_max', 'flux_mean', 'flux_median', 'flux_std', 'flux_skew']:
        xdf[f'undet_over_det_{stat}'] = xdf[f'undet_{stat}'] / xdf[f'det_{stat}']

import pickle
def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_default_index():
    if os.path.exists('default_index.pkl'):
        return pickle_load('default_index.pkl')
    else:
        return pickle_load('~/plasticc/default_index.pkl')


def add_ratio_inputs(xdf10, ratio_inputs):
    add_dope_features(xdf10)
    add_more_dope_features(xdf10)
    for c in ratio_inputs:
        xdf10[f'{c}_times_sq_dist'] = xdf10[c] * xdf10['sq_dist']
        xdf10[f'{c}_over_det_min'] = xdf10[c] / xdf10['det_flux_min']
        xdf10[f'{c}_over_det_max'] = xdf10[c] / xdf10['det_flux_max']
        xdf10[f'{c}_over_det_mean'] = xdf10[c] / xdf10['det_flux_mean']
        xdf10[f'{c}_over_det_median'] = xdf10[c] / xdf10['det_flux_median']
    return xdf10


DEFAULT_DENOMS = [
   '0__cid_ce__normalize_True_times_sq_dist',
   '0__fft_coefficient__coeff_0__attr_"abs"_times_sq_dist',
   '5__fft_coefficient__coeff_0__attr_"abs"_times_sq_dist',
   'flux__longest_strike_above_mean_times_sq_dist',
   'flux_max_times_sq_dist', 'flux_w_mean_times_sq_dist',
   'mjd_det__cid_ce__normalize_False', 'sq_dist'
]


def add_ratio_inputs2(xdf10, numerators, denoms=DEFAULT_DENOMS):
    add_more_dope_features(xdf10)
    for c in numerators:
        for d in denoms:
            if c == d:
                continue
            xdf10[f'{c}_over_{d}'] = xdf10[c] / xdf10[d]
    return xdf10





def make_fluxband_idx_feats(train):
    gb = train.groupby((OBJECT_ID, PASSBAND))
    passband_det_means = gb.detected.mean().unstack().add_prefix('mn_detected_')
    fluxband_idx_feats = pd.DataFrame(
        dict(
            max_std_fluxband=gb.flux.std().unstack().idxmax(1),
            max_fluxband=gb.flux.max().unstack().idxmax(1),
            min_fluxband=gb.flux.min().unstack().idxmin(1),
            max_absfluxband=gb.abs_flux.max().unstack().idxmax(1),
        )
    )
    passband_flux_stats = gb.flux.agg(['std', 'max']).add_suffix('_flux').unstack().pipe(
        flatten_cols)
    return fluxband_idx_feats.join(passband_det_means).join(passband_flux_stats)


def make_det_df_stats(train):
    det_df = train[train.detected == 1]
    undet_df = train[train.detected == 0]
    det_flux_stats = det_df.groupby(OBJECT_ID).flux.agg(BASE_AGGS)
    undet_flux_stats = undet_df.groupby(OBJECT_ID).flux.agg(BASE_AGGS)
    # ratios = (det_flux_stats / undet_flux_stats)
    flux_by_det_feats = pd.concat(
        [det_flux_stats.add_prefix('det_flux_'), undet_flux_stats.add_prefix('undet_flux_'),
         # ratios.add_prefix('ratio')
         ], axis=1)
    return flux_by_det_feats


def make_dec4_feats(train):
    return pd.concat([make_det_df_stats(train), make_fluxband_idx_feats(train)],
                     axis=1)

def idx_renamer(idx, rename_dct): return pd.Index([rename_dct.get(x, x) for x in idx])

def make_hostgal_ratio_feats(xdf4):
    """Useless I think, could try again."""
    xdf4['hostgal_photoz_certain_ratio'] = (xdf4['hostgal_photoz'] / xdf4['hostgal_photoz_certain'])
    xdf4['hostgal_err_ratio'] = xdf4['hostgal_photoz_err'] / xdf4['hostgal_photoz']
    xdf4['hostgal_err_certain'] = xdf4['hostgal_photoz_err'] / xdf4['hostgal_photoz']
    return xdf4


def get_membership_mask(candidates, collection_they_might_be_in) -> np.ndarray:
    """Return a boolean list where entry i indicates whether candidates[i] is in the second arg."""
    return np.array([x in collection_they_might_be_in for x in candidates])


# TODO add smart TQDM
def tqdm_chunks(collection, chunk_size, enum=False):
    """Call funcy.chunks and return the resulting generator wrapped in a progress bar."""
    tqdm_nice = tqdm_notebook  # if in_notebook() else tqdm
    chunks = funcy.chunks(chunk_size, collection)
    if enum:
        chunks = enumerate(chunks)

    return tqdm_nice(chunks, total=int(np.ceil(len(collection) / chunk_size)))


def difference_join(left_df, right_df):
    result = left_df.join(right_df[right_df.columns.difference(left_df.columns)])
    if result.count().min() < left_df.shape[0] / 2:
        print('There are many nans')
    return result


TSKW = dict(column_id=OBJECT_ID, column_sort='mjd')


def assign_feature_178(xdf13):
    """Doesn't obviously help."""
    num = ['flux_by_flux_ratio_sq1__quantile__q_0.7']
    denom = ['2__quantile__q_0.9_over_det_min']
    xdf14 = add_ratio_inputs2(xdf13, num, denom)
    return xdf14

def make_extra_ts_featurs(train, meta_train):
    feats = []
    feats2 = []
    feats3 = []
    for chk in tqdm_chunks(meta_train.object_id.unique(), 1000):
        slc = train[get_membership_mask(train[OBJECT_ID], set(chk))]
        extracted_features = extract_features(
            slc, EXTRA_FLUX_PARS,
            column_value='flux',
            disable_progressbar=True,
            **TSKW
        )
        feats.append(extracted_features)
        extracted_features2 = extract_features(
            slc, FC_PASSBAND_V2,
            column_value='flux',
            column_kind='passband',
            disable_progressbar=True,
            **TSKW
        )

        feats2.append(extracted_features2)
        extracted_features3 = extract_features(
            slc,
            column_value='flux_by_flux_ratio_sq',
            column_kind='passband',
            disable_progressbar=True,
            **TSKW
        )
        feats3.append(extracted_features3)
    new_feat_df = pd.concat(feats)
    new_feat_df2 = pd.concat(feats2)
    new_feat_df3 = pd.concat(feats3)

    catted = pd.concat([new_feat_df, new_feat_df2,
                        new_feat_df3.add_prefix('flux_by_flux_ratio_sq')], axis=1)
    return catted


def tsfresh_joiner(df, feat_df, settings, disable_bar=True, **kwargs):
    X = _tsfresh_extract(df, settings, disable_bar=disable_bar, **kwargs)
    return feat_df.join(X)


def _tsfresh_extract(df, settings, disable_bar=True, **kwargs):
    X = extract_features(
        df, default_fc_parameters=settings, column_id=OBJECT_ID, profile=True,
        column_sort='mjd', disable_progressbar=disable_bar, **kwargs
    ).rename_axis(OBJECT_ID)
    return X


from collections import defaultdict


def merge_pars(pars40):
    """BROKEN AF"""
    base = defaultdict(list)
    for k in list(pars40.values()):
        for c, v in k.items():
            if v and v not in base[c]:
                base[c].extend(v)
            else:
                base[c] = v
    return dict(base)



def fix_sq_colnames(x):
    if x.startswith('flux_by_flux_ratio_sq'):
        return 'sq_'.join(x.split('sq'))
    else:
        return x


def zip_to_series(a,b):
    assert len(a) == len(b)
    return pd.Series(funcy.zipdict(a,b))


def wt_test(sub1, sub2, y):
    wt = np.arange(0, 1. , .01); wt
    scores = {}
    for w in wt:
        preds = (sub1 * w) +  (sub2 * (1-w))
        scores[w] = multi_weighted_logloss(y, preds.values)
    scores = pd.Series(scores)
    print(f'best weight for sub1: {scores.idxmin():.2f} for OOF: {scores.min():.4f} ')
    return scores


def wt_test3(sub1, sub2, sub3, y, step=.03):
    wt = np.round(np.arange(0, 1, step),2);
    if 1 not in wt:
        wt = list(wt)
        wt.append(1)
    scores = {}
    for w1 in tqdm_notebook(wt):
        for w2 in wt:
            w3 = 1 - (w2 + w1)
            if w3 < 0:
                continue
            preds = (sub1 * w1) + (sub2 * w2) + (sub3 * w3)
            scores[(w1,w2,w3)] = multi_weighted_logloss(y, preds.values)
    scores = pd.Series(scores)
    print(f'best weights: {scores.idxmin()} for OOF: {scores.min():.4f} ')
    return scores


def divide_by_rowsum(y_preds):
    if isinstance(y_preds,pd.DataFrame):
        row_sum = y_preds.sum(1)
        return y_preds.apply(lambda x: x / row_sum)
    else:
        row_sum = y_preds.sum(1).reshape(y_preds.shape[0], 1)
        return y_preds / row_sum




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
