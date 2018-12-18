
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import chain
sns.set_style('whitegrid')
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', RuntimeWarning)
import cesium.featurize as featurize
from gatspy.periodic import LombScargleMultiband, LombScargleMultibandFast
import pdb


def make_lists(train_series):
    groups = train_series.groupby(['object_id', 'passband'])
    times = groups.apply(
        lambda block: block['mjd'].values).reset_index().rename(columns={0: 'seq'})
    flux = groups.apply(
        lambda block: block['flux'].values
    ).reset_index().rename(columns={0: 'seq'})
    err = groups.apply(
        lambda block: block['flux_err'].values
    ).reset_index().rename(columns={0: 'seq'})
    det = groups.apply(
        lambda block: block['detected'].astype(bool).values
    ).reset_index().rename(columns={0: 'seq'})
    times_list = times.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
    flux_list = flux.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
    err_list = err.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
    det_list = det.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
    return times_list, flux_list, err_list, det_list


def fit_multiband_freq(tup):
    idx, group = tup
    t, f, e, b = group['mjd'], group['flux'], group['flux_err'], group['passband']
    model = LombScargleMultiband(fit_period=True)
    model.optimizer.period_range = (0.1, int((group['mjd'].max() - group['mjd'].min()) / 2))
    model.fit(t, f, e, b)
    return model


def get_freq_features(N, train_series, times_list, flux_list, train_metadata,
                      subsetting_pos=None):
    if subsetting_pos is None:
        subset_times_list = times_list
        subset_flux_list = flux_list
    else:
        subset_times_list = [v for i, v in enumerate(times_list)
                             if i in set(subsetting_pos)]
        subset_flux_list = [v for i, v in enumerate(flux_list)
                            if i in set(subsetting_pos)]
    feats = featurize.featurize_time_series(
        times=subset_times_list[:N],
        values=subset_flux_list[:N],
        features_to_use=['skew',
                        'percent_beyond_1_std',
                        'percent_difference_flux_percentile'
                        ],
        scheduler=None,

    )
    subset = train_series[train_series['object_id'].isin(
        train_metadata['object_id'].iloc[subsetting_pos].iloc[:N])]
    models = list(map(fit_multiband_freq, subset.groupby('object_id')))
    feats['object_pos'] = subsetting_pos[:N]
    feats['freq1_freq'] = [model.best_period for model in models]
    return feats, models
