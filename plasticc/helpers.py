
from tsfresh import extract_features
import funcy
import pandas as pd
from tqdm import *

from plasticc.constants import OBJECT_ID
from .constants import *



def get_membership_mask(candidates, collection_they_might_be_in) -> np.ndarray:
    """Return a boolean list where entry i indicates whether candidates[i] is in the second arg."""
    return np.array([x in collection_they_might_be_in for x in candidates])


#TODO add smart TQDM
def tqdm_chunks(collection, chunk_size, enum=False):
    """Call funcy.chunks and return the resulting generator wrapped in a progress bar."""
    tqdm_nice = tqdm_notebook #if in_notebook() else tqdm
    chunks = funcy.chunks(chunk_size, collection)
    if enum:
        chunks = enumerate(chunks)

    return tqdm_nice(chunks, total=int(np.ceil(len(collection) / chunk_size)))

def difference_join(left_df, right_df):
    result  = left_df.join(right_df[right_df.columns.difference(left_df.columns)])
    if result.count().min() < left_df.shape[0] / 2:
        print('There are many nans')
    return result


TSKW  =dict(column_id=OBJECT_ID, column_sort='mjd')


def add_features(chunk_paths, save_dir):
    for pth in tqdm_notebook(chunk_paths):
        i_c = os.path.basename(pth)[:-3]
        feat_cache_path = f'{save_dir}/{i_c}.mp'
        df = pd.read_msgpack(pth).reset_index()
        feats = pd.read_msgpack(feat_cache_path)
        test_feat_df = tsfresh_joiner(df, feats, NEW_PARS['flux'], disable_bar=False)

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

    catted=  pd.concat([new_feat_df, new_feat_df2,
                        new_feat_df3.add_prefix('flux_by_flux_ratio_sq')], axis=1)
    return catted



def tsfresh_joiner(df, feat_df, settings, disable_bar=True, **kwargs):
    X = _tsfresh_extract(df, settings, disable_bar=disable_bar, **kwargs)
    return feat_df.join(X)


def _tsfresh_extract(df, settings, disable_bar=True, **kwargs):
    X = extract_features(
        df, default_fc_parameters=settings, column_id=OBJECT_ID, profile=True,
        column_sort='mjd',  disable_progressbar=disable_bar, **kwargs
    ).rename_axis(OBJECT_ID)
    return X

from collections import defaultdict
def merge_pars(pars40):
    # BROKEN AF
    base = defaultdict(list)
    for k in list(pars40.values()):
        for c, v in k.items():
            if v and v not in base[c]:
                base[c].extend(v)
            else:
                base[c] = v
    return dict(base)

