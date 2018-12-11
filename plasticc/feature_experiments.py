import pandas as pd
from tqdm import tqdm_notebook

from plasticc.constants import LGB_PARAMS
from plasticc.forked_lgbm_script import lgbm_modeling_cross_validation


def try_adding_features(xdf, y, fnames_start, cand_fnames, benchmark_score=.5109,
                        nr_fold=5):
    scores = {}
    cur_added = pd.Index([])
    best = benchmark_score
    for fname in tqdm_notebook(cand_fnames):
        cur_fnames = cur_added.union([fname])
        clfs, bm_score, imp_new14, oof_preds_cw, normed_score = lgbm_modeling_cross_validation(
            LGB_PARAMS, xdf[fnames_start.union(cur_fnames)], y, nr_fold=nr_fold, smote=False
        )
        scores.append((cur_fnames, bm_score))
        if bm_score < best:
            cur_added = cur_fnames
            print(f'adding {cur_added} for score={bm_score:.4f}')
            best = bm_score
    return scores


def try_removing_features(xdf, y, fnames_start, cand_fnames, benchmark_score=.5109,
                          nr_fold=5):
    scores = {}
    cur_dropped = pd.Index([])
    best = benchmark_score
    for fname in tqdm_notebook(cand_fnames):
        to_drop = cur_dropped.union([fname])
        clfs, bm_score, imp_new14, oof_preds_cw, normed_score = lgbm_modeling_cross_validation(
            LGB_PARAMS, xdf[fnames_start.drop(to_drop)], y, nr_fold=nr_fold, smote=False
        )
        scores.append((to_drop, bm_score))
        if bm_score < best:
            cur_dropped = to_drop
            print(f'dropping {cur_dropped} for score={bm_score:.4f}')
            best = bm_score
    return scores  # sorted(scores, key=lambda x: x[1])
