import pandas as pd

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import gc
import os
import matplotlib.pyplot as plt
import itertools
import pickle, gzip
import glob
from sklearn.preprocessing import StandardScaler
from tsfresh.feature_extraction import extract_features
np.warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix

from .constants import *


def read_importances(pth='lgbm_importances.csv'):
    return agg_importances(pd.read_csv(pth))

def agg_importances(df):
    return df.drop_duplicates(['feature'], keep='first').set_index('feature')['mean_gain'].sort_values(ascending=False)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def build_and_plot_cm(y, oof_preds):
    unique_y = np.unique(y)
    class_map = dict()
    for i, val in enumerate(unique_y):
        class_map[val] = i
    y_map = np.array([class_map[val] for val in y])

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_map, np.argmax(oof_preds, axis=-1))
    np.set_printoptions(precision=2)

    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    class_names = list(sample_sub.columns[1:-1])
    del sample_sub;
    gc.collect()

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(12, 12))
    foo = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                                title='Confusion matrix')
    return foo

classes = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53',
         'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 'class_90',
         'class_92', 'class_95']
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


def dedup_and_gen_unknown(df, new_path):
    df = dedup_sub(df, ['object_id'])
    if df.shape[0] != SUB_SIZE:
        print(f'df is only {df.shape[0]} rows. Expecting {SUB_SIZE:,}.')
    class_99 = GenUnknown(df)
    df['class_99'] = class_99
    assert 'object_id' in df.columns, df.columns
    df.to_csv(new_path, index=False)

    return new_path


def dedup_sub(df, subset):
    dups = df.duplicated(subset=subset)
    if dups.sum() > 0:
        print(f'Deleting {dups.sum()} duplicate {subset}')
        df = df.loc[~dups]
    return df


def compare_subs(sub1, sub2, n=300):
    deltas = sub1.sort_values('object_id').tail(n) - sub2.sort_values('object_id').tail(n)
    by_class = deltas.abs().describe()
    overall = deltas.stack().abs().describe()
    by_class.loc['OVERALL'] = overall
    return by_class

def put_obj_id_in_correct_chunk(dirname):

    raise NotImplementedError()


