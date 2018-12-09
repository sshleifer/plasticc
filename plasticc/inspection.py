import gc
import itertools

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

np.warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix

from .constants import *


def read_importances(pth='lgbm_importances.csv'):
    return agg_importances(pd.read_csv(pth))

def agg_importances(df):
    return df.drop_duplicates(['feature'], keep='first').set_index('feature')['mean_gain'].sort_values(ascending=False)


import matplotlib.pyplot as plt

import seaborn as sns
def plot_oid(train, oid):
    meta = meta_train[meta_train.object_id == oid].iloc[0]
    tit = f'id={meta.object_id} class={meta.target} photoz={meta.hostgal_photoz}'

    pl_data = train[train.object_id == oid]
    sns.scatterplot(x='mjd', y='flux', data=pl_data, hue='passband', )
    plt.title(tit)


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

def build_and_plot_cm(y, oof_preds, data_dir=DATA_DIR):
    cnf_matrix = make_cnf_matrix(oof_preds, y)
    np.set_printoptions(precision=2)
    sample_sub = pd.read_csv(data_dir / 'sample_submission.csv')
    class_names = list(sample_sub.columns[1:-1])
    del sample_sub;
    gc.collect()

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(12, 12))
    foo = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                                title='Confusion matrix')
    return foo


def make_cnf_matrix(oof_preds, y):
    unique_y = np.unique(y)
    class_map = dict()
    for i, val in enumerate(unique_y):
        class_map[val] = i
    y_map = np.array([class_map[val] for val in y])
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_map, np.argmax(oof_preds, axis=-1))
    return cnf_matrix


classes = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53',
         'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 'class_90',
         'class_92', 'class_95']


def put_obj_id_in_correct_chunk(dirname):

    raise NotImplementedError()


