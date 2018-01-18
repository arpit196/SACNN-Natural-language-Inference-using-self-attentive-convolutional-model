import numpy as np
import pandas as pd
from tflearn.data_utils import VocabularyProcessor


def load_snli(data_fn):
    data = pd.read_csv(data_fn, delimiter='\t', header=None, names=['sen1', 'sen2', 'labels'], na_values='')

    X = data[['sen1', 'sen2']].as_matrix()
    num_instances, num_classes = X.shape
    X = X.ravel()
    X = [str(x) for x in list(X)]
    max_sen_len = max([len(str(x).split(' ')) for x in list(X)])
    vocab_processor = VocabularyProcessor(max_sen_len)
    vec_X = np.array(list(vocab_processor.fit_transform(X)))
    voc_size = len(vocab_processor.vocabulary_._mapping)
    vec_X = vec_X.reshape(num_instances, num_classes, max_sen_len)

    sen1 = vec_X[:, 0, :]
    sen2 = vec_X[:, 1, :]
    labels = data['labels'].as_matrix()
    labels = np.reshape(labels, (-1, 1))
    return sen1, sen2, labels, max_sen_len, voc_size


def data_split():
    pass