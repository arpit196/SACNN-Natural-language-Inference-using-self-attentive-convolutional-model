import os

import tensorflow as tf
from tqdm import tqdm

from data.data_loader import load_snli
from models.sdot_product_attention import ScaledDotProductAttentionSiameseNet
from models.cnn import CNNbasedSiameseNet
import numpy as np

data_fn = 'train_snli.txt'
logs_path = 'logs/'

# ++++++++++++++++++++++++++++++++++
embedding_size = 64
hidden_size = 256
num_epochs = 10
batch_size = 128
eval_every = 10
# ++++++++++++++++++++++++++++++++++

sen1, sen2, labels, max_doc_len, vocabulary_size = load_snli(data_fn)
num_batches = len(labels) // batch_size
print('Num batches: ', num_batches)

train_sen1 = sen1[:-1000]
train_sen2 = sen2[:-1000]
train_labels = labels[:-1000]

eval_sen1 = sen1[-1000:]
eval_sen2 = sen2[-1000:]
eval_labels = labels[-1000:]

with tf.Session() as session:
    model = CNNbasedSiameseNet(max_doc_len, vocabulary_size, embedding_size, hidden_size)
    global_step = 0

    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    session.run(init)
    session.run(init_local)
    if not os.path.isdir(logs_path):
        os.makedirs(logs_path)
    eval_summary_writer = tf.summary.FileWriter(logs_path + 'eval', graph=session.graph)
    train_summary_writer = tf.summary.FileWriter(logs_path + 'train', graph=session.graph)

    metrics = {'acc': 0.0}
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        tqdm_iter = tqdm(range(num_batches),
                                total=num_batches,
                                desc="Batches",
                                leave=False,
                                postfix=metrics)
        shuffle_idxs = np.random.permutation(range(len(train_labels)))
        train_sen1 = train_sen1[shuffle_idxs]
        train_sen2 = train_sen2[shuffle_idxs]
        train_labels = train_labels[shuffle_idxs]

        # ++++++++
        # small train set for measuring train accuracy
        small_train1 = train_sen1[-1000:]
        small_train2 = train_sen2[-1000:]
        small_train_labels = train_labels[-1000:]

        for batch in tqdm_iter:
            global_step += 1
            x1_batch = train_sen1[batch * batch_size:(batch + 1) * batch_size]
            x2_batch = train_sen2[batch * batch_size:(batch + 1) * batch_size]
            y_batch = train_labels[batch * batch_size:(batch+1) * batch_size]
            feed_dict = {model.x1: x1_batch, model.x2: x2_batch, model.labels: y_batch}
            loss, _ = session.run([model.loss, model.opt], feed_dict=feed_dict)
            if batch % eval_every == 0:
                feed_dict = {model.x1: small_train1, model.x2: small_train2, model.labels: small_train_labels}
                train_summary = session.run(model.summary_op, feed_dict=feed_dict)
                train_summary_writer.add_summary(train_summary, global_step)

                feed_dict = {model.x1: eval_sen1, model.x2: eval_sen2, model.labels: eval_labels}
                accuracy, summary = session.run([model.accuracy, model.summary_op], feed_dict=feed_dict)
                eval_summary_writer.add_summary(summary, global_step)
                tqdm_iter.set_postfix(acc=accuracy)





