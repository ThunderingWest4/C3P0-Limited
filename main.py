import numpy as np
import tensorflow as tf
import dataset
from termcolor import colored
import network
import os.path
import os
import pickle

print(colored("Successfully imported packages", "green"))

CITATION="""
@inproceedings{
    Tiedemann2012ParallelData,
    author = {Tiedemann, J},
    title = {Parallel Data, Tools and Interfaces in OPUS},
    booktitle = {LREC}
    year = {2012}
}
"""
BATCH_SIZE = 64

if not (os.path.exists("train_ds.pickle")): #only check for one bc if one exists, both do and vice versa

    # Download the file
    path_to_zip = tf.keras.utils.get_file(
        'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract=True)

    en, es, en_map, es_map = dataset.getData(os.path.dirname(path_to_zip) + "\\spa-eng\\spa.txt")

    en_train = en[0:100000]
    es_train = es[0:100000]
    BUFFER_SIZE = len(en_train)

    en_eval = en[100000:]
    es_eval = es[100000:]

    train_ds = tf.data.Dataset.from_tensor_slices((en_train, es_train)).shuffle(BUFFER_SIZE)
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)

    eval_ds = tf.data.Dataset.from_tensor_slices((en_eval, es_eval)).shuffle(BUFFER_SIZE)
    eval_ds = eval_ds.batch(BATCH_SIZE, drop_remainder=True)
    
    pickle.dump(train_ds, "train_ds.pickle")
    pickle.dump(eval_ds, "eval_ds.pickle")
    pickle.dump(en_map, "en_map.pickle")
    pickle.dump(es_map, "es_map.pickle")

else:
    train_ds = pickle.load("train_ds.pickle")
    eval_ds = pickle.load("eval_ds.pickle")
    en_map = pickle.load("en_map.pickle")
    es_map = pickle.load("es_map.pickle")


print(colored("Train and Eval datasets created", "green"))

print(dataset.detokenize(en_train[0], en_map) + " " + dataset.detokenize(es_train[0], es_map))


BUFFER_SIZE = len(en_train)
STEPS_PER_EPOCH = BUFFER_SIZE // BATCH_SIZE
EMBEDDING_DIM = 256
UNITS = 1024
VOCAB_INP_SIZE = len(en_map.word_index)+1
VOCAB_OUT_SIZE = len(es_map.word_index)+1

NMTAttn = network.NMTAttn(VOCAB_INP_SIZE, VOCAB_OUT_SIZE, UNITS, n_encoder=3, n_decoder=3, n_attn_heads=1, dropout=0.03, mode='test')
NMTAttn.model(np.array([1, 2, 3]))