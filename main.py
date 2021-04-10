import numpy as np
import tensorflow as tf
from dataset import *
from EncoderDecoder import *
from termcolor import colored
#import network
import os.path
import os
import json
import random


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

# # detect and init the TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# # instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# # instantiating the model in the strategy scope creates the model on the TPU
# with tpu_strategy.scope():

# Download the file
#path_to_zip = tf.keras.utils.get_file(
#    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
#    extract=True)

en, es, en_map, es_map = getData("../input/spaeng/spa.txt")
# en_f = []
# es_f = []

# # espanol dict is larger than the eng dict so we pad for even
# target_length = len(en[0]) if len(en[0]) > len(es[0]) else len(es[0])
# for i in range(len(en)): # they have the same amount of samples so we can use one index
#     en_f.append(np.append(en[i], [0]*(target_length - len(en[i]))))
#     es_f.append(np.append(es[i], [0]*(target_length - len(es[i]))))

#     en_train = en_f[0:100000]
#     es_train = es_f[0:100000]
#     en_eval = en_f[100000:]
#     es_eval = es_f[100000:]

BATCH_SIZE=32
#     BUFFER_SIZE = len(en_train)
EMBEDDING_DIM = 512
#     STEP_EPOCH = len(en_train)//BATCH_SIZE
EPOCHS = 30
UNITS = 1024
VOCAB_INP_SIZE = len(en[0])
VOCAB_OUT_SIZE = len(es[0])

# VOCAB_INP_SIZE = 51
# VOCAB_OUT_SIZE = 53

# train_ds = tf.data.Dataset.from_tensor_slices((en_train, es_train)).shuffle(BUFFER_SIZE)
# train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)

# eval_ds = tf.data.Dataset.from_tensor_slices((en_eval, es_eval)).shuffle(BUFFER_SIZE)
# eval_ds = eval_ds.batch(BATCH_SIZE, drop_remainder=True)

# full_ds = tf.data.Dataset.from_tensor_slices((en_f, es_f)).shuffle(BUFFER_SIZE)#.batch(BATCH_SIZE, drop_remainder=True)


# print(colored("Train and Eval datasets created", "green"))

# print(detokenize(en_train[0], en_map) + " | " + detokenize(es_train[0], es_map))

# for i in range(0, 52):
#     print(detokenize([i], en_map))

# for i in range(0, 54):
#     print(detokenize([i], es_map))

"""
EncDecModel = EncDec(VOCAB_INP_SIZE, VOCAB_OUT_SIZE, EMBEDDING_DIM, UNITS, BATCH_SIZE, en_map, es_map)
EncDecModel.train(train_ds, EPOCHS, STEP_EPOCH)
test = preprocess("hi how are you")
print(f"Preprocessed: {test}")
testseq = en_map.text_to_sequences(test)
encout, weights = EncDecModel.encoder(testseq, hidden_init(BATCH_SIZE, UNITS))
end = EncDecModel.decoder(encout)
print(f"Output: {es_map.sequences_to_texts(end)}")
"""
# print(full_ds)
en_config = en_map.get_config()
es_config = es_map.get_config()
n_eng = len(json.loads(en_config['word_counts']).keys()) # going into the config dict, taking the dict with the word counts, and taking the n of keys to get the overall number of words because apparently my other method is broken :/
n_es = len(json.loads(es_config['word_counts']).keys())

# dec_out = [np.append(x[1:], [0]) for x in es_f]

# method header
# sep_data(enc_og, dec_og, n_eng_words, n_es_words)

# en_f, es_f, dec_out = sep_data(en, es, n_eng, n_es)
# print(dec_out[0])

ED2 = EnDe2(VOCAB_INP_SIZE, VOCAB_OUT_SIZE, EMBEDDING_DIM, UNITS, BATCH_SIZE, n_eng, n_es)
print(colored("About to start training", "green"))
# print(len(en_f), len(es_f), len(dec_out))
# print(en_f[0:5], es_f[0], dec_out[0])
# ED2.train(en_f[0:200], es_f[0:200], dec_out[0:200], EPOCHS)

"""
THINGS TO ADD: 

PARAMS W/ DIM, BATCH SIZE, N_CLASSES, N_CHANNELS, SHUFFLE
PARTITION W N_IDS
LABELS W N_LABELS
train_gen = DataGenerator(partition['train'], labels, **params)
eval_gen = DataGenerator(partition['train'], labels, **params)

ED2.train(train_gen, eval_gen)

MAKE SURE THAT THE DATA GENERATOR TECHNIQUE WILL WORK WITH THE KERAS FUNCTIONAL API THING
IF NOT, SCREW AROUND WITH THE FUNCTIONAL API AND INPUTS UNTIL IT DOES

"""
list_ids = sorted(np.load('../input/spa-eng-separated/archive/list_ids.npy'))
train_gen = DataGenerator(list_ids, n_eng, n_es)
# print(train_gen.__getitem__(0))
ED2.train(train_gen, EPOCHS)

test_string = "hi how are you doing"
tokenized_string = en_map.texts_to_sequences(test_string)
result = ED2.model.predict(tokenized_string)
translated = es_map.sequences_to_texts(result)
print(f"Model translated {test_string} (eng) to {translated} (esp)")

# NMTAttn = network.NMTAttn(VOCAB_INP_SIZE, VOCAB_OUT_SIZE, UNITS, n_encoder=3, n_decoder=3, n_attn_heads=1, dropout=0.03, mode='test')
# NMTAttn.model(np.array([1, 2, 3]))