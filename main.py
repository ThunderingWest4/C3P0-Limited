import numpy as np
import tensorflow as tf
import dataset
import EncoderDecoder
from termcolor import colored
import network
import os.path
import os


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

# detect and init the TPU
#tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#tf.config.experimental_connect_to_cluster(tpu)
#tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
#tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
#with tpu_strategy.scope():

# Download the file
#path_to_zip = tf.keras.utils.get_file(
#    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
#    extract=True)

en, es, en_map, es_map = dataset.getData("../input/spaeng/spa.txt")

en_train = en[0:100000]
es_train = es[0:100000]
en_eval = en[100000:]
es_eval = es[100000:]

BATCH_SIZE=64
BUFFER_SIZE = len(en_train)
EMBEDDING_DIM = 512
STEP_EPOCH = len(en_train)//BATCH_SIZE
EPOCHS = 10
UNITS = 1024
VOCAB_INP_SIZE = len(en_map.word_index)+1
VOCAB_OUT_SIZE = len(es_map.word_index)+1

train_ds = tf.data.Dataset.from_tensor_slices((en_train, es_train)).shuffle(BUFFER_SIZE)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)

eval_ds = tf.data.Dataset.from_tensor_slices((en_eval, es_eval)).shuffle(BUFFER_SIZE)
eval_ds = eval_ds.batch(BATCH_SIZE, drop_remainder=True)

full_ds = tf.data.Dataset.from_tensor_slices((en, es)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


print(colored("Train and Eval datasets created", "green"))

print(dataset.detokenize(en_train[0], en_map) + " | " + dataset.detokenize(es_train[0], es_map))

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

ED2 = EncoderDecoder.EnDe2(VOCAB_INP_SIZE, VOCAB_OUT_SIZE, EMBEDDING_DIM, UNITS, full_ds, BATCH_SIZE)

# NMTAttn = network.NMTAttn(VOCAB_INP_SIZE, VOCAB_OUT_SIZE, UNITS, n_encoder=3, n_decoder=3, n_attn_heads=1, dropout=0.03, mode='test')
# NMTAttn.model(np.array([1, 2, 3]))