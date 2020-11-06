import numpy as np
import tensorflow as tf
import dataset
from termcolor import colored
import network
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
# Download the file
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

en, es = dataset.getData(os.path.dirname(path_to_zip) + "\\spa-eng\\spa.txt")

en_train = en[0:100000]
es_train = es[0:100000]

en_eval = en[100000:]
es_eval = es[100000:]

NMTAttn = network.NMTAttn()