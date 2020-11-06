import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import dataset
from termcolor import colored

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

ds = dataset.getData()

for ex in ds.take(10):
    print(colored(ex['en'], "blue"))
    print(colored(ex['es'], "red"))