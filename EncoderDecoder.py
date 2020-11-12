from tensorflow.keras import layers as tfl
import tensorflow.keras as tk
import tensorflow as tf
#import tensorflow.lattice as tflat
import numpy as np
from math import log

# Architecture from: https://towardsdatascience.com/understanding-neural-machine-translation-encoder-decoder-architecture-80f205643ba4
# Bahdanau Attention: https://blog.floydhub.com/attention-mechanism/ 

class EncDec():

    def __init__(self, input_vocab, target_vocab, d_model):
        # constructs network
        encoder = self.encoder(input_vocab, d_model)
        decoder = self.decoder(target_vocab, d_model)

        model = tk.Sequential(

            # ENCODER SECTION
            # getting layers from encoder block to overall model
            [layer for layer in encoder.layers],

            # DECODER SECTION
            # getting layers from decoder sequential into overall model sequential
            [layer for layer in decoder.layers], 
            tfl.Lambda(function=tf.nn.log_softmax)

        )

        self.model = model

    def encoder(self, input_vocab_size, d_model) -> tk.Sequential:
        # LSTM VS GRUs
        # LSTMs are older and have 3 gates (input, output, forget) vs the GRU's 2 gates (update, reset) and greater efficiency
        return tk.Sequential(
            tfl.Embedding(input_vocab_size, d_model), 
            tfl.GRU(
              d_model, 
              return_sequences=True,
              return_state=True,
              recurrent_initializer = 'glorot_uniform' # draws samples (initial weights) from uniform distr btwn -lim, lim where lim = sqrt( 6 / (num_inps + num_outs) )
            )
        )

    def decoder(self, target_vocab_size, d_model) -> tk.Sequential:
        toret = tk.Sequential()
        toret.add(tfl.Embedding(target_vocab_size, d_model)) 
        toret.add(tfl.GRU(
            d_model, 
            return_sequences=True,
            return_state=True,
            recurrent_initializer = 'glorot_uniform')) # draws samples (initial weights) from uniform distr btwn -lim, lim where lim = sqrt( 6 / (num_inps + num_outs) )
        toret.add(tfl.Dense(target_vocab_size))
        #tfl.Softmax()            
        return toret
        


