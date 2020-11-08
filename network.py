from tensorflow.keras import layers as tfl
import tensorflow.keras as tk
import tensorflow as tf
import tensorflow.lattice as tflat
import numpy as np
import extraLayers as el

class NMTAttn():

    def __init__(self, input_vocab, target_vocab, d_model, n_encoder, n_decoder, n_attn_heads, dropout, mode):
        # constructs network
        inp_encoder = self.input_encoder_fn(input_vocab, d_model, n_encoder)
        pre_attention_dec = self.pre_attn_decoder(mode, target_vocab, d_model)

        model = tk.Sequential(

            # get the Select thing working - [0,1,0,1]

            tflat.ParallelCombination(inp_encoder, pre_attention_dec), 
            tfl.Lambda(function=pre_attn_inp, output_shape=4), 
            tfl.Residual(tfl.Attention(d_model, n_attn_heads, dropout, mode)),

            # Another Select layer - [0, 2]

            [tfl.LSTM(d_model) for _ in range(n_decoder)], 

            tfl.Dense(target_vocab),
            tf.nn.log_softmax()

        )

        return model

    def input_encoder_fn(input_vocab_size, d_model, n_encoder) -> tk.Sequential:
        return tk.Sequential(
            tfl.Embedding(input_vocab_size, d_model), 
            [tfl.LSTM(d_model) for i in range(n_encoder)]
        )

    def pre_attn_decoder(mode, target_vocab_size, d_model) -> tk.Sequential:
        return tk.Sequential(
            tfl.ShiftRight(mode=mode), 
            tfl.Embedding(target_vocab_size, d_model), 
            tfl.LSTM(d_model)
        )

    def pre_attn_inp(encoder_activ, decoder_activ, inps) -> (np.array, np.array, np.array, np.array):
        keys = encoder_activ
        vals = encoder_activ

        queries = decoder_activ

        mask = inps != 0
        mask = np.reshape(mask, (mask.shape[0], 1, 1, mask.shape[1]))
        mask += np.zeros((1, 1, decoder_activ.shape[1], 1))

        return queries, keys, vals, mask

    