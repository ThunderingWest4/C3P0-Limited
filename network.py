from tensorflow.keras import layers as tfl
import tensorflow.keras as tk
import tensorflow as tf
#import tensorflow.lattice as tflat
import numpy as np

# Architecture from: https://towardsdatascience.com/understanding-neural-machine-translation-encoder-decoder-architecture-80f205643ba4
# Bahdanau Attention: https://blog.floydhub.com/attention-mechanism/ 

class NMTAttn():

    def __init__(self, input_vocab, target_vocab, d_model, n_encoder, n_decoder, n_attn_heads, dropout, mode):
        # constructs network
        encoder = self.encoder(input_vocab, d_model, n_encoder)
        decoder = self.decoder(target_vocab, d_model)

        model = tk.Sequential(

            # ENCODER SECTION
            # getting layers from encoder block to overall model
            [layer for layer in encoder.layers],

            # ATTENTION SECTION
            # get the Select thing working - [0,1,0,1]

            #tflat.ParallelCombination(inp_encoder, pre_attention_dec), 
            tfl.Lambda(function=self.pre_attn_inp, output_shape=4), 
            tfl.Residual(tfl.Attention(d_model, n_attn_heads, dropout, mode)),

            # Another Select layer - [0, 2]

            # DECODER SECTION
            # getting layers from decoder sequential into overall model sequential
            [layer for layer in decoder.layers] 

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
        return tk.Sequential(
            tfl.Embedding(target_vocab_size, d_model), 
            tfl.GRU(
              d_model, 
              return_sequences=True,
              return_state=True,
              recurrent_initializer = 'glorot_uniform' # draws samples (initial weights) from uniform distr btwn -lim, lim where lim = sqrt( 6 / (num_inps + num_outs) )
            ),
            tfl.Dense(target_vocab_size),
            tf.nn.log_softmax()
        )

    def Attention():
      pass



    def pre_attn_inp(self, encoder_activ, decoder_activ, inps) -> (np.array, np.array, np.array, np.array):
        keys = encoder_activ
        vals = encoder_activ

        queries = decoder_activ

        mask = inps != 0
        mask = np.reshape(mask, (mask.shape[0], 1, 1, mask.shape[1]))
        mask += np.zeros((1, 1, decoder_activ.shape[1], 1))

        return queries, keys, vals, mask


"""
SHIFTRIGHT AND _ZERO_PAD METHOD COPIED AND MOFIDIED FROM TRAX GITHUB REPOSITORY
"""
def _zero_pad(x, pad, axis):
  """Helper for jnp.pad with 0s for single-axis case."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = pad  # Padding on axis.
  return np.pad(x, pad_widths, mode='constant',
                 constant_values=x.dtype.type(0))

def ShiftRight(n_positions=1, mode='train'):
  """Returns a layer that can insert padding to shift the input sequence.
  Args:
    n_positions: Number of positions to shift the input sequence rightward;
        initial positions freed by the shift get padded with zeros.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  def f(x):
    if mode == 'predict':
      return x
    padded = _zero_pad(x, (n_positions, 0), 1)
    return padded[:, :-n_positions]
    
  return tfl.Lambda(name=f'ShiftRight({n_positions})', function=f)