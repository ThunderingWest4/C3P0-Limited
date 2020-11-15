from tensorflow.keras import layers as layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from math import log

class EncDec():

    def __init__(self, input_vocab, target_vocab, embedding_dim, units, batch_size, epochs):

        # constructs network
        
        hidden = hidden_init(batch_size, units)
        self.input_dim = input_vocab
        self.out_dim = target_vocab
        self.batch_size = batch_size
        self.embed_dim = embedding_dim
        self.units = units
        self.epochs = epochs

        self.encoder = EncDec.Encoder(input_vocab, embedding_dim, units, batch_size)
        self.decoder = EncDec.Decoder(target_vocab, embedding_dim, units, batch_size)

        #self.model = model

    def train(self):
        self.optimizer = keras.optimizers.Adam(0.01)
        self.loss_obj = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none')

        
            
    def loss(self, ans, pred):
        mask = tf.math.logical_not(tf.math.equal(ans, 0))
        loss = self.loss_obj(ans, pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss*=mask

        return tf.reduce_mean(loss)

    def train_step(self, inp, targ, hid):
        pass

    class Encoder(keras.Model):
        def __init__(self, input_vocab, embed_dim, encoder_units, batch_size):
            super(EncDec.Encoder, self).__init__()
            self.embed = layers.Embedding(input_vocab, embed_dim)
            self.gru = layers.GRU(
                    encoder_units, 
                    return_sequences=True,
                    return_state=True,
                    recurrent_initializer = 'glorot_uniform') # draws samples (initial weights) from uniform distr btwn -lim, lim where lim = sqrt( 6 / (num_inps + num_outs) )

        def call(self, x, hidden):
            output, state = self.gru(self.embed(x), initial_state=hidden)
            return output, state
    

    class Decoder(keras.Model):
        def __init__(self, target_vocab, embed_dim, units, batch_size):
            super(Decoder, self).__init__()
            self.batch_size = batch_size
            self.dec_units = units
            self.embedding = layers.Embedding(target_vocab, embed_dim)
            self.gru = layers.GRU(self.dec_units, 
                                   return_sequences=True, 
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
            self.fc = layers.Dense(target_vocab)
            
        def call(self, x, enc_out):
            x = self.embedding(x)
            out, state = self.gru(x)
            out = tf.reshape(out, (-1, out.shape[2]))
            x = self.fc(out)
            x = tf.nn.log_softmax(x)

            return x, state

def hidden_init(batch, n_encoder):
    return tf.zeros((batch, n_encoder))

