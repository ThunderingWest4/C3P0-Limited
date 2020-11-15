from tensorflow.keras import layers as layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
from termcolor import colored

class EncDec():

    def __init__(self, input_vocab, target_vocab, embedding_dim, units, batch_size, inpmap, targmap):

        # constructs network
        
        hidden = hidden_init(batch_size, units)
        self.input_dim = input_vocab
        self.out_dim = target_vocab
        self.batch_size = batch_size
        self.embed_dim = embedding_dim
        self.units = units
        self.inmap = inpmap
        self.outmap = targmap

        self.encoder = EncDec.Encoder(input_vocab, embedding_dim, units, batch_size)
        self.decoder = EncDec.Decoder(target_vocab, embedding_dim, units, batch_size)

        print(colored("Encoder and Decoder models created! Ready for training", "green"))


    def train(self, data, epochs, steps_per_epoch):
        self.optimizer = keras.optimizers.Adam(0.01)
        self.loss_obj = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none')

        for ep in range(epochs):
            start = time.time()

            enchid = hidden_init(self.batch_size, self.units)
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(data.take(steps_per_epoch)):
                
                batch_loss = self.train_step(inp, targ, enchid)
                total_loss += batch_loss

                if(batch%100 == 0): 
                    print(f"Epoch {ep+1} | Batch {batch} | Loss {batch_loss}")
            
            print(colored(f"Epoch {ep+1} completed | Loss {total_loss/steps_per_epoch}", "green"))
            print(f"Time for epoch {ep+1}: {time.time()-start} seconds")

        
            
    def loss(self, ans, pred):
        mask = tf.math.logical_not(tf.math.equal(ans, 0))
        loss = self.loss_obj(ans, pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss*=mask

        return tf.reduce_mean(loss)

    def train_step(self, inp, targ, hid):
        loss = 0

        with tf.GradientTape() as tape:
            encout, enchid = self.encoder(inp, hid)

            decin = tf.expand_dims([self.outmap.word_index['<s>']]*self.batch_size, 1)

            for t in range(1, targ.shape[1]):
                pred, dec_hid, _ = self.decoder(decin, enchid, encout)

                loss += loss(self, targ[:,t], pred) # doing loss onto the predicted translation

                decin = tf.expand_dims(targ[:,t], 1) #teacher forcing - feeding in answer as input

            batch_loss = int(loss / targ.shape[1]) # total loss / n_examples = avg loss
            vars = self.encoder.trainable_variables + self.decoder.trainable_variables
            grads = tape.gradient(loss, vars) #finds gradient between loss and vars
            self.optimizer.apply_gradients(zip(grads, vars))
            
            return grads


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

