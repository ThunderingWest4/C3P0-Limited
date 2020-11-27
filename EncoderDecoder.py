from tensorflow.keras import layers as layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
from termcolor import colored
import sys

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

    @tf.function
    def train(self, data, epochs, steps_per_epoch):
        self.optimizer = keras.optimizers.Adam(0.01)
        self.loss_obj = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none')
        
        print(colored(f"Beginning training for {epochs} epochs", "green"))
            
        for ep in range(epochs):
            start = time.time()

            enchid = hidden_init(self.batch_size, self.units)
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(data.take(steps_per_epoch)):

                # print(inp.shape, targ.shape)
                batch_loss = 0
                for i in range(batch):
                    batch_loss += self.train_step(inp[i], targ[i], enchid)
                total_loss += batch_loss

                if(batch%100 == 0): 
                    print(f"Epoch {ep+1} | Batch {batch} | Loss {batch_loss}")

            print(colored(f"Epoch {ep+1} completed | Loss {total_loss/steps_per_epoch}", "green"))
            print(f"Time for epoch {ep+1}: {time.time()-start} seconds")
                
        print(colored("Training completed!", "green"))


            
    def loss(self, ans, pred):
        mask = tf.math.logical_not(tf.math.equal(ans, 0))
        #print(str(ans.shape) + " " + str(pred.shape))
        
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
                pred, dec_hid = self.decoder(decin)

                loss += EncDec.loss(self, targ[:,t], pred) # doing loss onto the predicted translation

                decin = tf.expand_dims(targ[:,t], 1) #teacher forcing - feeding in answer as input

            batch_loss = int(loss / targ.shape[1]) # total loss / n_examples = avg loss
            vs = self.encoder.trainable_variables + self.decoder.trainable_variables
            
            grads = tape.gradient(loss, vs) #finds gradient between loss and vars
            self.optimizer.apply_gradients(zip(grads, vs))
            
            return batch_loss


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
            super(EncDec.Decoder, self).__init__()
            self.batch_size = batch_size
            self.dec_units = units
            self.embedding = layers.Embedding(target_vocab, embed_dim)
            self.gru = layers.GRU(self.dec_units, 
                                   return_sequences=True, 
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
            self.fc = layers.Dense(target_vocab)
            
        def call(self, x):
            x = self.embedding(x)
            out, state = self.gru(x)
            out = tf.reshape(out, (-1, out.shape[2]))
            x = self.fc(out)
            x = tf.nn.log_softmax(x)

            return x, state

def hidden_init(batch, n_encoder):
    return tf.zeros((batch, n_encoder))

class EnDe2():
    def __init__(self, input_vocab, target_vocab, embedding_dim, units, ds, bat):
        
        
        #input -> encoder embedding -> encoder GRU -> save states -> decoder input -> decoder GRU w encoder states -> decoder LSTM -> dense w softmax activ
        
        enc_in = layers.Input(shape=(input_vocab,), batch_size=bat)
        enc_out = layers.Embedding(input_vocab, embedding_dim)(enc_in)
        enc_out, state = layers.GRU(
                                            units, 
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')(enc_out)
        
        dec_in = layers.Input(shape=(target_vocab,), batch_size=bat)
        dec_out = layers.Embedding(target_vocab, embedding_dim)(dec_in)
        dec_out = layers.GRU(units)(dec_out, initial_state=state)
        dec_out = layers.Dense(target_vocab, activation='softmax')(dec_out)
        
        self.model = keras.models.Model([enc_in, dec_in], dec_out)
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.model.summary()
        
        