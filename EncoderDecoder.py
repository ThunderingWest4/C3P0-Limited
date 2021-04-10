import tensorflow.keras as keras
import keras.layers as layers
import numpy as np

class EnDe2():
    def __init__(self, inp_size, targ_size, embedding_dim, units, bat, input_vocab, target_vocab):
        
        self.bat_size = bat
        
        # input -> encoder embedding -> encoder GRU -> 
        # save states -> decoder input -> decoder GRU w encoder states -> 
        # decoder LSTM -> dense w softmax activ
        
#        # e_temp: (32, 51, 12933) | np.ndarray | encoder input
#        # d_temp: (32, 53, 24794) | np.ndarray | decoder input
#        # o_temp: (32, 53, 24794) | np.ndarray | decoder output
        
        enc_in = layers.Input(shape=(inp_size, input_vocab), batch_size=bat)
#         print(input_vocab, embedding_dim)
#         enc_out = layers.Embedding(input_dim=input_vocab+1, output_dim=embedding_dim//2)(enc_in)
        enc = keras.layers.LSTM(embedding_dim, return_state=True)
        enc_out, state_h, state_c = enc(enc_in)
#         enc_out, state = layers.GRU(units//2, 
#                                     return_state=True,
#                                     recurrent_initializer='glorot_uniform')(enc_out)

        enc_states = [state_h, state_c]

        dec_in = layers.Input(shape=(targ_size, target_vocab), batch_size=bat)
#         print(target_vocab, embedding_dim
#         dec_out = layers.Embedding(input_dim=target_vocab+1, output_dim=embedding_dim//2)(dec_in)
#         dec_out = layers.GRU(units//2)(dec_out, initial_state=state)
#         dec_out = layers.Dense(targ_size, activation='softmax')(dec_out)
        dec_lstm = keras.layers.LSTM(embedding_dim, 
                                     return_sequences=True, 
                                     return_state=True)
        dec_out, _, _ = dec_lstm(dec_in, initial_state=enc_states)
        dec_dense = keras.layers.Dense(target_vocab, activation='softmax')
        dec_out = dec_dense(dec_out)
        
        self.model = keras.models.Model([enc_in, dec_in], dec_out)
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.model.summary()
        
    def train(self, train_gen, epochs):
        self.model.compile(
            optimizer="rmsprop", 
            loss="categorical_crossentropy", 
            metrics=["accuracy"]
        )
#         self.model.fit(
#             [e, d_in],
#             d_out,
#             batch_size=self.bat_size,
#             epochs=epochs,
#             validation_split=0.2, 
#         )
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="s2s",
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )

        for i in range(epochs):
            """
            APPARENTLY callbacks doesn't work so the data doesn't save
            and considering kaggle sessions last 9 hours max and this takes 2 hours per epoch, we *need* checkpoints very badly
            so yeah
            crappy but hopefully workable solution
            """
            self.model.fit(
                x=train_gen, # train_gen will return tuple ([encoder_in, decoder_in], decoder_out)
                epochs=1, 
                workers = 5,
                use_multiprocessing = True, 
                callbacks = [model_checkpoint_callback]

            )
            
            self.model.save("s2s")
        