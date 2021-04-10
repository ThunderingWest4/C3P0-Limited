import numpy as np
import tensorflow as tf
from EncoderDecoder import *
from dataset import *
import json
from termcolor import colored 


def separate_model(model_save_path, latent_dim): 
    model = tf.keras.models.load_model(model_save_path)

    enc_in = model.input[0]
    enc_out, s_h_enc, s_c_enc = model.layers[2].output # lstm_1, aka first layer after the inputs
    enc_states = [s_h_enc, s_c_enc]
    enc_model = keras.Model(enc_in, enc_states) # takes in the encoder inputs, returns encoder states

    dec_in = model.input[1] 
    dec_s_h = keras.Input(shape=(latent_dim,), name="input_3")
    dec_s_c = keras.Input(shape=(latent_dim,), name="input_4")
    d_s_inp = [dec_s_h, dec_s_c]
    dec_lstm = model.layers[3]

    dec_out, s_h_d, s_c_d = dec_lstm (
        dec_in, initial_state = d_s_inp
    )

    d_states = [s_h_d, s_c_d]
    dec_dense = model.layers[4]
    dec_out = dec_dense(dec_out)
    dec_model = keras.Model(
        [dec_in] + d_s_inp, [dec_out] + d_states
    )

    return enc_model, dec_model

def modify_input(data, batch_size, max_eng, n_eng):
    max_eng = 51 # longest english sequence

    inp = np.zeros((batch_size, max_eng, n_eng), dtype=int)

    for t, x in enumerate(data):

        inp[0][t][x] = 1.0

    return inp


latent_dim = 512 
batch_size = 32

enc_model, dec_model = separate_model('semi_en_es_s2s', latent_dim)

# print("Model architecture summary: ")
# model.summary()

en, _, en_map, es_map = getData("spa.txt")

# getting number of decoder tokens, aka number of words in dict
es_config = es_map.get_config()
num_decoder_tokens = len(json.loads(es_config['word_counts']).keys())

# getting input
prelim = input("Enter text to translate: \n")

# preprocessing
# <s>, <e> tags added, spaces inserted btwn words
processed = preprocess(prelim).split(" ")
# makes list of words formatted with an item for each word, tag, punctuation mark, etc

# taking english input text and tokenizing it to a Tensor sequence
tokenized = en_map.texts_to_sequences(processed)
print(tokenized)

formatted = modify_input( tokenized, batch_size, len(en[0]), len(json.loads(en_map.get_config()['word_counts']).keys()) )

# now that the text is tokenized, we can start translating with the RNN
tokens = [] # hardcoded in length 53, should be programatically done later
end = False
start_token = es_map.texts_to_sequences(['<s>'])[0] # token for SoS Start of String
end_token = es_map.texts_to_sequences(['<e>'])[0] # token for End of String EoS
decoded_seq = ""

print("################################################ \nPassing tokenized data into the encoder \n################################################")
# getting the the initial LSTM states, pass in to the decoder model
states = enc_model.predict(formatted)

# Generate empty target sequence of length 1.
target_seq = np.zeros((batch_size, 53, num_decoder_tokens))
# Populate the first character of target sequence with the start character.
target_seq[0, 0, start_token] = 1.0

print("################################################ \nPassing encoded states into the decoder to translate \n################################################")
counter = 0
while not end:
    out, h, c = dec_model.predict([target_seq] + states)

    token = np.argmax(out[0, -1, :])
    tokens.append(token)
    
    print(token)
    ch = es_map.sequences_to_texts([[token]])[0]
    decoded_seq += ch

    if ch == "<e>" or token == end_token or len(tokens) > 53:
        end = True

    # target_seq = np.zeros((batch_size, 53, num_decoder_tokens))
    target_seq[0][counter][token] = 1.0

    states = [h, c]
    
    counter += 1

print("Translation complete!")
print("---------------------")

print(f"Input text: {prelim}")

print(f"Translated text: {decoded_seq}")
