# Saving Data
import numpy as np
import os

def generate_data():
    # Clearing output to make sure we don't crash the notebook by writing too much to disc lol
    # for file in os.walk("./"):
    #     print(file)
    #     os.remove(file)

    DataGenNeeded = False
    batch = 32
    # leaving code so that i can generate new dataset thing whenever i need

    if DataGenNeeded:

        en, es, en_map, es_map = getData("../input/spaeng/spa.txt")

        list_ids = np.array([])
        counter = 0

        for i in range(0, len(en), batch):
            if i+batch < len(en):
                np.save(f"{counter}_en.npy", en[i:i+batch])
                np.save(f"{counter}_es.npy", es[i:i+batch])
                np.append(list_ids, counter)
                print(f"Saved values {i} to {i+batch} in file with id {counter}")

            else: 
                continue
                # ignore the files with batch size <32

            counter += 1

        np.save("list_ids.npy", list_ids)
        print("Data Save complete")

def sep_data(enc_og, dec_og, n_eng_words, n_es_words):
    
    
    n_pairs = len(enc_og)
    max_eng = len(enc_og[0])
    max_es = len(dec_og[0])
    
    enc_inp = np.zeros((n_pairs, max_eng, n_eng_words), dtype=int)
    dec_inp = np.zeros((n_pairs, max_es, n_es_words), dtype=int)
    dec_out = np.zeros((n_pairs, max_es, n_es_words), dtype=int) #going to be same dims as dec_inp but off by a timestep
    # so a specific index in dec_out would correspond to index-1 in dec_inp
    
    # actually putting data into the arrays no
    
    for i, (inp, targ) in enumerate(zip(enc_og, dec_og)):
        
        for t, c in enumerate(inp): # english data
            enc_inp[i, t, c] = 1 # c represents the index/token/code thing for a word so we're basically going to that index
            # and saying "hey, there's x word present at this timestamp in this data pair"
            # no need to add spacces bc it's doing words so by default, after predictions and joined, spaces will be added in
        
        for t, c in enumerate(targ): # spanish data
            dec_inp[i, t, int(c)] = 1
            if t>0:
                # dec out will be ahead by 1 timestep and not include first word
                # so second word of input will be first of output
                dec_out[i, t-1, int(c)] = 1
                         
    
    return (enc_inp, dec_inp, dec_out)