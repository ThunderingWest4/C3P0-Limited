import tensorflow as tf
import unicodedata
import re
import io

def getData(path):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    # n_total = 118964
    pairs = [[preprocess(x) for x in l.split('\t')] for l in lines]
    en, sp = zip(*pairs) #make tuples from pairs

    en_tensor, en_map = tokenize(en)
    sp_tensor, sp_map = tokenize(sp)

    return en_tensor, sp_tensor, en_map, sp_map

def uni_to_ascii(s) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn') # ensuring it's not an accent

def preprocess(w) -> str:
    w = uni_to_ascii(w.lower().strip())

    # make space between word and punct
    w = re.sub(r"([?.!,¿¡])", r" \1 ", w) # substitutes captured string thing (referenced w \1) with that thing + space
    # get rid of multiple space seq things
    w = re.sub(r'[" "]+', " ", w)

    # keep only letters and punct
    w = re.sub(r"[^a-zA-Z?.!,¿¡]+", " ", w).strip()

    # add start + end token for model
    w = '<s> ' + w + ' <e>'
    
    return w

def tokenize(lang) -> (tf.Tensor, tf.keras.preprocessing.text.Tokenizer):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer

def detokenize(sequence, lang_map) -> str:
    toRet = ""
    for x in sequence:
        if x != 0:
            toRet += lang_map.index_word[x] + " "
    return toRet