import os
import json
import copy
import tensorflow as tf
import numpy as np

from hedgetool import HEDGE 



MODEL_PATH = "/home/datduong/gdrive/projects/c11_text_interaction/01/"
DATA_PATH = (
    "/home/datduong/gdrive/CS/ML/SST/archive/SST2-Data/"
    "SST2-Data/stanfordSentimentTreebank/stanfordSentimentTreebank"
)


with open(os.path.join(DATA_PATH, "vocab.json"), "r") as f:
    vocab = json.load(f)

def encode_sentence(sentence, max_length, padding=-1):
    return np.array(
        [
            (
                [vocab.get(word, padding) for word in sentence.split(" ")] + [padding]*max_length
            )[:max_length]
        ]
    )

if __name__ == '__main__':

    model = tf.keras.models.load_model(MODEL_PATH)


    padding = vocab['#'] 
    max_length = 70

    class MyHEDGE(HEDGE):
        def __init__(self, *arg):
            super(MyHEDGE, self).__init__(*arg)

        def predict_prob(self, encoded_sentence):
            return self.model.predict(encoded_sentence)

    ids_to_tokens = {i:k for k, i in vocab.items()}
    hedge = MyHEDGE(model, max_length, padding, ids_to_tokens)

    sentence = 'i think the song sound very interesting'

    encoded_sentence = encode_sentence(sentence, max_length, padding)
    P_history, spans, contributions = hedge.main_algorithm(encoded_sentence)
    
    hedge.visualize(encoded_sentence, P_history, contributions)
