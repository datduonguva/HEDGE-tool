import os
import json
import copy
import tensorflow as tf
import numpy as np

from hedgetool import HEDGE 


MODEL_PATH = input("your model path: ")
VOCAB_PATH = input("your vocab path: ") 

with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

def encode_sentence(sentence, max_length, padding=-1):
    return np.array(
        [
            (
                [vocab.get(word, padding) for word in sentence.split(" ")] + [padding]*max_length
            )[:max_length]
        ]
    )

def main():
    """
    To generate the Hierachical Explaination of a model, please extend the HEDGE class and
    overwrite the `predict_prob` method to return the softmax output given the encoded
    input

    This step is different for each model.
    """
    # load the pretrained model
    model = tf.keras.models.load_model(MODEL_PATH)

    padding = vocab['#'] 
    max_length = 70

    sentence = 'i think the song sound very interesting'
    encoded_sentence = encode_sentence(sentence, max_length, padding)


    # extend the HEDGE class and overwrite the `predict_prob` method
    class MyHEDGE(HEDGE):
        def __init__(self, *arg):
            super(MyHEDGE, self).__init__(*arg)

        def predict_prob(self, encoded_sentence):
            return self.model.predict(encoded_sentence)

    ids_to_tokens = {i:k for k, i in vocab.items()}

    #initialize your extended HEDGE object
    hedge = MyHEDGE(model, max_length, padding, ids_to_tokens)

    # calling the main algorithm 
    P_history, spans, contributions = hedge.main_algorithm(encoded_sentence)
    
    # visualize the outcome
    hedge.visualize(encoded_sentence, P_history, contributions)

if __name__ == '__main__':
    main()
